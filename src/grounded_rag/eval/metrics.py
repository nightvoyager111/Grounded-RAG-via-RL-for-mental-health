"""Programmatic eval metrics for grounded RAG.

Metric semantics (MVP — whole-answer granularity):
- citation_precision: of the [chunk_id] markers appearing in the answer,
  fraction that refer to a chunk actually in the retrieved set. Guards
  against fabricated citations.
- citation_recall: of the retrieved chunks provided to the generator,
  fraction that were cited at least once in the answer. Measures whether
  the generator used the evidence at all (vs. answering from parametric
  knowledge and ignoring the passages).
  NOTE: This is a proxy. Classical citation recall is per-claim and
  requires claim decomposition + a verifier. Sentence-level is a stretch
  upgrade per CLAUDE.md.
- copy_rate: fraction of the answer's n-grams (default n=8) that also
  appear in at least one cited passage. High values → the generator is
  copying rather than composing (a known GRPO reward-hacking mode).
- abstention_probe: 1 if the answer matches an IDK/evasion pattern, else 0.
  Aggregated across a dataset, gives the abstention rate.
- groundedness_rate: delegates to a verifier callable
  (passages, answer) -> [0, 1]. With StubVerifier this is meaningless;
  the number becomes real once the NLI verifier is built in step 4.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Sequence

CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")

_ABSTENTION_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\bi (?:cannot|can't|am unable to) answer\b",
    r"\binsufficient (?:information|evidence|context)\b",
    r"\bnot (?:enough|sufficient) (?:information|evidence|context)\b",
    r"\bno (?:information|evidence) (?:is )?(?:available|provided)\b",
    r"\bthe (?:passages|sources|context) (?:do not|don't) (?:contain|mention|say|provide)\b",
    r"\bunable to determine\b",
]
_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


def extract_citations(answer: str) -> List[str]:
    """Return the ordered list of chunk_ids cited in the answer.

    We keep duplicates so callers can measure citation density if they want;
    precision/recall dedupe internally."""
    return CITATION_RE.findall(answer)


def citation_precision(answer: str, retrieved_ids: Sequence[str]) -> Optional[float]:
    """None when the answer contains no citations — undefined, not zero."""
    cits = extract_citations(answer)
    if not cits:
        return None
    valid_set = set(retrieved_ids)
    hits = sum(1 for c in cits if c in valid_set)
    return hits / len(cits)


def citation_recall(answer: str, retrieved_ids: Sequence[str]) -> Optional[float]:
    """None when no chunks were retrieved."""
    if not retrieved_ids:
        return None
    cited = set(extract_citations(answer))
    hit = sum(1 for r in retrieved_ids if r in cited)
    return hit / len(retrieved_ids)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def copy_rate(
    answer: str,
    cited_passages: Sequence[str],
    n: int = 8,
) -> Optional[float]:
    """Fraction of answer n-grams that appear in any cited passage.

    Returns None when the answer has fewer than n tokens (n-gram undefined)."""
    ans_toks = _tokenize(answer)
    ans_ngrams = _ngrams(ans_toks, n)
    if not ans_ngrams:
        return None
    passage_ngrams = set()
    for p in cited_passages:
        passage_ngrams.update(_ngrams(_tokenize(p), n))
    if not passage_ngrams:
        return 0.0
    hits = sum(1 for g in ans_ngrams if g in passage_ngrams)
    return hits / len(ans_ngrams)


def abstention_probe(answer: str) -> int:
    return 1 if _ABSTENTION_RE.search(answer) else 0


# Split the answer into sentences on . ! ? (followed by space or end-of-string).
# Per the system prompt, a well-formed sentence ends "...text [chunk_id]." — so
# we split BEFORE the period matters and check whether "...text [chunk_id]"
# ends in a bracketed valid chunk.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Matches a trailing "[chunk_id]" (possibly followed by the terminal punctuation).
_TAIL_CITE_RE = re.compile(r"\[([^\[\]]+?)\]\s*[.!?]*\s*$")


def citation_compliance(answer: str, retrieved_ids: Sequence[str]) -> Optional[float]:
    """Fraction of the answer's sentences that end with a valid [chunk_id]
    from the retrieved set. Sharper than citation_recall for RL: gives one
    signal per sentence instead of one aggregate per answer, so a policy
    that adds even one more citation gets rewarded proportionally.

    Returns None if the answer has no scorable sentences (empty, or matches
    the abstention pattern — abstentions shouldn't be citation-scored).
    """
    if abstention_probe(answer):
        return None
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(answer.strip()) if p.strip()]
    if not parts:
        return None
    valid = set(retrieved_ids)
    hits = 0
    for p in parts:
        m = _TAIL_CITE_RE.search(p)
        if m and m.group(1) in valid:
            hits += 1
    return hits / len(parts)


def groundedness_rate(
    passages: Sequence[str],
    answer: str,
    verifier: Callable[[Sequence[str], str], float],
) -> float:
    """Delegate to the verifier callable. Interface is fixed here so the
    real NLI verifier can drop in without changing eval code."""
    return float(verifier(passages, answer))


def score_example(
    *,
    question: str,
    answer: str,
    retrieved_ids: Sequence[str],
    retrieved_texts: Sequence[str],
    verifier: Callable[[Sequence[str], str], float],
    copy_ngram: int = 8,
) -> dict:
    """Compute all metrics for one (question, answer, retrieved) tuple.

    Returns a dict of metric_name → value. None means "undefined for this
    example" (e.g. no citations, no retrieved chunks); the aggregator should
    skip None values, not treat them as 0."""
    id_to_text = dict(zip(retrieved_ids, retrieved_texts))
    cited_ids = [c for c in extract_citations(answer) if c in id_to_text]
    cited_texts = [id_to_text[c] for c in cited_ids]

    return {
        "question": question,
        "answer": answer,
        "citation_precision": citation_precision(answer, retrieved_ids),
        "citation_recall": citation_recall(answer, retrieved_ids),
        "copy_rate": copy_rate(answer, cited_texts, n=copy_ngram),
        "abstention": abstention_probe(answer),
        "groundedness_rate": groundedness_rate(retrieved_texts, answer, verifier),
    }


def aggregate(rows: Iterable[dict]) -> dict:
    """Mean each metric across rows, skipping Nones. abstention averages
    to an abstention rate."""
    rows = list(rows)
    keys = ["citation_precision", "citation_recall", "copy_rate", "abstention", "groundedness_rate"]
    out = {"n_examples": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
        out[f"{k}_n"] = len(vals)
    return out
