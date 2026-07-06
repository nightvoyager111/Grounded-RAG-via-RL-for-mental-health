"""Unit tests for the eval metrics and runner.

Same patterns as test_retrieval.py:
- Stub external systems (retriever, generator, verifier) with fakes.
- Use tmp_path for on-disk artifacts.
- One behavior per test, name says what breaks if it fails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import pytest

from src.grounded_rag.eval.metrics import (
    abstention_probe,
    aggregate,
    citation_precision,
    citation_recall,
    copy_rate,
    extract_citations,
    groundedness_rate,
    score_example,
)
from src.grounded_rag.eval.runner import EvalConfig, run_eval
from src.grounded_rag.generation.prompt import Passage
from src.grounded_rag.verifier import StubVerifier


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


def test_extract_citations_finds_bracketed_ids():
    ans = "GAD involves excessive worry [icd11:6B00] and restlessness [nimh:gad:1]."
    assert extract_citations(ans) == ["icd11:6B00", "nimh:gad:1"]


def test_extract_citations_returns_empty_when_no_brackets():
    assert extract_citations("No citations here.") == []


def test_extract_citations_preserves_duplicates_and_order():
    assert extract_citations("[a] and [b] then [a] again") == ["a", "b", "a"]


# ---------------------------------------------------------------------------
# citation_precision / citation_recall
# ---------------------------------------------------------------------------


def test_citation_precision_full_match():
    ans = "Foo [a] bar [b]."
    assert citation_precision(ans, ["a", "b", "c"]) == 1.0


def test_citation_precision_partial_match_flags_fabricated_ids():
    ans = "Foo [a] bar [ghost]."
    assert citation_precision(ans, ["a", "b"]) == 0.5


def test_citation_precision_is_none_when_no_citations():
    """Undefined for uncited answers — should NOT collapse to 0."""
    assert citation_precision("no citations", ["a"]) is None


def test_citation_recall_measures_evidence_usage():
    ans = "Only [a] cited."
    assert citation_recall(ans, ["a", "b"]) == 0.5


def test_citation_recall_is_none_when_no_retrieval():
    assert citation_recall("[a]", []) is None


# ---------------------------------------------------------------------------
# copy_rate
# ---------------------------------------------------------------------------


def test_copy_rate_full_verbatim_is_one():
    passage = "the quick brown fox jumps over the lazy dog every morning"
    assert copy_rate(passage, [passage], n=3) == 1.0


def test_copy_rate_novel_composition_is_zero():
    ans = "alpha beta gamma delta epsilon zeta eta theta"
    passage = "totally different words unrelated to the answer content whatsoever"
    assert copy_rate(ans, [passage], n=3) == 0.0


def test_copy_rate_none_when_answer_shorter_than_ngram():
    """copy_rate on a 3-word answer with n=8 has no ngrams to score."""
    assert copy_rate("too short", ["some passage"], n=8) is None


def test_copy_rate_zero_when_no_cited_passages():
    """Answer has ngrams but nothing was cited → 0, not None."""
    ans = "one two three four five six seven eight nine"
    assert copy_rate(ans, [], n=4) == 0.0


# ---------------------------------------------------------------------------
# abstention_probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I do not know based on the passages.",
        "The passages do not contain that information.",
        "Insufficient information to answer.",
        "Unable to determine from the sources.",
    ],
)
def test_abstention_probe_hits_idk_patterns(answer):
    assert abstention_probe(answer) == 1


def test_abstention_probe_zero_for_confident_answer():
    assert abstention_probe("GAD is characterized by excessive worry [a].") == 0


# ---------------------------------------------------------------------------
# groundedness_rate — delegates to verifier
# ---------------------------------------------------------------------------


def test_groundedness_rate_delegates_to_callable():
    calls = []

    def verifier(passages, answer):
        calls.append((tuple(passages), answer))
        return 0.73

    score = groundedness_rate(["p1", "p2"], "an answer", verifier)
    assert score == 0.73
    assert calls == [(("p1", "p2"), "an answer")]


def test_groundedness_rate_with_stub_verifier():
    assert groundedness_rate(["p"], "a", StubVerifier(0.5)) == 0.5


# ---------------------------------------------------------------------------
# score_example + aggregate
# ---------------------------------------------------------------------------


def test_score_example_returns_all_metric_keys():
    row = score_example(
        question="q",
        answer="Answer citing [a] and [ghost].",
        retrieved_ids=["a", "b"],
        retrieved_texts=["passage A text", "passage B text"],
        verifier=StubVerifier(0.5),
    )
    for k in ("citation_precision", "citation_recall", "copy_rate",
              "abstention", "groundedness_rate"):
        assert k in row
    assert row["citation_precision"] == 0.5   # 1 of 2 cited ids valid
    assert row["citation_recall"] == 0.5      # 1 of 2 retrieved cited
    assert row["abstention"] == 0
    assert row["groundedness_rate"] == 0.5


def test_aggregate_skips_none_values():
    rows = [
        {"citation_precision": 1.0, "citation_recall": 0.5, "copy_rate": None,
         "abstention": 0, "groundedness_rate": 0.9},
        {"citation_precision": None, "citation_recall": 1.0, "copy_rate": 0.2,
         "abstention": 1, "groundedness_rate": 0.4},
    ]
    agg = aggregate(rows)
    assert agg["n_examples"] == 2
    assert agg["citation_precision"] == 1.0     # only the non-None row counted
    assert agg["citation_precision_n"] == 1
    assert agg["citation_recall"] == 0.75
    assert agg["abstention"] == 0.5             # 1/2 abstained → abstention rate
    assert agg["copy_rate"] == 0.2


# ---------------------------------------------------------------------------
# run_eval — end-to-end with fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeChunk:
    chunk_id: str
    title: str
    text: str


class _FakeRetriever:
    def __init__(self, chunks: List[_FakeChunk]):
        self.chunks = chunks

    def retrieve(self, query):
        return self.chunks


class _FakeGenerator:
    """Emits a deterministic answer that cites the first two retrieved ids."""

    def __init__(self):
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        cits = " ".join(f"[{p.chunk_id}]" for p in passages[:2])
        return f"Answer to {question} {cits}."


def test_run_eval_writes_rows_and_report(tmp_path):
    chunks = [
        _FakeChunk("a", "Title A", "passage A text here"),
        _FakeChunk("b", "Title B", "passage B text here"),
    ]
    cfg = EvalConfig(copy_ngram=4, output_dir=str(tmp_path / "out"))
    gen = _FakeGenerator()

    report = run_eval(
        questions=["What is A?", "What is B?"],
        retriever=_FakeRetriever(chunks),
        generator=gen,
        verifier=StubVerifier(0.5),
        cfg=cfg,
    )
    assert gen.calls == 2
    assert report["n_examples"] == 2

    rows_path = tmp_path / "out" / "rows.jsonl"
    report_path = tmp_path / "out" / "report.json"
    assert rows_path.exists()
    assert report_path.exists()

    rows = [json.loads(l) for l in open(rows_path)]
    assert len(rows) == 2
    # Both fake citations point at retrieved ids → precision should be 1.0.
    assert all(r["citation_precision"] == 1.0 for r in rows)
