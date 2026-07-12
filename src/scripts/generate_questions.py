"""Auto-generate factual questions from corpus chunks via Cohere.

CLAUDE.md Act 2 prep: 25 questions was fine for a demo; the citation-repair
GRPO run needs a wider prompt pool for stability and eval power. This
script samples chunks from the corpus, asks command-r for one factual
question per chunk, dedupes against the existing baseline set, and writes
an expanded JSONL.

Design notes:
- We DO NOT touch data/qa/baseline_questions.jsonl. The 25-question set
  is the "same eval as Act 1" harness and must stay reproducible. Output
  goes to data/qa/expanded_questions.jsonl.
- The system prompt matches the corpus's "Is NOT" list from CLAUDE.md:
  factual definitions/criteria only, no personal-advice/self-harm framing.
  A per-chunk skip is emitted when the model can't produce a compliant Q.
- Dedupe is prefix-based (first 40 chars, lowercased); it's crude but
  catches near-duplicates without pulling in an embedding call.
- Same key + rate-limit pattern as the judge (LLMJudgeVerifier).

Usage:
    python -m src.scripts.generate_questions
    python -m src.scripts.generate_questions --target 100 --seed 20260712
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You write factual retrieval-QA questions for a mental-health clinical "
    "reference corpus. Given a passage, produce ONE question the passage "
    "clearly and directly answers.\n\n"
    "Rules:\n"
    "- The question must be about DEFINITIONS, DIAGNOSTIC CRITERIA, SYMPTOMS, "
    "TREATMENT DESCRIPTIONS, or PREVALENCE — factual reference content only.\n"
    "- The question must be answerable from the passage alone. No outside knowledge.\n"
    "- DO NOT ask personal-advice questions ('what should I do about...', "
    "'am I...', 'how can I...'). DO NOT ask about self-harm or suicide.\n"
    "- DO NOT ask questions the passage cannot answer.\n"
    "- One sentence, ends with a question mark.\n\n"
    'Output ONLY a JSON object: {"question": "<...>"}. If no compliant question '
    'can be derived, output {"question": null}.'
)


def _load_corpus(paths: List[str]) -> List[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _load_questions(path: str) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_question(text: str):
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        obj = json.loads(m.group(0))
    q = obj.get("question")
    if not q or not isinstance(q, str) or not q.strip().endswith("?"):
        return None
    return q.strip()


def _dedup_key(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())[:40]


def _call_with_retry(fn, max_attempts=3, base_sleep=1.0):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("timeout" in msg or "temporarily" in msg
                         or "connection" in msg or "503" in msg or "502" in msg
                         or "429" in msg)
            if not transient or attempt == max_attempts - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+",
                    default=["data/corpus/icd11.jsonl", "data/corpus/nimh.jsonl"])
    ap.add_argument("--existing", default="data/qa/baseline_questions.jsonl")
    ap.add_argument("--out", default="data/qa/expanded_questions.jsonl")
    ap.add_argument("--target", type=int, default=100,
                    help="total questions in the OUTPUT file (existing + new)")
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--model", default="command-r-08-2024")
    args = ap.parse_args()

    load_dotenv()
    import cohere
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise SystemExit("COHERE_API_KEY not set")
    client = cohere.ClientV2(api_key=api_key)

    corpus = _load_corpus(args.corpus)
    existing = _load_questions(args.existing)
    print(f"Corpus chunks: {len(corpus)} | existing questions: {len(existing)} "
          f"| target: {args.target}")

    # Start the output as a superset of existing (so the file is drop-in
    # replaceable for baseline_questions.jsonl in any config).
    out_rows: List[dict] = list(existing)
    seen: Set[str] = {_dedup_key(r["question"]) for r in out_rows}

    rng = random.Random(args.seed)
    # Shuffle a copy so we don't re-hit the same chunks if we rerun.
    shuffled = list(corpus)
    rng.shuffle(shuffled)

    needed = args.target - len(out_rows)
    print(f"Need {needed} new questions.")
    if needed <= 0:
        print("Already at target — nothing to do.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_added = n_skipped = n_dup = 0
    for chunk in shuffled:
        if len(out_rows) >= args.target:
            break
        text = chunk.get("chunk_text", "").strip()
        if len(text) < 120:
            n_skipped += 1
            continue

        def _call():
            return client.chat(
                model=args.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Passage:\n{text}"},
                ],
            )
        try:
            resp = _call_with_retry(_call)
        except Exception as e:
            print(f"  API error, skipping chunk {chunk.get('chunk_id')}: {e}")
            n_skipped += 1
            continue
        content = resp.message.content
        raw = content[0].text if content else ""
        q = _parse_question(raw)
        if q is None:
            n_skipped += 1
            continue
        key = _dedup_key(q)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out_rows.append({"question": q, "source_chunk_id": chunk.get("chunk_id")})
        n_added += 1
        if n_added % 10 == 0:
            print(f"  +{n_added} ({len(out_rows)}/{args.target})")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(out_rows)} rows → {args.out}")
    print(f"  added: {n_added} | skipped: {n_skipped} | dupes: {n_dup}")
    if len(out_rows) < args.target:
        print(f"WARN: fell short of target ({len(out_rows)}/{args.target}). "
              "Corpus may not have enough compliant chunks; consider lowering target.")


if __name__ == "__main__":
    main()
