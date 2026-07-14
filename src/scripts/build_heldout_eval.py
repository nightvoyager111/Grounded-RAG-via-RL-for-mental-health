"""Build the held-out eval subset: questions never seen by any training run.

Motivation: `data/qa/expanded_questions.jsonl` is a superset of the DPO
pair-construction pool (`data/qa/baseline_questions.jsonl`) and the
GRPO rollout pool (`data/grpo/prompts_v2.jsonl`). Evaluating GRPO v3 on
the full 200-Q file would double-count questions the reward function
already saw during training — an unfair test.

This script writes a strict held-out subset by removing:
  - every question in `baseline_questions.jsonl` (DPO source)
  - every question that appears (by exact question text) in the GRPO
    prompts file

Matching is on the `question` field. Whitespace-normalized string
equality, case-insensitive.

Usage:
    python -m src.scripts.build_heldout_eval

Writes:
    data/qa/heldout_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Set


def _load_jsonl(path: str) -> list:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _questions_from(rows: Iterable[dict], key: str = "question") -> Set[str]:
    return {_norm(r[key]) for r in rows if key in r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool",         default="data/qa/expanded_questions.jsonl",
                    help="the full eval pool (superset)")
    ap.add_argument("--dpo-source",   default="data/qa/baseline_questions.jsonl",
                    help="questions DPO's pair construction was seeded from")
    ap.add_argument("--grpo-prompts", default="data/grpo/prompts_v2.jsonl",
                    help="rollout prompts file GRPO trained on")
    ap.add_argument("--out",          default="data/qa/heldout_questions.jsonl")
    args = ap.parse_args()

    pool = _load_jsonl(args.pool)
    dpo_seen = _questions_from(_load_jsonl(args.dpo_source))
    grpo_seen = _questions_from(_load_jsonl(args.grpo_prompts))
    seen = dpo_seen | grpo_seen

    kept = [r for r in pool if _norm(r["question"]) not in seen]
    dropped = len(pool) - len(kept)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"Pool:            {len(pool):>4}  ({args.pool})")
    print(f"DPO source:      {len(dpo_seen):>4}  ({args.dpo_source})")
    print(f"GRPO training:   {len(grpo_seen):>4}  ({args.grpo_prompts})")
    print(f"Union seen:      {len(seen):>4}")
    print(f"Held-out kept:   {len(kept):>4}")
    print(f"Dropped:         {dropped:>4}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
