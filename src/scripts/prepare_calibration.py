"""Turn baseline rows.jsonl into a labeling template for calibration.

The eval runner already produced (question, answer, retrieved passages)
tuples in src/results/baseline/rows.jsonl. We just need to reshape them
into calibration records with a `label: null` field for the human to fill.

Usage:
    python -m src.scripts.prepare_calibration \\
        --rows src/results/baseline/rows.jsonl \\
        --out  data/calibration/pending.jsonl \\
        --n 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config
from src.grounded_rag.verifier.calibration import CalibrationRecord, write_records


def _load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rehydrate_passages(retriever: Retriever, question: str, answer: str) -> list:
    """rows.jsonl doesn't store the retrieved passages (only ids/metrics),
    so we replay retrieval to reconstruct them. Deterministic because
    retrieval is deterministic for a fixed query + index."""
    return [r.text for r in retriever.retrieve(question)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="src/results/baseline/rows.jsonl")
    ap.add_argument("--out", default="data/calibration/pending.jsonl")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--n", type=int, default=40, help="how many rows to sample")
    ap.add_argument(
        "--hard-negatives",
        type=int,
        default=0,
        help="also emit N passage-swapped rows: an answer paired with passages "
             "retrieved for a different question. These are candidate negatives "
             "(you still label them). Ids are prefixed cal:neg:.",
    )
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    rows = list(_load_rows(args.rows))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if len(rows) < args.n + args.hard_negatives:
        raise SystemExit(
            f"Not enough rows: need {args.n + args.hard_negatives}, have {len(rows)}"
        )
    chosen = rows[: args.n]
    neg_rows = rows[args.n : args.n + args.hard_negatives]

    retr_cfg = load_retrieval_config(args.retrieval_config)
    retriever = Retriever(retr_cfg)

    records = []
    for i, r in enumerate(chosen):
        passages = _rehydrate_passages(retriever, r["question"], r["answer"])
        records.append(
            CalibrationRecord(
                id=f"cal:{i:03d}",
                question=r["question"],
                answer=r["answer"],
                passages=passages,
                label=None,
                score=None,
            )
        )

    # Hard negatives: pair each answer with passages retrieved for a *different*
    # question. The answer may still be factually true, but it is not supported
    # by these passages — which is what the verifier must catch. Human still
    # labels; a swap can occasionally yield a supported answer by coincidence.
    if args.hard_negatives:
        if len(neg_rows) < 2:
            raise SystemExit("--hard-negatives requires at least 2 donor rows")
        for i, r in enumerate(neg_rows):
            donor = neg_rows[(i + 1) % len(neg_rows)]
            mismatched_passages = _rehydrate_passages(
                retriever, donor["question"], donor["answer"]
            )
            records.append(
                CalibrationRecord(
                    id=f"cal:neg:{i:03d}",
                    question=r["question"],
                    answer=r["answer"],
                    passages=mismatched_passages,
                    label=None,
                    score=None,
                )
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_records(args.out, records)
    print(f"Wrote {len(records)} unlabeled records → {args.out}")
    print("\nHow to label:")
    print(f"  Open {args.out}. For each line, set \"label\" to 1 if the")
    print("  answer is fully supported by the passages, else 0. Save as")
    print("  data/calibration/labeled.jsonl and run run_calibration.py.")


if __name__ == "__main__":
    main()
