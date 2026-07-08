"""Calibration harness for the groundedness verifier.

CLAUDE.md step 5: hand-label 30–50 (passage, answer, grounded?) examples,
compute verifier ↔ human agreement, gate at ~85% before any training.

This module is pure math + I/O over labeled records. The actual model
scoring happens in nli.py; here we just compare labels vs. scores."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass
class CalibrationRecord:
    """One labeled example. `passages` and `answer` are the inputs the
    verifier sees; `label` is 1 if a human judged the answer grounded in
    the passages, else 0. `score` is populated after the verifier runs."""

    id: str
    question: str
    answer: str
    passages: List[str]
    label: Optional[int] = None
    score: Optional[float] = None


def read_labeled(path: str | Path) -> List[CalibrationRecord]:
    out: List[CalibrationRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                CalibrationRecord(
                    id=d["id"],
                    question=d["question"],
                    answer=d["answer"],
                    passages=list(d["passages"]),
                    label=d.get("label"),
                    score=d.get("score"),
                )
            )
    return out


def write_records(path: str | Path, records: Iterable[CalibrationRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "question": r.question,
                        "answer": r.answer,
                        "passages": r.passages,
                        "label": r.label,
                        "score": r.score,
                    }
                )
                + "\n"
            )


def agreement_at_threshold(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict:
    """Fraction of examples where (score >= threshold) matches the label."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be same length")
    if not labels:
        raise ValueError("empty inputs")
    tp = fp = tn = fn = 0
    for l, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1:
            tp += 1
        elif pred == 1 and l == 0:
            fp += 1
        elif pred == 0 and l == 0:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": n,
        "threshold": float(threshold),
        "agreement": (tp + tn) / n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def sweep_thresholds(
    labels: Sequence[int],
    scores: Sequence[float],
    n_steps: int = 21,
) -> Tuple[dict, List[dict]]:
    """Pick the threshold that maximizes agreement. Returns (best, all_rows).

    Ties broken toward 0.5 (least-committal middle threshold), so the picked
    threshold reflects the score's actual decision boundary rather than an
    accidental edge case."""
    if n_steps < 2:
        raise ValueError("need at least 2 steps")
    step = 1.0 / (n_steps - 1)
    rows = [
        agreement_at_threshold(labels, scores, i * step) for i in range(n_steps)
    ]
    best = min(rows, key=lambda r: (-r["agreement"], abs(r["threshold"] - 0.5)))
    return best, rows


def calibration_report(
    labeled: Sequence[CalibrationRecord],
    default_threshold: float = 0.5,
) -> dict:
    """Assemble the numbers CLAUDE.md's calibration gate needs."""
    labels = [r.label for r in labeled]
    scores = [r.score for r in labeled]
    if any(l is None for l in labels):
        raise ValueError("all records must be labeled (0/1) before reporting")
    if any(s is None for s in scores):
        raise ValueError("all records must have a score (run verifier first)")

    at_default = agreement_at_threshold(labels, scores, default_threshold)
    best, sweep = sweep_thresholds(labels, scores)
    pos_rate = sum(labels) / len(labels)
    return {
        "n_examples": len(labeled),
        "positive_rate": pos_rate,
        "at_default_threshold": at_default,
        "best_threshold": best,
        "sweep": sweep,
        "gate_pass_at_default": at_default["agreement"] >= 0.85,
        "gate_pass_at_best": best["agreement"] >= 0.85,
    }


def score_records(
    records: Sequence[CalibrationRecord],
    verifier: Callable[[Sequence[str], str], float],
) -> List[CalibrationRecord]:
    """Fill in `score` on each record using the verifier. Returns new list;
    input records are not mutated."""
    out: List[CalibrationRecord] = []
    for r in records:
        s = float(verifier(r.passages, r.answer))
        out.append(
            CalibrationRecord(
                id=r.id,
                question=r.question,
                answer=r.answer,
                passages=r.passages,
                label=r.label,
                score=s,
            )
        )
    return out
