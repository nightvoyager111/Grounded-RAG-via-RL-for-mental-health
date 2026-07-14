"""Bootstrap confidence intervals for eval metrics.

Point estimates (like `groundedness_rate=0.88`) hide their own uncertainty.
With n=200 questions, a 2-point difference between two models may or may
not be within noise. Percentile-bootstrap CIs quantify that directly.

Two modes:
  Single-file:  metric ± CI for each metric in a rows.jsonl
  Two-file:     both above, plus bootstrapped Δ = model_B - model_A with CI
                and the fraction of resamples where the sign of Δ agrees
                with the point estimate (a rough one-sided uncertainty).

Usage:
    # One model
    python -m src.scripts.bootstrap_ci src/results/grpo_v3_n200/rows.jsonl

    # Compare two models on the SAME question set (paired bootstrap)
    python -m src.scripts.bootstrap_ci \\
        src/results/baseline_n200/rows.jsonl \\
        src/results/grpo_v3_n200/rows.jsonl \\
        --n-resamples 2000 --seed 20260714

Requires numpy only. Skips None metrics per-example (matching eval.metrics.aggregate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


METRIC_KEYS = [
    "groundedness_rate",
    "citation_precision",
    "citation_recall",
    "copy_rate",
    "abstention",
]


def _load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mean_ignore_none(vals: np.ndarray) -> float:
    """Mean of an object array that may contain None."""
    xs = [x for x in vals if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _bootstrap_ci(values: List[object], n: int, rng: np.random.Generator,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point, low, high) for percentile bootstrap on values.
    values may contain None; we take mean-ignoring-None over each resample."""
    arr = np.array(values, dtype=object)
    N = len(arr)
    if N == 0:
        return float("nan"), float("nan"), float("nan")
    resample_means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        resample_means[i] = _mean_ignore_none(arr[idx])
    point = _mean_ignore_none(arr)
    lo = float(np.quantile(resample_means, alpha / 2))
    hi = float(np.quantile(resample_means, 1 - alpha / 2))
    return point, lo, hi


def _paired_delta_bootstrap(a_vals: List[object], b_vals: List[object],
                            n: int, rng: np.random.Generator,
                            alpha: float = 0.05) -> dict:
    """Paired bootstrap: for each resample, draw the same indices from
    both models, compute both means, take the delta. Assumes rows are
    aligned 1:1 (same question index in both files)."""
    assert len(a_vals) == len(b_vals), "row counts must match for paired bootstrap"
    a = np.array(a_vals, dtype=object)
    b = np.array(b_vals, dtype=object)
    N = len(a)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        deltas[i] = _mean_ignore_none(b[idx]) - _mean_ignore_none(a[idx])
    point = _mean_ignore_none(b) - _mean_ignore_none(a)
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    # Directional agreement — what fraction of resamples share the sign
    # of the point estimate. >0.975 ≈ "significant" at 95% one-sided.
    if point >= 0:
        p_dir = float(np.mean(deltas >= 0))
    else:
        p_dir = float(np.mean(deltas <= 0))
    return {"point": point, "low": lo, "high": hi, "p_directional": p_dir}


def _print_single(rows: List[dict], label: str, n: int, rng: np.random.Generator) -> None:
    print(f"\n{label} (n={len(rows)}, bootstrap n_resamples={n}):")
    for k in METRIC_KEYS:
        vals = [r.get(k) for r in rows]
        n_def = sum(1 for v in vals if v is not None)
        point, lo, hi = _bootstrap_ci(vals, n, rng)
        print(f"  {k:<20} {point:.3f}  [{lo:.3f}, {hi:.3f}]   (defined on {n_def}/{len(rows)})")


def _print_pair(a_rows: List[dict], b_rows: List[dict], a_label: str,
                b_label: str, n: int, rng: np.random.Generator) -> None:
    # Align by question text to be safe (rows may be in different order).
    b_by_q = {r["question"]: r for r in b_rows}
    aligned_a, aligned_b = [], []
    dropped = 0
    for r in a_rows:
        q = r["question"]
        if q in b_by_q:
            aligned_a.append(r)
            aligned_b.append(b_by_q[q])
        else:
            dropped += 1
    print(f"\nPaired bootstrap: {b_label} − {a_label} "
          f"(aligned n={len(aligned_a)}, dropped={dropped})")
    for k in METRIC_KEYS:
        av = [r.get(k) for r in aligned_a]
        bv = [r.get(k) for r in aligned_b]
        d = _paired_delta_bootstrap(av, bv, n, rng)
        signif = "  ***" if d["p_directional"] > 0.975 else \
                 "  **"  if d["p_directional"] > 0.95 else ""
        print(f"  {k:<20} Δ={d['point']:+.3f}  [{d['low']:+.3f}, {d['high']:+.3f}]"
              f"   dir={d['p_directional']:.2f}{signif}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_a", type=Path,
                    help="rows.jsonl (or the only file, in single mode)")
    ap.add_argument("rows_b", type=Path, nargs="?", default=None,
                    help="optional second rows.jsonl; triggers paired-delta mode")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    a_rows = _load(str(args.rows_a))
    label_a = args.label_a or args.rows_a.parent.name

    _print_single(a_rows, label_a, args.n_resamples, rng)

    if args.rows_b:
        b_rows = _load(str(args.rows_b))
        label_b = args.label_b or args.rows_b.parent.name
        _print_single(b_rows, label_b, args.n_resamples, rng)
        _print_pair(a_rows, b_rows, label_a, label_b,
                    args.n_resamples, rng)
        print("\nLegend: dir = fraction of resamples where sign(Δ) matches "
              "point estimate. ** > 0.95, *** > 0.975.")


if __name__ == "__main__":
    main()
