"""Build the eval_compare JSON that plot_grpo.py consumes, with bootstrap
CIs baked in.

Reads N rows.jsonl files (`--model NAME:path`), computes point estimate +
percentile bootstrap CI for each metric, writes:

    {
      "<model_name>": {
        "groundedness_rate": {"value": 0.88, "low": 0.82, "high": 0.94},
        ...
      },
      ...
    }

Downstream: `plot_grpo.py --eval-compare <this_file>` renders bars with
error bars.

Usage:
    python -m src.scripts.build_eval_compare \\
        --model Baseline:src/results/baseline_n200/rows.jsonl \\
        --model DPO:src/results/dpo_n200/rows.jsonl \\
        --model "GRPO v3":src/results/grpo_v3_n200/rows.jsonl \\
        --out reports/grpo_viz/eval_compare_n200.json
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

# Derived metrics: per-row bernoulli indicators computed from the raw row
# dict, then bootstrapped like any other metric. `citation_coverage` is the
# fraction of answers that emitted at least one valid citation. It's the
# sharpest signal of the DPO-collapse / GRPO-recovery story (baseline 92% →
# DPO 48% → GRPO v3 72%), so it needs to be in the eval-compare JSON
# explicitly, not just derivable from a raw count.
DERIVED_METRICS = {
    # A row "counts as cited" iff citation_precision was defined for it
    # (i.e. the answer contained at least one bracketed id that matched
    # a retrieved chunk — matches the eval.metrics.citation_precision
    # None-when-no-citations contract).
    "citation_coverage": lambda r: 1 if r.get("citation_precision") is not None else 0,
}


def _load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mean_ignore_none(vals: np.ndarray) -> float:
    xs = [x for x in vals if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _bootstrap_ci(values: List[object], n: int, rng: np.random.Generator,
                  alpha: float = 0.05) -> dict:
    arr = np.array(values, dtype=object)
    N = len(arr)
    if N == 0:
        return {"value": None, "low": None, "high": None, "n_defined": 0}
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        means[i] = _mean_ignore_none(arr[idx])
    return {
        "value": _mean_ignore_none(arr),
        "low":   float(np.quantile(means, alpha / 2)),
        "high":  float(np.quantile(means, 1 - alpha / 2)),
        "n_defined": int(sum(1 for v in arr if v is not None)),
    }


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Split "NAME:path" — NAME may contain spaces, but not ':'."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME:path, got {spec!r}")
    name, path = spec.split(":", 1)
    return name.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", type=_parse_model_spec,
                    required=True, metavar="NAME:path",
                    help="Repeatable. e.g. --model DPO:src/results/dpo_n200/rows.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--drop-n-defined", action="store_true",
                    help="strip n_defined from output (leaner JSON)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {}
    print(f"Computing bootstrap CIs (n_resamples={args.n_resamples})...")
    for name, path in args.model:
        rows = _load(path)
        metrics_out = {}
        # Raw metrics (skip None per-example, mean).
        for k in METRIC_KEYS:
            vals = [r.get(k) for r in rows]
            ci = _bootstrap_ci(vals, args.n_resamples, rng)
            if args.drop_n_defined:
                ci.pop("n_defined", None)
            metrics_out[k] = ci
        # Derived metrics (bernoulli indicator per row → bootstrap the mean).
        for k, indicator in DERIVED_METRICS.items():
            vals = [indicator(r) for r in rows]
            ci = _bootstrap_ci(vals, args.n_resamples, rng)
            if args.drop_n_defined:
                ci.pop("n_defined", None)
            metrics_out[k] = ci
        out[name] = metrics_out
        # human-readable log line
        g = metrics_out["groundedness_rate"]
        cov = metrics_out["citation_coverage"]
        cr = metrics_out["citation_recall"]
        print(f"  {name:<12} n={len(rows)}  g={g['value']:.3f} [{g['low']:.3f},{g['high']:.3f}]"
              f"  cov={cov['value']:.3f} [{cov['low']:.3f},{cov['high']:.3f}]"
              f"  cite_r={cr['value']:.3f} [{cr['low']:.3f},{cr['high']:.3f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
