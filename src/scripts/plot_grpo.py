"""Generate GRPO evaluation visuals.

Single-panel grouped bar chart across arbitrary models. Includes:
  - groundedness_rate
  - citation_coverage — (# answers with >=1 valid citation) / n. The metric
    that visually carries the DPO-collapse / GRPO-recovery story.
    Auto-derived from citation_precision_n / n_examples if the JSON only
    has raw counts; emitted as first-class by build_eval_compare.py.
  - citation_recall
  - copy_rate

Dropped from the default view:
  - citation_precision (flat ~0.96-0.98 across models — a sentence, not a bar)
  - abstention (~0.01 everywhere — dead visual space)

Error bars are rendered when the JSON is in the {value, low, high} shape
produced by build_eval_compare.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

MODEL_COLORS = ["#7F8C8D", "#2A9D8F", "#8E44AD", "#E76F51", "#2E4A8A"]

METRIC_ORDER = [
    "groundedness_rate",
    "citation_coverage",
    "citation_recall",
    "copy_rate",
]
METRIC_LABEL = {
    "groundedness_rate": "Groundedness",
    "citation_coverage": "Citation coverage",
    "citation_recall": "Cite recall",
    "copy_rate": "Copy rate",
}


def _entry(model_metrics: dict, key: str):
    """(value, low, high) for one metric, or (None, None, None) if absent."""
    v = model_metrics.get(key)
    if v is None:
        return None, None, None
    if isinstance(v, dict):
        return v.get("value"), v.get("low"), v.get("high")
    return float(v), None, None


def _inject_coverage(data: dict) -> None:
    """Add citation_coverage to each model in-place when it can be derived
    from citation_precision_n / total set size."""
    for name, m in data.items():
        if "citation_coverage" in m:
            continue
        cov_n = m.get("citation_precision_n")
        total = m.get("n_examples")
        if total is None:
            for k in ("groundedness_rate", "copy_rate", "citation_recall"):
                e = m.get(k)
                if isinstance(e, dict) and e.get("n_defined"):
                    total = e["n_defined"]
                    break
        if cov_n is not None and total:
            m["citation_coverage"] = cov_n / total


def plot_eval_compare(eval_path: Path, out: Path):
    data = json.loads(eval_path.read_text())
    _inject_coverage(data)
    models = list(data.keys())
    metrics = [m for m in METRIC_ORDER
               if any(_entry(data[k], m)[0] is not None for k in models)]

    n_models = len(models)
    x = np.arange(len(metrics))
    w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    any_ci = False
    for i, name in enumerate(models):
        ys, err_lo, err_hi, has_ci = [], [], [], False
        for m in metrics:
            v, lo, hi = _entry(data[name], m)
            v = 0.0 if v is None else v
            ys.append(v)
            if lo is not None and hi is not None:
                err_lo.append(v - lo)
                err_hi.append(hi - v)
                has_ci = True
                any_ci = True
            else:
                err_lo.append(0.0)
                err_hi.append(0.0)

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        pos = x + (i - (n_models - 1) / 2) * w
        yerr = np.array([err_lo, err_hi]) if has_ci else None
        bars = ax.bar(pos, ys, w, color=color, label=name,
                      edgecolor="white", linewidth=1,
                      yerr=yerr, capsize=3,
                      error_kw={"ecolor": "#333", "lw": 1, "alpha": 0.7})
        for r, e_hi in zip(bars, err_hi):
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + e_hi + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    title = "Eval Comparison (LLM-judge verifier)"
    if any_ci:
        title += " · 95% bootstrap CIs"
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, ncol=min(n_models, 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-compare", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("reports/grpo_viz"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_eval_compare(args.eval_compare, args.outdir / "model_comparison.png")


if __name__ == "__main__":
    main()
