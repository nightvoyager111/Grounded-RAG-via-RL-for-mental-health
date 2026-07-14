"""Generate GRPO training + evaluation visuals.

Changes vs. the original plot_eval_compare:
  - Adds `citation_coverage` = (# answers with >=1 citation) / n, the metric
    that actually carries the DPO-collapse / GRPO-recovery story. Derived from
    citation_precision_n / n_defined when a coverage value isn't supplied
    directly. Coverage is a proportion over the whole eval set, so it renders
    without a bootstrap whisker unless a {value, low, high} dict is provided.
  - Default view drops `citation_precision` (flat across models -> a sentence,
    not a bar group) and `abstention` (~0.01 everywhere -> dead space).
  - Faceted layout: metrics living near the ceiling (groundedness, coverage)
    get one panel with a tight y-range; low-valued rate metrics (copy_rate,
    cite_recall) get another. A shared 0-1.2 axis buried the real gains.
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

METRIC_LABEL = {
    "groundedness_rate": "Groundedness",
    "citation_coverage": "Citation coverage",
    "copy_rate": "Copy rate",
    "citation_recall": "Cite recall",
}

# Which metrics belong in which panel, and the y-range that keeps their
# differences legible instead of squashed against a shared 0-1.2 axis.
PANELS = [
    ("Faithfulness (higher = better)",
     ["groundedness_rate", "citation_coverage"], (0.0, 1.05)),
    ("Copy / recall (lower copy, higher recall)",
     ["copy_rate", "citation_recall"], (0.0, 0.45)),
]


def _entry(model_metrics: dict, key: str):
    """(value, low, high) for one metric, or (None, None, None) if absent.
    Accepts a plain number or a {value, low, high} dict."""
    v = model_metrics.get(key)
    if v is None:
        return None, None, None
    if isinstance(v, dict):
        return v.get("value"), v.get("low"), v.get("high")
    return float(v), None, None


def _inject_coverage(data: dict) -> None:
    """Add citation_coverage to each model in-place when it can be derived.
    Prefers an explicit value; else citation_precision_n / n_defined (the
    share of answers that contained at least one citation)."""
    for name, m in data.items():
        if "citation_coverage" in m:
            continue
        cov_n = m.get("citation_precision_n")
        # total set size: prefer an explicit n_examples, else fall back to
        # the n_defined recorded on any metric (they share the eval set).
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
    n_models = len(models)

    fig, axes = plt.subplots(
        1, len(PANELS),
        figsize=(6.2 * len(PANELS), 5.2),
        constrained_layout=True,
    )
    if len(PANELS) == 1:
        axes = [axes]

    any_ci = False
    for ax, (panel_title, metric_keys, ylim) in zip(axes, PANELS):
        metrics = [m for m in metric_keys
                   if any(_entry(data[k], m)[0] is not None for k in models)]
        x = np.arange(len(metrics))
        w = 0.8 / n_models

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
            top_whisker = max(err_hi) if has_ci else 0.0
            for r, e_hi in zip(bars, err_hi):
                h = r.get_height()
                ax.text(r.get_x() + r.get_width() / 2,
                        h + e_hi + 0.012 * (ylim[1] - ylim[0]) / 1.0,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
        ax.set_ylim(*ylim)
        ax.set_ylabel("score")
        ax.set_title(panel_title, fontweight="bold", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(n_models, 4),
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    sup = "Eval Comparison (LLM-judge verifier)"
    if any_ci:
        sup += " · 95% bootstrap CIs"
    fig.suptitle(sup, fontsize=13, fontweight="bold")
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