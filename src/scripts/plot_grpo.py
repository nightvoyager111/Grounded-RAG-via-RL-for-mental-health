"""Generate GRPO training + evaluation visuals.

Mirrors plot_dpo.py structure but adds two RL-specific views:
- Per-step reward trace with component decomposition (from reward_trace.jsonl).
- Rollout attractor-basin scatter (groundedness vs copy_rate, colored by reward).

Inputs (all optional; each drives one figure):
  --reward-trace   PATH   JSONL emitted by GroundednessReward (per-completion diagnostics).
  --eval-compare   PATH   JSON: {"<model_name>": {...metrics}, ...}. 2–5 keys.
  --outdir         DIR    Where to write PNGs (default: reports/grpo_viz).

Metric keys expected in eval-compare: groundedness_rate, abstention, copy_rate,
citation_precision, citation_recall.
"""
from __future__ import annotations

import argparse
import collections
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

PALETTE = {
    "reward": "#2E4A8A",
    "g": "#2A9D8F",
    "copy": "#E76F51",
    "cite_r": "#2E4A8A",
    "comp": "#8E44AD",
    "abst": "#B58900",
}

# 4-model default palette. plot_eval_compare picks colors by index, so any
# order the user hands in still maps to a distinct hue.
MODEL_COLORS = ["#7F8C8D", "#2A9D8F", "#8E44AD", "#E76F51", "#2E4A8A"]

METRIC_ORDER = [
    "groundedness_rate",
    "copy_rate",
    "citation_precision",
    "citation_recall",
    "abstention",
]
METRIC_LABEL = {
    "groundedness_rate": "Groundedness",
    "copy_rate": "Copy rate",
    "citation_precision": "Cite precision",
    "citation_recall": "Cite recall",
    "abstention": "Abstention",
}


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def _plot_smoothed(ax, xs, ys, color, label):
    ax.plot(xs, ys, color=color, alpha=0.3, lw=1)
    sm = _smooth(ys)
    off = (len(xs) - len(sm)) // 2
    ax.plot(xs[off:off + len(sm)], sm, color=color, lw=2.5, label=label)


def plot_training_curves(trace_path: Path, out: Path):
    """4-panel training diagnostics from reward_trace.jsonl."""
    buckets = collections.defaultdict(list)
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            buckets[r["step"]].append(r)

    steps = sorted(buckets)
    if not steps:
        raise ValueError(f"No steps in {trace_path}")

    def _avg(step, key):
        vals = [d[key] for d in buckets[step] if d.get(key) is not None]
        return np.mean(vals) if vals else np.nan

    keys = ("reward", "groundedness", "copy_rate", "citation_recall",
            "citation_compliance", "abstention")
    series = {k: np.array([_avg(s, k) for s in steps]) for k in keys}
    xs = np.array(steps)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

    # (0,0) reward
    ax = axes[0, 0]
    _plot_smoothed(ax, xs, series["reward"], PALETTE["reward"], "reward (5-step MA)")
    ax.set_title("GRPO Reward Over Training", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("R (compound)")
    ax.legend(frameon=False)

    # (0,1) citation repair — the v3 story
    ax = axes[0, 1]
    if not np.all(np.isnan(series["citation_compliance"])):
        _plot_smoothed(ax, xs, series["citation_compliance"], PALETTE["comp"],
                       "compliance (per-sentence, v3 reward)")
    _plot_smoothed(ax, xs, series["citation_recall"], PALETTE["cite_r"],
                   "citation_recall (aggregate)")
    ax.set_title("Citation Repair Signal", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)

    # (1,0) attractor basin scatter — one point per rollout
    ax = axes[1, 0]
    all_rows = [d for lst in buckets.values() for d in lst]
    gs = np.array([d["groundedness"] for d in all_rows])
    cs = np.array([d["copy_rate"] if d["copy_rate"] is not None else 0
                   for d in all_rows])
    rs = np.array([d["reward"] for d in all_rows])
    sc = ax.scatter(gs, cs, c=rs, cmap="RdYlGn", s=14, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="reward")
    ax.set_title("Rollout Attractor Basins (colored by reward)", fontweight="bold")
    ax.set_xlabel("groundedness")
    ax.set_ylabel("copy_rate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.annotate("verbatim copy\nmode", xy=(0.95, 0.85), fontsize=9,
                ha="right", color="#666")
    ax.annotate("abstention\nmode", xy=(0.5, 0.05), fontsize=9,
                ha="center", color="#666")
    ax.annotate("target\n(cite + paraphrase)", xy=(0.95, 0.15), fontsize=9,
                ha="right", color="#333", fontweight="bold")

    # (1,1) groundedness / copy / abstention
    ax = axes[1, 1]
    _plot_smoothed(ax, xs, series["copy_rate"], PALETTE["copy"], "copy_rate")
    _plot_smoothed(ax, xs, series["groundedness"], PALETTE["g"], "groundedness")
    _plot_smoothed(ax, xs, series["abstention"], PALETTE["abst"], "abstention")
    ax.set_title("Groundedness / Copy / Abstention Over Training", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)

    fig.suptitle("GRPO Training Diagnostics", fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_eval_compare(eval_path: Path, out: Path):
    """Grouped bar chart across arbitrary models. Expects
    {"<model_name>": {metric: value, ...}, ...}."""
    data = json.loads(eval_path.read_text())
    models = list(data.keys())
    metrics = [m for m in METRIC_ORDER if all(m in data[k] for k in models)]

    n_models = len(models)
    x = np.arange(len(metrics))
    w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for i, name in enumerate(models):
        ys = [data[name].get(m) or 0 for m in metrics]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        bars = ax.bar(x + (i - (n_models - 1) / 2) * w, ys, w,
                      color=color, label=name, edgecolor="white", linewidth=1)
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("Eval Comparison (n=25, LLM-judge verifier)", fontweight="bold")
    ax.legend(frameon=False, ncol=min(n_models, 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-trace", type=Path)
    ap.add_argument("--eval-compare", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("reports/grpo_viz"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.reward_trace:
        plot_training_curves(args.reward_trace, args.outdir / "training_diagnostics.png")
    if args.eval_compare:
        plot_eval_compare(args.eval_compare, args.outdir / "model_comparison.png")

    if not any([args.reward_trace, args.eval_compare]):
        ap.error("provide at least one of --reward-trace / --eval-compare")


if __name__ == "__main__":
    main()
