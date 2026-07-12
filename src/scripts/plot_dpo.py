"""Generate DPO training + evaluation visuals.

Inputs (all optional; each drives one figure):
  --trainer-state PATH   HF trainer_state.json (from any DPO checkpoint dir).
  --eval-compare  PATH   JSON: {"baseline": {...metrics}, "dpo": {...metrics}}.
  --eval-sweep    PATH   JSON: {"<ckpt-or-epoch>": {...metrics}, ...}.
  --outdir        DIR    Where to write PNGs (default: reports/dpo_viz).

Metric keys expected: groundedness_rate, abstention, copy_rate,
citation_precision, citation_recall.
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

PALETTE = {
    "loss": "#2E4A8A",
    "chosen": "#2A9D8F",
    "rejected": "#E76F51",
    "margin": "#8E44AD",
    "kl": "#B58900",
    "acc": "#264653",
    "baseline": "#7F8C8D",
    "dpo": "#2A9D8F",
    "grid": "#CCCCCC",
}

METRIC_ORDER = [
    "groundedness_rate",
    "abstention",
    "copy_rate",
    "citation_precision",
    "citation_recall",
]
METRIC_LABEL = {
    "groundedness_rate": "Groundedness",
    "abstention": "Abstention",
    "copy_rate": "Copy rate",
    "citation_precision": "Cite precision",
    "citation_recall": "Cite recall",
}


def _series(log_history, key):
    xs, ys = [], []
    for row in log_history:
        if key in row and "step" in row:
            xs.append(row["step"])
            ys.append(row[key])
    return np.array(xs), np.array(ys)


def plot_training_curves(trainer_state_path: Path, out: Path):
    state = json.loads(trainer_state_path.read_text())
    hist = state.get("log_history", [])
    if not hist:
        raise ValueError(f"No log_history in {trainer_state_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

    # 1. Loss
    ax = axes[0, 0]
    xs, ys = _series(hist, "loss")
    if len(xs):
        ax.plot(xs, ys, color=PALETTE["loss"], lw=2, label="train loss")
    xe, ye = _series(hist, "eval_loss")
    if len(xe):
        ax.plot(xe, ye, color=PALETTE["rejected"], lw=2, ls="--", marker="o",
                ms=4, label="eval loss")
    ax.set_title("DPO Loss", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)

    # 2. Rewards chosen / rejected
    ax = axes[0, 1]
    for key, color, label in [
        ("rewards/chosen", PALETTE["chosen"], "chosen"),
        ("rewards/rejected", PALETTE["rejected"], "rejected"),
    ]:
        xs, ys = _series(hist, key)
        if len(xs):
            ax.plot(xs, ys, color=color, lw=2, label=label)
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_title("Implicit Rewards", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\beta \log \pi_\theta/\pi_{\mathrm{ref}}$")
    ax.legend(frameon=False)

    # 3. Reward margin & accuracy
    ax = axes[1, 0]
    xs, ys = _series(hist, "rewards/margins")
    if len(xs):
        ax.fill_between(xs, 0, ys, color=PALETTE["margin"], alpha=0.18)
        ax.plot(xs, ys, color=PALETTE["margin"], lw=2, label="margin (chosen − rejected)")
    xa, ya = _series(hist, "rewards/accuracies")
    if len(xa):
        ax2 = ax.twinx()
        ax2.plot(xa, ya, color=PALETTE["acc"], lw=2, ls=":", marker="s", ms=3,
                 label="accuracy")
        ax2.set_ylabel("accuracy", color=PALETTE["acc"])
        ax2.set_ylim(0, 1.02)
        ax2.grid(False)
        ax2.tick_params(axis="y", colors=PALETTE["acc"])
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_title("Reward Margin & Pair Accuracy", fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("margin")
    ax.legend(frameon=False, loc="upper left")

    # 4. KL to reference
    ax = axes[1, 1]
    plotted = False
    for key, ls, label in [
        ("rewards/kl", "-", "KL(π‖π_ref)"),
        ("kl", "-", "KL"),
        ("logps/chosen", ":", "logp chosen"),
        ("logps/rejected", ":", "logp rejected"),
    ]:
        xs, ys = _series(hist, key)
        if len(xs):
            ax.plot(xs, ys, lw=2, ls=ls, label=label)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no KL / logp fields logged",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.set_title("Policy Drift from Reference", fontweight="bold")
    ax.set_xlabel("step")
    ax.legend(frameon=False)

    fig.suptitle("DPO Training Diagnostics — Qwen2.5-1.5B-Instruct + LoRA",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_eval_compare(eval_path: Path, out: Path):
    data = json.loads(eval_path.read_text())
    base = data["baseline"]
    dpo = data["dpo"]
    metrics = [m for m in METRIC_ORDER if m in base and m in dpo]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    b = ax.bar(x - w / 2, [base[m] for m in metrics], w,
               color=PALETTE["baseline"], label="Baseline RAG",
               edgecolor="white", linewidth=1.2)
    d = ax.bar(x + w / 2, [dpo[m] for m in metrics], w,
               color=PALETTE["dpo"], label="DPO (ckpt-21)",
               edgecolor="white", linewidth=1.2)

    for bars in (b, d):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # deltas
    for i, m in enumerate(metrics):
        delta = dpo[m] - base[m]
        color = "#2A9D8F" if (delta > 0) == (m in {"groundedness_rate", "citation_precision", "citation_recall"}) else "#E76F51"
        ax.annotate(f"Δ {delta:+.2f}", xy=(i, max(base[m], dpo[m]) + 0.09),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("score")
    ax.set_title("Baseline vs DPO — Act 1 Evaluation (n=25, LLM-judge verifier)",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_ckpt_sweep(sweep_path: Path, out: Path):
    data = json.loads(sweep_path.read_text())
    # keys can be checkpoint names or epoch numbers; sort by extracted int
    def key_num(k):
        digits = "".join(ch for ch in k if ch.isdigit())
        return int(digits) if digits else 0
    ckpts = sorted(data.keys(), key=key_num)
    xs = [key_num(k) for k in ckpts]

    metrics = [m for m in METRIC_ORDER if all(m in data[k] for k in ckpts)]
    colors = ["#2A9D8F", "#E76F51", "#8E44AD", "#2E4A8A", "#B58900"]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for m, c in zip(metrics, colors):
        ys = [data[k][m] for k in ckpts]
        ax.plot(xs, ys, marker="o", lw=2, color=c, label=METRIC_LABEL[m])

    # highlight the winner
    winner_idx = int(np.argmax([data[k].get("groundedness_rate", 0) for k in ckpts]))
    ax.axvline(xs[winner_idx], color="k", ls="--", lw=1, alpha=0.4)
    ax.text(xs[winner_idx], ax.get_ylim()[1] * 0.98,
            f" winner: {ckpts[winner_idx]}",
            va="top", fontsize=9, color="#333")

    ax.set_xlabel("checkpoint / epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Checkpoint Evaluation Sweep", fontweight="bold")
    ax.legend(frameon=False, ncol=len(metrics), loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer-state", type=Path)
    p.add_argument("--eval-compare", type=Path)
    p.add_argument("--eval-sweep", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("reports/dpo_viz"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.trainer_state:
        plot_training_curves(args.trainer_state, args.outdir / "training_curves.png")
    if args.eval_compare:
        plot_eval_compare(args.eval_compare, args.outdir / "baseline_vs_dpo.png")
    if args.eval_sweep:
        plot_ckpt_sweep(args.eval_sweep, args.outdir / "checkpoint_sweep.png")

    if not any([args.trainer_state, args.eval_compare, args.eval_sweep]):
        p.error("provide at least one of --trainer-state / --eval-compare / --eval-sweep")


if __name__ == "__main__":
    main()
