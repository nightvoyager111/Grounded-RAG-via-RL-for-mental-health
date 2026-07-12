"""GRPO reward function for the grounded generator.

CLAUDE.md step 8: `R = groundedness - lambda * copy_penalty`. Helpfulness
and citation compliance are DELIBERATELY omitted at first — the point of
Act 2 is to observe collapse under a partial reward, then repair it in
step 10.

Signature matches TRL GRPOTrainer's `reward_funcs` contract:
    reward_fn(prompts, completions, **kwargs) -> list[float]

Extra per-example fields (retrieved_ids, retrieved_texts) are passed
through **kwargs from the dataset columns.

Diagnostics (citation_precision/recall, abstention) are computed and
optionally written to a JSONL trace so we can prove *when* collapse
happens and *which* metric degrades first — they never enter the reward.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from src.grounded_rag.eval.metrics import (
    abstention_probe,
    citation_compliance,
    citation_precision,
    citation_recall,
    copy_rate,
    extract_citations,
)


@dataclass
class RewardConfig:
    copy_penalty_lambda: float = 0.5      # weight on n-gram copy penalty
    copy_ngram: int = 8
    judge_max_workers: int = 8            # parallel Cohere calls per batch
    trace_path: Optional[str] = None      # JSONL of every scored completion
    # If groundedness is None (judge failed), fall back to this so training
    # doesn't crash on a transient API error.
    fallback_score: float = 0.5

    # Compound-reward knobs (Act 2 step 10). All default to 0 → the
    # v1 collapse-inducing reward (g - lambda*copy) is preserved.
    #
    # citation_bonus_mu: weight on citation_recall reward. Fixes the
    #   DPO-inherited citation collapse (0.34 → 0.10) by rewarding the
    #   policy for using [chunk_id] markers on retrieved evidence. We
    #   use recall (not precision) because precision was already 1.0
    #   at baseline — the failure mode is dropping citations, not
    #   inventing them.
    # abstention_penalty_rho: subtracted when the answer abstains AND
    #   the passages actually support an answer. Without this, "I don't
    #   know" is a stable local optimum (R ≈ 0.5).
    # abstention_ignore_score: if judge score >= this, we consider the
    #   passages to have supported an answer, so abstention is penalized.
    citation_bonus_mu: float = 0.0
    abstention_penalty_rho: float = 0.0
    abstention_ignore_score: float = 0.5

    # v3 knob: per-sentence citation compliance bonus. Sharper than
    # citation_recall aggregate because each sentence contributes its own
    # signal — an extra citation added → proportional reward bump. This is
    # the direct RL target of the "every factual sentence must end with a
    # [chunk_id]" system-prompt rule.
    citation_compliance_bonus_nu: float = 0.0


class GroundednessReward:
    """Reward = groundedness - lambda * copy_penalty.

    `judge` is any callable (passages, answer) -> float in [0, 1] — this
    is the same interface `LLMJudgeVerifier.score` exposes, so the Act 1
    verifier drops in unchanged.

    Copy rate is measured against *cited* passages when the answer cites
    valid chunks, else against all retrieved passages (worst case for the
    generator — discourages "just copy the first passage" strategies).
    """

    # TRL GRPOTrainer reads reward_func.__name__ for its metrics log; class
    # instances don't have __name__ by default, so expose one explicitly.
    __name__ = "groundedness_minus_copy"

    def __init__(self, judge: Callable[[Sequence[str], str], float], cfg: RewardConfig):
        self.judge = judge
        self.cfg = cfg
        self._trace_fh = None
        if cfg.trace_path:
            Path(cfg.trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(cfg.trace_path, "a", encoding="utf-8")
        self._step = 0

    def close(self) -> None:
        if self._trace_fh:
            self._trace_fh.close()
            self._trace_fh = None

    def _score_one(self, texts: Sequence[str], answer: str) -> float:
        try:
            return float(self.judge(texts, answer))
        except Exception:
            return self.cfg.fallback_score

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        retrieved_ids: List[List[str]],
        retrieved_texts: List[List[str]],
        **kwargs,
    ) -> List[float]:
        # 1. Score all completions with the judge in parallel — Cohere is
        #    the throughput bottleneck; sequential kills GRPO step time.
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.cfg.judge_max_workers) as ex:
            groundedness = list(ex.map(
                self._score_one, retrieved_texts, completions
            ))
        judge_seconds = time.time() - t0

        rewards: List[float] = []
        diagnostics: List[dict] = []
        for i, ans in enumerate(completions):
            ids = retrieved_ids[i]
            texts = retrieved_texts[i]
            id_to_text = dict(zip(ids, texts))
            cited_ids = [c for c in extract_citations(ans) if c in id_to_text]
            cited_texts = [id_to_text[c] for c in cited_ids] or list(texts)

            g = groundedness[i]
            c = copy_rate(ans, cited_texts, n=self.cfg.copy_ngram)
            c_val = 0.0 if c is None else c
            cr = citation_recall(ans, ids)
            cr_val = 0.0 if cr is None else cr
            comp = citation_compliance(ans, ids)
            comp_val = 0.0 if comp is None else comp
            abst = abstention_probe(ans)
            # Only penalize abstention when the retrieved passages actually
            # look supportive (judge saw them as at least partially grounding
            # something) — abstaining on a genuinely unanswerable question
            # should not be punished.
            abst_penalty = self.cfg.abstention_penalty_rho * abst * (
                1.0 if g >= self.cfg.abstention_ignore_score else 0.0
            )
            r = (
                g
                - self.cfg.copy_penalty_lambda * c_val
                + self.cfg.citation_bonus_mu * cr_val
                + self.cfg.citation_compliance_bonus_nu * comp_val
                - abst_penalty
            )
            rewards.append(r)

            diagnostics.append({
                "step": self._step,
                "groundedness": g,
                "copy_rate": c,
                "reward": r,
                "citation_precision": citation_precision(ans, ids),
                "citation_recall": cr,
                "citation_compliance": comp,
                "abstention": abst,
                "answer": ans,
            })

        if self._trace_fh:
            for d in diagnostics:
                self._trace_fh.write(json.dumps(d) + "\n")
            self._trace_fh.flush()

        self._step += 1

        # One-line stdout summary so a Colab log tells the collapse story
        # without opening the trace file.
        n = len(rewards)
        avg_g = sum(groundedness) / n
        avg_c = sum(0.0 if d["copy_rate"] is None else d["copy_rate"]
                    for d in diagnostics) / n
        avg_r = sum(rewards) / n
        abst = sum(d["abstention"] for d in diagnostics) / n
        cp_vals = [d["citation_precision"] for d in diagnostics
                   if d["citation_precision"] is not None]
        cr_vals = [d["citation_recall"] for d in diagnostics
                   if d["citation_recall"] is not None]
        comp_vals = [d["citation_compliance"] for d in diagnostics
                     if d["citation_compliance"] is not None]
        avg_cp = sum(cp_vals) / len(cp_vals) if cp_vals else float("nan")
        avg_cr = sum(cr_vals) / len(cr_vals) if cr_vals else float("nan")
        avg_comp = sum(comp_vals) / len(comp_vals) if comp_vals else float("nan")
        print(
            f"[reward step {self._step:04d}] "
            f"R={avg_r:.3f} g={avg_g:.3f} copy={avg_c:.3f} "
            f"cite_p={avg_cp:.2f} cite_r={avg_cr:.2f} comp={avg_comp:.2f} "
            f"abst={abst:.2f} judge={judge_seconds:.1f}s (n={n})"
        )
        return rewards
