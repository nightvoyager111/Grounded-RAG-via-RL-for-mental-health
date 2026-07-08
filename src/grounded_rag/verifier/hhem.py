"""Vectara HHEM-2.1 groundedness verifier.

HHEM (Hughes Hallucination Evaluation Model) is trained specifically for
RAG-style faithfulness: given (premise, hypothesis) it returns a single
scalar in [0, 1] where higher = more consistent with the premise. This
matches our `(passages, answer) -> float` contract directly — no 3-way
softmax to reduce, no `P_entail - P_contradict` to normalize.

Uses `trust_remote_code=True` because HHEM ships a custom `predict`
method that batches (premise, hypothesis) pairs on the model side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class HHEMConfig:
    model_name: str = "vectara/hallucination_evaluation_model"
    revision: str | None = None       # pin if you want reproducibility
    device: str = "auto"              # "auto" | "cpu" | "cuda" | "mps"
    passage_join: str = "\n\n"        # only used when aggregate == "concat"
    # Same aggregation choice as the NLI verifier:
    #   "max"    — score answer against each passage, take max (default).
    #   "concat" — join passages, single call. HHEM handles long context
    #              better than DeBERTa-NLI but still benefits from "max".
    aggregate: str = "max"


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HHEMVerifier:
    """Groundedness verifier backed by Vectara HHEM-2.1.

    score(passages, answer) -> float in [0, 1]. Higher means the answer is
    more consistent with the passages. Calibration picks the threshold."""

    def __init__(self, cfg: HHEMConfig):
        self.cfg = cfg
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, PreTrainedModel

        # HHEM's custom modeling code targets an older transformers that used
        # `_tied_weights_keys`. Current transformers looks up
        # `all_tied_weights_keys` during from_pretrained finalization; shim it
        # so loading doesn't crash. Safe: the shim just forwards to the old
        # attribute (defaulting to []) — no behavior change for models that
        # already define `all_tied_weights_keys`.
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(
                lambda self: getattr(self, "_tied_weights_keys", None) or []
            )

        kwargs = {"trust_remote_code": True}
        if self.cfg.revision:
            kwargs["revision"] = self.cfg.revision
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, **kwargs
        )
        device = _pick_device(self.cfg.device)
        model.to(device).eval()
        self._model = model
        self._device = device

    def _predict(self, pairs):
        # HHEM's custom predict() handles tokenization + batching.
        scores = self._model.predict(pairs)
        # Returns a torch tensor of shape [N]; convert to list of floats.
        return [float(s) for s in scores.detach().cpu().tolist()]

    def score(self, passages: Sequence[str], answer: str) -> float:
        self._load()
        if not passages:
            return 0.5  # no evidence → neutral
        if self.cfg.aggregate == "concat":
            premise = self.cfg.passage_join.join(passages)
            return self._predict([(premise, answer)])[0]
        if self.cfg.aggregate == "max":
            scores = self._predict([(p, answer) for p in passages])
            return max(scores)
        raise ValueError(f"unknown aggregate: {self.cfg.aggregate!r}")

    def __call__(self, passages: Sequence[str], answer: str) -> float:
        return self.score(passages, answer)
