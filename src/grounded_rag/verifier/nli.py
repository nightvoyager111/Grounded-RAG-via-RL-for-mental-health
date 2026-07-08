"""NLI-based groundedness verifier.

Per CLAUDE.md: passages as premise, answer as hypothesis, three-way NLI
softmax, score = P(entail) - P(contradict), rescaled to [0, 1] so it plugs
into the same `(passages, answer) -> float` contract StubVerifier defines.

This is entailment, NOT cosine similarity. Do not swap in embedding
similarity here — that is a retrieval concern, not a faithfulness one.

Model default: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli — 184M params,
label order [entailment, neutral, contradiction] per the model card. We
detect label order at load time from id2label so a different NLI checkpoint
still scores correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class NLIConfig:
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"        # NLI heads are small; fp32 avoids softmax noise
    max_length: int = 512
    passage_join: str = "\n\n"    # used only when aggregate == "concat"
    # How to combine multiple passages into a groundedness score:
    #   "concat" — legacy: join all passages, run NLI once. Noisy on long premises.
    #   "max"    — score answer against each passage separately, take the max.
    #              Standard trick for RAG groundedness: an answer is grounded if
    #              *any* passage entails it. Reduces long-premise dilution.
    aggregate: str = "max"


def _resolve_dtype(name: str):
    import torch

    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLIVerifier:
    """Groundedness verifier via a 3-way NLI model.

    score(passages, answer) returns a scalar in [0, 1]:
        raw = P(entail) - P(contradict)   ∈ [-1, 1]
        score = (raw + 1) / 2             ∈ [0, 1]

    So 0.5 = neutral (equal entail/contradict mass), > 0.5 = grounded,
    < 0.5 = contradicted. The calibration harness picks the decision
    threshold; do not hard-code 0.5 as "grounded"."""

    def __init__(self, cfg: NLIConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None
        self._label_map = None  # e.g. {"entailment": 0, "neutral": 1, "contradiction": 2}
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        kwargs = {}
        dtype = _resolve_dtype(self.cfg.dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, **kwargs
        )
        device = _pick_device(self.cfg.device)
        model.to(device).eval()

        # Detect label positions from id2label; NLI checkpoints disagree on
        # which index is entail vs contradict (MNLI vs some XNLI variants).
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self._label_map = {v: k for k, v in id2label.items()}
        for req in ("entailment", "contradiction"):
            if req not in self._label_map:
                raise ValueError(
                    f"NLI model {self.cfg.model_name} does not expose an "
                    f"'{req}' label (got {id2label}). Choose a 3-way NLI model."
                )

        self._tokenizer = tok
        self._model = model
        self._device = device

    def _premise(self, passages: Sequence[str]) -> str:
        return self.cfg.passage_join.join(passages)

    def _score_one(self, premise: str, answer: str) -> float:
        import torch

        enc = self._tokenizer(
            premise,
            answer,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        p_entail = probs[self._label_map["entailment"]].item()
        p_contra = probs[self._label_map["contradiction"]].item()
        raw = p_entail - p_contra
        return (raw + 1.0) / 2.0

    def score(self, passages: Sequence[str], answer: str) -> float:
        self._load()
        if not passages:
            return 0.5  # no evidence → neutral
        if self.cfg.aggregate == "concat":
            return self._score_one(self._premise(passages), answer)
        if self.cfg.aggregate == "max":
            return max(self._score_one(p, answer) for p in passages)
        raise ValueError(f"unknown aggregate: {self.cfg.aggregate!r}")

    def __call__(self, passages: Sequence[str], answer: str) -> float:
        return self.score(passages, answer)

    def score_many(self, items: Sequence[tuple]) -> List[float]:
        """items: iterable of (passages, answer). Convenience for calibration."""
        return [self.score(p, a) for p, a in items]
