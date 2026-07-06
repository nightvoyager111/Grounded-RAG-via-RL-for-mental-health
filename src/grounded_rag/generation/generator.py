"""HF causal-LM wrapper for the grounded generator.

The wrapper is intentionally thin: it holds tokenizer + model, applies the
chat template, decodes greedily (or with sampling), and returns just the
assistant's new tokens. Training paths (DPO/GRPO) will subclass or reuse
the same tokenizer/model — do NOT reimplement prompt formatting there."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .prompt import Passage, build_messages


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 320
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"          
    dtype: str = "auto"            # "auto" | "float16" | "bfloat16" | "float32"
    seed: int = 20260704
    stop_strings: List[str] = field(default_factory=list)


def _resolve_dtype(name: str):
    import torch

    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


class HFGenerator:
    """Wraps a HuggingFace causal LM behind a `generate(question, passages)` API.

    Lazy-loads the model on first call so `import` is cheap for tests."""

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(self.cfg.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        kwargs = {}
        dtype = _resolve_dtype(self.cfg.dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)
        if self.cfg.device != "auto":
            self._model.to(self.cfg.device)
        self._model.eval()

    def generate(self, question: str, passages: Sequence[Passage]) -> str:
        self._load()
        import torch

        messages = build_messages(question, passages)
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else 1.0,
                top_p=self.cfg.top_p if self.cfg.do_sample else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        for s in self.cfg.stop_strings:
            if s in text:
                text = text.split(s)[0]
        return text.strip()
