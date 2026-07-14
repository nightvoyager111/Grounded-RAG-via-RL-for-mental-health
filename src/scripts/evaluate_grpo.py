"""Evaluate a GRPO-trained policy on the same 25-question harness used
for baseline + DPO. Produces the third column of the CLAUDE.md step-11 table.

Wire matters here: `train_grpo.py` merges the DPO LoRA into the base
weights *before* attaching a fresh LoRA for GRPO to train. To reproduce
that policy at eval time we must do the same stack — base → merge(DPO)
→ load(GRPO) — otherwise the GRPO adapter lands on the wrong base
weights and the numbers are garbage. That's why this can't be
`run_baseline.py --lora-adapter checkpoints/grpo`.

Usage:
    python -m src.scripts.evaluate_grpo
    python -m src.scripts.evaluate_grpo --grpo-adapter checkpoints/grpo --output-dir src/results/grpo
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import yaml
from dotenv import load_dotenv

from src.grounded_rag.eval.runner import EvalConfig, run_eval
from src.grounded_rag.generation.generator import GenerationConfig, HFGenerator
from src.grounded_rag.generation.prompt import Passage
from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config
from src.grounded_rag.verifier import LLMJudgeConfig, LLMJudgeVerifier


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_questions(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["question"]


class StackedAdapterGenerator(HFGenerator):
    """HFGenerator variant that merges a first LoRA (DPO) into the base
    before attaching a second LoRA (GRPO). Same public API — run_eval
    only needs `.generate(question, passages)`."""

    def __init__(self, cfg: GenerationConfig, base_adapter: str | None, top_adapter: str | None):
        super().__init__(cfg)
        self._base_adapter = base_adapter
        self._top_adapter = top_adapter

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(self.cfg.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        kwargs = {}
        from src.grounded_rag.generation.generator import _resolve_dtype
        dtype = _resolve_dtype(self.cfg.dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)

        if self._base_adapter:
            from peft import PeftModel
            print(f"Merging base adapter from {self._base_adapter}...")
            model = PeftModel.from_pretrained(model, self._base_adapter)
            model = model.merge_and_unload()
        if self._top_adapter:
            from peft import PeftModel
            print(f"Attaching top adapter from {self._top_adapter}...")
            model = PeftModel.from_pretrained(model, self._top_adapter)

        self._model = model
        if self.cfg.device != "auto":
            self._model.to(self.cfg.device)
        self._model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--generation-config", default="configs/generation.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--verifier-config", default="configs/verifier.yaml")
    ap.add_argument("--grpo-config", default="configs/grpo.yaml")
    ap.add_argument("--dpo-adapter", default=None,
                    help="override configs/grpo.yaml:dpo_adapter (set to '' to skip DPO merge)")
    ap.add_argument("--grpo-adapter", default=None,
                    help="override configs/grpo.yaml:output_dir")
    ap.add_argument("--output-dir", default="src/results/grpo")
    ap.add_argument("--questions-file", default=None,
                    help="override eval questions_file (e.g. the n=200 pool)")
    args = ap.parse_args()

    load_dotenv()

    retr_cfg = load_retrieval_config(args.retrieval_config)
    gen_cfg = GenerationConfig(**_load_yaml(args.generation_config))
    eval_raw = _load_yaml(args.eval_config)
    grpo_raw = _load_yaml(args.grpo_config)

    dpo_adapter = args.dpo_adapter if args.dpo_adapter is not None else grpo_raw.get("dpo_adapter")
    if dpo_adapter == "":
        dpo_adapter = None
    grpo_adapter = args.grpo_adapter or grpo_raw["output_dir"]

    eval_cfg = EvalConfig(
        copy_ngram=eval_raw.get("copy_ngram", 8),
        output_dir=args.output_dir,
    )

    retriever = Retriever(retr_cfg)
    generator = StackedAdapterGenerator(gen_cfg, base_adapter=dpo_adapter, top_adapter=grpo_adapter)

    vraw = _load_yaml(args.verifier_config)
    verifier = LLMJudgeVerifier(LLMJudgeConfig(
        model_name=vraw["judge_model_name"],
        temperature=vraw["judge_temperature"],
        api_key_env=vraw["judge_api_key_env"],
    ))

    questions_file = args.questions_file or eval_raw["questions_file"]
    report = run_eval(
        questions=_iter_questions(questions_file),
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        cfg=eval_cfg,
    )
    print(json.dumps(report, indent=2))
    print(f"\nRows: {Path(eval_cfg.output_dir) / 'rows.jsonl'}")
    print(f"Report: {Path(eval_cfg.output_dir) / 'report.json'}")


if __name__ == "__main__":
    main()
