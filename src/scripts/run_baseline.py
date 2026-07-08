"""Run the baseline RAG eval: retrieve → generate → score.

Usage:
    python -m src.scripts.run_baseline \
        --retrieval-config configs/retrieval.yaml \
        --generation-config configs/generation.yaml \
        --eval-config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.grounded_rag.eval.runner import EvalConfig, run_eval
from src.grounded_rag.generation.generator import GenerationConfig, HFGenerator
from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config
from src.grounded_rag.verifier import (
    LLMJudgeConfig,
    LLMJudgeVerifier,
    NLIConfig,
    NLIVerifier,
    StubVerifier,
)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--generation-config", default="configs/generation.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--verifier-config", default="configs/verifier.yaml")
    ap.add_argument("--verifier", choices=["stub", "nli", "llm_judge"], default="stub",
                    help="stub=constant 0.5 placeholder (fast smoke test); "
                         "nli=DeBERTa-MNLI (calibrated poorly on this domain); "
                         "llm_judge=Cohere Command judge (the actual calibrated verifier).")
    ap.add_argument("--lora-adapter", default=None,
                    help="path to a PEFT-saved LoRA adapter (e.g. checkpoints/dpo). "
                         "Overrides generation.yaml if set.")
    ap.add_argument("--output-dir", default=None,
                    help="override eval output_dir (e.g. src/results/dpo-epoch5)")
    args = ap.parse_args()

    load_dotenv()

    retr_cfg = load_retrieval_config(args.retrieval_config)
    gen_cfg = GenerationConfig(**_load_yaml(args.generation_config))
    if args.lora_adapter:
        gen_cfg.lora_adapter = args.lora_adapter
    eval_raw = _load_yaml(args.eval_config)
    eval_cfg = EvalConfig(
        copy_ngram=eval_raw.get("copy_ngram", 8),
        output_dir=args.output_dir or eval_raw["output_dir"],
    )
    questions_file = eval_raw["questions_file"]

    retriever = Retriever(retr_cfg)
    generator = HFGenerator(gen_cfg)
    if args.verifier == "llm_judge":
        vraw = _load_yaml(args.verifier_config)
        verifier = LLMJudgeVerifier(LLMJudgeConfig(
            model_name=vraw["judge_model_name"],
            temperature=vraw["judge_temperature"],
            api_key_env=vraw["judge_api_key_env"],
        ))
    elif args.verifier == "nli":
        vraw = _load_yaml(args.verifier_config)
        verifier = NLIVerifier(NLIConfig(
            model_name=vraw["model_name"],
            device=vraw["device"],
            dtype=vraw["dtype"],
            max_length=vraw["max_length"],
            passage_join=vraw["passage_join"],
        ))
    else:
        verifier = StubVerifier()

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
