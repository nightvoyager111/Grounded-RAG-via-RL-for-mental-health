"""Materialize a GRPO rollout dataset from the question set.

CLAUDE.md Act 2 prep. Retrieval is expensive (Cohere Embed + Rerank) and
we don't want to re-run it every GRPO step. This script does it once per
question and writes a JSONL that train_grpo.py consumes directly:

    {"prompt": <chat-templated str>, "question": ..., "retrieved_ids": [...],
     "retrieved_texts": [...]}

Reusing the DPO prompt renderer means the tokenized string GRPOTrainer
sees at rollout time is identical to what the baseline generator saw
during eval — no silent drift in prompt shape.

Usage:
    python -m src.scripts.prepare_grpo_prompts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from src.grounded_rag.generation.generator import GenerationConfig, HFGenerator
from src.grounded_rag.generation.prompt import Passage, build_messages
from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_questions(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo-config", default="configs/grpo.yaml")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--generation-config", default="configs/generation.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_dotenv()
    gcfg = _load_yaml(args.grpo_config)
    retr_cfg = load_retrieval_config(args.retrieval_config)
    gen_raw = _load_yaml(args.generation_config)

    questions = _load_questions(gcfg["questions_file"])
    if args.limit:
        questions = questions[: args.limit]
    print(f"Preparing {len(questions)} rollout prompts...")

    retriever = Retriever(retr_cfg)
    # We only need the tokenizer for chat templating; skip loading weights
    # if the constructor allows it. HFGenerator._load() lazily loads model
    # too, but on a Mac this is fine for a one-shot prep.
    generator = HFGenerator(GenerationConfig(**gen_raw))
    generator._load()
    tokenizer = generator._tokenizer

    out_path = Path(gcfg["prompts_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in questions:
            q = row["question"]
            retrieved = retriever.retrieve(q)
            passages = [Passage(r.chunk_id, r.title, r.text) for r in retrieved]
            prompt = tokenizer.apply_chat_template(
                build_messages(q, passages),
                tokenize=False,
                add_generation_prompt=True,
            )
            f.write(json.dumps({
                "question": q,
                "prompt": prompt,
                "retrieved_ids": [p.chunk_id for p in passages],
                "retrieved_texts": [p.text for p in passages],
            }) + "\n")
            n += 1
    print(f"Wrote {n} rows → {out_path}")


if __name__ == "__main__":
    main()
