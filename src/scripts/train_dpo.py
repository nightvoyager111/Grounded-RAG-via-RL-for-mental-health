"""TRL DPO trainer for the grounded generator.

CLAUDE.md step 7. Reads (prompt, chosen, rejected) triples produced by
build_dpo_pairs.py, trains a LoRA adapter on the base generator, and saves
it to `checkpoints/dpo/`.

Designed to run on Colab. Local Mac (MPS) will work but slowly; Unsloth is
CUDA-only, so on MPS we fall back to plain TRL + PEFT.

Usage:
    python -m src.scripts.train_dpo
    python -m src.scripts.train_dpo --pairs data/dpo/pairs.jsonl --output-dir checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import yaml


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_pairs(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
    return out


def _try_unsloth_load(model_name: str, max_length: int):
    """Return (model, tokenizer) via Unsloth if available and CUDA is present.
    Otherwise return (None, None) so caller falls back to plain HF+PEFT."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        from unsloth import FastLanguageModel  # type: ignore
    except Exception:
        return None, None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=True,
    )
    return model, tokenizer


def _plain_load(model_name: str):
    """Standard HF load path — works on CPU, CUDA, and MPS."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-config", default="configs/dpo.yaml")
    ap.add_argument("--pairs", default=None, help="override pairs path")
    ap.add_argument("--output-dir", default=None, help="override output dir")
    args = ap.parse_args()

    cfg = _load_yaml(args.dpo_config)
    pairs_path = args.pairs or cfg["pairs_out"]
    output_dir = args.output_dir or cfg["output_dir"]

    pairs = _load_pairs(pairs_path)
    if not pairs:
        raise SystemExit(f"No pairs in {pairs_path}. Run build_dpo_pairs.py first.")
    print(f"Loaded {len(pairs)} pairs from {pairs_path}")

    # --- Load model (Unsloth if available, else plain HF) ---
    model, tokenizer = (None, None)
    if cfg.get("use_unsloth", True):
        model, tokenizer = _try_unsloth_load(cfg["base_model"], cfg["max_length"])
        if model is not None:
            print("Loaded via Unsloth (CUDA + 4-bit).")
    if model is None:
        model, tokenizer = _plain_load(cfg["base_model"])
        print("Loaded via plain HF (Unsloth unavailable or non-CUDA device).")

    # --- LoRA ---
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    # Unsloth patches PEFT itself; only wrap here if we're on the plain path.
    if not hasattr(model, "peft_config"):
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # --- Dataset ---
    from datasets import Dataset

    ds = Dataset.from_list(pairs)

    # --- DPO trainer ---
    from trl import DPOConfig, DPOTrainer

    dpo_cfg = DPOConfig(
        output_dir=output_dir,
        beta=cfg["beta"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved LoRA adapter → {output_dir}")


if __name__ == "__main__":
    main()
