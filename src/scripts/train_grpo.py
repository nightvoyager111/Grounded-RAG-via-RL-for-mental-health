"""TRL GRPO trainer for the grounded generator.

CLAUDE.md step 9. Rolls out G completions per prompt, scores each with
GroundednessReward (= judge - lambda * copy), and updates the policy.
Reward is *intentionally partial* here — the Act 2 story is watching
groundedness rise while the answer degrades (verbatim copying, terse
citations, or evasive abstentions). Step 10 adds `+ mu * helpfulness`.

Designed for Colab T4. On MPS/CPU it will run but very slowly.

Usage:
    python -m src.scripts.train_grpo
    python -m src.scripts.train_grpo --prompts data/grpo/prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "prompt": r["prompt"],
                "retrieved_ids": r["retrieved_ids"],
                "retrieved_texts": r["retrieved_texts"],
            })
    return out


def _load_model(base_model: str, dpo_adapter: str | None):
    """Load base + optionally merge the DPO LoRA adapter so GRPO starts
    from the Act 1 policy. We merge (not stack) because TRL GRPOTrainer
    will attach its own LoRA on top."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)

    if dpo_adapter:
        from peft import PeftModel
        print(f"Merging DPO adapter from {dpo_adapter}...")
        model = PeftModel.from_pretrained(model, dpo_adapter)
        model = model.merge_and_unload()
        print("DPO adapter merged.")
    return model, tokenizer


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo-config", default="configs/grpo.yaml")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = _load_yaml(args.grpo_config)
    prompts_path = args.prompts or cfg["prompts_out"]
    output_dir = args.output_dir or cfg["output_dir"]

    rows = _load_prompts(prompts_path)
    if not rows:
        raise SystemExit(f"No prompts in {prompts_path}. Run prepare_grpo_prompts.py first.")
    print(f"Loaded {len(rows)} rollout prompts from {prompts_path}")

    # --- Model (base + merged DPO adapter) ---
    model, tokenizer = _load_model(cfg["base_model"], cfg.get("dpo_adapter"))

    # --- Fresh LoRA on top for GRPO to train ---
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset ---
    from datasets import Dataset
    ds = Dataset.from_list(rows)

    # --- Reward ---
    from src.grounded_rag.verifier import LLMJudgeConfig, LLMJudgeVerifier
    from src.grounded_rag.training.grpo import GroundednessReward, RewardConfig

    judge = LLMJudgeVerifier(LLMJudgeConfig(
        model_name=cfg["judge_model_name"],
        temperature=cfg["judge_temperature"],
        api_key_env=cfg["judge_api_key_env"],
    ))
    reward = GroundednessReward(
        judge=judge.score,
        cfg=RewardConfig(
            copy_penalty_lambda=cfg["copy_penalty_lambda"],
            copy_ngram=cfg["copy_ngram"],
            judge_max_workers=cfg["judge_max_workers"],
            trace_path=cfg.get("reward_trace"),
            # Optional compound-reward knobs (grpo_v2.yaml). Default to
            # 0 so the old grpo.yaml still reproduces the v1 reward.
            citation_bonus_mu=cfg.get("citation_bonus_mu", 0.0),
            abstention_penalty_rho=cfg.get("abstention_penalty_rho", 0.0),
            abstention_ignore_score=cfg.get("abstention_ignore_score", 0.5),
        ),
    )

    # --- GRPO trainer ---
    import inspect
    import torch
    from trl import GRPOConfig, GRPOTrainer

    # TRL renames/removes GRPOConfig args between minor versions
    # (e.g. max_prompt_length → tokenizer-side truncation). Filter by
    # what this installed version actually accepts.
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_generations=cfg["num_generations"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        num_train_epochs=cfg["num_train_epochs"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        report_to=[],
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,   # keep retrieved_ids/texts for reward
    )
    accepted = set(inspect.signature(GRPOConfig).parameters)
    dropped = [k for k in grpo_kwargs if k not in accepted]
    if dropped:
        print(f"GRPOConfig dropped unsupported kwargs for this TRL version: {dropped}")
    grpo_cfg = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in accepted})
    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        reward_funcs=[reward],
        train_dataset=ds,
        processing_class=tokenizer,
    )
    try:
        trainer.train()
    finally:
        reward.close()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved GRPO LoRA adapter → {output_dir}")


if __name__ == "__main__":
    main()
