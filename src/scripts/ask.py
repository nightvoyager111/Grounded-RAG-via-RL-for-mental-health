"""Ask one question interactively against any of baseline / DPO / GRPO.

Wires Retriever → HFGenerator (or StackedAdapterGenerator for GRPO, which
needs base → merge(DPO) → load(GRPO) per evaluate_grpo.py) and prints
the answer plus the retrieved passages it was allowed to cite.

"""
from __future__ import annotations

import argparse
import textwrap

import yaml
from dotenv import load_dotenv

from src.grounded_rag.generation.generator import GenerationConfig, HFGenerator
from src.grounded_rag.generation.prompt import Passage
from src.grounded_rag.retrieval import Retriever, load_config as load_retrieval_config
from src.scripts.evaluate_grpo import StackedAdapterGenerator


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", choices=["baseline", "dpo", "grpo"], default="grpo")
    ap.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    ap.add_argument("--generation-config", default="configs/generation.yaml")
    ap.add_argument("--grpo-config", default="configs/grpo_v3.yaml")
    ap.add_argument("--dpo-adapter", default="checkpoints/dpo")
    ap.add_argument("--grpo-adapter", default=None,
                    help="override configs/grpo.yaml:output_dir")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--sample", action="store_true",
                    help="turn on temperature sampling — makes baseline vs "
                         "DPO vs GRPO differences more visible than greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    load_dotenv()

    retr_cfg = load_retrieval_config(args.retrieval_config)
    gen_cfg = GenerationConfig(**_load_yaml(args.generation_config))
    if args.sample:
        gen_cfg.do_sample = True
        gen_cfg.temperature = args.temperature

    if args.model == "baseline":
        generator = HFGenerator(gen_cfg)
    elif args.model == "dpo":
        gen_cfg.lora_adapter = args.dpo_adapter
        generator = HFGenerator(gen_cfg)
    else:
        grpo_raw = _load_yaml(args.grpo_config)
        grpo_adapter = args.grpo_adapter or grpo_raw["output_dir"]
        generator = StackedAdapterGenerator(
            gen_cfg, base_adapter=args.dpo_adapter, top_adapter=grpo_adapter
        )

    retriever = Retriever(retr_cfg)
    results = retriever.retrieve(args.question, top_n=args.top_n, top_k=args.top_k)
    passages = [Passage(chunk_id=r.chunk_id, title=r.title, text=r.text) for r in results]

    answer = generator.generate(args.question, passages)

    print(f"\nModel: {args.model}")
    print(f"Question: {args.question}\n")
    print("Retrieved passages:")
    for i, p in enumerate(passages, 1):
        preview = textwrap.shorten(p.text, width=200, placeholder=" …")
        print(f"  [{i}] {p.chunk_id}  {p.title}")
        print(f"      {preview}")
    print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
