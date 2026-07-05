"""Build the vector index from configs/retrieval.yaml.

Usage:
    python -m src.scripts.build_index [--config configs/retrieval.yaml]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.grounded_rag.retrieval import build_index, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    embs, chunks = build_index(cfg)
    print(f"Indexed {len(chunks)} chunks → {cfg.index_dir} (dim={embs.shape[1]})")


if __name__ == "__main__":
    main()
