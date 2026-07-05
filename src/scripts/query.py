"""End-to-end retrieval smoke test.

Usage:
    python -m src.scripts.query "what are the diagnostic criteria for generalized anxiety disorder?"
    python -m src.scripts.query "..." --top-k 3 --config configs/retrieval.yaml
"""
from __future__ import annotations

import argparse
import textwrap

from dotenv import load_dotenv

from src.grounded_rag.retrieval import Retriever, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="the question to retrieve for")
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    retriever = Retriever(cfg)
    results = retriever.retrieve(args.query, top_n=args.top_n, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    for i, r in enumerate(results, 1):
        preview = textwrap.shorten(r.text, width=400, placeholder=" …")
        print(f"[{i}] rerank={r.rerank_score:.3f}  cos={r.score:.3f}  "
              f"source={r.source}  id={r.chunk_id}")
        print(f"    title: {r.title}")
        print(f"    text : {preview}\n")


if __name__ == "__main__":
    main()
