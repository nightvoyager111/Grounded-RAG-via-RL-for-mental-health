from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class RetrievalConfig:
    embed_model: str
    embed_dim: int
    rerank_model: str
    top_n: int
    top_k: int
    embed_input_type_doc: str
    embed_input_type_query: str
    index_dir: str
    corpus_files: List[str] = field(default_factory=list)
    seed: int = 0
    cohere_client_version: str = "v2"


def load_config(path: str | Path) -> RetrievalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RetrievalConfig(**raw)
