from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .config import RetrievalConfig

EMBED_BATCH = 96


def _corpus_signature(chunks: List[dict], cfg: RetrievalConfig) -> str:
    """Fingerprint the inputs that would change the embeddings. If this matches
    the fingerprint on disk, re-embedding is a waste of API calls."""
    h = hashlib.sha256()
    h.update(f"{cfg.embed_model}|{cfg.embed_dim}|{cfg.embed_input_type_doc}".encode())
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(b"\0")
        h.update(c["chunk_text"].encode())
        h.update(b"\0")
    return h.hexdigest()


def _iter_chunks(paths: Iterable[str]) -> List[dict]:
    chunks: List[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
    return chunks


def _embed_batches(client, texts: List[str], model: str, input_type: str, dim: int) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embed(
            texts=batch,
            model=model,
            input_type=input_type,
            output_dimension=dim,
            embedding_types=["float"],
        )
        vecs.extend(resp.embeddings.float)
    arr = np.asarray(vecs, dtype=np.float32)
    # L2-normalize so cosine == dot product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_index(
    cfg: RetrievalConfig, cohere_client=None, force: bool = False
) -> Tuple[np.ndarray, List[dict]]:
    chunks = _iter_chunks(cfg.corpus_files)
    sig = _corpus_signature(chunks, cfg)

    out_dir = Path(cfg.index_dir)
    meta_path = out_dir / "meta.json"
    if not force and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("corpus_signature") == sig and (out_dir / "embeddings.npy").exists():
            embs = np.load(out_dir / "embeddings.npy")
            return embs, chunks

    if cohere_client is None:
        import cohere

        cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    texts = [c["chunk_text"] for c in chunks]
    embs = _embed_batches(
        cohere_client, texts, cfg.embed_model, cfg.embed_input_type_doc, cfg.embed_dim
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embs)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    meta = {
        "embed_model": cfg.embed_model,
        "embed_dim": cfg.embed_dim,
        "n_chunks": len(chunks),
        "corpus_signature": sig,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return embs, chunks


def load_index(cfg: RetrievalConfig) -> Tuple[np.ndarray, List[dict]]:
    d = Path(cfg.index_dir)
    embs = np.load(d / "embeddings.npy")
    with open(d / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return embs, chunks
