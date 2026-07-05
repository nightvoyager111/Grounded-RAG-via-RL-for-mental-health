from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import RetrievalConfig
from .index import load_index


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    source: str
    score: float
    rerank_score: Optional[float] = None


class Retriever:
    def __init__(self, cfg: RetrievalConfig, cohere_client=None):
        self.cfg = cfg
        if cohere_client is None:
            import cohere

            cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.client = cohere_client
        self.embs, self.chunks = load_index(cfg)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embed(
            texts=[query],
            model=self.cfg.embed_model,
            input_type=self.cfg.embed_input_type_query,
            output_dimension=self.cfg.embed_dim,
            embedding_types=["float"],
        )
        v = np.asarray(resp.embeddings.float[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def _vector_topn(self, qv: np.ndarray, n: int) -> List[int]:
        # embeddings are L2-normalized → dot product == cosine
        scores = self.embs @ qv
        if n >= len(scores):
            return list(np.argsort(-scores))
        idx = np.argpartition(-scores, n)[:n]
        return list(idx[np.argsort(-scores[idx])])

    def retrieve(self, query: str, top_n: Optional[int] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_n = top_n or self.cfg.top_n
        top_k = top_k or self.cfg.top_k

        qv = self._embed_query(query)
        cand_idx = self._vector_topn(qv, top_n)
        cand_texts = [self.chunks[i]["chunk_text"] for i in cand_idx]
        cand_scores = (self.embs[cand_idx] @ qv).tolist()

        rerank = self.client.rerank(
            model=self.cfg.rerank_model,
            query=query,
            documents=cand_texts,
            top_n=top_k,
        )

        out: List[RetrievedChunk] = []
        for r in rerank.results:
            local = r.index
            global_i = cand_idx[local]
            c = self.chunks[global_i]
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["chunk_text"],
                    title=c.get("title", ""),
                    source=c.get("source", ""),
                    score=float(cand_scores[local]),
                    rerank_score=float(r.relevance_score),
                )
            )
        return out
