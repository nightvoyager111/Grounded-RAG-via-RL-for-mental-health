from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .config import RetrievalConfig
from .index import load_index


def _call_with_retry(fn: Callable, *, max_attempts: int = 6, base_sleep: float = 8.0):
    """Retry a Cohere client call on 429 / transient errors with exponential
    backoff. Trial keys allow 10 calls/min, so base_sleep=8s covers the worst
    case of hitting the limit right after a burst."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg or "too many" in msg or "rate limit" in msg
                         or "timeout" in msg or "temporarily" in msg)
            if not transient or attempt == max_attempts - 1:
                raise
            sleep_s = base_sleep * (2 ** attempt)
            time.sleep(sleep_s)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    source: str
    score: float
    rerank_score: Optional[float] = None


class Retriever:
    def __init__(self, cfg: RetrievalConfig, cohere_client=None, min_call_interval: float = 6.5):
        """min_call_interval: seconds to wait between successive API calls.
        Trial keys allow 10 calls/min; each retrieve() makes 2 calls (embed +
        rerank), so 6.5s gap keeps us safely under the ceiling."""
        self.cfg = cfg
        if cohere_client is None:
            import cohere

            cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.client = cohere_client
        self.embs, self.chunks = load_index(cfg)
        self.min_call_interval = float(min_call_interval)
        self._last_call_at = 0.0

    def _pace(self) -> None:
        gap = time.time() - self._last_call_at
        if gap < self.min_call_interval:
            time.sleep(self.min_call_interval - gap)
        self._last_call_at = time.time()

    def _embed_query(self, query: str) -> np.ndarray:
        self._pace()
        resp = _call_with_retry(lambda: self.client.embed(
            texts=[query],
            model=self.cfg.embed_model,
            input_type=self.cfg.embed_input_type_query,
            output_dimension=self.cfg.embed_dim,
            embedding_types=["float"],
        ))
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

        self._pace()
        rerank = _call_with_retry(lambda: self.client.rerank(
            model=self.cfg.rerank_model,
            query=query,
            documents=cand_texts,
            top_n=top_k,
        ))

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
