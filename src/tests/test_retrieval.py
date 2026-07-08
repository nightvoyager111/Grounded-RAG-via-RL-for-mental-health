"""Unit tests for the retrieval module.

Run:  pytest        (from the repo root)

Testing 101 for this file:
- Each `test_*` function is one test. Pytest discovers them automatically.
- We use `assert` — if the condition is False, the test fails.
- `tmp_path` is a pytest fixture: a fresh temporary directory per test.
  Anything we write there is cleaned up automatically, so tests don't
  pollute the repo or each other.
- We never call the real Cohere API in unit tests. Real API calls are
  slow, cost money, need network, and give non-deterministic results.
  Instead we build a *fake* client (`FakeCohereClient`) that mimics the
  shape of the real one. This is called "stubbing" or "mocking".
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from src.grounded_rag.retrieval.config import RetrievalConfig, load_config
from src.grounded_rag.retrieval.index import (
    _corpus_signature,
    _iter_chunks,
    build_index,
)
from src.grounded_rag.retrieval.search import Retriever


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeCohereClient:
    """Mimics the parts of cohere.ClientV2 that our code uses.

    - `embed(texts=..., ...)` returns an object with `.embeddings.float`
      shaped like the real SDK: a list of vectors (one per input text).
    - `rerank(query=..., documents=..., top_n=...)` returns an object with
      a `.results` list, each element has `.index` (into `documents`) and
      `.relevance_score`.

    We control what vectors and rerank scores it returns so tests are
    deterministic. We also count calls so we can assert e.g. the cache
    was used (0 embed calls means we short-circuited)."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.embed_calls = 0
        self.rerank_calls = 0

    def embed(self, *, texts, model, input_type, output_dimension, embedding_types):
        self.embed_calls += 1
        # Deterministic pseudo-embedding: seed off the text so two calls with
        # the same text give the same vector. Real Cohere embeddings are
        # meaningful; ours just need to be stable + distinct per input.
        vecs: List[List[float]] = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vecs.append(rng.standard_normal(output_dimension).tolist())
        return SimpleNamespace(embeddings=SimpleNamespace(float=vecs))

    def rerank(self, *, model, query, documents, top_n):
        self.rerank_calls += 1
        # Deterministic "relevance": prefer shorter documents. We just need
        # SOME ordering so we can check the retriever wires indices right.
        scored = sorted(
            enumerate(documents), key=lambda kv: len(kv[1])
        )[:top_n]
        results = [
            SimpleNamespace(index=i, relevance_score=1.0 / (1 + len(doc)))
            for i, doc in scored
        ]
        return SimpleNamespace(results=results)


def _make_chunks(n: int) -> List[dict]:
    return [
        {
            "chunk_id": f"c:{i}",
            "source": "test",
            "title": f"Title {i}",
            "chunk_text": f"chunk text number {i}" + (" extra" * (i % 3)),
        }
        for i in range(n)
    ]


def _make_cfg(tmp_path, corpus_files: List[str]) -> RetrievalConfig:
    return RetrievalConfig(
        embed_model="fake-embed",
        embed_dim=4,
        rerank_model="fake-rerank",
        top_n=5,
        top_k=2,
        embed_input_type_doc="search_document",
        embed_input_type_query="search_query",
        index_dir=str(tmp_path / "index"),
        corpus_files=corpus_files,
    )


def _write_corpus(tmp_path, chunks: List[dict]) -> str:
    p = tmp_path / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return str(p)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    """load_config() should parse the YAML file into a RetrievalConfig."""
    p = tmp_path / "retrieval.yaml"
    p.write_text(
        "embed_model: embed-v4.0\n"
        "embed_dim: 1024\n"
        "rerank_model: rerank-v3.5\n"
        "top_n: 50\n"
        "top_k: 5\n"
        "embed_input_type_doc: search_document\n"
        "embed_input_type_query: search_query\n"
        "index_dir: data/index\n"
        "corpus_files:\n"
        "  - a.jsonl\n"
        "  - b.jsonl\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.embed_model == "embed-v4.0"
    assert cfg.embed_dim == 1024
    assert cfg.top_n == 50
    assert cfg.corpus_files == ["a.jsonl", "b.jsonl"]
    # Optional fields fall back to defaults (seed=0, client v2)
    assert cfg.seed == 0
    assert cfg.cohere_client_version == "v2"


# ---------------------------------------------------------------------------
# index.py — corpus loading + signature + caching
# ---------------------------------------------------------------------------


def test_iter_chunks_reads_multiple_files_and_skips_blank_lines(tmp_path):
    """_iter_chunks should concatenate JSONL files and ignore empty lines."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"chunk_id":"a:0","chunk_text":"hi"}\n\n', encoding="utf-8")
    b.write_text('{"chunk_id":"b:0","chunk_text":"yo"}\n', encoding="utf-8")
    chunks = _iter_chunks([str(a), str(b)])
    assert [c["chunk_id"] for c in chunks] == ["a:0", "b:0"]


def test_corpus_signature_is_stable_and_content_sensitive(tmp_path):
    """The signature must change when content changes, and NOT change when
    it doesn't. This is the whole point of the cache-skip in build_index."""
    cfg = _make_cfg(tmp_path, corpus_files=[])
    chunks = _make_chunks(3)

    sig_a = _corpus_signature(chunks, cfg)
    sig_b = _corpus_signature(chunks, cfg)
    assert sig_a == sig_b, "same input should give same signature"

    # Change one character of one chunk → signature must change.
    chunks[1] = {**chunks[1], "chunk_text": chunks[1]["chunk_text"] + "!"}
    sig_c = _corpus_signature(chunks, cfg)
    assert sig_c != sig_a, "content change must invalidate the signature"


def test_build_index_writes_expected_files_and_shape(tmp_path):
    """First-time build: should call embed, write embeddings.npy of the right
    shape, chunks.jsonl in the right order, and a meta.json with a signature."""
    chunks = _make_chunks(6)
    corpus = _write_corpus(tmp_path, chunks)
    cfg = _make_cfg(tmp_path, corpus_files=[corpus])
    fake = FakeCohereClient(dim=cfg.embed_dim)

    embs, out_chunks = build_index(cfg, cohere_client=fake)

    assert fake.embed_calls == 1, "6 chunks fit in one batch of 96"
    assert embs.shape == (6, cfg.embed_dim)
    # L2-normalized → each row has unit length
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # On-disk artifacts exist and match
    from pathlib import Path

    d = Path(cfg.index_dir)
    assert (d / "embeddings.npy").exists()
    assert (d / "chunks.jsonl").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["n_chunks"] == 6
    assert meta["embed_dim"] == cfg.embed_dim
    assert "corpus_signature" in meta


def test_build_index_uses_cache_on_second_call(tmp_path):
    """Second build with unchanged corpus should NOT call the embed API."""
    corpus = _write_corpus(tmp_path, _make_chunks(4))
    cfg = _make_cfg(tmp_path, corpus_files=[corpus])

    fake1 = FakeCohereClient(dim=cfg.embed_dim)
    build_index(cfg, cohere_client=fake1)
    assert fake1.embed_calls == 1

    fake2 = FakeCohereClient(dim=cfg.embed_dim)
    build_index(cfg, cohere_client=fake2)
    assert fake2.embed_calls == 0, "signature match should skip re-embedding"


def test_build_index_reembeds_when_corpus_changes(tmp_path):
    """If the corpus content changes, cache must be invalidated."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"chunk_id":"c:0","chunk_text":"original"}\n', encoding="utf-8"
    )
    cfg = _make_cfg(tmp_path, corpus_files=[str(corpus)])

    fake1 = FakeCohereClient(dim=cfg.embed_dim)
    build_index(cfg, cohere_client=fake1)

    # Modify the corpus.
    corpus.write_text(
        '{"chunk_id":"c:0","chunk_text":"CHANGED"}\n', encoding="utf-8"
    )
    fake2 = FakeCohereClient(dim=cfg.embed_dim)
    build_index(cfg, cohere_client=fake2)
    assert fake2.embed_calls == 1, "content change must trigger re-embed"


def test_build_index_force_flag_bypasses_cache(tmp_path):
    """force=True should re-embed even if the signature matches."""
    corpus = _write_corpus(tmp_path, _make_chunks(2))
    cfg = _make_cfg(tmp_path, corpus_files=[corpus])

    build_index(cfg, cohere_client=FakeCohereClient(dim=cfg.embed_dim))
    fake = FakeCohereClient(dim=cfg.embed_dim)
    build_index(cfg, cohere_client=fake, force=True)
    assert fake.embed_calls == 1


# ---------------------------------------------------------------------------
# search.py — retriever
# ---------------------------------------------------------------------------


@pytest.fixture
def built_retriever(tmp_path):
    """Fixture: build an index and return a Retriever wired to a fake client.

    A pytest fixture is a reusable setup step. Any test that lists it as a
    parameter gets the return value injected. Keeps tests DRY."""
    chunks = _make_chunks(10)
    corpus = _write_corpus(tmp_path, chunks)
    cfg = _make_cfg(tmp_path, corpus_files=[corpus])
    build_index(cfg, cohere_client=FakeCohereClient(dim=cfg.embed_dim))
    fake = FakeCohereClient(dim=cfg.embed_dim)
    return Retriever(cfg, cohere_client=fake), fake, cfg


def test_retriever_loads_index_on_init(built_retriever):
    retr, _fake, cfg = built_retriever
    assert retr.embs.shape == (10, cfg.embed_dim)
    assert len(retr.chunks) == 10


def test_retrieve_returns_top_k_in_rerank_order(built_retriever):
    """Contract: retrieve() returns exactly top_k RetrievedChunk objects,
    in the order the reranker produced (rerank_score descending)."""
    retr, _fake, cfg = built_retriever
    results = retr.retrieve("what is anxiety?")
    assert len(results) == cfg.top_k
    # rerank_score should be monotonically non-increasing
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True), scores


def test_retrieve_hits_embed_once_and_rerank_once(built_retriever):
    """One user query should trigger exactly one embed call and one rerank
    call. Catches regressions like embedding each candidate individually."""
    retr, fake, _cfg = built_retriever
    retr.retrieve("something")
    assert fake.embed_calls == 1
    assert fake.rerank_calls == 1


def test_retrieve_result_fields_come_from_the_right_chunk(built_retriever):
    """Each result's chunk_id/title/text must match a chunk that actually
    exists in the index. Guards against index-arithmetic bugs where local
    (post-topN) indices get confused with global (corpus) indices."""
    retr, _fake, _cfg = built_retriever
    ids_in_corpus = {c["chunk_id"] for c in retr.chunks}
    by_id = {c["chunk_id"]: c for c in retr.chunks}
    for r in retr.retrieve("anything"):
        assert r.chunk_id in ids_in_corpus
        assert r.text == by_id[r.chunk_id]["chunk_text"]
        assert r.title == by_id[r.chunk_id]["title"]


def test_retrieve_respects_topk_override(built_retriever):
    retr, _fake, _cfg = built_retriever
    results = retr.retrieve("q", top_k=3)
    assert len(results) == 3


def test_vector_topn_returns_sorted_indices(built_retriever):
    """_vector_topn is the pre-rerank stage. Given a query vector, it must
    return the n indices with highest dot product, sorted best-first."""
    retr, _fake, _cfg = built_retriever
    # Pick a query = one of the stored vectors → it should be its own top-1.
    target = 3
    qv = retr.embs[target]
    top = retr._vector_topn(qv, n=5)
    assert top[0] == target
    # Scores at the returned indices should be monotonically non-increasing.
    got = retr.embs[top] @ qv
    assert list(got) == sorted(got, reverse=True)
