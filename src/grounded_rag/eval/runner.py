"""End-to-end eval harness: retrieve → generate → score.

Separates the pieces so tests can swap in fakes:
- `run_eval(...)` takes a retriever, a generator, a verifier, a question
  iterable, and writes per-example rows + an aggregate report.
- Neither the retriever nor the generator is constructed here; the caller
  wires them. That keeps this file free of network / model imports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Protocol, Sequence

from ..generation.prompt import Passage
from .metrics import aggregate, score_example


class RetrieverLike(Protocol):
    def retrieve(self, query: str): ...  # returns list w/ .chunk_id, .text, .title


class GeneratorLike(Protocol):
    def generate(self, question: str, passages: Sequence[Passage]) -> str: ...


@dataclass
class EvalConfig:
    copy_ngram: int = 8
    output_dir: str = "src/results/baseline"


def _to_passages(retrieved) -> List[Passage]:
    return [
        Passage(chunk_id=r.chunk_id, title=r.title, text=r.text) for r in retrieved
    ]


def run_eval(
    *,
    questions: Iterable[str],
    retriever: RetrieverLike,
    generator: GeneratorLike,
    verifier: Callable[[Sequence[str], str], float],
    cfg: EvalConfig,
) -> dict:
    """Score every question. Writes rows.jsonl + report.json under output_dir."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    rows: List[dict] = []
    with open(rows_path, "w", encoding="utf-8") as f:
        for q in questions:
            retrieved = retriever.retrieve(q)
            passages = _to_passages(retrieved)
            answer = generator.generate(q, passages)

            row = score_example(
                question=q,
                answer=answer,
                retrieved_ids=[p.chunk_id for p in passages],
                retrieved_texts=[p.text for p in passages],
                verifier=verifier,
                copy_ngram=cfg.copy_ngram,
            )
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    report = aggregate(rows)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
