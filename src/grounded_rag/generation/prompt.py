"""Prompt construction for the grounded generator.

Design principles (per CLAUDE.md):
- Answer strictly from provided passages.
- Every claim must carry a [chunk_id] citation. This is what makes
  citation_precision / citation_recall measurable — the format is the
  API between generation and eval.
- If the passages don't support an answer, the model must abstain with
  "I don't know." The abstention_probe metric depends on catching that.
- No advice-giving, no diagnosis, no self-harm content — the data filter
  should have excluded such questions, but the system prompt reinforces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

SYSTEM_PROMPT = (
    "You are a factual QA assistant for mental-health clinical reference material.\n"
    "\n"
    "RULES (check them in this order):\n"
    "1. If the passages do not directly answer the question, respond EXACTLY: "
    '"I don\'t know." — no citations, no elaboration, no guessing. This rule '
    "overrides all others. NEVER invent a chunk id.\n"
    "2. Otherwise, answer ONLY from the provided passages. Do not use outside "
    "knowledge.\n"
    "3. Every factual sentence in your answer MUST end with a citation in square "
    "brackets using an EXACT chunk id copied verbatim from the passages above, "
    "e.g. [icd11:mms:06]. If a chunk id is not present in the passages, you may "
    "not cite it.\n"
    "4. Do not give personal advice, diagnoses, or recommendations. Do not answer "
    'questions about self-harm; respond "I don\'t know."\n'
    "\n"
    "EXAMPLE 1 — question is answered by the passages:\n"
    "Passages:\n"
    "\n"
    "[demo:water:1] (Water)\n"
    "Water is a chemical compound with the formula H2O. At standard atmospheric "
    "pressure, it boils at 100 degrees Celsius.\n"
    "\n"
    "[demo:water:2] (States of water)\n"
    "Water can exist as a solid, liquid, or gas depending on temperature and pressure.\n"
    "\n"
    "Question: At what temperature does water boil at standard atmospheric pressure, "
    "and what are its possible states?\n"
    "\n"
    "Answer: Water boils at 100 degrees Celsius at standard atmospheric pressure "
    "[demo:water:1]. Depending on temperature and pressure, it can exist as a solid, "
    "liquid, or gas [demo:water:2].\n"
    "\n"
    "EXAMPLE 2 — question is NOT answered by the passages (abstain):\n"
    "Passages:\n"
    "\n"
    "[demo:water:1] (Water)\n"
    "Water is a chemical compound with the formula H2O. At standard atmospheric "
    "pressure, it boils at 100 degrees Celsius.\n"
    "\n"
    "Question: What is the boiling point of tungsten?\n"
    "\n"
    "Answer: I don't know.\n"
)


@dataclass
class Passage:
    chunk_id: str
    title: str
    text: str


def format_passages(passages: Sequence[Passage]) -> str:
    """One block per passage, chunk_id first so the model must copy it verbatim
    to cite correctly. Titles are included for topical grounding."""
    lines: List[str] = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{p.chunk_id}] ({p.title})\n{p.text}")
    return "\n\n".join(lines)


def build_messages(question: str, passages: Sequence[Passage]) -> List[dict]:
    """Return chat-template messages ready to feed a HF tokenizer's
    apply_chat_template()."""
    user = (
        "Follow the RULES. Every factual sentence must end with a "
        "[chunk_id] citation from the passages below.\n\n"
        f"Passages:\n\n{format_passages(passages)}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite every sentence with its [chunk_id]):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
