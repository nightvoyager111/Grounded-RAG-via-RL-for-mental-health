"""Tests for the generation module.

We test prompt construction and message shape — NOT model output. Loading
Qwen2.5-1.5B in a unit test would be slow and defeats the point of a fast
test suite. The generator's `generate()` method is exercised end-to-end
via a fake in test_eval.py's test_run_eval_writes_rows_and_report.
"""
from __future__ import annotations

from src.grounded_rag.generation.prompt import (
    Passage,
    SYSTEM_PROMPT,
    build_messages,
    format_passages,
)


def test_format_passages_starts_each_block_with_chunk_id():
    ps = [
        Passage("a:1", "Anxiety", "worry is the core feature"),
        Passage("b:2", "Depression", "low mood is central"),
    ]
    out = format_passages(ps)
    assert "[a:1] (Anxiety)" in out
    assert "[b:2] (Depression)" in out
    assert "worry is the core feature" in out
    # Blocks separated by blank line so the model can tell them apart.
    assert "\n\n" in out


def test_build_messages_has_system_and_user_roles():
    msgs = build_messages("What is X?", [Passage("a", "T", "text")])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "What is X?" in msgs[1]["content"]
    assert "[a]" in msgs[1]["content"]


def test_build_messages_instructs_citation_and_abstention():
    """The system prompt is what the eval metrics rely on. If someone
    edits it and drops the citation or IDK instruction, catch it here."""
    assert "chunk id" in SYSTEM_PROMPT.lower() or "[chunk_id]" in SYSTEM_PROMPT
    assert "i don't know" in SYSTEM_PROMPT.lower()
    assert "advice" in SYSTEM_PROMPT.lower()
