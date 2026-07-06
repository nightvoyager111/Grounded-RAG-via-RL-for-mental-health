from __future__ import annotations

from typing import Sequence


class StubVerifier:
    """Placeholder verifier — returns a constant score.

    Exists so eval/runner.py can be wired end-to-end before the real NLI
    verifier (step 4) lands. Do NOT report the number produced by this
    verifier as a baseline groundedness_rate — it is meaningless."""

    def __init__(self, score: float = 0.5):
        self.score = float(score)

    def __call__(self, passages: Sequence[str], answer: str) -> float:
        return self.score
