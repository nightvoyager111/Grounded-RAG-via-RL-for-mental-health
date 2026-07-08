"""LLM-as-judge groundedness verifier.

CLAUDE.md step 5 explicitly names this as the calibration-gate fallback:
"either swap to a small-LLM judge or narrow the corpus to harder-fact
items". We chose the judge.

Cohere Command is used because the retriever already depends on Cohere,
so no new API surface. The judge is asked for a coarse three-level score
(0.0 / 0.5 / 1.0) instead of a boolean — this preserves a "partial /
abstention" middle bucket for the calibration threshold sweep, and LLMs
are more reliable at trichotomies than continuous scoring.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence



JUDGE_SYSTEM = (
    "You are a text-comparison evaluator. You are NOT giving medical advice, "
    "diagnosing, or providing clinical guidance — you are only comparing two "
    "pieces of text to check whether one is supported by the other. The topic "
    "of the text is incidental; treat the task as a factual entailment check.\n\n"
    "Given evidence passages and an answer, judge how well the answer is "
    "SUPPORTED by the passages. Focus on the CORE CLAIMS — the substantive "
    "facts the answer is trying to convey — not on incidental phrasing.\n\n"
    "Guidelines:\n"
    "- Paraphrase is fully acceptable. If the passages state a fact and the "
    "answer restates it in different words, that is supported. Do NOT require "
    "verbatim overlap.\n"
    "- Reordering, summarizing, and combining information from multiple "
    "passages are all fine.\n"
    "- What counts as UNsupported: a specific factual claim in the answer "
    "(a name, number, symptom, mechanism, definition) that is NOT stated or "
    "clearly implied by any passage. This is parametric hallucination, which "
    "is what this judge exists to catch.\n"
    "- Judge SUPPORT, not helpfulness or completeness. An answer that omits "
    "detail but is faithful is still grounded.\n"
    "- Abstentions ('I don't know', refusals) are neither grounded nor "
    "ungrounded — score them 0.5.\n\n"
    "Output ONLY a JSON object, no prose, in this exact schema:\n"
    '  {"score": <one of 0.0, 0.25, 0.5, 0.75, 1.0>, "reason": "<one short sentence>"}\n'
    "Rubric:\n"
    "  1.00 — every core claim is supported (paraphrase is fine)\n"
    "  0.75 — core claims supported; one minor detail is not in the passages\n"
    "  0.50 — partial support (either a mix of supported / unsupported claims, "
    "or an abstention)\n"
    "  0.25 — most claims are not supported\n"
    "  0.00 — the central claim is a hallucination (not in any passage)"
)


def _build_user_prompt(passages: Sequence[str], answer: str) -> str:
    parts = ["Passages:"]
    for i, p in enumerate(passages, 1):
        parts.append(f"\n[{i}] {p.strip()}")
    parts.append(f"\n\nAnswer:\n{answer.strip()}")
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_score(text: str) -> float:
    """Extract {"score": ...} from the model output. Falls back to a
    permissive regex if the model wrapped the JSON in prose despite the
    instruction to output only JSON."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"judge did not return JSON: {text!r}")
        obj = json.loads(m.group(0))
    score = float(obj["score"])
    buckets = (0.0, 0.25, 0.5, 0.75, 1.0)
    if score not in buckets:
        # Clip to the nearest valid bucket rather than erroring — LLMs
        # occasionally emit off-rubric values.
        score = min(buckets, key=lambda v: abs(v - score))
    return score


@dataclass
class LLMJudgeConfig:
    model_name: str = "command-r-plus-08-2024"
    temperature: float = 0.0          # deterministic judging
    api_key_env: str = "COHERE_API_KEY"
    passage_join: str = "\n\n"        # unused; passages are formatted per-message


def _call_with_retry(fn: Callable, *, max_attempts: int = 3, base_sleep: float = 1.0):
    """Retry only on genuine network/5xx blips. Production keys don't hit 429s."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("timeout" in msg or "temporarily" in msg
                         or "connection" in msg or "503" in msg or "502" in msg)
            if not transient or attempt == max_attempts - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt))


class LLMJudgeVerifier:
    """Cohere-Command-backed groundedness judge.

    score(passages, answer) → {0.0, 0.5, 1.0}.  Same contract as the NLI
    and HHEM verifiers, so it plugs into the calibration harness and, later,
    the DPO pair-construction step."""

    def __init__(self, cfg: LLMJudgeConfig, client=None):
        self.cfg = cfg
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        import cohere

        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.cfg.api_key_env} not set — needed for LLMJudgeVerifier"
            )
        self._client = cohere.ClientV2(api_key=api_key)
        return self._client

    def score(self, passages: Sequence[str], answer: str) -> float:
        if not passages:
            return 0.5
        client = self._get_client()
        resp = _call_with_retry(lambda: client.chat(
            model=self.cfg.model_name,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _build_user_prompt(passages, answer)},
            ],
        ))
        # Cohere ClientV2 chat returns .message.content as a list of content
        # blocks; grab the text of the first (and typically only) block.
        content = resp.message.content
        text = content[0].text if content else ""
        return _parse_score(text)

    def __call__(self, passages: Sequence[str], answer: str) -> float:
        return self.score(passages, answer)
