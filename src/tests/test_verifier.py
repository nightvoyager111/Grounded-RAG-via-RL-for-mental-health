"""Tests for the verifier + calibration harness.

We never load the real DeBERTa NLI model in tests. Instead we:
- Test the *score arithmetic* (P_entail - P_contradict → [0, 1]) via a
  monkeypatched forward path.
- Test calibration math (agreement, threshold sweep) with pure numbers.
- Test I/O round-trip for calibration records.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.grounded_rag.verifier import StubVerifier
from src.grounded_rag.verifier.calibration import (
    CalibrationRecord,
    agreement_at_threshold,
    calibration_report,
    read_labeled,
    score_records,
    sweep_thresholds,
    write_records,
)
from src.grounded_rag.verifier.nli import NLIConfig, NLIVerifier


# ---------------------------------------------------------------------------
# StubVerifier
# ---------------------------------------------------------------------------


def test_stub_verifier_returns_configured_constant():
    v = StubVerifier(0.42)
    assert v(["p"], "a") == 0.42


# ---------------------------------------------------------------------------
# NLIVerifier score arithmetic
# ---------------------------------------------------------------------------


class _FakeLogits:
    """Tensor-like object exposing just what NLIVerifier.score touches."""
    def __init__(self, values):
        import torch
        self.logits = torch.tensor([values], dtype=torch.float32)


class _FakeNLIModel:
    """Deterministic 3-class model: returns preset logits regardless of input."""
    def __init__(self, logits):
        self._logits = logits

        # Label order fixed as [entailment, neutral, contradiction] — mirrors
        # the default DeBERTa-MNLI checkpoint used in the config.
        self.config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return _FakeLogits(self._logits)


class _FakeTokenizer:
    """Returns an input dict shaped like a BatchEncoding; ignores content."""
    def __call__(self, premise, hypothesis, **kwargs):
        import torch
        d = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        d["to"] = lambda device: d
        return SimpleNamespace(**{**d, "to": lambda device: SimpleNamespace(**d, to=lambda dev: None) if False else _FakeBatch(d)})


class _FakeBatch:
    """BatchEncoding-like with .to(device) and ** unpacking support."""
    def __init__(self, d):
        self._d = d
    def to(self, device):
        return self
    def keys(self):
        return [k for k in self._d if k != "to"]
    def __getitem__(self, k):
        return self._d[k]
    # Allow **enc unpacking
    def __iter__(self):
        return iter(self.keys())


def _install_fake_nli(monkeypatch, logits):
    """Patch NLIVerifier so _load() installs our fakes instead of downloading."""

    def fake_load(self):
        self._tokenizer = _FakeTokenizer()
        self._model = _FakeNLIModel(logits)
        self._label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self._device = "cpu"

    monkeypatch.setattr(NLIVerifier, "_load", fake_load)


def test_nli_score_pure_entailment_is_one(monkeypatch):
    # Extreme logits: entail wins by a mile → softmax ~ [1, 0, 0]
    _install_fake_nli(monkeypatch, [50.0, 0.0, 0.0])
    v = NLIVerifier(NLIConfig())
    s = v.score(["passage"], "answer")
    assert s == pytest.approx(1.0, abs=1e-4)


def test_nli_score_pure_contradiction_is_zero(monkeypatch):
    _install_fake_nli(monkeypatch, [0.0, 0.0, 50.0])
    v = NLIVerifier(NLIConfig())
    assert v.score(["p"], "a") == pytest.approx(0.0, abs=1e-4)


def test_nli_score_balanced_is_half(monkeypatch):
    # Equal entail/contradict mass, any neutral → raw = 0 → score = 0.5
    _install_fake_nli(monkeypatch, [1.0, 5.0, 1.0])
    v = NLIVerifier(NLIConfig())
    assert v.score(["p"], "a") == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# calibration math
# ---------------------------------------------------------------------------


def test_agreement_at_threshold_counts_confusion_matrix():
    #  score,  label
    #  0.9,    1     tp
    #  0.8,    1     tp
    #  0.6,    0     fp
    #  0.3,    0     tn
    #  0.1,    1     fn
    labels = [1, 1, 0, 0, 1]
    scores = [0.9, 0.8, 0.6, 0.3, 0.1]
    r = agreement_at_threshold(labels, scores, threshold=0.5)
    assert r == {
        "n": 5, "threshold": 0.5,
        "agreement": 3 / 5,
        "tp": 2, "fp": 1, "tn": 1, "fn": 1,
        "precision": 2 / 3, "recall": 2 / 3,
    }


def test_agreement_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        agreement_at_threshold([1], [0.5, 0.6], threshold=0.5)


def test_sweep_thresholds_picks_best_and_prefers_middle_on_ties():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    best, rows = sweep_thresholds(labels, scores, n_steps=11)
    assert best["agreement"] == 1.0
    # Multiple thresholds achieve perfect agreement → tiebreak toward 0.5
    assert 0.3 <= best["threshold"] <= 0.8
    assert len(rows) == 11


def test_calibration_report_flags_gate():
    """≥ 0.85 agreement → gate passes; below → gate fails."""
    labels = [1] * 9 + [0]                # 10 examples
    scores = [0.9] * 9 + [0.9]            # 9 correct, 1 wrong = 0.9 agreement
    records = [
        CalibrationRecord(id=str(i), question="q", answer="a", passages=["p"],
                          label=l, score=s)
        for i, (l, s) in enumerate(zip(labels, scores))
    ]
    report = calibration_report(records)
    assert report["n_examples"] == 10
    assert report["at_default_threshold"]["agreement"] == 0.9
    assert report["gate_pass_at_default"] is True


def test_calibration_report_gate_fails_below_threshold():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.1, 0.1, 0.9, 0.1]     # 2/5 = 0.4 agreement at t=0.5
    records = [
        CalibrationRecord(id=str(i), question="q", answer="a", passages=["p"],
                          label=l, score=s)
        for i, (l, s) in enumerate(zip(labels, scores))
    ]
    report = calibration_report(records)
    assert report["gate_pass_at_default"] is False


def test_calibration_report_rejects_unlabeled_records():
    records = [CalibrationRecord(id="1", question="q", answer="a",
                                  passages=["p"], label=None, score=0.5)]
    with pytest.raises(ValueError):
        calibration_report(records)


# ---------------------------------------------------------------------------
# score_records + I/O
# ---------------------------------------------------------------------------


def test_score_records_populates_score_using_verifier():
    records = [
        CalibrationRecord(id="1", question="q", answer="a1", passages=["p"], label=1),
        CalibrationRecord(id="2", question="q", answer="a2", passages=["p"], label=0),
    ]
    scored = score_records(records, verifier=StubVerifier(0.7))
    assert [r.score for r in scored] == [0.7, 0.7]
    # Input records are not mutated
    assert records[0].score is None


def test_calibration_records_roundtrip(tmp_path):
    records = [
        CalibrationRecord(id="a", question="q1", answer="ans1",
                          passages=["p1", "p2"], label=1, score=0.9),
        CalibrationRecord(id="b", question="q2", answer="ans2",
                          passages=["p3"], label=0, score=0.1),
    ]
    p = tmp_path / "cal.jsonl"
    write_records(p, records)
    back = read_labeled(p)
    assert len(back) == 2
    assert back[0].id == "a"
    assert back[0].passages == ["p1", "p2"]
    assert back[0].label == 1
    assert back[0].score == 0.9
    assert back[1].label == 0
