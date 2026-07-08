"""Groundedness verifier.

Interface (any callable that takes retrieved passages + answer and returns
a scalar score in [0, 1]) is what the eval harness and both training paths
depend on. The real NLI-based implementation is built in step 4; for now
the stub keeps the harness plumbed."""
from .stub import StubVerifier
from .nli import NLIConfig, NLIVerifier
from .hhem import HHEMConfig, HHEMVerifier
from .llm_judge import LLMJudgeConfig, LLMJudgeVerifier
from .calibration import (
    CalibrationRecord,
    agreement_at_threshold,
    calibration_report,
    read_labeled,
    score_records,
    sweep_thresholds,
    write_records,
)

__all__ = [
    "StubVerifier",
    "NLIConfig",
    "NLIVerifier",
    "HHEMConfig",
    "HHEMVerifier",
    "LLMJudgeConfig",
    "LLMJudgeVerifier",
    "CalibrationRecord",
    "agreement_at_threshold",
    "calibration_report",
    "read_labeled",
    "score_records",
    "sweep_thresholds",
    "write_records",
]
