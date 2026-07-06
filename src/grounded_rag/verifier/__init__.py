"""Groundedness verifier.

Interface (any callable that takes retrieved passages + answer and returns
a scalar score in [0, 1]) is what the eval harness and both training paths
depend on. The real NLI-based implementation is built in step 4; for now
the stub keeps the harness plumbed."""
from .stub import StubVerifier

__all__ = ["StubVerifier"]
