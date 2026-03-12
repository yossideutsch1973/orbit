"""Custom exception classes for KoopSim."""

from __future__ import annotations


class KoopSimError(Exception):
    """Base exception for all KoopSim errors."""


class NotFittedError(KoopSimError):
    """Raised when a model is used before fitting."""


class DimensionMismatchError(KoopSimError):
    """Raised when input dimensions don't match expected dimensions."""
