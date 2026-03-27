"""KoopSim — Koopman Operator Simulation Toolkit."""

from __future__ import annotations

__version__ = "0.1.0"

from koopsim.koopsim import KoopSim
from koopsim.core.edmd import EDMD
from koopsim.core.base import KoopmanModel
from koopsim.core.prediction import PredictionEngine
from koopsim.core.exceptions import KoopSimError, NotFittedError, DimensionMismatchError
