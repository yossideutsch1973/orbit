"""KoopSim — Koopman Operator Simulation Toolkit."""

from __future__ import annotations

__version__ = "0.1.0"

from koopsim.core.base import KoopmanModel as KoopmanModel
from koopsim.core.edmd import EDMD as EDMD
from koopsim.core.exceptions import DimensionMismatchError as DimensionMismatchError
from koopsim.core.exceptions import KoopSimError as KoopSimError
from koopsim.core.exceptions import NotFittedError as NotFittedError
from koopsim.core.prediction import PredictionEngine as PredictionEngine
from koopsim.koopsim import KoopSim as KoopSim
