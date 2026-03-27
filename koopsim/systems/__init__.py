"""Domain-specific dynamical system implementations."""

from __future__ import annotations

from koopsim.systems.base import DynamicalSystem
from koopsim.systems.chaotic import LorenzAttractor, LotkaVolterra
from koopsim.systems.circuit import RLCCircuit
from koopsim.systems.fluid_grid import DoubleGyre
from koopsim.systems.fluid_particles import HopfBifurcation, PointVortexSystem
from koopsim.systems.mechanical import (
    EulerBernoulliBeam,
    SpringMassDamper,
    VanDerPolOscillator,
)

__all__ = [
    "DynamicalSystem",
    "DoubleGyre",
    "EulerBernoulliBeam",
    "HopfBifurcation",
    "LorenzAttractor",
    "LotkaVolterra",
    "PointVortexSystem",
    "RLCCircuit",
    "SpringMassDamper",
    "VanDerPolOscillator",
]
