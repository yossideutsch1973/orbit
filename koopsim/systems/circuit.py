"""Electrical circuit dynamical systems.

Provides canonical circuit models for benchmarking Koopman operator learning
on systems governed by linear ODEs with known analytical solutions.
"""

from __future__ import annotations

import numpy as np

from koopsim.systems.base import DynamicalSystem


class RLCCircuit(DynamicalSystem):
    """Series RLC circuit.

    State: ``[q, i]`` where *q* is the charge on the capacitor and
    *i = dq/dt* is the current.

    .. math::

        \\dot{q} = i

        \\dot{i} = -\\frac{R}{L} i - \\frac{1}{LC} q

    Parameters
    ----------
    R : float
        Resistance [Ohm].
    L : float
        Inductance [Henry].
    C : float
        Capacitance [Farad].
    """

    def __init__(self, R: float = 1.0, L: float = 1.0, C: float = 1.0) -> None:
        self._R = R
        self._L = L
        self._C = C

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        q, i = state
        dq = i
        di = -(self._R / self._L) * i - (1.0 / (self._L * self._C)) * q
        return np.array([dq, di])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "RLCCircuit"
