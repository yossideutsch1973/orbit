"""Grid-based fluid flow systems.

Provides Eulerian flow fields commonly used as benchmarks for Lagrangian
coherent structure (LCS) analysis and Koopman operator learning.
"""

from __future__ import annotations

import numpy as np

from koopsim.systems.base import DynamicalSystem


class DoubleGyre(DynamicalSystem):
    """Time-dependent double-gyre flow.

    The velocity field is defined by:

    .. math::

        \\dot{x} = -\\pi A \\sin(\\pi f(x,t)) \\cos(\\pi y)

        \\dot{y} = \\pi A \\cos(\\pi f(x,t)) \\sin(\\pi y) \\, \\frac{\\partial f}{\\partial x}

    with

    .. math::

        f(x, t) = \\varepsilon \\sin(\\omega t) x^2
                   + (1 - 2\\varepsilon \\sin(\\omega t)) x

    The domain is :math:`[0, 2] \\times [0, 1]`.

    Parameters
    ----------
    A : float
        Amplitude of the gyre velocity.
    epsilon : float
        Amplitude of time-dependent perturbation.
    omega : float
        Angular frequency of the perturbation.
    """

    def __init__(
        self,
        A: float = 0.1,
        epsilon: float = 0.25,
        omega: float = 2 * np.pi,
    ) -> None:
        self._A = A
        self._epsilon = epsilon
        self._omega = omega

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y = state
        a = self._epsilon * np.sin(self._omega * t)
        b = 1.0 - 2.0 * a

        f = a * x**2 + b * x
        df_dx = 2.0 * a * x + b

        dx = -np.pi * self._A * np.sin(np.pi * f) * np.cos(np.pi * y)
        dy = np.pi * self._A * np.cos(np.pi * f) * np.sin(np.pi * y) * df_dx

        return np.array([dx, dy])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "DoubleGyre"
