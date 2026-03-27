"""Chaotic dynamical systems.

Provides classic chaotic systems for benchmarking Koopman methods
on systems with sensitive dependence on initial conditions.
"""

from __future__ import annotations

import numpy as np

from koopsim.systems.base import DynamicalSystem


class LorenzAttractor(DynamicalSystem):
    """Lorenz attractor — the classic chaotic system.

    .. math::

        \\dot{x} = \\sigma (y - x)

        \\dot{y} = x (\\rho - z) - y

        \\dot{z} = x y - \\beta z

    State: ``[x, y, z]``.

    Parameters
    ----------
    sigma : float
        Prandtl number. Default 10.0 (classic value).
    rho : float
        Rayleigh number. Default 28.0 (classic chaotic regime).
    beta : float
        Geometric factor. Default 8/3 (classic value).
    """

    def __init__(
        self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
    ) -> None:
        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self._sigma * (y - x)
        dy = x * (self._rho - z) - y
        dz = x * y - self._beta * z
        return np.array([dx, dy, dz])

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "LorenzAttractor"


class LotkaVolterra(DynamicalSystem):
    """Lotka-Volterra predator-prey equations.

    .. math::

        \\dot{x} = \\alpha x - \\beta x y

        \\dot{y} = \\delta x y - \\gamma y

    State: ``[x, y]`` where *x* is prey population and *y* is predator
    population.

    Parameters
    ----------
    alpha : float
        Prey birth rate. Default 1.0.
    beta : float
        Predation rate. Default 0.5.
    gamma : float
        Predator death rate. Default 0.5.
    delta : float
        Predator reproduction rate per prey consumed. Default 0.25.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.25,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y = state
        dx = self._alpha * x - self._beta * x * y
        dy = self._delta * x * y - self._gamma * y
        return np.array([dx, dy])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "LotkaVolterra"
