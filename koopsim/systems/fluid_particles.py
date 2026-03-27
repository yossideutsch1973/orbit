"""Particle-based fluid / dynamical systems.

Provides canonical systems used for testing and demonstrating
Koopman operator methods on low-dimensional flows.
"""

from __future__ import annotations

import numpy as np

from koopsim.systems.base import DynamicalSystem


class HopfBifurcation(DynamicalSystem):
    """Hopf bifurcation normal form: canonical limit-cycle system.

    .. math::

        \\dot{x} = \\mu x - y - x(x^2 + y^2)

        \\dot{y} = x + \\mu y - y(x^2 + y^2)

    For :math:`\\mu > 0` the origin is unstable and trajectories converge to
    a stable limit cycle of radius :math:`\\sqrt{\\mu}`.

    Parameters
    ----------
    mu : float
        Bifurcation parameter.  Positive values yield a stable limit cycle.
    """

    def __init__(self, mu: float = 1.0) -> None:
        self._mu = mu

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y = state
        r2 = x**2 + y**2
        dx = self._mu * x - y - x * r2
        dy = x + self._mu * y - y * r2
        return np.array([dx, dy])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "HopfBifurcation"


class PointVortexSystem(DynamicalSystem):
    """System of point vortices in 2D inviscid flow.

    Each vortex induces a velocity on all others via the Biot-Savart law.
    The state vector is the flattened positions: ``[x1, y1, x2, y2, ...]``.

    Parameters
    ----------
    n_vortices : int
        Number of point vortices.
    strengths : np.ndarray or None
        Circulation strengths for each vortex.  If ``None``, alternating
        ``+1`` / ``-1`` strengths are used.
    """

    def __init__(self, n_vortices: int = 3, strengths: np.ndarray | None = None) -> None:
        self._n_vortices = n_vortices
        if strengths is None:
            self._strengths = np.array([(-1.0) ** i for i in range(n_vortices)])
        else:
            self._strengths = np.asarray(strengths, dtype=np.float64)
            if len(self._strengths) != n_vortices:
                raise ValueError(
                    f"Length of strengths ({len(self._strengths)}) must match "
                    f"n_vortices ({n_vortices})."
                )

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self._n_vortices
        # Reshape to (n_vortices, 2)
        pos = state.reshape(n, 2)
        vel = np.zeros_like(pos)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r2 = dx**2 + dy**2
                # Regularise to avoid singularity
                r2 = max(r2, 1e-10)
                # Biot-Savart: velocity induced by vortex j on vortex i
                # v_x = -Gamma_j * dy / (2*pi*r^2)
                # v_y =  Gamma_j * dx / (2*pi*r^2)
                factor = self._strengths[j] / (2.0 * np.pi * r2)
                vel[i, 0] += -factor * dy
                vel[i, 1] += factor * dx

        return vel.ravel()

    @property
    def state_dim(self) -> int:
        return 2 * self._n_vortices

    @property
    def name(self) -> str:
        return "PointVortexSystem"
