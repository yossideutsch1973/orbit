"""Mechanical / structural dynamical systems.

Provides classical mechanical systems for benchmarking Koopman methods
on vibration, structural dynamics, and nonlinear oscillator problems.
"""

from __future__ import annotations

import numpy as np

from koopsim.systems.base import DynamicalSystem


class SpringMassDamper(DynamicalSystem):
    """Chain of *n* masses connected by springs and dampers.

    The first mass is connected to a fixed wall, and each subsequent mass
    is connected to its predecessor.  The state vector is
    ``[q1, ..., qn, qdot1, ..., qdotn]`` (positions followed by velocities).

    Parameters
    ----------
    n_masses : int
        Number of masses in the chain.
    k : float
        Spring stiffness (uniform for all springs).
    c : float
        Damping coefficient (uniform for all dampers).
    m : float
        Mass of each body (uniform).
    """

    def __init__(
        self, n_masses: int = 3, k: float = 1.0, c: float = 0.1, m: float = 1.0
    ) -> None:
        self._n = n_masses
        self._k = k
        self._c = c
        self._m = m

        # Pre-assemble stiffness and damping matrices
        self._K_mat = self._build_tridiag(n_masses, k)
        self._C_mat = self._build_tridiag(n_masses, c)

    @staticmethod
    def _build_tridiag(n: int, val: float) -> np.ndarray:
        """Build the tridiagonal stiffness/damping matrix for a chain."""
        M = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                M[i, i] = 2.0 * val
            else:
                M[i, i] = 2.0 * val
            if i > 0:
                M[i, i - 1] = -val
                M[i - 1, i] = -val
        # First mass connected to wall (already accounted for by 2*val on diagonal)
        # Last mass has free end: only one spring connecting it to predecessor
        M[n - 1, n - 1] = val
        return M

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self._n
        q = state[:n]
        v = state[n:]

        # M * a = -K*q - C*v  =>  a = -(K*q + C*v) / m
        accel = -(self._K_mat @ q + self._C_mat @ v) / self._m

        return np.concatenate([v, accel])

    @property
    def state_dim(self) -> int:
        return 2 * self._n

    @property
    def name(self) -> str:
        return "SpringMassDamper"


class EulerBernoulliBeam(DynamicalSystem):
    """Euler-Bernoulli beam discretised with finite elements.

    Uses a modal truncation approach: the beam is discretised into
    *n_elements* equal-length elements with standard Hermite shape functions.
    The resulting mass and stiffness matrices are diagonalised (undamped
    free vibration modes), and the state is represented in modal coordinates:
    ``[eta_1, ..., eta_n, eta_dot_1, ..., eta_dot_n]``.

    A cantilever (fixed-free) boundary condition is assumed.

    Parameters
    ----------
    n_elements : int
        Number of finite elements.
    E : float
        Young's modulus [Pa].
    I : float
        Second moment of area [m^4].
    rho : float
        Material density [kg/m^3].
    A : float
        Cross-sectional area [m^2].
    L : float
        Total beam length [m].
    """

    def __init__(
        self,
        n_elements: int = 5,
        E: float = 2e11,
        I: float = 1e-6,
        rho: float = 7800.0,
        A: float = 1e-4,
        L: float = 1.0,
    ) -> None:
        self._n_elements = n_elements
        self._E = E
        self._I = I
        self._rho = rho
        self._A = A
        self._L = L

        # Compute natural frequencies (modal approach)
        self._omega_n = self._compute_natural_frequencies()

    def _compute_natural_frequencies(self) -> np.ndarray:
        """Compute approximate natural frequencies for a cantilever beam.

        Uses the analytical formula for a continuous cantilever beam:
        omega_n = (beta_n * L)^2 * sqrt(EI / (rho * A * L^4))

        where beta_n * L are roots of 1 + cos(beta*L)*cosh(beta*L) = 0.
        First few: 1.8751, 4.6941, 7.8548, 10.9955, 14.1372, ...
        """
        # Approximate eigenvalues (beta_n * L) for cantilever
        beta_L = np.array([1.8751, 4.6941, 7.8548, 10.9955, 14.1372])
        # Extend for more elements using the asymptotic formula
        n = self._n_elements
        if n > len(beta_L):
            extra = np.array([(2 * k + 1) * np.pi / 2.0 for k in range(len(beta_L), n)])
            beta_L = np.concatenate([beta_L, extra])
        beta_L = beta_L[:n]

        EI = self._E * self._I
        rhoA = self._rho * self._A
        L = self._L

        omega_n = (beta_L / L) ** 2 * np.sqrt(EI / rhoA)
        return omega_n

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self._n_elements
        eta = state[:n]      # modal displacements
        eta_dot = state[n:]  # modal velocities

        # Undamped modal equations: eta_ddot_i = -omega_i^2 * eta_i
        eta_ddot = -(self._omega_n ** 2) * eta

        return np.concatenate([eta_dot, eta_ddot])

    @property
    def state_dim(self) -> int:
        return 2 * self._n_elements

    @property
    def name(self) -> str:
        return "EulerBernoulliBeam"


class VanDerPolOscillator(DynamicalSystem):
    """Van der Pol oscillator.

    .. math::

        \\ddot{x} - \\mu (1 - x^2) \\dot{x} + x = 0

    State: ``[x, x_dot]``.

    Parameters
    ----------
    mu : float
        Nonlinearity parameter.  Larger values produce more relaxation
        oscillation behaviour.
    """

    def __init__(self, mu: float = 1.0) -> None:
        self._mu = mu

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        x, x_dot = state
        x_ddot = self._mu * (1.0 - x ** 2) * x_dot - x
        return np.array([x_dot, x_ddot])

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "VanDerPolOscillator"
