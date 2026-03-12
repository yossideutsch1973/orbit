"""Abstract base class for dynamical systems that generate training data."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

logger = logging.getLogger("koopsim")


class DynamicalSystem(ABC):
    """Abstract base for dynamical systems that generate training data.

    Subclasses define a continuous-time ODE via :meth:`rhs`, and this base
    class provides trajectory generation and Koopman snapshot-pair extraction.
    """

    @abstractmethod
    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Right-hand side of dx/dt = f(t, x).

        Parameters
        ----------
        t : float
            Current time.
        state : np.ndarray
            Current state vector of length :attr:`state_dim`.

        Returns
        -------
        np.ndarray
            Time derivative of the state, same shape as *state*.
        """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable system name."""

    def generate_trajectory(self, x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
        """Generate a single trajectory by integrating the ODE.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state of length :attr:`state_dim`.
        dt : float
            Time step between consecutive snapshots.
        n_steps : int
            Number of integration steps (the trajectory has ``n_steps + 1`` points).

        Returns
        -------
        np.ndarray, shape (n_steps + 1, state_dim)
            The trajectory, including the initial condition at index 0.
        """
        times = np.arange(n_steps + 1) * dt
        sol = solve_ivp(
            self.rhs,
            [times[0], times[-1]],
            x0,
            t_eval=times,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.status != 0:
            logger.warning(
                "ODE solver did not converge for %s: %s", self.name, sol.message
            )
        return sol.y.T  # (n_steps+1, state_dim)

    def generate_snapshots(
        self,
        x0: np.ndarray | list[np.ndarray],
        dt: float,
        n_steps: int,
        n_trajectories: int = 1,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate snapshot pairs (X, Y) for Koopman learning.

        If *x0* is a single state vector, it is randomly perturbed to produce
        *n_trajectories* initial conditions.  If *x0* is a list of arrays,
        each element is used as an initial condition and *n_trajectories* is
        ignored.

        From each trajectory of length ``n_steps + 1``, consecutive pairs
        ``(traj[j], traj[j+1])`` are extracted and vertically stacked.

        Parameters
        ----------
        x0 : np.ndarray or list[np.ndarray]
            A single initial state (perturbed *n_trajectories* times) or an
            explicit list of initial conditions.
        dt : float
            Time step between snapshots.
        n_steps : int
            Number of steps per trajectory.
        n_trajectories : int
            Number of trajectories when *x0* is a single state.
        noise_std : float
            Standard deviation of additive Gaussian noise applied to the
            snapshot pairs.
        rng : np.random.Generator or None
            Random number generator.  If ``None``, a default one is created.

        Returns
        -------
        X : np.ndarray, shape (n_total_pairs, state_dim)
            Current-state snapshots.
        Y : np.ndarray, shape (n_total_pairs, state_dim)
            Next-state snapshots.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Build list of initial conditions
        if isinstance(x0, list):
            initial_conditions = [np.asarray(ic, dtype=np.float64) for ic in x0]
        else:
            x0 = np.asarray(x0, dtype=np.float64)
            if n_trajectories == 1:
                initial_conditions = [x0]
            else:
                perturbation = rng.standard_normal((n_trajectories, len(x0))) * 0.1
                initial_conditions = [x0 + perturbation[i] for i in range(n_trajectories)]

        X_parts: list[np.ndarray] = []
        Y_parts: list[np.ndarray] = []

        for ic in tqdm(initial_conditions, desc=f"Generating {self.name} trajectories"):
            traj = self.generate_trajectory(ic, dt, n_steps)
            # Extract consecutive snapshot pairs
            X_parts.append(traj[:-1])  # (n_steps, state_dim)
            Y_parts.append(traj[1:])   # (n_steps, state_dim)

        X = np.vstack(X_parts)
        Y = np.vstack(Y_parts)

        if noise_std > 0.0:
            X = X + rng.normal(0.0, noise_std, X.shape)
            Y = Y + rng.normal(0.0, noise_std, Y.shape)

        logger.info(
            "%s: generated %d snapshot pairs from %d trajectories.",
            self.name,
            X.shape[0],
            len(initial_conditions),
        )
        return X, Y
