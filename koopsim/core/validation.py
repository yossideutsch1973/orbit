"""Model validation utilities for Koopman operator models."""

from __future__ import annotations

import numpy as np

from koopsim.core.base import KoopmanModel
from koopsim.core.prediction import PredictionEngine


class ModelValidator:
    """Static methods for validating Koopman model quality."""

    @staticmethod
    def prediction_error(
        model: KoopmanModel,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        metric: str = "rmse",
    ) -> float:
        """Compute one-step prediction error.

        Lifts X_test into the Koopman space, applies K, unlifts,
        and compares the result to Y_test.

        Parameters
        ----------
        model : KoopmanModel
            A fitted Koopman model.
        X_test : np.ndarray, shape (n_samples, n_features)
            Pre-snapshot test data.
        Y_test : np.ndarray, shape (n_samples, n_features)
            Post-snapshot test data (ground truth).
        metric : str
            Error metric: ``'rmse'``, ``'mae'``, or ``'relative'``.

        Returns
        -------
        float
            The computed error value.

        Raises
        ------
        ValueError
            If metric is not recognized.
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        Y_test = np.asarray(Y_test, dtype=np.float64)

        K = model.get_koopman_matrix()
        Psi_X = model.lift(X_test)
        Psi_Y_pred = Psi_X @ K
        Y_pred = model.unlift(Psi_Y_pred)

        diff = Y_pred - Y_test

        if metric == "rmse":
            return float(np.sqrt(np.mean(diff ** 2)))
        elif metric == "mae":
            return float(np.mean(np.abs(diff)))
        elif metric == "relative":
            norm_Y = np.linalg.norm(Y_test, "fro")
            if norm_Y == 0:
                return float(np.linalg.norm(diff, "fro"))
            return float(np.linalg.norm(diff, "fro") / norm_Y)
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from 'rmse', 'mae', 'relative'."
            )

    @staticmethod
    def multi_step_error(
        model: KoopmanModel,
        trajectory: np.ndarray,
        dt: float,
        n_steps: int,
    ) -> np.ndarray:
        """Compute multi-step prediction error at each step.

        Uses the PredictionEngine to predict from trajectory[0] at
        t = dt, 2*dt, ..., n_steps*dt and compares to
        trajectory[1], trajectory[2], ..., trajectory[n_steps].

        Parameters
        ----------
        model : KoopmanModel
            A fitted Koopman model.
        trajectory : np.ndarray, shape (n_steps + 1, n_features)
            Ground-truth trajectory. Must have at least n_steps + 1 rows.
        dt : float
            Time step between consecutive trajectory snapshots.
        n_steps : int
            Number of prediction steps.

        Returns
        -------
        np.ndarray, shape (n_steps,)
            RMSE at each prediction step.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        engine = PredictionEngine(model)

        x0 = trajectory[0]
        errors = np.empty(n_steps, dtype=np.float64)

        for i in range(n_steps):
            t = (i + 1) * dt
            x_pred = engine.predict(x0, t)
            x_true = trajectory[i + 1]
            errors[i] = np.sqrt(np.mean((x_pred - x_true) ** 2))

        return errors

    @staticmethod
    def spectral_analysis(model: KoopmanModel) -> dict:
        """Analyze eigenvalues of the Koopman matrix.

        Parameters
        ----------
        model : KoopmanModel
            A fitted Koopman model.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``eigenvalues``: complex eigenvalues of K
            - ``frequencies``: oscillation frequencies (from angle(lambda) / dt)
            - ``growth_rates``: growth/decay rates (from log|lambda| / dt)
            - ``dominant_mode_indices``: indices sorted by |lambda| descending
        """
        K = model.get_koopman_matrix()
        dt = model.dt

        eigenvalues = np.linalg.eigvals(K)

        # Frequencies from the phase angle of eigenvalues
        frequencies = np.angle(eigenvalues) / dt

        # Growth rates from log of magnitude
        magnitudes = np.abs(eigenvalues)
        # Avoid log(0) by clamping
        safe_magnitudes = np.where(magnitudes > 0, magnitudes, 1e-300)
        growth_rates = np.log(safe_magnitudes) / dt

        # Dominant mode indices sorted by magnitude (descending)
        dominant_mode_indices = np.argsort(magnitudes)[::-1]

        return {
            "eigenvalues": eigenvalues,
            "frequencies": frequencies.real,
            "growth_rates": growth_rates.real,
            "dominant_mode_indices": dominant_mode_indices,
        }
