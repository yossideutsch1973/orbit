"""Prediction engine for Koopman operator models.

Given a fitted KoopmanModel, predicts future states at arbitrary times
using the matrix exponential or eigendecomposition — no time-stepping loops.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.linalg import expm, logm

from koopsim.core.base import KoopmanModel
from koopsim.core.constants import (
    CONDITION_NUMBER_WARNING_THRESHOLD,
    EIGENVALUE_STABILITY_THRESHOLD,
    EXPM_SIZE_THRESHOLD,
)
from koopsim.core.exceptions import KoopSimError, NotFittedError

logger = logging.getLogger("koopsim")


class PredictionEngine:
    """Predicts future states using a fitted Koopman model.

    Two backends:

    - ``'expm'``: compute ``L = logm(K) / dt``, then ``K_t = expm(L * t)``.
      Good for ``n_obs <= 200``.
    - ``'eigen'``: decompose ``K = V diag(lambda) V^{-1}``, then
      ``K_t = V diag(lambda^{t/dt}) V^{-1}``.  Fast for large systems.
    - ``'auto'``: uses ``expm`` if ``n_koopman_dims <= EXPM_SIZE_THRESHOLD``,
      otherwise ``eigen``.

    Parameters
    ----------
    model : KoopmanModel
        A fitted Koopman model.
    method : str
        One of ``'auto'``, ``'expm'``, ``'eigen'``.
    """

    def __init__(self, model: KoopmanModel, method: str = "auto") -> None:
        if not model._is_fitted():
            raise NotFittedError(
                "PredictionEngine requires a fitted model. Call model.fit() first."
            )

        if method not in ("auto", "expm", "eigen"):
            raise ValueError(
                f"method must be 'auto', 'expm', or 'eigen', got '{method}'."
            )

        self._model = model
        self._K = model.get_koopman_matrix()
        self._dt = model.dt
        self._n_koopman = model.n_koopman_dims

        # Resolve 'auto'
        if method == "auto":
            method = "expm" if self._n_koopman <= EXPM_SIZE_THRESHOLD else "eigen"
        self._method = method

        # Stability analysis — always check eigenvalues
        eigenvalues = np.linalg.eigvals(self._K)
        abs_eigs = np.abs(eigenvalues)
        unstable_mask = abs_eigs > EIGENVALUE_STABILITY_THRESHOLD
        if np.any(unstable_mask):
            max_abs = np.max(abs_eigs)
            n_unstable = int(np.sum(unstable_mask))
            warnings.warn(
                f"Koopman matrix has {n_unstable} eigenvalue(s) with "
                f"|lambda| > {EIGENVALUE_STABILITY_THRESHOLD:.6f} "
                f"(max |lambda| = {max_abs:.6f}). "
                f"Predictions may diverge for large t.",
                stacklevel=2,
            )
            logger.warning(
                "Unstable eigenvalues detected: %d modes with max |lambda| = %.6f.",
                n_unstable,
                max_abs,
            )

        # Precompute backend-specific data
        if self._method == "expm":
            self._precompute_expm()
        else:
            self._precompute_eigen()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_expm(self) -> None:
        """Compute L = logm(K) / dt for the expm backend."""
        L_raw = logm(self._K)

        # If imaginary part is negligible, discard it
        if np.isrealobj(L_raw):
            self._L = L_raw / self._dt
        else:
            imag_norm = np.linalg.norm(L_raw.imag)
            real_norm = np.linalg.norm(L_raw.real)
            ref_norm = max(real_norm, 1e-15)
            if imag_norm < 1e-10 * ref_norm:
                self._L = L_raw.real / self._dt
            else:
                # Keep complex; we will take real part after expm
                self._L = L_raw / self._dt

        logger.debug("PredictionEngine (expm): L computed, shape %s.", self._L.shape)

    def _precompute_eigen(self) -> None:
        """Eigendecompose K for the eigen backend.

        Falls back to expm if V is singular (defective K).
        """
        eigenvalues, V = np.linalg.eig(self._K)

        # Check condition of V
        cond = np.linalg.cond(V)
        if cond > CONDITION_NUMBER_WARNING_THRESHOLD:
            warnings.warn(
                f"Eigenvector matrix has high condition number ({cond:.4e}). "
                f"Eigendecomposition may be inaccurate.",
                stacklevel=2,
            )
            logger.warning(
                "High eigenvector condition number: %.4e.", cond
            )

        # Try to invert V; fall back to expm if singular
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            logger.warning(
                "Eigenvector matrix is singular; falling back to expm backend."
            )
            self._method = "expm"
            self._precompute_expm()
            return

        self._eigenvalues = eigenvalues
        self._V = V
        self._V_inv = V_inv

        logger.debug(
            "PredictionEngine (eigen): %d eigenvalues computed.", len(eigenvalues)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, x0: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Predict state at time *t* (scalar) or times *t* (array).

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape ``(n_state,)`` or ``(n_samples, n_state)``.
        t : float or np.ndarray
            A single time (scalar) or 1-D array of query times.

        Returns
        -------
        np.ndarray
            - Single x0 + scalar t: ``(n_state,)``
            - Batch x0 + scalar t: ``(n_samples, n_state)``
            - Single x0 + array t: ``(len(t), n_state)``
        """
        x0 = np.asarray(x0, dtype=np.float64)
        scalar_t = np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0)

        if scalar_t:
            return self._predict_single_time(x0, float(t))

        # Array of times
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        single_x0 = x0.ndim == 1

        if single_x0:
            x0_2d = x0.reshape(1, -1)
        else:
            x0_2d = x0

        results = []
        for ti in t_arr:
            res = self._predict_single_time(x0_2d, ti)
            results.append(res)

        # Stack results: (len(t), n_samples, n_state) then handle shapes
        stacked = np.stack(results, axis=0)  # (len(t), n_samples, n_state)

        if single_x0:
            # Squeeze out the n_samples=1 dimension -> (len(t), n_state)
            return stacked[:, 0, :]
        else:
            return stacked

    def predict_trajectory(self, x0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict trajectory at specified times.

        Parameters
        ----------
        x0 : np.ndarray, shape ``(n_state,)``
            Initial state vector.
        times : np.ndarray, shape ``(n_times,)``
            Array of query times.

        Returns
        -------
        np.ndarray, shape ``(n_times, n_state)``
        """
        x0 = np.asarray(x0, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64).ravel()

        if x0.ndim != 1:
            raise ValueError(
                f"predict_trajectory expects a 1-D initial state, got shape {x0.shape}."
            )

        return self.predict(x0, times)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_single_time(self, x0: np.ndarray, t: float) -> np.ndarray:
        """Predict state at a single time for x0 of shape (n_state,) or (n_samples, n_state)."""
        squeeze = x0.ndim == 1
        if squeeze:
            x0_2d = x0.reshape(1, -1)
        else:
            x0_2d = x0

        # Lift
        z0 = self._model.lift(x0_2d)  # (n_samples, n_koopman)

        # Compute K_t
        K_t = self._compute_K_t(t)  # (n_koopman, n_koopman)

        # Propagate (row-vector convention)
        z_t = z0 @ K_t  # (n_samples, n_koopman)

        # Take real part if complex
        if np.iscomplexobj(z_t):
            z_t = z_t.real

        # Unlift
        x_t = self._model.unlift(z_t)  # (n_samples, n_state)

        if squeeze:
            return x_t[0]
        return x_t

    def _compute_K_t(self, t: float) -> np.ndarray:
        """Compute the Koopman matrix at arbitrary time t."""
        if self._method == "expm":
            K_t = expm(self._L * t)
            if np.iscomplexobj(K_t):
                K_t = K_t.real
            return K_t
        else:
            # eigen backend
            powers = self._eigenvalues ** (t / self._dt)
            K_t = self._V @ np.diag(powers) @ self._V_inv
            if np.iscomplexobj(K_t):
                K_t = K_t.real
            return K_t

    @property
    def method(self) -> str:
        """The resolved prediction method ('expm' or 'eigen')."""
        return self._method

    @property
    def model(self) -> KoopmanModel:
        """The underlying Koopman model."""
        return self._model
