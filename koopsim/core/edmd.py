"""Extended Dynamic Mode Decomposition (EDMD) implementation.

Fits a finite-dimensional approximation of the Koopman operator from
snapshot pairs using a user-specified observable dictionary.
"""

from __future__ import annotations

import logging

import numpy as np

from koopsim.core.base import KoopmanModel
from koopsim.core.constants import (
    CONDITION_NUMBER_WARNING_THRESHOLD,
    DEFAULT_REGULARIZATION,
)
from koopsim.core.exceptions import DimensionMismatchError, NotFittedError
from koopsim.utils.dictionary import (
    IdentityDictionary,
    ObservableDictionary,
)

logger = logging.getLogger("koopsim")


class EDMD(KoopmanModel):
    """Extended Dynamic Mode Decomposition.

    Approximates the Koopman operator in a finite-dimensional observable space
    defined by *dictionary*.  Uses SVD-based Tikhonov-regularised least squares
    for numerical stability.

    Row-vector convention: ``Psi_Y ≈ Psi_X @ K``.

    Parameters
    ----------
    dictionary : ObservableDictionary | None
        Observable dictionary for lifting.  If ``None``, an
        :class:`IdentityDictionary` is used (pure DMD).
    regularization : float
        Tikhonov regularisation parameter (alpha).  Applied as
        ``(Psi_X.T @ Psi_X + alpha * I)^{-1} @ Psi_X.T @ Psi_Y``.
    svd_rank : int | None
        If set, truncate to the top *svd_rank* singular values before
        solving.  Useful for filtering noise in rank-deficient data.
    """

    def __init__(
        self,
        dictionary: ObservableDictionary | None = None,
        regularization: float = DEFAULT_REGULARIZATION,
        svd_rank: int | None = None,
    ) -> None:
        self._dictionary_input = dictionary
        self._regularization = regularization
        self._svd_rank = svd_rank

        # Fitted attributes (set by fit)
        self.K_: np.ndarray | None = None
        self.dictionary_: ObservableDictionary | None = None
        self._n_state_dims: int | None = None
        self._n_koopman_dims: int | None = None
        self._dt: float | None = None

    # ------------------------------------------------------------------
    # KoopmanModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, dt: float) -> EDMD:
        """Fit the EDMD model from snapshot pairs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Pre-snapshot data (row vectors).
        Y : np.ndarray, shape (n_samples, n_features)
            Post-snapshot data (row vectors), taken *dt* time later.
        dt : float
            Time step between snapshot pairs.

        Returns
        -------
        self

        Raises
        ------
        DimensionMismatchError
            If X and Y have incompatible shapes.
        ValueError
            If dt <= 0.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or Y.ndim != 2:
            raise DimensionMismatchError(
                f"X and Y must be 2D arrays. Got X.ndim={X.ndim}, Y.ndim={Y.ndim}."
            )
        if X.shape != Y.shape:
            raise DimensionMismatchError(
                f"X and Y must have the same shape. Got X.shape={X.shape}, Y.shape={Y.shape}."
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

        n_samples, n_features = X.shape
        self._n_state_dims = n_features
        self._dt = dt

        # 1. Build / wrap dictionary
        if self._dictionary_input is None:
            self.dictionary_ = IdentityDictionary()
        else:
            self.dictionary_ = self._dictionary_input

        # 2. Fit dictionary on X, then lift both X and Y
        self.dictionary_.fit(X)
        Psi_X = self.dictionary_.transform(X)  # (n_samples, n_obs)
        Psi_Y = self.dictionary_.transform(Y)  # (n_samples, n_obs)

        self._n_koopman_dims = Psi_X.shape[1]

        logger.info(
            "EDMD fitting: %d samples, %d state dims -> %d Koopman dims.",
            n_samples,
            n_features,
            self._n_koopman_dims,
        )

        # 3. Solve K via SVD-based Tikhonov regression
        #    Psi_Y ≈ Psi_X @ K
        #    K = (Psi_X.T Psi_X + alpha I)^{-1} Psi_X.T Psi_Y
        self.K_ = self._solve_svd(Psi_X, Psi_Y)

        logger.info("EDMD fit complete. K shape: %s.", self.K_.shape)
        return self

    def get_koopman_matrix(self) -> np.ndarray:
        """Return the fitted Koopman matrix K.

        Returns
        -------
        np.ndarray, shape (n_koopman_dims, n_koopman_dims)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self.K_ is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        return self.K_

    def lift(self, X: np.ndarray) -> np.ndarray:
        """Lift state-space data into the Koopman observable space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_koopman_dims)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self.dictionary_ is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        return self.dictionary_.transform(X)

    def unlift(self, Z: np.ndarray) -> np.ndarray:
        """Project Koopman observables back to state space.

        For dictionaries that prepend identity (e.g. ``CompositeDictionary``),
        the first ``n_state_dims`` columns *are* the state. For a plain
        ``IdentityDictionary`` the full output is the state.

        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, n_koopman_dims)

        Returns
        -------
        np.ndarray, shape (n_samples, n_state_dims)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self.dictionary_ is None or self._n_state_dims is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        Z = np.asarray(Z, dtype=np.float64)
        return Z[:, : self._n_state_dims]

    @property
    def n_state_dims(self) -> int:
        """Number of original state-space dimensions."""
        if self._n_state_dims is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        return self._n_state_dims

    @property
    def n_koopman_dims(self) -> int:
        """Number of Koopman observable dimensions."""
        if self._n_koopman_dims is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        return self._n_koopman_dims

    @property
    def dt(self) -> float:
        """Time step between snapshot pairs used for fitting."""
        if self._dt is None:
            raise NotFittedError("EDMD has not been fitted. Call fit() first.")
        return self._dt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_svd(self, Psi_X: np.ndarray, Psi_Y: np.ndarray) -> np.ndarray:
        """Solve for K using SVD-based Tikhonov regression.

        Solves ``Psi_Y ≈ Psi_X @ K`` as:
            K = V diag(sigma / (sigma^2 + alpha)) U^T Psi_Y

        where ``Psi_X = U Sigma V^T`` (economy SVD).

        Parameters
        ----------
        Psi_X : np.ndarray, shape (n_samples, n_obs)
        Psi_Y : np.ndarray, shape (n_samples, n_obs)

        Returns
        -------
        np.ndarray, shape (n_obs, n_obs)
            The Koopman matrix K.
        """
        U, sigma, Vt = np.linalg.svd(Psi_X, full_matrices=False)

        # Log condition number
        cond = sigma[0] / sigma[-1] if sigma[-1] > 0 else np.inf
        logger.debug("EDMD SVD condition number: %.4e", cond)
        if cond > CONDITION_NUMBER_WARNING_THRESHOLD:
            logger.warning(
                "High condition number (%.4e) detected in EDMD solve. "
                "Consider increasing regularization or reducing dictionary size.",
                cond,
            )

        # Optional SVD rank truncation
        if self._svd_rank is not None and self._svd_rank < len(sigma):
            rank = self._svd_rank
            logger.info("Truncating SVD: %d -> %d singular values.", len(sigma), rank)
            U = U[:, :rank]
            sigma = sigma[:rank]
            Vt = Vt[:rank, :]

        # Tikhonov filter factors: sigma_i / (sigma_i^2 + alpha)
        alpha = self._regularization
        filter_factors = sigma / (sigma**2 + alpha)

        # K = V @ diag(filter_factors) @ U^T @ Psi_Y
        #   = Vt.T @ diag(filter_factors) @ U.T @ Psi_Y
        K = Vt.T @ np.diag(filter_factors) @ (U.T @ Psi_Y)

        return K
