"""Abstract base class for all Koopman operator models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class KoopmanModel(ABC):
    """Abstract base for all Koopman operator models.

    All implementations must follow the row-vector convention:
    X is (n_samples, n_features), K is (n_obs, n_obs), and Psi_Y ≈ Psi_X @ K.

    Predictions use the Koopman form: state(t) = K**t @ initial_state
    (or expm(t * logK)), with no time-stepping loops.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, dt: float) -> KoopmanModel:
        """Fit the Koopman model from snapshot pairs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Pre-snapshot data (row vectors).
        Y : np.ndarray, shape (n_samples, n_features)
            Post-snapshot data (row vectors), taken dt time later.
        dt : float
            Time step between snapshot pairs.

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def get_koopman_matrix(self) -> np.ndarray:
        """Return the fitted Koopman matrix K.

        Returns
        -------
        np.ndarray, shape (n_koopman_dims, n_koopman_dims)
        """
        ...

    @abstractmethod
    def lift(self, X: np.ndarray) -> np.ndarray:
        """Lift state-space data into the Koopman observable space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_koopman_dims)
        """
        ...

    @abstractmethod
    def unlift(self, Z: np.ndarray) -> np.ndarray:
        """Project Koopman observables back to state space.

        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, n_koopman_dims)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        ...

    @property
    @abstractmethod
    def n_state_dims(self) -> int:
        """Number of original state-space dimensions."""
        ...

    @property
    @abstractmethod
    def n_koopman_dims(self) -> int:
        """Number of Koopman observable dimensions."""
        ...

    @property
    @abstractmethod
    def dt(self) -> float:
        """Time step between snapshot pairs used for fitting."""
        ...

    def metadata(self) -> dict:
        """Return a dictionary with model metadata.

        Returns
        -------
        dict
            Keys include 'model_type', 'n_state_dims', 'n_koopman_dims', 'dt',
            and 'is_fitted'.
        """
        is_fitted = self._is_fitted()
        info: dict = {
            "model_type": type(self).__name__,
            "is_fitted": is_fitted,
        }
        if is_fitted:
            info["n_state_dims"] = self.n_state_dims
            info["n_koopman_dims"] = self.n_koopman_dims
            info["dt"] = self.dt
        return info

    def _is_fitted(self) -> bool:
        """Check whether the model has been fitted.

        Subclasses may override for a more specific check.
        """
        try:
            self.get_koopman_matrix()
            return True
        except Exception:
            return False
