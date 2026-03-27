"""Observable dictionary functions for Koopman operator learning.

Provides lifting maps that embed state-space data into higher-dimensional
observable spaces where the Koopman operator acts linearly.
"""

from __future__ import annotations

import abc
import logging
from typing import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures

from koopsim.core.exceptions import DimensionMismatchError, NotFittedError

logger = logging.getLogger("koopsim")


class ObservableDictionary(abc.ABC):
    """Abstract base class for observable dictionary functions.

    All dictionaries follow the row-vector convention: X is (n_samples, n_features).
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> ObservableDictionary:
        """Fit the dictionary to training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through the dictionary.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Lifted data of shape (n_samples, n_output_features).
        """

    @property
    @abc.abstractmethod
    def n_output_features(self) -> int:
        """Number of output features after transformation."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Lifted data of shape (n_samples, n_output_features).
        """
        return self.fit(X).transform(X)


class IdentityDictionary(ObservableDictionary):
    """Identity (passthrough) dictionary.

    Returns input data unchanged. Used as the base observable that preserves
    raw state variables.
    """

    def __init__(self) -> None:
        self._n_features: int | None = None

    def fit(self, X: np.ndarray) -> IdentityDictionary:
        """Fit by recording the number of input features.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        self._n_features = X.shape[1]
        logger.debug("IdentityDictionary fitted with %d features.", self._n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return input data unchanged.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Same data, shape (n_samples, n_features).
        """
        if self._n_features is None:
            raise NotFittedError("IdentityDictionary has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        if X.shape[1] != self._n_features:
            raise DimensionMismatchError(
                f"Expected {self._n_features} features, got {X.shape[1]}."
            )
        return X

    @property
    def n_output_features(self) -> int:
        """Number of output features (equal to input features)."""
        if self._n_features is None:
            raise NotFittedError("IdentityDictionary has not been fitted.")
        return self._n_features


class PolynomialDictionary(ObservableDictionary):
    """Polynomial observable dictionary.

    Wraps scikit-learn's PolynomialFeatures but excludes the constant term
    and linear terms (degree 1), since identity observables handle those.
    Only returns higher-order polynomial features (degree >= 2).

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (must be >= 2).
    """

    def __init__(self, degree: int = 2) -> None:
        if degree < 2:
            raise ValueError("Polynomial degree must be >= 2.")
        self._degree = degree
        self._poly: PolynomialFeatures | None = None
        self._n_features: int | None = None
        self._n_output: int | None = None

    def fit(self, X: np.ndarray) -> PolynomialDictionary:
        """Fit the polynomial dictionary.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        self._n_features = X.shape[1]
        # include_bias=False excludes the constant term.
        # We still get linear terms (degree 1) which we need to strip.
        self._poly = PolynomialFeatures(degree=self._degree, include_bias=False)
        self._poly.fit(X)
        # Identify which output columns correspond to degree >= 2.
        # powers_ has shape (n_output_features, n_features) with exponent sums.
        degrees = self._poly.powers_.sum(axis=1)
        self._higher_order_mask = degrees >= 2
        self._n_output = int(self._higher_order_mask.sum())
        logger.debug(
            "PolynomialDictionary fitted: degree=%d, %d input features -> %d "
            "higher-order features.",
            self._degree,
            self._n_features,
            self._n_output,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to higher-order polynomial features.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Polynomial features of shape (n_samples, n_output_features),
            excluding constant and linear terms.
        """
        if self._poly is None:
            raise NotFittedError("PolynomialDictionary has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        if X.shape[1] != self._n_features:
            raise DimensionMismatchError(
                f"Expected {self._n_features} features, got {X.shape[1]}."
            )
        all_features = self._poly.transform(X)
        return all_features[:, self._higher_order_mask]

    @property
    def n_output_features(self) -> int:
        """Number of higher-order polynomial features."""
        if self._n_output is None:
            raise NotFittedError("PolynomialDictionary has not been fitted.")
        return self._n_output


class RBFDictionary(ObservableDictionary):
    """Radial Basis Function observable dictionary.

    Places RBF centers via k-means clustering on training data, then
    computes Gaussian RBF activations for each center.

    Parameters
    ----------
    n_centers : int
        Number of RBF centers (k-means clusters).
    kernel : str
        Kernel type. Currently only ``'rbf'`` (Gaussian) is supported.
    gamma : float | str
        Kernel bandwidth parameter. If ``'auto'``, uses the median heuristic:
        ``gamma = 1 / (2 * median(pairwise_distances)^2)``.
    """

    def __init__(
        self,
        n_centers: int = 100,
        kernel: str = "rbf",
        gamma: float | str = "auto",
    ) -> None:
        if kernel != "rbf":
            raise ValueError(f"Unsupported kernel '{kernel}'. Only 'rbf' is supported.")
        if n_centers < 1:
            raise ValueError("n_centers must be >= 1.")
        self._n_centers = n_centers
        self._kernel = kernel
        self._gamma_param = gamma
        self._gamma: float | None = None
        self._centers: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> RBFDictionary:
        """Fit RBF centers using k-means and compute gamma.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")

        # Use MiniBatchKMeans for efficiency on larger datasets.
        n_centers = min(self._n_centers, X.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_centers, random_state=0, batch_size=min(1024, X.shape[0])
        )
        kmeans.fit(X)
        self._centers = kmeans.cluster_centers_  # (n_centers, n_features)

        if self._gamma_param == "auto":
            # Median heuristic: gamma = 1 / (2 * median_dist^2)
            # Compute pairwise distances between a subsample and centers
            # for efficiency.
            n_sub = min(500, X.shape[0])
            indices = np.linspace(0, X.shape[0] - 1, n_sub, dtype=int)
            X_sub = X[indices]
            # Squared distances from subsample to centers
            dists_sq = self._squared_distances(X_sub, self._centers)
            median_dist_sq = np.median(dists_sq[dists_sq > 0])
            if median_dist_sq <= 0:
                median_dist_sq = 1.0
            self._gamma = 1.0 / (2.0 * median_dist_sq)
        else:
            self._gamma = float(self._gamma_param)

        logger.debug(
            "RBFDictionary fitted: %d centers, gamma=%.6g.",
            n_centers,
            self._gamma,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute RBF activations for each center.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            RBF activations of shape (n_samples, n_centers).
        """
        if self._centers is None or self._gamma is None:
            raise NotFittedError("RBFDictionary has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        if X.shape[1] != self._centers.shape[1]:
            raise DimensionMismatchError(
                f"Expected {self._centers.shape[1]} features, got {X.shape[1]}."
            )
        dists_sq = self._squared_distances(X, self._centers)
        return np.exp(-self._gamma * dists_sq)

    @property
    def n_output_features(self) -> int:
        """Number of RBF centers."""
        if self._centers is None:
            raise NotFittedError("RBFDictionary has not been fitted.")
        return self._centers.shape[0]

    @staticmethod
    def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between rows of A and B.

        Parameters
        ----------
        A : np.ndarray
            Shape (n, d).
        B : np.ndarray
            Shape (m, d).

        Returns
        -------
        np.ndarray
            Shape (n, m) of squared distances.
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        A_sq = np.sum(A**2, axis=1, keepdims=True)  # (n, 1)
        B_sq = np.sum(B**2, axis=1, keepdims=True)  # (m, 1)
        cross = A @ B.T  # (n, m)
        dists_sq = A_sq + B_sq.T - 2.0 * cross
        # Clamp negative values arising from floating-point errors.
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return dists_sq


class CompositeDictionary(ObservableDictionary):
    """Composite dictionary that horizontally stacks multiple dictionaries.

    Always prepends an :class:`IdentityDictionary` so that the first
    ``n_state`` columns of the lifted output are the raw state variables.
    This is critical for unlifting predictions back to state space.

    Parameters
    ----------
    dictionaries : Sequence[ObservableDictionary]
        List of dictionaries to compose. An :class:`IdentityDictionary` is
        automatically prepended; any ``IdentityDictionary`` instances in the
        input list are silently skipped to avoid duplication.
    """

    def __init__(self, dictionaries: Sequence[ObservableDictionary]) -> None:
        # Filter out IdentityDictionary instances from input; we prepend our own.
        non_identity = [d for d in dictionaries if not isinstance(d, IdentityDictionary)]
        self._identity = IdentityDictionary()
        self._dictionaries: list[ObservableDictionary] = [self._identity] + non_identity
        self._fitted = False

    def fit(self, X: np.ndarray) -> CompositeDictionary:
        """Fit all constituent dictionaries.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise DimensionMismatchError(f"Expected 2D array, got {X.ndim}D array.")
        for d in self._dictionaries:
            d.fit(X)
        self._fitted = True
        logger.debug(
            "CompositeDictionary fitted: %d sub-dictionaries, %d total features.",
            len(self._dictionaries),
            self.n_output_features,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform through all dictionaries and horizontally stack.

        The first ``n_state`` columns are always the raw state (identity).

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Lifted data of shape (n_samples, n_output_features).
        """
        if not self._fitted:
            raise NotFittedError("CompositeDictionary has not been fitted.")
        parts = [d.transform(X) for d in self._dictionaries]
        return np.hstack(parts)

    @property
    def n_output_features(self) -> int:
        """Total number of output features across all dictionaries."""
        if not self._fitted:
            raise NotFittedError("CompositeDictionary has not been fitted.")
        return sum(d.n_output_features for d in self._dictionaries)

    @property
    def n_state(self) -> int:
        """Number of raw state features (identity columns)."""
        return self._identity.n_output_features
