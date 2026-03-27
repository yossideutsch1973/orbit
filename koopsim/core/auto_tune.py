"""Automatic hyperparameter selection for EDMD via cross-validation.

Given snapshot pairs (X, Y), evaluates candidate dictionary configurations
and regularization values, and returns the best combination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from koopsim.core.edmd import EDMD
from koopsim.core.exceptions import KoopSimError
from koopsim.core.validation import ModelValidator
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
    RBFDictionary,
)

logger = logging.getLogger("koopsim")


@dataclass
class AutoTuneResult:
    """Result of automatic hyperparameter selection.

    Attributes
    ----------
    poly_degree : int or None
        Best polynomial degree, or None if polynomial not selected.
    rbf_centers : int or None
        Best number of RBF centers, or None if RBF not selected.
    regularization : float
        Best regularization value.
    cv_error : float
        Cross-validation error of the best configuration.
    all_results : list[dict]
        Results for all evaluated configurations.
    """

    poly_degree: int | None
    rbf_centers: int | None
    regularization: float
    cv_error: float
    all_results: list[dict]


def auto_tune(
    X: np.ndarray,
    Y: np.ndarray,
    dt: float,
    *,
    poly_degrees: list[int] | None = None,
    rbf_centers_list: list[int] | None = None,
    regularizations: list[float] | None = None,
    n_folds: int = 5,
    metric: str = "rmse",
    verbose: bool = True,
) -> AutoTuneResult:
    """Find the best EDMD hyperparameters via k-fold cross-validation.

    Evaluates all combinations of dictionary type, complexity, and
    regularization, using k-fold CV error to select the winner.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Pre-snapshot data.
    Y : np.ndarray, shape (n_samples, n_features)
        Post-snapshot data.
    dt : float
        Time step between snapshot pairs.
    poly_degrees : list[int] or None
        Polynomial degrees to try. Default: [2, 3, 4].
    rbf_centers_list : list[int] or None
        RBF center counts to try. Default: [10, 25, 50].
    regularizations : list[float] or None
        Regularization values to try. Default: [1e-8, 1e-6, 1e-4, 1e-2].
    n_folds : int
        Number of CV folds. Default: 5.
    metric : str
        Error metric: 'rmse', 'mae', or 'relative'. Default: 'rmse'.
    verbose : bool
        Log progress.

    Returns
    -------
    AutoTuneResult
        Best hyperparameters and all evaluation results.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] < n_folds:
        raise ValueError(
            f"n_samples ({X.shape[0]}) must be >= n_folds ({n_folds}). "
            f"Reduce n_folds or provide more data."
        )

    if poly_degrees is None:
        poly_degrees = [2, 3, 4]
    if rbf_centers_list is None:
        n_samples = X.shape[0]
        rbf_centers_list = [c for c in [10, 25, 50] if c < n_samples]
        if not rbf_centers_list:
            rbf_centers_list = [max(2, n_samples // 2)]
    if regularizations is None:
        regularizations = [1e-8, 1e-6, 1e-4, 1e-2]

    # Build candidate configurations: (poly_degree, rbf_centers) pairs
    candidates: list[tuple[int | None, int | None]] = []

    # Identity only
    candidates.append((None, None))

    # Polynomial only
    for deg in poly_degrees:
        candidates.append((deg, None))

    # RBF only
    for n_c in rbf_centers_list:
        candidates.append((None, n_c))

    # Polynomial + RBF combined
    for deg in poly_degrees:
        for n_c in rbf_centers_list:
            candidates.append((deg, n_c))

    # Generate fold indices
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    all_results: list[dict] = []
    best_error = np.inf
    best_config: dict = {}

    total = len(candidates) * len(regularizations)
    evaluated = 0

    for poly_deg, rbf_n in candidates:
        for reg in regularizations:
            fold_errors = []

            for fold_idx in range(n_folds):
                # Split train/val
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

                X_train, Y_train = X[train_idx], Y[train_idx]
                X_val, Y_val = X[val_idx], Y[val_idx]

                # Build dictionary
                dictionary = _build_dictionary(poly_deg, rbf_n)

                # Fit
                model = EDMD(dictionary=dictionary, regularization=reg)
                try:
                    model.fit(X_train, Y_train, dt)
                    error = ModelValidator.prediction_error(model, X_val, Y_val, metric=metric)
                except Exception:
                    error = np.inf
                fold_errors.append(error)

            mean_error = float(np.mean(fold_errors))
            result = {
                "poly_degree": poly_deg,
                "rbf_centers": rbf_n,
                "regularization": reg,
                "cv_error": mean_error,
                "fold_errors": fold_errors,
            }
            all_results.append(result)

            if mean_error < best_error:
                best_error = mean_error
                best_config = result

            evaluated += 1
            if verbose:
                _log_progress(evaluated, total, poly_deg, rbf_n, reg, mean_error)

    if not best_config or not np.isfinite(best_error):
        raise KoopSimError(
            "Auto-tune failed: no configuration produced a finite CV error. "
            "Check your data or try different hyperparameter ranges."
        )

    if verbose:
        logger.info(
            "Auto-tune best: poly=%s, rbf=%s, reg=%.1e, cv_%s=%.6f",
            best_config["poly_degree"],
            best_config["rbf_centers"],
            best_config["regularization"],
            metric,
            best_error,
        )

    return AutoTuneResult(
        poly_degree=best_config["poly_degree"],
        rbf_centers=best_config["rbf_centers"],
        regularization=best_config["regularization"],
        cv_error=best_error,
        all_results=all_results,
    )


def _build_dictionary(poly_degree: int | None, rbf_centers: int | None):
    """Build a dictionary from the given configuration."""
    dictionaries: list = []
    if poly_degree is not None:
        dictionaries.append(PolynomialDictionary(poly_degree))
    if rbf_centers is not None:
        dictionaries.append(RBFDictionary(n_centers=rbf_centers))
    if dictionaries:
        return CompositeDictionary(dictionaries)
    return IdentityDictionary()


def _log_progress(
    evaluated: int,
    total: int,
    poly_deg: int | None,
    rbf_n: int | None,
    reg: float,
    error: float,
) -> None:
    """Log a single evaluation result."""
    dict_desc = "identity"
    if poly_deg is not None and rbf_n is not None:
        dict_desc = f"poly({poly_deg})+rbf({rbf_n})"
    elif poly_deg is not None:
        dict_desc = f"poly({poly_deg})"
    elif rbf_n is not None:
        dict_desc = f"rbf({rbf_n})"

    logger.info(
        "[%d/%d] %s, reg=%.1e -> cv_error=%.6f",
        evaluated,
        total,
        dict_desc,
        reg,
        error,
    )
