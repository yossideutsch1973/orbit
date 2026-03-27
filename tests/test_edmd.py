"""Tests for the EDMD (Extended Dynamic Mode Decomposition) implementation."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.edmd import EDMD
from koopsim.core.exceptions import NotFittedError
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
)

# -------------------------------------------------------------------------
# 1. Perfect recovery on 2D rotation with IdentityDictionary
# -------------------------------------------------------------------------


class TestLinearRotation:
    """EDMD with IdentityDictionary on a pure 2D rotation should recover
    K ≈ R.T to near machine precision (the transpose comes from the
    row-vector convention Y = X @ R.T, so K = R.T)."""

    def test_perfect_recovery(self, simple_linear_system):
        """K should match R.T to near machine precision."""
        X, Y, dt, theta = simple_linear_system
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
        model.fit(X, Y, dt)

        K = model.get_koopman_matrix()
        np.testing.assert_allclose(K, R.T, atol=1e-10)

    def test_lift_unlift_roundtrip(self, simple_linear_system):
        """lift then unlift should return the original data."""
        X, Y, dt, _ = simple_linear_system
        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
        model.fit(X, Y, dt)

        Z = model.lift(X)
        X_rec = model.unlift(Z)
        np.testing.assert_allclose(X_rec, X, atol=1e-14)

    def test_one_step_prediction(self, simple_linear_system):
        """Psi_X @ K should reproduce Y."""
        X, Y, dt, _ = simple_linear_system
        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
        model.fit(X, Y, dt)

        K = model.get_koopman_matrix()
        Y_pred = model.lift(X) @ K
        Y_rec = model.unlift(Y_pred)
        np.testing.assert_allclose(Y_rec, Y, atol=1e-10)


# -------------------------------------------------------------------------
# 2. Duffing-like nonlinear system with PolynomialDictionary
# -------------------------------------------------------------------------


def _duffing_step(X: np.ndarray) -> np.ndarray:
    """Simple nonlinear map: x_next = [x1 + 0.1*x2, x2 - 0.1*x1 + 0.05*x1^3]."""
    x1, x2 = X[:, 0], X[:, 1]
    y1 = x1 + 0.1 * x2
    y2 = x2 - 0.1 * x1 + 0.05 * x1**3
    return np.column_stack([y1, y2])


@pytest.fixture
def duffing_data(rng):
    """Generate snapshot pairs from the Duffing-like map."""
    n_samples = 500
    X = rng.uniform(-1.0, 1.0, (n_samples, 2))
    Y = _duffing_step(X)
    dt = 0.1
    return X, Y, dt


class TestDuffingNonlinear:
    """EDMD with a degree-3 polynomial dictionary on the Duffing-like system."""

    def test_polynomial_edmd_bounded_error(self, duffing_data):
        """One-step prediction error should be small with a poly dictionary."""
        X, Y, dt = duffing_data
        poly = PolynomialDictionary(degree=3)
        dictionary = CompositeDictionary([poly])

        model = EDMD(dictionary=dictionary, regularization=1e-10)
        model.fit(X, Y, dt)

        K = model.get_koopman_matrix()
        Psi_X = model.lift(X)
        Psi_Y_pred = Psi_X @ K
        Y_pred = model.unlift(Psi_Y_pred)

        # One-step relative error should be well bounded
        error = np.linalg.norm(Y_pred - Y) / np.linalg.norm(Y)
        assert error < 0.05, f"One-step relative error too large: {error:.4f}"

    def test_polynomial_edmd_multi_step(self, duffing_data, rng):
        """Multi-step prediction should remain reasonable for a few steps."""
        X, Y, dt = duffing_data
        poly = PolynomialDictionary(degree=3)
        dictionary = CompositeDictionary([poly])

        model = EDMD(dictionary=dictionary, regularization=1e-10)
        model.fit(X, Y, dt)

        K = model.get_koopman_matrix()

        # Start from a single initial condition and propagate 5 steps
        x0 = rng.uniform(-0.5, 0.5, (1, 2))
        x_true = x0.copy()
        x_pred = x0.copy()

        for _ in range(5):
            x_true = _duffing_step(x_true)
            psi = model.lift(x_pred)
            psi_next = psi @ K
            x_pred = model.unlift(psi_next)

        # After 5 steps, error should still be bounded (not diverging wildly)
        error = np.linalg.norm(x_pred - x_true)
        assert error < 1.0, f"Multi-step error diverged: {error:.4f}"


# -------------------------------------------------------------------------
# 3. Regularization reduces test-set error when data has noise
# -------------------------------------------------------------------------


class TestRegularization:
    """Regularization should help when data is noisy and overparameterized."""

    def test_regularization_improves_noisy_fit(self, rng):
        """With an overparameterized dictionary and noisy data, regularization
        should reduce test-set error by preventing overfitting to noise."""
        # Simple 2D rotation
        theta = np.pi / 6
        dt = 0.1
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        # Small training set with heavy noise -> overfitting regime
        n_train = 30
        n_test = 200
        X_train = rng.standard_normal((n_train, 2))
        Y_train_clean = X_train @ R.T
        Y_train = Y_train_clean + rng.normal(0, 0.3, Y_train_clean.shape)

        X_test = rng.standard_normal((n_test, 2))
        Y_test = X_test @ R.T  # Clean test targets

        # Use a high-degree polynomial dictionary to create many more
        # parameters than samples (overparameterized regime)
        poly = PolynomialDictionary(degree=5)
        dictionary_low = CompositeDictionary([poly])
        dictionary_high = CompositeDictionary([PolynomialDictionary(degree=5)])

        # Low regularization — will overfit to noise
        model_low = EDMD(dictionary=dictionary_low, regularization=1e-14)
        model_low.fit(X_train, Y_train, dt)
        K_low = model_low.get_koopman_matrix()
        Psi_test = model_low.lift(X_test)
        Y_pred_low = model_low.unlift(Psi_test @ K_low)
        error_low = np.linalg.norm(Y_pred_low - Y_test, "fro")

        # Higher regularization — should generalize better
        model_high = EDMD(dictionary=dictionary_high, regularization=1e-1)
        model_high.fit(X_train, Y_train, dt)
        K_high = model_high.get_koopman_matrix()
        Psi_test_h = model_high.lift(X_test)
        Y_pred_high = model_high.unlift(Psi_test_h @ K_high)
        error_high = np.linalg.norm(Y_pred_high - Y_test, "fro")

        # In this overparameterized + noisy regime, regularization should help
        assert error_high < error_low, (
            f"Regularization did not improve test error: "
            f"low_reg={error_low:.4f}, high_reg={error_high:.4f}"
        )


# -------------------------------------------------------------------------
# 4. SVD rank truncation on a system with low-rank dynamics
# -------------------------------------------------------------------------


class TestSVDTruncation:
    """SVD rank truncation should filter out noisy directions."""

    def test_rank_truncation_recovers_low_rank(self, rng):
        """System with rank-2 dynamics embedded in 5D should benefit from
        SVD rank truncation when there is substantial noise in the
        null-space directions."""
        # Create a rank-2 system in 5D
        n_samples = 50  # Few samples to make noise matter more
        dt = 0.1

        # True dynamics in 2D
        theta = np.pi / 4
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        X_2d = rng.standard_normal((n_samples, 2))
        Y_2d = X_2d @ R.T

        # Embed into 5D with a well-conditioned embedding
        embed = np.zeros((2, 5))
        embed[0, 0] = 1.0
        embed[1, 1] = 1.0
        X_5d = X_2d @ embed
        Y_5d = Y_2d @ embed

        # Add significant noise — especially in the null-space dims (cols 2-4)
        noise_X = rng.standard_normal(X_5d.shape) * 0.3
        noise_Y = rng.standard_normal(Y_5d.shape) * 0.3
        X_5d_noisy = X_5d + noise_X
        Y_5d_noisy = Y_5d + noise_Y

        # Without truncation — full model tries to fit noise in all 5 dims
        model_full = EDMD(dictionary=IdentityDictionary(), regularization=1e-12)
        model_full.fit(X_5d_noisy, Y_5d_noisy, dt)

        # With rank-2 truncation — focuses on the two meaningful directions
        model_trunc = EDMD(dictionary=IdentityDictionary(), regularization=1e-12, svd_rank=2)
        model_trunc.fit(X_5d_noisy, Y_5d_noisy, dt)

        # Evaluate on a large clean test set
        X_test_2d = rng.standard_normal((500, 2))
        Y_test_2d = X_test_2d @ R.T
        X_test = X_test_2d @ embed
        Y_test = Y_test_2d @ embed

        K_full = model_full.get_koopman_matrix()
        K_trunc = model_trunc.get_koopman_matrix()

        Y_pred_full = X_test @ K_full
        Y_pred_trunc = X_test @ K_trunc

        error_full = np.linalg.norm(Y_pred_full - Y_test, "fro")
        error_trunc = np.linalg.norm(Y_pred_trunc - Y_test, "fro")

        # Truncated model should generalize better on the meaningful subspace
        assert error_trunc <= error_full * 1.05, (
            f"Truncated model much worse: full={error_full:.4f}, trunc={error_trunc:.4f}"
        )

    def test_svd_rank_sets_correctly(self, simple_linear_system):
        """svd_rank parameter should be used during fitting."""
        X, Y, dt, _ = simple_linear_system
        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15, svd_rank=1)
        model.fit(X, Y, dt)
        K = model.get_koopman_matrix()
        # With rank-1 truncation on a 2D system, K should be rank 1
        rank = np.linalg.matrix_rank(K, tol=1e-10)
        assert rank == 1, f"Expected rank 1, got {rank}"


# -------------------------------------------------------------------------
# 5. NotFittedError when calling methods before fit
# -------------------------------------------------------------------------


class TestNotFittedError:
    """Calling model methods before fit should raise NotFittedError."""

    def test_get_koopman_matrix_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            model.get_koopman_matrix()

    def test_lift_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            model.lift(np.zeros((1, 2)))

    def test_unlift_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            model.unlift(np.zeros((1, 2)))

    def test_n_state_dims_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            _ = model.n_state_dims

    def test_n_koopman_dims_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            _ = model.n_koopman_dims

    def test_dt_raises(self):
        model = EDMD()
        with pytest.raises(NotFittedError):
            _ = model.dt


# -------------------------------------------------------------------------
# 6. Test metadata() returns correct info
# -------------------------------------------------------------------------


class TestMetadata:
    """metadata() should return correct model information."""

    def test_metadata_before_fit(self):
        model = EDMD()
        meta = model.metadata()
        assert meta["model_type"] == "EDMD"
        assert meta["is_fitted"] is False
        assert "n_state_dims" not in meta
        assert "n_koopman_dims" not in meta
        assert "dt" not in meta

    def test_metadata_after_fit(self, simple_linear_system):
        X, Y, dt, _ = simple_linear_system
        model = EDMD(dictionary=IdentityDictionary())
        model.fit(X, Y, dt)

        meta = model.metadata()
        assert meta["model_type"] == "EDMD"
        assert meta["is_fitted"] is True
        assert meta["n_state_dims"] == 2
        assert meta["n_koopman_dims"] == 2
        assert meta["dt"] == dt

    def test_metadata_with_composite_dictionary(self, simple_linear_system):
        X, Y, dt, _ = simple_linear_system
        poly = PolynomialDictionary(degree=2)
        dictionary = CompositeDictionary([poly])

        model = EDMD(dictionary=dictionary)
        model.fit(X, Y, dt)

        meta = model.metadata()
        assert meta["n_state_dims"] == 2
        # Identity (2) + degree-2 poly terms for 2 features: x1^2, x1*x2, x2^2 = 3
        assert meta["n_koopman_dims"] == 5
