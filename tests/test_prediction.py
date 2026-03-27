"""Tests for the Prediction Engine (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.constants import EIGENVALUE_STABILITY_THRESHOLD
from koopsim.core.edmd import EDMD
from koopsim.core.exceptions import NotFittedError
from koopsim.core.prediction import PredictionEngine
from koopsim.utils.dictionary import IdentityDictionary


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _fit_rotation_model(X, Y, dt):
    """Fit a minimal EDMD model on the given rotation data."""
    model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
    model.fit(X, Y, dt)
    return model


# -------------------------------------------------------------------------
# 1. Rotation agreement: expm and eigen give same results
# -------------------------------------------------------------------------


class TestRotationAgreement:
    """On a 2D rotation system, expm and eigen backends should agree."""

    def test_expm_eigen_agreement(self, simple_linear_system, rng):
        """Both methods should produce nearly identical predictions."""
        X, Y, dt, theta = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)

        engine_expm = PredictionEngine(model, method="expm")
        engine_eigen = PredictionEngine(model, method="eigen")

        x0 = rng.standard_normal(2)
        t_query = 0.5

        pred_expm = engine_expm.predict(x0, t_query)
        pred_eigen = engine_eigen.predict(x0, t_query)

        np.testing.assert_allclose(pred_expm, pred_eigen, atol=1e-8)

    def test_expm_eigen_agreement_trajectory(self, simple_linear_system, rng):
        """Trajectory predictions should also agree between methods."""
        X, Y, dt, theta = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)

        engine_expm = PredictionEngine(model, method="expm")
        engine_eigen = PredictionEngine(model, method="eigen")

        x0 = rng.standard_normal(2)
        times = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

        traj_expm = engine_expm.predict_trajectory(x0, times)
        traj_eigen = engine_eigen.predict_trajectory(x0, times)

        np.testing.assert_allclose(traj_expm, traj_eigen, atol=1e-8)


# -------------------------------------------------------------------------
# 2. Trajectory at integer multiples matches repeated matrix multiply
# -------------------------------------------------------------------------


class TestIntegerMultiples:
    """Predictions at t = k*dt should match K^k applied directly."""

    def test_trajectory_matches_matrix_power(self, simple_linear_system, rng):
        """predict_trajectory at t = dt, 2*dt, 3*dt should match
        repeated application of K."""
        X, Y, dt, theta = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        K = model.get_koopman_matrix()

        engine = PredictionEngine(model, method="expm")

        x0 = rng.standard_normal(2)
        times = np.array([dt, 2 * dt, 3 * dt])
        traj = engine.predict_trajectory(x0, times)

        # Manual repeated multiply
        z = model.lift(x0.reshape(1, -1))
        expected = []
        for k in range(1, 4):
            z_k = z @ np.linalg.matrix_power(K, k)
            x_k = model.unlift(z_k)
            expected.append(x_k[0])
        expected = np.array(expected)

        np.testing.assert_allclose(traj, expected, atol=1e-10)


# -------------------------------------------------------------------------
# 3. Single time prediction: correct shape
# -------------------------------------------------------------------------


class TestSingleTimePrediction:
    """predict() with a scalar t should return the right shape."""

    def test_single_x0_scalar_t(self, simple_linear_system, rng):
        """1-D x0 + scalar t -> (n_state,)."""
        X, Y, dt, _ = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        engine = PredictionEngine(model)

        x0 = rng.standard_normal(2)
        result = engine.predict(x0, 0.5)
        assert result.shape == (2,)

    def test_batch_x0_scalar_t(self, simple_linear_system, rng):
        """2-D x0 (batch) + scalar t -> (n_samples, n_state)."""
        X, Y, dt, _ = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        engine = PredictionEngine(model)

        x0_batch = rng.standard_normal((5, 2))
        result = engine.predict(x0_batch, 0.5)
        assert result.shape == (5, 2)


# -------------------------------------------------------------------------
# 4. Batch time prediction: correct shape
# -------------------------------------------------------------------------


class TestBatchTimePrediction:
    """predict() with an array of times should return the right shape."""

    def test_single_x0_array_t(self, simple_linear_system, rng):
        """1-D x0 + array t -> (len(t), n_state)."""
        X, Y, dt, _ = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        engine = PredictionEngine(model)

        x0 = rng.standard_normal(2)
        times = np.array([0.1, 0.2, 0.3, 0.5])
        result = engine.predict(x0, times)
        assert result.shape == (4, 2)


# -------------------------------------------------------------------------
# 5. Auto method selection
# -------------------------------------------------------------------------


class TestAutoMethod:
    """auto should select expm for small systems."""

    def test_auto_selects_expm_for_small(self, simple_linear_system):
        """A 2D system should use expm with auto."""
        X, Y, dt, _ = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        engine = PredictionEngine(model, method="auto")
        assert engine.method == "expm"

    def test_explicit_method_is_respected(self, simple_linear_system):
        """If we explicitly request eigen, it should be used."""
        X, Y, dt, _ = simple_linear_system
        model = _fit_rotation_model(X, Y, dt)
        engine = PredictionEngine(model, method="eigen")
        assert engine.method == "eigen"


# -------------------------------------------------------------------------
# 6. Stability warning for unstable eigenvalues
# -------------------------------------------------------------------------


class TestStabilityWarning:
    """A system with |lambda| > 1 should trigger a warning."""

    def test_unstable_eigenvalue_warns(self, rng):
        """Build a model whose K has an eigenvalue > 1, verify warning."""
        # Create an expanding system: Y = 1.5 * X (eigenvalue = 1.5)
        n_samples = 100
        dt = 0.1
        X = rng.standard_normal((n_samples, 2))
        scale = 1.5
        Y = scale * X

        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
        model.fit(X, Y, dt)

        # The Koopman matrix should have eigenvalues near 1.5
        K = model.get_koopman_matrix()
        eigs = np.abs(np.linalg.eigvals(K))
        assert np.any(eigs > EIGENVALUE_STABILITY_THRESHOLD), (
            f"Expected unstable eigenvalue, got |eigs| = {eigs}"
        )

        with pytest.warns(UserWarning, match="eigenvalue"):
            PredictionEngine(model)


# -------------------------------------------------------------------------
# 7. Not fitted error
# -------------------------------------------------------------------------


class TestNotFittedError:
    """PredictionEngine should refuse an unfitted model."""

    def test_unfitted_model_raises(self):
        """Constructing PredictionEngine with unfitted model raises NotFittedError."""
        model = EDMD(dictionary=IdentityDictionary())
        with pytest.raises(NotFittedError):
            PredictionEngine(model)
