"""Tests for model validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.edmd import EDMD
from koopsim.core.validation import ModelValidator
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_linear_model(simple_linear_system):
    """EDMD perfectly fitted on a 2D rotation system."""
    X, Y, dt, theta = simple_linear_system
    model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
    model.fit(X, Y, dt)
    return model, X, Y, dt, theta


def _rotation_trajectory(theta, n_steps, x0, dt):
    """Generate a trajectory from the 2D rotation system.

    Returns array of shape (n_steps + 1, 2) starting from x0.
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    traj = np.empty((n_steps + 1, 2))
    traj[0] = x0
    for i in range(n_steps):
        traj[i + 1] = traj[i] @ R.T
    return traj


# ---------------------------------------------------------------------------
# prediction_error
# ---------------------------------------------------------------------------


class TestPredictionError:
    """Tests for ModelValidator.prediction_error."""

    def test_rmse_near_zero_for_perfect_model(self, perfect_linear_model):
        """RMSE should be near zero for a perfect linear model."""
        model, X, Y, dt, _ = perfect_linear_model
        rmse = ModelValidator.prediction_error(model, X, Y, metric="rmse")
        assert rmse < 1e-8, f"RMSE too large for perfect model: {rmse}"

    def test_mae_near_zero_for_perfect_model(self, perfect_linear_model):
        """MAE should be near zero for a perfect linear model."""
        model, X, Y, dt, _ = perfect_linear_model
        mae = ModelValidator.prediction_error(model, X, Y, metric="mae")
        assert mae < 1e-8, f"MAE too large for perfect model: {mae}"

    def test_relative_near_zero_for_perfect_model(self, perfect_linear_model):
        """Relative error should be near zero for a perfect linear model."""
        model, X, Y, dt, _ = perfect_linear_model
        rel = ModelValidator.prediction_error(model, X, Y, metric="relative")
        assert rel < 1e-8, f"Relative error too large for perfect model: {rel}"

    def test_unknown_metric_raises(self, perfect_linear_model):
        """Unknown metric should raise ValueError."""
        model, X, Y, _, _ = perfect_linear_model
        with pytest.raises(ValueError, match="Unknown metric"):
            ModelValidator.prediction_error(model, X, Y, metric="r2")

    def test_noisy_model_has_positive_error(self, rng):
        """A model fitted on noisy data should have positive error on clean data."""
        theta = np.pi / 6
        dt = 0.1
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        X = rng.standard_normal((100, 2))
        Y_clean = X @ R.T
        Y_noisy = Y_clean + rng.normal(0, 0.5, Y_clean.shape)

        model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
        model.fit(X, Y_noisy, dt)

        rmse = ModelValidator.prediction_error(model, X, Y_clean, metric="rmse")
        assert rmse > 0.01, "Error should be positive for noisy model"


# ---------------------------------------------------------------------------
# multi_step_error
# ---------------------------------------------------------------------------


class TestMultiStepError:
    """Tests for ModelValidator.multi_step_error."""

    def test_returns_correct_length(self, perfect_linear_model):
        """multi_step_error should return an array of length n_steps."""
        model, _, _, dt, theta = perfect_linear_model
        n_steps = 10
        x0 = np.array([1.0, 0.0])
        trajectory = _rotation_trajectory(theta, n_steps, x0, dt)

        errors = ModelValidator.multi_step_error(model, trajectory, dt, n_steps)
        assert errors.shape == (n_steps,)

    def test_errors_small_for_perfect_model(self, perfect_linear_model):
        """Errors should be small for a perfect linear model."""
        model, _, _, dt, theta = perfect_linear_model
        n_steps = 5
        x0 = np.array([1.0, 0.0])
        trajectory = _rotation_trajectory(theta, n_steps, x0, dt)

        errors = ModelValidator.multi_step_error(model, trajectory, dt, n_steps)
        # For a perfect linear model, multi-step errors should be small
        assert np.all(errors < 0.1), f"Multi-step errors too large: {errors}"

    def test_single_step(self, perfect_linear_model):
        """Single step error should match prediction_error."""
        model, _, _, dt, theta = perfect_linear_model
        x0 = np.array([1.0, 0.0])
        trajectory = _rotation_trajectory(theta, 1, x0, dt)

        errors = ModelValidator.multi_step_error(model, trajectory, dt, 1)
        assert errors.shape == (1,)


# ---------------------------------------------------------------------------
# spectral_analysis
# ---------------------------------------------------------------------------


class TestSpectralAnalysis:
    """Tests for ModelValidator.spectral_analysis."""

    def test_eigenvalue_count(self, perfect_linear_model):
        """Number of eigenvalues should match n_koopman_dims."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        assert len(result["eigenvalues"]) == model.n_koopman_dims

    def test_frequencies_are_real(self, perfect_linear_model):
        """Frequencies should be real-valued."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        assert np.isrealobj(result["frequencies"])

    def test_growth_rates_are_real(self, perfect_linear_model):
        """Growth rates should be real-valued."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        assert np.isrealobj(result["growth_rates"])

    def test_dominant_mode_indices(self, perfect_linear_model):
        """Dominant mode indices should be a valid permutation."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        indices = result["dominant_mode_indices"]
        n = model.n_koopman_dims
        assert len(indices) == n
        assert set(indices) == set(range(n))

    def test_rotation_eigenvalues_unit_magnitude(self, perfect_linear_model):
        """For a pure rotation, eigenvalue magnitudes should be ~1."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        magnitudes = np.abs(result["eigenvalues"])
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-8)

    def test_rotation_growth_rates_near_zero(self, perfect_linear_model):
        """For a pure rotation, growth rates should be ~0 (no growth/decay)."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        np.testing.assert_allclose(result["growth_rates"], 0.0, atol=1e-6)

    def test_result_keys(self, perfect_linear_model):
        """Result should have the expected keys."""
        model, _, _, _, _ = perfect_linear_model
        result = ModelValidator.spectral_analysis(model)
        expected_keys = {"eigenvalues", "frequencies", "growth_rates", "dominant_mode_indices"}
        assert set(result.keys()) == expected_keys
