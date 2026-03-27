"""Tests for Monte Carlo uncertainty quantification."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.edmd import EDMD
from koopsim.core.uncertainty import MonteCarloUQ
from koopsim.utils.dictionary import IdentityDictionary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rotation_model(simple_linear_system):
    """Fitted EDMD on 2D rotation system."""
    X, Y, dt, theta = simple_linear_system
    model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
    model.fit(X, Y, dt)
    return model, dt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonteCarloUQ:
    """Tests for MonteCarloUQ."""

    def test_returns_correct_keys(self, rotation_model):
        """Result dict should contain mean, std, percentiles, samples."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=20, noise_scale=0.01)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)

        assert "mean" in result
        assert "std" in result
        assert "percentiles" in result
        assert "samples" in result

    def test_percentile_keys(self, rotation_model):
        """Percentiles dict should have keys 5, 25, 50, 75, 95."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=20, noise_scale=0.01)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)

        expected_keys = {5, 25, 50, 75, 95}
        assert set(result["percentiles"].keys()) == expected_keys

    def test_shapes_correct(self, rotation_model):
        """All returned arrays should have correct shapes."""
        model, dt = rotation_model
        n_samples = 50
        uq = MonteCarloUQ(model, n_samples=n_samples, noise_scale=0.01)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)

        n_features = 2
        assert result["mean"].shape == (n_features,)
        assert result["std"].shape == (n_features,)
        assert result["samples"].shape == (n_samples, n_features)
        for p in (5, 25, 50, 75, 95):
            assert result["percentiles"][p].shape == (n_features,)

    def test_mean_close_to_deterministic(self, rotation_model):
        """For small noise, mean should be close to deterministic prediction."""
        from koopsim.core.prediction import PredictionEngine

        model, dt = rotation_model
        engine = PredictionEngine(model)
        x0 = np.array([1.0, 0.0])

        deterministic = engine.predict(x0, t=dt)

        uq = MonteCarloUQ(model, n_samples=200, noise_scale=0.001)
        result = uq.predict_with_uncertainty(x0, t=dt)

        np.testing.assert_allclose(
            result["mean"],
            deterministic,
            atol=0.01,
            err_msg="MC mean should be close to deterministic prediction for small noise.",
        )

    def test_std_scales_with_noise(self, rotation_model):
        """Larger noise_scale should produce larger std."""
        model, dt = rotation_model
        x0 = np.array([1.0, 0.0])

        uq_small = MonteCarloUQ(model, n_samples=200, noise_scale=0.001)
        result_small = uq_small.predict_with_uncertainty(x0, t=dt)

        uq_large = MonteCarloUQ(model, n_samples=200, noise_scale=0.1)
        result_large = uq_large.predict_with_uncertainty(x0, t=dt)

        # Std with larger noise should be larger
        assert np.all(result_large["std"] > result_small["std"]), (
            f"Larger noise should produce larger std: "
            f"small={result_small['std']}, large={result_large['std']}"
        )

    def test_zero_noise_gives_zero_std(self, rotation_model):
        """Zero noise_scale should give effectively zero std."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=50, noise_scale=0.0)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)

        np.testing.assert_allclose(result["std"], 0.0, atol=1e-14)

    def test_gaussian_noise_model(self, rotation_model):
        """Gaussian noise model should work without errors."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=20, noise_model="gaussian", noise_scale=0.01)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)
        assert result["samples"].shape[0] == 20

    def test_uniform_noise_model(self, rotation_model):
        """Uniform noise model should work without errors."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=20, noise_model="uniform", noise_scale=0.01)
        x0 = np.array([1.0, 0.0])
        result = uq.predict_with_uncertainty(x0, t=dt)
        assert result["samples"].shape[0] == 20

    def test_invalid_noise_model_raises(self, rotation_model):
        """Invalid noise model should raise ValueError."""
        model, _ = rotation_model
        with pytest.raises(ValueError, match="Unknown noise_model"):
            MonteCarloUQ(model, noise_model="poisson")

    def test_2d_x0_input(self, rotation_model):
        """Should accept (1, n_features) shaped x0."""
        model, dt = rotation_model
        uq = MonteCarloUQ(model, n_samples=10, noise_scale=0.01)
        x0 = np.array([[1.0, 0.0]])
        result = uq.predict_with_uncertainty(x0, t=dt)
        assert result["mean"].shape == (2,)
