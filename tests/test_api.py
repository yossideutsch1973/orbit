"""Tests for the KoopSim public Python API (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim import KoopSim
from koopsim.core.exceptions import NotFittedError
from koopsim.systems import HopfBifurcation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_rotation_data(rng):
    """2D rotation snapshot pairs for quick EDMD tests."""
    theta = np.pi / 6
    dt = 0.1
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    n_samples = 200
    X = rng.standard_normal((n_samples, 2))
    Y = X @ R.T
    return X, Y, dt


@pytest.fixture
def hopf_data():
    """Snapshot data from HopfBifurcation system."""
    system = HopfBifurcation(mu=1.0)
    x0 = np.array([0.5, 0.5])
    X, Y = system.generate_snapshots(x0, dt=0.01, n_steps=100, n_trajectories=5)
    return X, Y, 0.01


# ---------------------------------------------------------------------------
# 1. Basic EDMD workflow
# ---------------------------------------------------------------------------


class TestBasicEDMDWorkflow:
    def test_fit_predict_shape_and_finite(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)
        x0 = X[0]
        result = sim.predict(x0, t=0.1)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_fit_returns_self(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        returned = sim.fit(X, Y, dt)
        assert returned is sim


# ---------------------------------------------------------------------------
# 2. Polynomial dictionary
# ---------------------------------------------------------------------------


class TestPolynomialDictionary:
    def test_poly_predict_on_hopf(self, hopf_data):
        X, Y, dt = hopf_data
        sim = KoopSim(method="edmd", poly_degree=3)
        sim.fit(X, Y, dt)
        x0 = X[0]
        result = sim.predict(x0, t=0.05)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# 3. RBF dictionary
# ---------------------------------------------------------------------------


class TestRBFDictionary:
    def test_rbf_predict(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd", rbf_centers=20)
        sim.fit(X, Y, dt)
        x0 = X[0]
        result = sim.predict(x0, t=0.1)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# 4. predict_trajectory
# ---------------------------------------------------------------------------


class TestPredictTrajectory:
    def test_trajectory_shape(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)
        times = np.linspace(0, 1.0, 50)
        x0 = X[0]
        traj = sim.predict_trajectory(x0, times)
        assert traj.shape == (50, 2)
        assert np.all(np.isfinite(traj))


# ---------------------------------------------------------------------------
# 5. predict_with_uncertainty
# ---------------------------------------------------------------------------


class TestPredictWithUncertainty:
    def test_returns_correct_keys_and_shapes(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)
        x0 = X[0]
        result = sim.predict_with_uncertainty(x0, t=0.1, n_samples=50)
        assert "mean" in result
        assert "std" in result
        assert "percentiles" in result
        assert "samples" in result
        assert result["mean"].shape == (2,)
        assert result["std"].shape == (2,)
        assert result["samples"].shape == (50, 2)
        for p in (5, 25, 50, 75, 95):
            assert p in result["percentiles"]
            assert result["percentiles"][p].shape == (2,)


# ---------------------------------------------------------------------------
# 6. validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_returns_float(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)
        error = sim.validate(X, Y, metric="rmse")
        assert isinstance(error, float)
        assert np.isfinite(error)
        assert error >= 0.0


# ---------------------------------------------------------------------------
# 7. spectral_analysis
# ---------------------------------------------------------------------------


class TestSpectralAnalysis:
    def test_returns_dict_with_eigenvalues(self, simple_rotation_data):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)
        analysis = sim.spectral_analysis()
        assert "eigenvalues" in analysis
        assert "frequencies" in analysis
        assert "growth_rates" in analysis
        assert "dominant_mode_indices" in analysis
        assert len(analysis["eigenvalues"]) == sim.model.n_koopman_dims


# ---------------------------------------------------------------------------
# 8. save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_predict_same_result(self, simple_rotation_data, tmp_path):
        X, Y, dt = simple_rotation_data
        sim = KoopSim(method="edmd")
        sim.fit(X, Y, dt)

        x0 = X[0]
        pred_before = sim.predict(x0, t=0.3)

        save_path = str(tmp_path / "model.koop")
        sim.save(save_path)

        sim_loaded = KoopSim.load(save_path)
        pred_after = sim_loaded.predict(x0, t=0.3)

        np.testing.assert_allclose(pred_before, pred_after, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. from_system convenience
# ---------------------------------------------------------------------------


class TestFromSystem:
    def test_from_system_hopf(self):
        system = HopfBifurcation(mu=1.0)
        sim = KoopSim.from_system(system, dt=0.01, n_steps=50, n_trajectories=5)
        x0 = np.array([0.3, 0.3])
        result = sim.predict(x0, t=0.05)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# 10. Not fitted errors
# ---------------------------------------------------------------------------


class TestNotFittedErrors:
    def test_predict_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        with pytest.raises(NotFittedError):
            sim.predict(np.array([1.0, 2.0]), t=0.1)

    def test_predict_trajectory_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        with pytest.raises(NotFittedError):
            sim.predict_trajectory(np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    def test_predict_with_uncertainty_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        with pytest.raises(NotFittedError):
            sim.predict_with_uncertainty(np.array([1.0, 2.0]), t=0.1)

    def test_validate_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        X = np.array([[1.0, 2.0]])
        with pytest.raises(NotFittedError):
            sim.validate(X, X)

    def test_spectral_analysis_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        with pytest.raises(NotFittedError):
            sim.spectral_analysis()

    def test_save_before_fit_raises(self):
        sim = KoopSim(method="edmd")
        with pytest.raises(NotFittedError):
            sim.save("/tmp/should_not_exist.koop")
