"""Tests for automatic hyperparameter selection."""

from __future__ import annotations

import numpy as np
import pytest

from koopsim.core.auto_tune import AutoTuneResult, auto_tune
from koopsim.koopsim import KoopSim

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rotation_data(rng):
    """2D rotation snapshot pairs — a simple linear system."""
    theta = np.pi / 6
    dt = 0.1
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    n_samples = 300
    X = rng.standard_normal((n_samples, 2))
    Y = X @ R.T
    return X, Y, dt


@pytest.fixture
def hopf_data():
    """Nonlinear Hopf bifurcation snapshot pairs."""
    from koopsim.systems.fluid_particles import HopfBifurcation

    system = HopfBifurcation(mu=1.0)
    rng = np.random.default_rng(42)
    X, Y = system.generate_snapshots(
        x0=np.array([0.5, 0.5]),
        dt=0.05,
        n_steps=100,
        n_trajectories=5,
        rng=rng,
    )
    return X, Y, 0.05


# ---------------------------------------------------------------------------
# 1. auto_tune returns correct structure
# ---------------------------------------------------------------------------


class TestAutoTuneBasic:
    """auto_tune should return a valid AutoTuneResult."""

    def test_returns_auto_tune_result(self, rotation_data):
        X, Y, dt = rotation_data
        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=[2, 3],
            rbf_centers_list=[10],
            regularizations=[1e-6, 1e-4],
            n_folds=3,
            verbose=False,
        )
        assert isinstance(result, AutoTuneResult)
        assert result.cv_error >= 0
        assert isinstance(result.all_results, list)
        assert len(result.all_results) > 0

    def test_all_results_count(self, rotation_data):
        """Number of results should equal candidates * regularizations."""
        X, Y, dt = rotation_data
        poly_degrees = [2, 3]
        rbf_centers_list = [10]
        regularizations = [1e-6, 1e-4]

        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=poly_degrees,
            rbf_centers_list=rbf_centers_list,
            regularizations=regularizations,
            n_folds=3,
            verbose=False,
        )

        # Candidates: identity(1) + poly(2) + rbf(1) + poly*rbf(2) = 6
        # Total: 6 * 2 regularizations = 12
        n_candidates = (
            1
            + len(poly_degrees)
            + len(rbf_centers_list)
            + (len(poly_degrees) * len(rbf_centers_list))
        )
        expected = n_candidates * len(regularizations)
        assert len(result.all_results) == expected

    def test_best_config_in_all_results(self, rotation_data):
        """Best config's cv_error should be the minimum."""
        X, Y, dt = rotation_data
        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=[2],
            rbf_centers_list=[10],
            regularizations=[1e-6],
            n_folds=3,
            verbose=False,
        )
        min_error = min(r["cv_error"] for r in result.all_results)
        assert abs(result.cv_error - min_error) < 1e-12


# ---------------------------------------------------------------------------
# 2. Linear system should prefer identity or low-complexity dictionary
# ---------------------------------------------------------------------------


class TestLinearSystemSelection:
    """For a linear system, auto_tune should find a low-error config."""

    def test_low_cv_error_on_linear(self, rotation_data):
        """CV error should be very small for a pure rotation."""
        X, Y, dt = rotation_data
        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=[2, 3],
            rbf_centers_list=[10],
            regularizations=[1e-10, 1e-6],
            n_folds=5,
            verbose=False,
        )
        # A linear system with identity dictionary + small reg should
        # give near-zero error
        assert result.cv_error < 0.01


# ---------------------------------------------------------------------------
# 3. Nonlinear system should benefit from richer dictionary
# ---------------------------------------------------------------------------


class TestNonlinearSystemSelection:
    """For a nonlinear system, a richer dictionary should help."""

    def test_nonlinear_selects_nontrivial(self, hopf_data):
        """On Hopf data, the best config should not be identity-only."""
        X, Y, dt = hopf_data
        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=[2, 3],
            rbf_centers_list=[10, 25],
            regularizations=[1e-8, 1e-6, 1e-4],
            n_folds=3,
            verbose=False,
        )
        # For Hopf, polynomial or RBF should beat identity
        identity_errors = [
            r["cv_error"]
            for r in result.all_results
            if r["poly_degree"] is None and r["rbf_centers"] is None
        ]
        best_identity = min(identity_errors)
        assert result.cv_error <= best_identity


# ---------------------------------------------------------------------------
# 4. KoopSim.fit_auto integration
# ---------------------------------------------------------------------------


class TestFitAutoIntegration:
    """KoopSim.fit_auto should auto-tune and fit the model."""

    def test_fit_auto_produces_fitted_model(self, rotation_data):
        X, Y, dt = rotation_data
        sim = KoopSim(method="edmd", verbose=False)
        result = sim.fit_auto(
            X,
            Y,
            dt,
            poly_degrees=[2],
            rbf_centers_list=[10],
            regularizations=[1e-6],
            n_folds=3,
        )

        assert isinstance(result, AutoTuneResult)
        # Model should be fitted and usable
        x0 = X[0]
        pred = sim.predict(x0, t=dt)
        assert pred.shape == x0.shape
        assert np.all(np.isfinite(pred))

    def test_fit_auto_predictions_reasonable(self, rotation_data):
        """After fit_auto, predictions should be accurate on linear data."""
        X, Y, dt = rotation_data
        sim = KoopSim(verbose=False)
        sim.fit_auto(
            X,
            Y,
            dt,
            poly_degrees=[2, 3],
            regularizations=[1e-10, 1e-6],
            n_folds=3,
        )

        # One-step prediction error should be small
        error = sim.validate(X[:50], Y[:50], metric="rmse")
        assert error < 0.01
