"""Smoke tests for the visualization module.

Verifies that figures are created without errors — no visual assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if matplotlib not available
pytest.importorskip("matplotlib")

from koopsim.utils.visualization import (
    animate_trajectory,
    plot_eigenspectrum,
    plot_particle_field,
    plot_phase_portrait,
    plot_prediction_error,
    plot_trajectory_comparison,
    plot_uncertainty_band,
    plot_vector_field,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def times():
    return np.linspace(0, 1, 50)


@pytest.fixture
def trajectory_2d(rng, times):
    """Simple 2D trajectory."""
    t = times
    x = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    return x


class TestPlotTrajectoryComparison:
    def test_creates_matplotlib_figure(self, times, trajectory_2d, rng):
        import matplotlib.figure

        predicted = trajectory_2d + rng.normal(0, 0.05, trajectory_2d.shape)
        fig = plot_trajectory_comparison(times, trajectory_2d, predicted)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_labels(self, times, trajectory_2d, rng):
        import matplotlib.figure

        predicted = trajectory_2d + rng.normal(0, 0.05, trajectory_2d.shape)
        fig = plot_trajectory_comparison(times, trajectory_2d, predicted, labels=["x", "y"])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_1d_trajectories(self, times):
        import matplotlib.figure

        true = np.sin(times)
        predicted = np.sin(times) + 0.01
        fig = plot_trajectory_comparison(times, true, predicted)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotPhasePortrait:
    def test_single_trajectory(self, trajectory_2d):
        import matplotlib.figure

        fig = plot_phase_portrait(trajectory_2d)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_multiple_trajectories(self, trajectory_2d, rng):
        import matplotlib.figure

        traj2 = trajectory_2d * 0.5 + rng.normal(0, 0.01, trajectory_2d.shape)
        fig = plot_phase_portrait([trajectory_2d, traj2])
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotEigenspectrum:
    def test_with_fitted_edmd(self, simple_linear_system):
        import matplotlib.figure

        from koopsim.core.edmd import EDMD

        X, Y, dt, _ = simple_linear_system
        model = EDMD()
        model.fit(X, Y, dt)

        fig = plot_eigenspectrum(model)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotParticleField:
    def test_positions_only(self, rng):
        import matplotlib.figure

        positions = rng.standard_normal((20, 2))
        fig = plot_particle_field(positions)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_velocities(self, rng):
        import matplotlib.figure

        positions = rng.standard_normal((20, 2))
        velocities = rng.standard_normal((20, 2)) * 0.1
        fig = plot_particle_field(positions, velocities=velocities)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_trails(self, rng):
        import matplotlib.figure

        n_particles = 10
        n_times = 15
        positions = rng.standard_normal((n_particles, 2))
        trails = rng.standard_normal((n_times, n_particles, 2))
        fig = plot_particle_field(positions, trails=trails)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotVectorField:
    def test_simple_grid(self):
        import matplotlib.figure

        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        gx, gy = np.meshgrid(x, y)
        u = -gy  # Simple rotation field
        v = gx
        fig = plot_vector_field(gx, gy, u, v)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotPredictionError:
    def test_basic_usage(self):
        import matplotlib.figure

        steps = np.arange(1, 51)
        errors = np.exp(-0.1 * steps) + 0.01
        fig = plot_prediction_error(steps, errors)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotUncertaintyBand:
    def test_without_true(self, times):
        import matplotlib.figure

        mean = np.sin(2 * np.pi * times)
        std = np.ones_like(times) * 0.1
        fig = plot_uncertainty_band(times, mean, std)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_true(self, times):
        import matplotlib.figure

        mean = np.sin(2 * np.pi * times)
        std = np.ones_like(times) * 0.1
        true = np.sin(2 * np.pi * times) + 0.05
        fig = plot_uncertainty_band(times, mean, std, true=true)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_multi_dim(self, times):
        import matplotlib.figure

        mean = np.column_stack([np.sin(times), np.cos(times)])
        std = np.ones_like(mean) * 0.1
        fig = plot_uncertainty_band(times, mean, std)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestAnimateTrajectory:
    def test_creates_gif(self, trajectory_2d, tmp_path):
        import os

        output = str(tmp_path / "test_anim.gif")
        result = animate_trajectory(trajectory_2d, dt=0.02, output_path=output, fps=10)
        assert result == output
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 0


class TestPlotlyBackend:
    @pytest.fixture(autouse=True)
    def _skip_if_no_plotly(self):
        pytest.importorskip("plotly")

    def test_trajectory_comparison(self, times, trajectory_2d, rng):
        import plotly.graph_objects as go

        predicted = trajectory_2d + rng.normal(0, 0.05, trajectory_2d.shape)
        fig = plot_trajectory_comparison(times, trajectory_2d, predicted, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_phase_portrait(self, trajectory_2d):
        import plotly.graph_objects as go

        fig = plot_phase_portrait(trajectory_2d, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_eigenspectrum(self, simple_linear_system):
        import plotly.graph_objects as go

        from koopsim.core.edmd import EDMD

        X, Y, dt, _ = simple_linear_system
        model = EDMD()
        model.fit(X, Y, dt)
        fig = plot_eigenspectrum(model, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_prediction_error(self):
        import plotly.graph_objects as go

        steps = np.arange(1, 51)
        errors = np.exp(-0.1 * steps)
        fig = plot_prediction_error(steps, errors, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_uncertainty_band(self, times):
        import plotly.graph_objects as go

        mean = np.sin(times)
        std = np.ones_like(times) * 0.1
        fig = plot_uncertainty_band(times, mean, std, backend="plotly")
        assert isinstance(fig, go.Figure)
