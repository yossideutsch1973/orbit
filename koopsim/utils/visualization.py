"""Visualization utilities for KoopSim.

All functions return figure objects and support both matplotlib and plotly backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from koopsim.core.base import KoopmanModel

logger = logging.getLogger("koopsim")


def _get_matplotlib():
    """Lazily import matplotlib with non-interactive backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for safety
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install koopsim[viz]"
        )


def _get_plotly():
    """Lazily import plotly graph_objects."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "plotly is required for visualization. "
            "Install with: pip install koopsim[viz]"
        )


def plot_trajectory_comparison(
    times: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    labels: list[str] | None = None,
    backend: str = "matplotlib",
):
    """Plot true vs predicted trajectories for each state dimension.

    Parameters
    ----------
    times : np.ndarray, shape (n_times,)
        Time values for the x-axis.
    true : np.ndarray, shape (n_times, n_dims)
        Ground truth trajectory.
    predicted : np.ndarray, shape (n_times, n_dims)
        Predicted trajectory.
    labels : list[str] | None
        Labels for each state dimension. If None, uses "dim 0", "dim 1", etc.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    times = np.asarray(times)
    true = np.asarray(true)
    predicted = np.asarray(predicted)

    if true.ndim == 1:
        true = true[:, np.newaxis]
    if predicted.ndim == 1:
        predicted = predicted[:, np.newaxis]

    n_dims = true.shape[1]
    if labels is None:
        labels = [f"dim {i}" for i in range(n_dims)]

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for i in range(n_dims):
            fig.add_trace(go.Scatter(
                x=times, y=true[:, i],
                mode="lines", name=f"{labels[i]} (true)",
                line=dict(dash="solid"),
            ))
            fig.add_trace(go.Scatter(
                x=times, y=predicted[:, i],
                mode="lines", name=f"{labels[i]} (predicted)",
                line=dict(dash="dash"),
            ))
        fig.update_layout(
            title="Trajectory Comparison",
            xaxis_title="Time",
            yaxis_title="State",
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), squeeze=False)
    for i in range(n_dims):
        ax = axes[i, 0]
        ax.plot(times, true[:, i], label="True", linewidth=1.5)
        ax.plot(times, predicted[:, i], "--", label="Predicted", linewidth=1.5)
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time")
    fig.suptitle("Trajectory Comparison")
    plt.tight_layout()
    return fig


def plot_phase_portrait(
    trajectories: list[np.ndarray] | np.ndarray,
    dims: tuple[int, int] = (0, 1),
    backend: str = "matplotlib",
):
    """Plot phase portrait of trajectories in 2D.

    Parameters
    ----------
    trajectories : np.ndarray or list[np.ndarray]
        Single trajectory of shape (n_times, n_dims) or list of such arrays.
    dims : tuple[int, int]
        Which two dimensions to plot.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 2:
        trajectories = [trajectories]

    d0, d1 = dims

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for idx, traj in enumerate(trajectories):
            traj = np.asarray(traj)
            fig.add_trace(go.Scatter(
                x=traj[:, d0], y=traj[:, d1],
                mode="lines", name=f"Trajectory {idx}",
            ))
            # Mark start
            fig.add_trace(go.Scatter(
                x=[traj[0, d0]], y=[traj[0, d1]],
                mode="markers", marker=dict(size=8, symbol="circle"),
                showlegend=False,
            ))
        fig.update_layout(
            title="Phase Portrait",
            xaxis_title=f"Dim {d0}",
            yaxis_title=f"Dim {d1}",
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        ax.plot(traj[:, d0], traj[:, d1], label=f"Trajectory {idx}", linewidth=1.2)
        ax.plot(traj[0, d0], traj[0, d1], "o", markersize=6)
    ax.set_xlabel(f"Dim {d0}")
    ax.set_ylabel(f"Dim {d1}")
    ax.set_title("Phase Portrait")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_eigenspectrum(
    model: KoopmanModel,
    backend: str = "matplotlib",
):
    """Plot eigenvalues of Koopman matrix on the complex plane with unit circle.

    Parameters
    ----------
    model : KoopmanModel
        A fitted KoopmanModel instance.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    K = model.get_koopman_matrix()
    eigenvalues = np.linalg.eigvals(K)
    magnitudes = np.abs(eigenvalues)

    # Unit circle points
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        # Unit circle
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode="lines", name="Unit circle",
            line=dict(color="gray", dash="dash"),
        ))
        # Eigenvalues colored by magnitude
        fig.add_trace(go.Scatter(
            x=eigenvalues.real, y=eigenvalues.imag,
            mode="markers", name="Eigenvalues",
            marker=dict(
                size=8,
                color=magnitudes,
                colorscale="RdYlBu_r",
                colorbar=dict(title="|λ|"),
            ),
        ))
        fig.update_layout(
            title="Koopman Eigenspectrum",
            xaxis_title="Re(λ)",
            yaxis_title="Im(λ)",
            xaxis=dict(scaleanchor="y"),
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(circle_x, circle_y, "k--", alpha=0.4, label="Unit circle")
    scatter = ax.scatter(
        eigenvalues.real, eigenvalues.imag,
        c=magnitudes, cmap="RdYlBu_r", s=50, edgecolors="k", linewidths=0.5,
        zorder=5,
    )
    fig.colorbar(scatter, ax=ax, label="|λ|")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Koopman Eigenspectrum")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_particle_field(
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
    trails: np.ndarray | None = None,
    backend: str = "matplotlib",
):
    """Plot particle positions with optional velocity arrows and trails.

    Parameters
    ----------
    positions : np.ndarray, shape (n_particles, 2)
        Current particle positions.
    velocities : np.ndarray | None, shape (n_particles, 2)
        Optional velocity vectors for quiver arrows.
    trails : np.ndarray | None, shape (n_times, n_particles, 2)
        Optional historical positions for drawing particle trails.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    positions = np.asarray(positions)

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        # Trails
        if trails is not None:
            trails = np.asarray(trails)
            n_particles = trails.shape[1]
            for p in range(n_particles):
                fig.add_trace(go.Scatter(
                    x=trails[:, p, 0], y=trails[:, p, 1],
                    mode="lines", line=dict(width=1, color="lightgray"),
                    showlegend=False,
                ))
        # Positions
        fig.add_trace(go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode="markers", name="Particles",
            marker=dict(size=6),
        ))
        fig.update_layout(
            title="Particle Field",
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(scaleanchor="y"),
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))
    # Trails
    if trails is not None:
        trails = np.asarray(trails)
        n_particles = trails.shape[1]
        for p in range(n_particles):
            ax.plot(
                trails[:, p, 0], trails[:, p, 1],
                color="lightgray", linewidth=0.8, zorder=1,
            )
    # Positions
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=20, zorder=3, label="Particles",
    )
    # Velocity arrows
    if velocities is not None:
        velocities = np.asarray(velocities)
        ax.quiver(
            positions[:, 0], positions[:, 1],
            velocities[:, 0], velocities[:, 1],
            angles="xy", scale_units="xy", scale=1, alpha=0.6, zorder=2,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Particle Field")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_vector_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    backend: str = "matplotlib",
):
    """Plot vector field as quiver plot.

    Parameters
    ----------
    grid_x : np.ndarray
        X coordinates of the grid (meshgrid format).
    grid_y : np.ndarray
        Y coordinates of the grid (meshgrid format).
    u : np.ndarray
        X-component of the velocity field on the grid.
    v : np.ndarray
        Y-component of the velocity field on the grid.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    u = np.asarray(u)
    v = np.asarray(v)

    if backend == "plotly":
        go = _get_plotly()
        # Flatten for plotly quiver-like representation using cones (2D scatter + annotations)
        xf = grid_x.ravel()
        yf = grid_y.ravel()
        uf = u.ravel()
        vf = v.ravel()
        magnitude = np.sqrt(uf ** 2 + vf ** 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xf, y=yf,
            mode="markers",
            marker=dict(size=3, color=magnitude, colorscale="Viridis",
                        colorbar=dict(title="Magnitude")),
            name="Grid points",
        ))
        # Approximate quiver with line segments
        scale = 0.3 * (np.max(xf) - np.min(xf)) / max(np.max(magnitude), 1e-15)
        x_end = xf + uf * scale
        y_end = yf + vf * scale
        for i in range(len(xf)):
            fig.add_trace(go.Scatter(
                x=[xf[i], x_end[i]], y=[yf[i], y_end[i]],
                mode="lines", line=dict(color="steelblue", width=1),
                showlegend=False,
            ))
        fig.update_layout(
            title="Vector Field",
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(scaleanchor="y"),
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))
    magnitude = np.sqrt(u ** 2 + v ** 2)
    q = ax.quiver(grid_x, grid_y, u, v, magnitude, cmap="viridis", alpha=0.8)
    fig.colorbar(q, ax=ax, label="Magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector Field")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction_error(
    steps: np.ndarray,
    errors: np.ndarray,
    backend: str = "matplotlib",
):
    """Plot prediction error vs number of steps.

    Parameters
    ----------
    steps : np.ndarray, shape (n_steps,)
        Step numbers or time values.
    errors : np.ndarray, shape (n_steps,)
        Error values at each step.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    steps = np.asarray(steps)
    errors = np.asarray(errors)

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=errors,
            mode="lines+markers", name="Error",
        ))
        fig.update_layout(
            title="Prediction Error",
            xaxis_title="Steps",
            yaxis_title="Error",
            yaxis_type="log",
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(steps, errors, "o-", markersize=4, linewidth=1.5)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Error")
    ax.set_title("Prediction Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_uncertainty_band(
    times: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    true: np.ndarray | None = None,
    backend: str = "matplotlib",
):
    """Plot prediction mean with uncertainty bands (plus/minus 1 and 2 sigma).

    Parameters
    ----------
    times : np.ndarray, shape (n_times,)
        Time values.
    mean : np.ndarray, shape (n_times,) or (n_times, n_dims)
        Predicted mean.
    std : np.ndarray
        Same shape as mean. Standard deviation.
    true : np.ndarray | None
        Optional ground truth, same shape as mean.
    backend : str
        ``"matplotlib"`` (default) or ``"plotly"``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    times = np.asarray(times)
    mean = np.asarray(mean)
    std = np.asarray(std)

    # Handle 1-D inputs
    if mean.ndim == 1:
        mean = mean[:, np.newaxis]
    if std.ndim == 1:
        std = std[:, np.newaxis]
    if true is not None:
        true = np.asarray(true)
        if true.ndim == 1:
            true = true[:, np.newaxis]

    n_dims = mean.shape[1]

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        colors = ["blue", "red", "green", "orange", "purple"]
        for i in range(n_dims):
            color = colors[i % len(colors)]
            # 2-sigma band
            fig.add_trace(go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mean[:, i] + 2 * std[:, i],
                                  (mean[:, i] - 2 * std[:, i])[::-1]]),
                fill="toself", fillcolor=f"rgba(0,100,200,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name=f"dim {i} ±2σ",
            ))
            # 1-sigma band
            fig.add_trace(go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([mean[:, i] + std[:, i],
                                  (mean[:, i] - std[:, i])[::-1]]),
                fill="toself", fillcolor=f"rgba(0,100,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name=f"dim {i} ±1σ",
            ))
            # Mean
            fig.add_trace(go.Scatter(
                x=times, y=mean[:, i],
                mode="lines", name=f"dim {i} (mean)",
            ))
            # True
            if true is not None:
                fig.add_trace(go.Scatter(
                    x=times, y=true[:, i],
                    mode="lines", name=f"dim {i} (true)",
                    line=dict(dash="dash"),
                ))
        fig.update_layout(
            title="Prediction with Uncertainty",
            xaxis_title="Time",
            yaxis_title="State",
        )
        return fig

    # matplotlib
    plt = _get_matplotlib()
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), squeeze=False)
    for i in range(n_dims):
        ax = axes[i, 0]
        ax.fill_between(
            times, mean[:, i] - 2 * std[:, i], mean[:, i] + 2 * std[:, i],
            alpha=0.15, color="steelblue", label="±2σ",
        )
        ax.fill_between(
            times, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
            alpha=0.3, color="steelblue", label="±1σ",
        )
        ax.plot(times, mean[:, i], linewidth=1.5, color="steelblue", label="Mean")
        if true is not None:
            ax.plot(times, true[:, i], "--", linewidth=1.5, color="darkorange", label="True")
        ax.set_ylabel(f"Dim {i}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time")
    fig.suptitle("Prediction with Uncertainty")
    plt.tight_layout()
    return fig


def animate_trajectory(
    trajectory: np.ndarray,
    dt: float,
    output_path: str,
    fps: int = 30,
    dims: tuple[int, int] = (0, 1),
) -> str:
    """Create animated GIF/MP4 of trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_times, n_dims)
        Trajectory data.
    dt : float
        Time step between frames.
    output_path : str
        Path for the output file (e.g. ``"trajectory.gif"``).
    fps : int
        Frames per second.
    dims : tuple[int, int]
        Which two dimensions to plot.

    Returns
    -------
    str
        The output file path.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for animation. "
                "Install with: pip install imageio"
            )

    plt = _get_matplotlib()
    trajectory = np.asarray(trajectory)
    d0, d1 = dims
    n_frames = len(trajectory)

    # Compute axis limits from full trajectory with padding
    x_min, x_max = trajectory[:, d0].min(), trajectory[:, d0].max()
    y_min, y_max = trajectory[:, d1].min(), trajectory[:, d1].max()
    x_pad = (x_max - x_min) * 0.1 + 1e-6
    y_pad = (y_max - y_min) * 0.1 + 1e-6

    # Subsample if too many frames (target ~200 frames max)
    max_frames = 200
    if n_frames > max_frames:
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(n_frames)

    images: list[np.ndarray] = []
    for frame_idx in indices:
        fig, ax = plt.subplots(figsize=(6, 6))
        # Trail up to current frame
        trail_end = frame_idx + 1
        ax.plot(
            trajectory[:trail_end, d0], trajectory[:trail_end, d1],
            color="steelblue", linewidth=1, alpha=0.6,
        )
        # Current position
        ax.plot(
            trajectory[frame_idx, d0], trajectory[frame_idx, d1],
            "o", color="red", markersize=8, zorder=5,
        )
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(f"Dim {d0}")
        ax.set_ylabel(f"Dim {d1}")
        ax.set_title(f"t = {frame_idx * dt:.3f}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        images.append(image)
        plt.close(fig)

    # Write output
    duration = 1000.0 / fps  # milliseconds per frame
    imageio.mimwrite(output_path, images, duration=duration)
    logger.info("Animation saved to %s (%d frames at %d fps).", output_path, len(images), fps)
    return output_path
