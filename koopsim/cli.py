"""Command-line interface for KoopSim.

Provides commands to generate training data, train Koopman models,
predict future states, validate models, and inspect saved models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger("koopsim")


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------


def _load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load snapshot pairs (X, Y) from CSV, NPY, or HDF5.

    For CSV and NPY files the data is treated as a single trajectory
    (rows = time steps, columns = state dims).  X = data[:-1], Y = data[1:].

    For HDF5 files, datasets named ``"X"`` and ``"Y"`` are expected.

    Parameters
    ----------
    path : str
        File path with extension ``.csv``, ``.npy``, or ``.h5`` / ``.hdf5``.

    Returns
    -------
    X, Y : np.ndarray
        Snapshot pair arrays.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        data = np.loadtxt(p, delimiter=",")
        X = data[:-1]
        Y = data[1:]
    elif ext == ".npy":
        loaded = np.load(p, allow_pickle=True)
        # If saved as a dict-like .npy (from np.save on a dict-wrapped array)
        if loaded.ndim == 0:
            d = loaded.item()
            X = d["X"]
            Y = d["Y"]
        else:
            # Plain trajectory array
            X = loaded[:-1]
            Y = loaded[1:]
    elif ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(p, "r") as f:
            X = np.array(f["X"])
            Y = np.array(f["Y"])
    else:
        raise click.BadParameter(f"Unsupported data file extension: {ext}")

    return X, Y


def _save_data(path: str, X: np.ndarray, Y: np.ndarray) -> None:
    """Save snapshot pairs (X, Y) to CSV, NPY, or HDF5.

    For CSV, the data is saved as vertically stacked ``[X; Y[-1:]]`` so the
    full trajectory can be reconstructed.  A comment header notes the format.

    For NPY, a dictionary ``{"X": X, "Y": Y}`` is saved.

    For HDF5, datasets ``"X"`` and ``"Y"`` are created.

    Parameters
    ----------
    path : str
        Destination file path.
    X, Y : np.ndarray
        Snapshot pair arrays.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        # Reconstruct trajectory: X rows, then the last Y row
        trajectory = np.vstack([X, Y[-1:]])
        header = "KoopSim trajectory data (rows=timesteps, cols=state dims)"
        np.savetxt(p, trajectory, delimiter=",", header=header)
    elif ext == ".npy":
        np.save(p, {"X": X, "Y": Y})
    elif ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(p, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("Y", data=Y)
    else:
        raise click.BadParameter(f"Unsupported output file extension: {ext}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose/--quiet", default=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose):
    """KoopSim -- Koopman Operator Simulation Toolkit."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s -- %(levelname)s -- %(message)s")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True),
    help="Training data file (CSV, NPY, or HDF5)",
)
@click.option("--method", type=click.Choice(["edmd", "neural"]), default="edmd")
@click.option("--dt", required=True, type=float, help="Time step between snapshots")
@click.option("--output", "-o", default="model.koop", help="Output model file")
@click.option("--poly-degree", type=int, default=None, help="Polynomial dictionary degree")
@click.option("--rbf-centers", type=int, default=None, help="Number of RBF centers")
@click.option("--regularization", type=float, default=1e-6)
@click.option("--svd-rank", type=int, default=None)
@click.pass_context
def train(ctx, data, method, dt, output, poly_degree, rbf_centers, regularization, svd_rank):
    """Train a Koopman model on snapshot data."""
    from koopsim import KoopSim

    X, Y = _load_data(data)

    sim = KoopSim(
        method=method,
        poly_degree=poly_degree,
        rbf_centers=rbf_centers,
        regularization=regularization,
        svd_rank=svd_rank,
        verbose=ctx.obj["verbose"],
    )
    sim.fit(X, Y, dt)
    sim.save(output)
    click.echo(f"Model saved to {output}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Model .koop file")
@click.option("--initial-state", required=True, type=str, help="Comma-separated initial state")
@click.option("--time", "-t", required=True, type=float, help="Prediction time")
@click.pass_context
def predict(ctx, model, initial_state, time):
    """Predict system state at a future time."""
    from koopsim import KoopSim

    sim = KoopSim.load(model)
    x0 = np.array([float(x) for x in initial_state.split(",")])
    result = sim.predict(x0, time)
    click.echo(f"State at t={time}: {result}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--test-data", required=True, type=click.Path(exists=True))
@click.option("--dt", required=True, type=float)
@click.option("--metric", type=click.Choice(["rmse", "mae", "relative"]), default="rmse")
@click.pass_context
def validate(ctx, model, test_data, dt, metric):
    """Validate model against test data."""
    from koopsim import KoopSim

    sim = KoopSim.load(model)
    X_test, Y_test = _load_data(test_data)
    error = sim.validate(X_test, Y_test, metric=metric)
    click.echo(f"Validation {metric}: {error:.6f}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--system",
    required=True,
    type=click.Choice(
        [
            "hopf",
            "double-gyre",
            "spring-mass",
            "rlc",
            "vanderpol",
            "beam",
            "point-vortex",
            "lorenz",
            "lotka-volterra",
        ]
    ),
)
@click.option("--output", "-o", required=True, help="Output data file (.csv, .npy, or .h5)")
@click.option("--dt", type=float, default=0.01)
@click.option("--n-steps", type=int, default=200)
@click.option("--n-trajectories", type=int, default=10)
@click.option("--noise-std", type=float, default=0.0)
@click.pass_context
def generate(ctx, system, output, dt, n_steps, n_trajectories, noise_std):
    """Generate training data from a built-in system."""
    from koopsim.systems import (
        DoubleGyre,
        EulerBernoulliBeam,
        HopfBifurcation,
        LorenzAttractor,
        LotkaVolterra,
        PointVortexSystem,
        RLCCircuit,
        SpringMassDamper,
        VanDerPolOscillator,
    )

    system_map = {
        "hopf": HopfBifurcation(),
        "double-gyre": DoubleGyre(),
        "spring-mass": SpringMassDamper(),
        "rlc": RLCCircuit(),
        "vanderpol": VanDerPolOscillator(),
        "beam": EulerBernoulliBeam(),
        "point-vortex": PointVortexSystem(n_vortices=3),
        "lorenz": LorenzAttractor(),
        "lotka-volterra": LotkaVolterra(),
    }

    sys_instance = system_map[system]
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(sys_instance.state_dim) * 0.5
    X, Y = sys_instance.generate_snapshots(x0, dt, n_steps, n_trajectories, noise_std)

    _save_data(output, X, Y)
    click.echo(f"Generated {X.shape[0]} snapshot pairs -> {output}")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, type=click.Path(exists=True))
@click.pass_context
def info(ctx, model):
    """Display information about a saved model."""
    from koopsim import KoopSim

    sim = KoopSim.load(model)
    meta = sim.model.metadata()
    for k, v in meta.items():
        click.echo(f"  {k}: {v}")
    # Spectral summary
    spec = sim.spectral_analysis()
    click.echo(f"  n_eigenvalues: {len(spec['eigenvalues'])}")
    click.echo(f"  max |lambda|: {np.max(np.abs(spec['eigenvalues'])):.6f}")
