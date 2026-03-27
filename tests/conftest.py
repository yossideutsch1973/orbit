"""Shared test fixtures for KoopSim."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_linear_system(rng):
    """2D rotation system with snapshot pairs.

    Returns (X, Y, dt, rotation_angle) where Y = X @ R.T
    for a rotation matrix R with angle theta.
    """
    theta = np.pi / 6  # 30 degrees
    dt = 0.1
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    n_samples = 200
    X = rng.standard_normal((n_samples, 2))
    Y = X @ R.T
    return X, Y, dt, theta
