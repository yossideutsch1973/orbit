"""Spectral analysis and interpretability tools for DKN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from dkn.network import KoopmanNet


def extract_eigenvalues(net: KoopmanNet) -> list[list[torch.Tensor]]:
    """Extract eigenvalues from all K matrices.

    Returns:
        Nested list: eigenvalues[layer_idx][neuron_idx] = complex tensor.
    """
    return [
        [neuron.eigenvalues() for neuron in layer.neurons]
        for layer in net.backbone
    ]


def spectral_radius(net: KoopmanNet) -> float:
    """Maximum |eigenvalue| across all K matrices."""
    max_radius = 0.0
    for layer_eigs in extract_eigenvalues(net):
        for eigs in layer_eigs:
            radius = eigs.abs().max().item()
            max_radius = max(max_radius, radius)
    return max_radius


def eigenvalue_summary(net: KoopmanNet) -> dict[str, float]:
    """Summary statistics of all eigenvalues across the network."""
    all_magnitudes: list[float] = []
    for layer_eigs in extract_eigenvalues(net):
        for eigs in layer_eigs:
            all_magnitudes.extend(eigs.abs().tolist())
    mags = np.array(all_magnitudes)
    return {
        "spectral_radius": float(mags.max()),
        "mean_magnitude": float(mags.mean()),
        "std_magnitude": float(mags.std()),
        "min_magnitude": float(mags.min()),
        "n_unstable": int((mags > 1.0).sum()),
    }


def k_matrices(net: KoopmanNet) -> list[list[torch.Tensor]]:
    """Extract raw K matrices from all neurons."""
    return [
        [neuron.K.data.clone() for neuron in layer.neurons]
        for layer in net.backbone
    ]
