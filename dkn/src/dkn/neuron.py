"""Koopman neuron: the atomic unit of a Distributed Koopman Network."""

import torch
import torch.nn as nn


class KoopmanNeuron(nn.Module):
    """Single Koopman operator neuron: lift -> K @ z -> project."""

    def __init__(self, d_in: int, d_lift: int, d_out: int, eps: float = 0.01) -> None:
        """
        Args:
            d_in: Input dimension (slice of layer input).
            d_lift: Lifted (latent) space dimension.
            d_out: Output dimension after projection.
            eps: Scale of random perturbation for near-identity K init.
        """
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(d_in, d_lift), nn.ReLU())
        self.K = nn.Parameter(torch.eye(d_lift) + eps * torch.randn(d_lift, d_lift))
        self.proj = nn.Linear(d_lift, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: lift (nonlinear) -> K @ z -> project."""
        g = self.lift(x)
        Kg = g @ self.K.T
        return self.proj(Kg)

    def spectral_penalty(self) -> torch.Tensor:
        """Penalty for eigenvalue magnitudes exceeding 1 (unit circle)."""
        eigs = torch.linalg.eigvals(self.K)
        magnitudes = eigs.abs()
        return torch.relu(magnitudes - 1.0).pow(2).mean()

    def eigenvalues(self) -> torch.Tensor:
        """Compute eigenvalues of the K matrix."""
        return torch.linalg.eigvals(self.K.data)
