"""Koopman neuron: the atomic unit of a Distributed Koopman Network."""

import torch
import torch.nn as nn


class KoopmanNeuron(nn.Module):
    """Single Koopman operator neuron: lift -> K @ z -> project."""

    def __init__(
        self, d_in: int, d_lift: int, d_out: int, eps: float = 0.01, k_rank: int = 0,
    ) -> None:
        """
        Args:
            d_in: Input dimension (slice of layer input).
            d_lift: Lifted (latent) space dimension.
            d_out: Output dimension after projection.
            eps: Scale of random perturbation for near-identity K init.
            k_rank: Low-rank factorization rank. 0 = full K matrix.
        """
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(d_in, d_lift), nn.Tanh())
        self.proj = nn.Linear(d_lift, d_out)
        self.k_rank = k_rank
        scaled_eps = eps / (d_lift ** 0.5)

        if k_rank > 0:
            # K = I + U @ V^T  (low-rank perturbation of identity)
            self.K_U = nn.Parameter(scaled_eps * torch.randn(d_lift, k_rank))
            self.K_V = nn.Parameter(scaled_eps * torch.randn(d_lift, k_rank))
            self.register_buffer("_I", torch.eye(d_lift))
        else:
            self.K = nn.Parameter(torch.eye(d_lift) + scaled_eps * torch.randn(d_lift, d_lift))

    @property
    def K_matrix(self) -> torch.Tensor:
        """Reconstruct full K matrix."""
        if self.k_rank > 0:
            return self._I + self.K_U @ self.K_V.T
        return self.K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: lift (nonlinear) -> K @ z -> project."""
        g = self.lift(x)
        Kg = g @ self.K_matrix.T
        return self.proj(Kg)

    def spectral_penalty(self) -> torch.Tensor:
        """Two-sided penalty pushing eigenvalue magnitudes toward 1 (unit circle)."""
        eigs = torch.linalg.eigvals(self.K_matrix)
        return (eigs.abs() - 1.0).pow(2).mean()

    def project_to_unit_circle(self, alpha: float = 0.2) -> None:
        """Soft-project K eigenvalues toward the unit circle (preserves phases)."""
        with torch.no_grad():
            K = self.K_matrix
            eigvals, V = torch.linalg.eig(K)
            unit_eigvals = eigvals / eigvals.abs().clamp(min=1e-8)
            soft_eigvals = (1 - alpha) * eigvals + alpha * unit_eigvals
            try:
                K_new = (V @ torch.diag(soft_eigvals) @ torch.linalg.inv(V)).real
                if self.k_rank > 0:
                    # Re-factorize perturbation via truncated SVD
                    delta = K_new - self._I
                    U_s, S_s, Vt_s = torch.linalg.svd(delta)
                    r = self.k_rank
                    self.K_U.data.copy_(U_s[:, :r] @ torch.diag(S_s[:r]))
                    self.K_V.data.copy_(Vt_s[:r].T)
                else:
                    self.K.data.copy_(K_new)
            except torch.linalg.LinAlgError:
                pass

    def eigenvalues(self) -> torch.Tensor:
        """Compute eigenvalues of the K matrix."""
        return torch.linalg.eigvals(self.K_matrix.data)
