"""Koopman layer: routes input slices to neurons with optional mixing."""

import torch
import torch.nn as nn

from dkn.neuron import KoopmanNeuron


class KoopmanLayer(nn.Module):
    """Layer of Koopman neurons with input routing and optional output mixing."""

    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        d_lift: int,
        d_out_per_neuron: int,
        mix: bool = False,
        eps: float = 0.01,
        broadcast: bool = False,
    ) -> None:
        """
        Args:
            d_in: Total input dimension.
            n_neurons: Number of Koopman neurons.
            d_lift: Lifted dimension per neuron.
            d_out_per_neuron: Output dimension per neuron.
            mix: If True, apply a learned mixing layer after concatenation.
            eps: Near-identity init scale for K matrices.
            broadcast: If True, every neuron sees full input instead of a slice.
        """
        super().__init__()
        self.broadcast = broadcast
        if broadcast:
            self.slices = [(0, d_in)] * n_neurons
            neuron_d_in = d_in
        else:
            base = d_in // n_neurons
            remainder = d_in % n_neurons
            self.slices = []
            offset = 0
            for i in range(n_neurons):
                size = base + (1 if i < remainder else 0)
                self.slices.append((offset, offset + size))
                offset += size
            neuron_d_in = -1  # each neuron has its own size
        neurons = []
        for i in range(n_neurons):
            nd_in = neuron_d_in if broadcast else (self.slices[i][1] - self.slices[i][0])
            neurons.append(KoopmanNeuron(nd_in, d_lift, d_out_per_neuron, eps))
        self.neurons = nn.ModuleList(neurons)
        self.d_out = n_neurons * d_out_per_neuron
        self.mixer = nn.Linear(self.d_out, self.d_out) if mix else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route inputs to neurons, concatenate, optionally mix."""
        parts = [neuron(x[:, s:e]) for neuron, (s, e) in zip(self.neurons, self.slices)]
        out = torch.cat(parts, dim=-1)
        if self.mixer is not None:
            out = self.mixer(out)
        return out
