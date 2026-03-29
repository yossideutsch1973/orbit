"""KoopmanNet: stacked Koopman layers with actor/critic heads for RL."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dkn.layer import KoopmanLayer


@dataclass
class DKNConfig:
    """Configuration for KoopmanNet architecture."""

    state_dim: int
    action_dim: int
    n_layers: int
    neurons_per_layer: int
    d_lift: int
    d_out_per_neuron: int
    head_hidden: int = 0
    broadcast: bool = True
    k_rank: int = 0


class KoopmanNet(nn.Module):
    """Distributed Koopman Network with actor-critic heads."""

    def __init__(self, cfg: DKNConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        d_in = cfg.state_dim
        layers = []
        for _ in range(cfg.n_layers):
            layer = KoopmanLayer(
                d_in, cfg.neurons_per_layer, cfg.d_lift, cfg.d_out_per_neuron,
                broadcast=cfg.broadcast, k_rank=cfg.k_rank,
            )
            layers.append(layer)
            d_in = layer.d_out
        self.backbone = nn.ModuleList(layers)
        self.backbone_out_dim = d_in

        if cfg.head_hidden > 0:
            self.actor = nn.Sequential(
                nn.Linear(d_in, cfg.head_hidden), nn.ReLU(),
                nn.Linear(cfg.head_hidden, cfg.action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(d_in, cfg.head_hidden), nn.ReLU(),
                nn.Linear(cfg.head_hidden, 1),
            )
        else:
            self.actor = nn.Linear(d_in, cfg.action_dim)
            self.critic = nn.Linear(d_in, 1)

        self.to(device)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.backbone:
            x = layer(x)
        return x

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value)."""
        features = self._backbone_forward(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action for environment interaction. Returns (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update. Returns (log_probs, values, entropy)."""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()

    def spectral_penalty(self) -> torch.Tensor:
        """Sum of spectral penalties across all K matrices."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.backbone:
            for neuron in layer.neurons:
                penalty = penalty + neuron.spectral_penalty()
        return penalty

    def project_k_matrices(self) -> None:
        """Project all K eigenvalues onto the unit circle."""
        for layer in self.backbone:
            for neuron in layer.neurons:
                neuron.project_to_unit_circle()
