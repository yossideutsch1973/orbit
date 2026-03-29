"""MLP actor-critic baseline for PPO comparison."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
    """Two-headed MLP with shared nothing (separate actor and critic)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int, device: torch.device) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.to(device)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value)."""
        return self.actor(obs), self.critic(obs).squeeze(-1)

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
