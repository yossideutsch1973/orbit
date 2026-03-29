"""Rollout buffer with GAE for PPO training."""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Stores rollout data and computes GAE advantages."""

    gamma: float = 0.99
    gae_lambda: float = 0.95

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def store(
        self, obs: np.ndarray, action: int, reward: float,
        done: bool, log_prob: float, value: float,
    ) -> None:
        """Store a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns(self, last_value: float, device: torch.device) -> dict[str, torch.Tensor]:
        """Compute GAE advantages and returns, return tensors on device."""
        values = self.values + [last_value]
        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        val_t = torch.tensor(self.values, dtype=torch.float32, device=device)
        return {
            "obs": torch.tensor(np.array(self.observations), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "advantages": adv_t,
            "returns": adv_t + val_t,
        }

    def clear(self) -> None:
        """Reset all storage."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.observations)
