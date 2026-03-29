"""Batched PPO update, shared by all actor-critic architectures."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    lr: float = 3e-4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    spectral_coef: float = 0.1


class PPOUpdater:
    """Performs batched PPO updates on an actor-critic network."""

    def __init__(self, network: nn.Module, cfg: PPOConfig) -> None:
        self.network = network
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(network.parameters(), lr=cfg.lr)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run PPO epochs on a batch. Returns average losses."""
        obs, actions = batch["obs"], batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(n, device=obs.device)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = indices[start : start + self.cfg.minibatch_size]
                new_log_probs, values, entropy = self.network.evaluate(obs[idx], actions[idx])

                ratio = (new_log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (values - returns[idx]).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_mean

                if self.cfg.spectral_coef > 0 and hasattr(self.network, "spectral_penalty"):
                    loss = loss + self.cfg.spectral_coef * self.network.spectral_penalty()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["total_loss"] += loss.item()
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in totals.items()}
