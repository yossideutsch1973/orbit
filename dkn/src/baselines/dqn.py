"""DQN baseline agent with replay buffer."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DQNConfig:
    """DQN hyperparameters."""

    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 5000


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig, device: torch.device) -> None:
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device
        self.q_net = QNetwork(state_dim, action_dim, cfg.hidden).to(device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer: deque[tuple] = deque(maxlen=cfg.buffer_size)
        self.step_count = 0

    def epsilon(self) -> float:
        """Current epsilon for exploration."""
        progress = min(self.step_count / self.cfg.eps_decay_steps, 1.0)
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * progress

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=-1).item()

    def store(self, obs: np.ndarray, action: int, reward: float,
              next_obs: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def update(self) -> float | None:
        """Sample batch and do TD update. Returns loss or None if buffer too small."""
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = random.sample(list(self.buffer), self.cfg.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=-1).values
            targets = rewards_t + self.cfg.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)
