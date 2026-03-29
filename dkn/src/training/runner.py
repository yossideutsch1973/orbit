"""Training loop with logging and eigenvalue snapshots."""

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from dkn.analysis import spectral_radius
from training.buffer import RolloutBuffer
from training.ppo import PPOConfig, PPOUpdater


@dataclass
class TrainResult:
    """Results from a training run."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    final_reward_mean: float = 0.0
    final_reward_std: float = 0.0
    best_reward: float = float("-inf")
    total_params: int = 0
    wall_time: float = 0.0
    loss_history: list[dict[str, float]] = field(default_factory=list)


def count_parameters(net: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def train_ppo(
    network: nn.Module,
    env_name: str,
    total_steps: int,
    steps_per_update: int,
    ppo_cfg: PPOConfig,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    seed: int = 42,
    log_interval: int = 10,
    is_dkn: bool = False,
) -> TrainResult:
    """Train an actor-critic network with PPO."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    env.reset(seed=seed)

    buf = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
    updater = PPOUpdater(network, ppo_cfg)

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    loss_history: list[dict[str, float]] = []
    ep_reward, ep_len = 0.0, 0
    best_reward = float("-inf")

    obs, _ = env.reset()
    start = time.time()
    step = 0

    while step < total_steps:
        for _ in range(steps_per_update):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = network.act(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            buf.store(obs, action.item(), reward, done, log_prob.item(), value.item())
            ep_reward += reward
            ep_len += 1
            obs = next_obs
            step += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                best_reward = max(best_reward, ep_reward)
                if len(ep_rewards) % log_interval == 0:
                    recent = ep_rewards[-log_interval:]
                    msg = f"Ep {len(ep_rewards)} | R: {np.mean(recent):.1f}+/-{np.std(recent):.1f} | step {step}"
                    if is_dkn:
                        msg += f" | rho: {spectral_radius(network):.3f}"
                    print(msg)
                ep_reward, ep_len = 0.0, 0
                obs, _ = env.reset()
            if step >= total_steps:
                break

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_val = network(obs_t)
        batch = buf.compute_returns(last_val.item(), device)
        stats = updater.update(batch)
        loss_history.append(stats)
        buf.clear()

    env.close()
    final = ep_rewards[-100:] if len(ep_rewards) >= 100 else ep_rewards
    return TrainResult(
        episode_rewards=ep_rewards,
        episode_lengths=ep_lengths,
        final_reward_mean=float(np.mean(final)) if final else 0.0,
        final_reward_std=float(np.std(final)) if final else 0.0,
        best_reward=best_reward,
        total_params=count_parameters(network),
        wall_time=time.time() - start,
        loss_history=loss_history,
    )
