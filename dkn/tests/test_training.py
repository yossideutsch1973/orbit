"""Tests for training infrastructure."""

import numpy as np
import torch
import pytest

from dkn.network import KoopmanNet, DKNConfig
from training.buffer import RolloutBuffer
from training.ppo import PPOConfig, PPOUpdater


DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_buffer_store_and_clear():
    buf = RolloutBuffer()
    for _ in range(10):
        buf.store(np.zeros(4), 0, 1.0, False, -0.5, 0.9)
    assert len(buf) == 10
    buf.clear()
    assert len(buf) == 0


def test_buffer_compute_returns():
    buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for _ in range(64):
        buf.store(np.random.randn(4), np.random.randint(2), 1.0, False, -0.5, 0.9)
    batch = buf.compute_returns(0.0, DEVICE)
    assert batch["obs"].shape == (64, 4)
    assert batch["actions"].shape == (64,)
    assert batch["advantages"].shape == (64,)
    assert batch["returns"].shape == (64,)


def test_ppo_update_runs():
    cfg = DKNConfig(state_dim=4, action_dim=2, n_layers=1,
                    neurons_per_layer=2, d_lift=4, d_out_per_neuron=2)
    net = KoopmanNet(cfg, DEVICE)
    updater = PPOUpdater(net, PPOConfig(lr=3e-4, ppo_epochs=2, minibatch_size=32))

    buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for _ in range(64):
        obs = np.random.randn(4)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = net.act(obs_t)
        buf.store(obs, action.item(), 1.0, False, log_prob.item(), value.item())

    stats = updater.update(buf.compute_returns(0.0, DEVICE))
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy" in stats


def test_training_loss_no_explosion():
    """5-iteration smoke test: loss must not explode."""
    import gymnasium as gym

    cfg = DKNConfig(state_dim=4, action_dim=2, n_layers=1,
                    neurons_per_layer=2, d_lift=4, d_out_per_neuron=2)
    net = KoopmanNet(cfg, DEVICE)
    updater = PPOUpdater(net, PPOConfig(lr=3e-4, ppo_epochs=4, minibatch_size=32))

    env = gym.make("CartPole-v1")
    losses = []
    for _ in range(5):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        obs, _ = env.reset(seed=42)
        for _ in range(128):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = net.act(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            buf.store(obs, action.item(), reward, terminated or truncated,
                      log_prob.item(), value.item())
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        with torch.no_grad():
            _, last_val = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        stats = updater.update(buf.compute_returns(last_val.item(), DEVICE))
        losses.append(stats["total_loss"])
    env.close()

    assert all(not np.isnan(l) for l in losses)
    assert losses[-1] < losses[0] * 10
