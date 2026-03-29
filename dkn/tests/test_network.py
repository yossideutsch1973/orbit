"""Tests for KoopmanNet."""

import torch
import pytest

from dkn.network import KoopmanNet, DKNConfig


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


DEVICE = torch.device("cpu")


@pytest.fixture
def cartpole_cfg():
    return DKNConfig(
        state_dim=4, action_dim=2, n_layers=2,
        neurons_per_layer=2, d_lift=8, d_out_per_neuron=4,
    )


@pytest.fixture
def lunar_cfg():
    return DKNConfig(
        state_dim=8, action_dim=4, n_layers=2,
        neurons_per_layer=4, d_lift=16, d_out_per_neuron=8, head_hidden=64,
    )


def test_forward_shape(cartpole_cfg):
    net = KoopmanNet(cartpole_cfg, DEVICE)
    logits, values = net(torch.randn(16, 4))
    assert logits.shape == (16, 2)
    assert values.shape == (16,)


def test_act_shape(cartpole_cfg):
    net = KoopmanNet(cartpole_cfg, DEVICE)
    action, log_prob, value = net.act(torch.randn(1, 4))
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_evaluate_shape(cartpole_cfg):
    net = KoopmanNet(cartpole_cfg, DEVICE)
    obs = torch.randn(32, 4)
    actions = torch.randint(0, 2, (32,))
    log_probs, values, entropy = net.evaluate(obs, actions)
    assert log_probs.shape == (32,)
    assert values.shape == (32,)
    assert entropy.shape == (32,)


def test_head_hidden(lunar_cfg):
    net = KoopmanNet(lunar_cfg, DEVICE)
    logits, values = net(torch.randn(8, 8))
    assert logits.shape == (8, 4)
    assert values.shape == (8,)


def test_gradient_flow_to_k_matrices(cartpole_cfg):
    net = KoopmanNet(cartpole_cfg, DEVICE)
    logits, values = net(torch.randn(16, 4))
    (logits.sum() + values.sum()).backward()
    for layer in net.backbone:
        for neuron in layer.neurons:
            assert neuron.K.grad is not None


def test_parameter_count(cartpole_cfg):
    net = KoopmanNet(cartpole_cfg, DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0
    # DKN should be relatively small
    assert n_params < 5000
