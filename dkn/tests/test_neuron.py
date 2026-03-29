"""Tests for KoopmanNeuron."""

import torch
import pytest

from dkn.neuron import KoopmanNeuron


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def test_output_shape():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3)
    x = torch.randn(16, 4)
    assert neuron(x).shape == (16, 3)


def test_single_sample():
    neuron = KoopmanNeuron(d_in=2, d_lift=4, d_out=2)
    x = torch.randn(1, 2)
    assert neuron(x).shape == (1, 2)


def test_gradient_flow():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3)
    x = torch.randn(8, 4)
    neuron(x).sum().backward()
    assert neuron.lift[0].weight.grad is not None
    assert neuron.K.grad is not None
    assert neuron.proj.weight.grad is not None


def test_near_identity_init():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3, eps=0.01)
    K = neuron.K_matrix
    diff = (K - torch.eye(8)).abs().max().item()
    assert diff < 0.1


def test_eigenvalues():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3)
    eigs = neuron.eigenvalues()
    assert eigs.shape == (8,)
    assert (eigs.abs() - 1.0).abs().mean().item() < 0.5


# --- Low-rank K tests ---


def test_lowrank_output_shape():
    neuron = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, k_rank=4)
    assert neuron(torch.randn(8, 4)).shape == (8, 3)


def test_lowrank_gradient_flow():
    neuron = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, k_rank=4)
    neuron(torch.randn(8, 4)).sum().backward()
    assert neuron.K_U.grad is not None
    assert neuron.K_V.grad is not None
    assert neuron.proj.weight.grad is not None


def test_lowrank_near_identity():
    neuron = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, eps=0.01, k_rank=2)
    K = neuron.K_matrix
    diff = (K - torch.eye(16)).abs().max().item()
    assert diff < 0.1


def test_lowrank_fewer_params():
    full = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, k_rank=0)
    low = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, k_rank=4)
    full_p = sum(p.numel() for p in full.parameters())
    low_p = sum(p.numel() for p in low.parameters())
    # Low-rank K: 2*16*4=128 vs full K: 16*16=256, saving 128 params
    assert low_p < full_p
    assert full_p - low_p == 16 * 16 - 2 * 16 * 4


def test_lowrank_eigenvalues():
    neuron = KoopmanNeuron(d_in=4, d_lift=16, d_out=3, k_rank=4)
    eigs = neuron.eigenvalues()
    assert eigs.shape == (16,)


def test_lowrank_spectral_penalty():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3, k_rank=2)
    penalty = neuron.spectral_penalty()
    assert penalty.shape == ()
    penalty.backward()
    assert neuron.K_U.grad is not None
