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
    assert neuron.lift.weight.grad is not None
    assert neuron.K.grad is not None
    assert neuron.proj.weight.grad is not None


def test_near_identity_init():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3, eps=0.01)
    diff = (neuron.K.data - torch.eye(8)).abs().max().item()
    assert diff < 0.1


def test_eigenvalues():
    neuron = KoopmanNeuron(d_in=4, d_lift=8, d_out=3)
    eigs = neuron.eigenvalues()
    assert eigs.shape == (8,)
    # Near-identity K -> eigenvalues near 1
    assert (eigs.abs() - 1.0).abs().mean().item() < 0.5
