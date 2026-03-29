"""Tests for KoopmanLayer."""

import torch
import pytest

from dkn.layer import KoopmanLayer


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def test_output_shape():
    layer = KoopmanLayer(d_in=8, n_neurons=4, d_lift=16, d_out_per_neuron=4)
    x = torch.randn(16, 8)
    assert layer(x).shape == (16, 16)


def test_d_out_matches_output():
    layer = KoopmanLayer(d_in=6, n_neurons=3, d_lift=8, d_out_per_neuron=5)
    assert layer.d_out == 15
    assert layer(torch.randn(4, 6)).shape[1] == layer.d_out


def test_uneven_input_division():
    layer = KoopmanLayer(d_in=7, n_neurons=3, d_lift=8, d_out_per_neuron=4)
    assert layer.slices == [(0, 3), (3, 5), (5, 7)]
    assert layer(torch.randn(8, 7)).shape == (8, 12)


def test_gradient_flow():
    layer = KoopmanLayer(d_in=4, n_neurons=2, d_lift=8, d_out_per_neuron=3)
    layer(torch.randn(8, 4)).sum().backward()
    for neuron in layer.neurons:
        assert neuron.K.grad is not None
        assert neuron.lift.weight.grad is not None


def test_with_mixing():
    layer = KoopmanLayer(d_in=8, n_neurons=2, d_lift=16, d_out_per_neuron=4, mix=True)
    assert layer(torch.randn(16, 8)).shape == (16, 8)
    assert layer.mixer is not None
