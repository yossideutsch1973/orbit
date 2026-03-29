# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Distributed Koopman Networks (DKN): a neural architecture where each neuron is a small Koopman operator (`input -> lift -> K*z -> project`). Many small K matrices composed in layers replace one monolithic operator, avoiding the curse of lifting while preserving interpretability (eigenvalue spectra = learned dynamical modes). Early results show 1.6x DQN reward with 10% of parameters on CartPole.

## Architecture

```
src/
  dkn/
    neuron.py      # KoopmanNeuron: lift -> K @ z -> project (atomic unit)
    layer.py       # KoopmanLayer: routes input slices to neurons, optional mixing
    network.py     # KoopmanNet: stacks layers, actor/critic heads
    analysis.py    # Eigenvalue extraction, spectral analysis, interpretability
  training/
    ppo.py         # Batched PPO (shared by all architectures)
    buffer.py      # Rollout buffer
    runner.py      # Training loop with logging
  baselines/
    mlp.py         # MLP actor-critic baseline
    dqn.py         # DQN baseline
  experiments/
    configs/       # YAML experiment configs
    run.py         # Entrypoint: python -m experiments.run --config <name>
tests/
  test_neuron.py
  test_layer.py
  test_network.py
  test_training.py
```

## Commands

```bash
# Setup
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy pyyaml gymnasium pytest

# Tests
pytest tests/ -x -q                          # run tests, stop on first failure
pytest tests/test_neuron.py -x -q            # run a single test file

# Experiments (requires PYTHONPATH or pip install -e .)
PYTHONPATH=src python -m experiments.run --config cartpole
PYTHONPATH=src python -m experiments.run --config lunarlander --device cuda
PYTHONPATH=src python -m experiments.run --config cartpole --arch dkn  # DKN only
```

## Tech Stack

- Python 3.10+, PyTorch, Gymnasium (with box2d, mujoco)
- SymPy for symbolic analysis (used by the math-researcher agent)
- pytest for testing, W&B for experiment tracking

## Custom Agents

Two agent definitions live at the project root:

- **experiment-runner.md** (Sonnet): Runs training, monitors convergence (NaN losses, eigenvalue explosion |lambda| > 10, reward plateau), produces comparison tables. Stops early if training diverges.
- **math-researcher.md** (Opus): Spectral analysis of K matrices, lifting function rank conditions, layer fusion opportunities, convergence analysis. Uses SymPy in `/tmp/math_scratch/`.

## Conventions

IMPORTANT: Each module must be under 150 lines. If it grows beyond that, decompose it.

- Type hints on all public functions. No `Any` unless unavoidable.
- Every class gets a one-line docstring. Complex methods get parameter docs.
- K matrices initialize near identity: `torch.eye(d) + eps * torch.randn(d, d)`.
- `@dataclass` for configs. No magic numbers in training code -- all hyperparams come from config YAML.
- Device handling: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Set once in runner, pass explicitly. Never hardcode "cpu" or "cuda".
- GPU auto-detect via `torch.cuda.is_available()`; all tensors respect a global `DEVICE`.

## Testing Rules

- Every PR must pass `pytest tests/ -x`. No exceptions.
- Neuron/layer tests: check output shapes, gradient flow, K spectrum extraction.
- Training tests: run 5 episodes, verify loss decreases. Use `CartPole-v1` (fast).
- Use `torch.manual_seed(42)` in all tests for reproducibility.

## What Claude Gets Wrong on This Project

- Tends to write monolithic training loops. ALWAYS decompose: buffer, runner, policy are separate.
- Forgets GPU device propagation. Every `.to(device)` must be explicit and tested.
- Over-engineers the Koopman neuron. The forward pass is three lines: `g = lift(x)`, `Kg = g @ K.T`, `out = proj(Kg)`. Keep it that simple.
- Creates overly large files. If a file exceeds 150 lines, split it before continuing.

## Experiment Config Structure

Configs define: env name, state/action dims, DKN architecture (layers, neurons, lift dim), MLP baseline size, PPO hyperparams, and logging intervals. See `cartpole.yaml` for a minimal example.

## When Compacting

Preserve: current experiment results, file modification list, any discovered bugs or architectural decisions.
