# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KoopSim is a Koopman Operator Simulation Toolkit for engineers (mechanical, aerospace, electrical, fluid-dynamics). The core promise: "Train once on snapshot data, then instantly query system state at any future time t" — no time-stepping loops.

## Status

Project is in initial development. The README contains the full specification/plan.

## Tech Stack

- **Core:** NumPy, SciPy, scikit-learn
- **Neural:** PyTorch + Lightning
- **Viz:** Matplotlib, Plotly, imageio
- **UI:** Streamlit (optional dashboard)
- **Data:** HDF5 for large snapshot datasets

## Architecture

Two learning modes:
- **Classical EDMD** (`core/edmd.py`): Extended Dynamic Mode Decomposition with polynomial + RBF dictionary
- **Neural Koopman** (`core/neural_koopman.py`): Autoencoder + linear Koopman layer in PyTorch

Prediction engine (`core/prediction.py`):
- Matrix exponential (`scipy.linalg.expm`) for small/medium systems
- Eigen-decomposition + fast reconstruction for large systems

Domain systems in `systems/`: fluid_particles, fluid_grid, mechanical, circuit.

Entry points: `cli.py` (CLI), `dashboard.py` (Streamlit), `koopsim.py` (Python API).

## Mathematical Constraint

All predictions must use the Koopman form:
```python
state(t) = K**t @ initial_state   # or expm(t * logK)
```
No time-stepping loops in final prediction — this is a hard requirement.

## Planned Project Layout

```
koopsim/
├── core/          # EDMD, Neural Koopman, prediction engine
├── systems/       # Domain-specific system implementations
├── utils/         # Dictionary functions, visualization, I/O (.koop files)
├── examples/      # Jupyter notebooks (Hopf vortex, double gyre, beam)
├── cli.py
├── dashboard.py
└── koopsim.py     # Main public API
```
