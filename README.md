# KoopSim

**Koopman Operator Simulation Toolkit** -- Train once on snapshot data, then instantly query system state at any future time `t`.

## What is KoopSim?

KoopSim implements Koopman operator methods for dynamical systems analysis. Instead of time-stepping simulations, it learns a linear operator that advances the system state:

```
state(t) = K^t @ initial_state
```

This enables instant prediction at any future time without iterative simulation.

## Features

- **Classical EDMD**: Extended Dynamic Mode Decomposition with polynomial + RBF dictionaries
- **Neural Koopman**: Autoencoder-based deep learning approach (requires PyTorch)
- **Instant Prediction**: Matrix exponential or eigendecomposition -- no time-stepping loops
- **Built-in Systems**: Hopf bifurcation, double gyre, spring-mass-damper, beam, RLC circuit, Van der Pol
- **Uncertainty Quantification**: Monte Carlo UQ for prediction confidence
- **Visualization**: Trajectory plots, phase portraits, eigenspectra, animations
- **CLI + Dashboard**: Command-line interface and Streamlit web dashboard

## Installation

```bash
pip install -e .              # Core only
pip install -e ".[viz]"       # + visualization
pip install -e ".[neural]"    # + PyTorch neural Koopman
pip install -e ".[all]"       # Everything
```

## Quick Start

### Python API

```python
from koopsim import KoopSim
from koopsim.systems import HopfBifurcation
import numpy as np

# Generate data from a dynamical system
system = HopfBifurcation(mu=1.0)
sim = KoopSim.from_system(system, poly_degree=3)

# Predict at any future time -- no time-stepping!
x0 = np.array([0.5, 0.0])
state = sim.predict(x0, t=5.0)

# Predict full trajectory
times = np.linspace(0, 10, 200)
trajectory = sim.predict_trajectory(x0, times)

# Uncertainty quantification
result = sim.predict_with_uncertainty(x0, t=5.0, n_samples=200)
print(f"Mean: {result['mean']}, Std: {result['std']}")
```

### CLI

```bash
# Generate training data
koopsim generate --system hopf -o data.h5

# Train a model
koopsim train --data data.h5 --method edmd --dt 0.01 -o model.koop

# Predict
koopsim predict --model model.koop --initial-state "0.5,0.0" --time 5.0

# Model info
koopsim info --model model.koop
```

### Streamlit Dashboard

```bash
streamlit run koopsim/dashboard.py
```

## Architecture

```
koopsim/
├── core/           # EDMD, Neural Koopman, prediction engine, validation, UQ
├── systems/        # Built-in dynamical systems (fluid, mechanical, circuit)
├── utils/          # Dictionary functions, visualization, I/O
├── cli.py          # Command-line interface
├── dashboard.py    # Streamlit web dashboard
└── koopsim.py      # High-level Python API
```

## Examples

See the `examples/` directory for Jupyter notebooks:
- `01_hopf_vortex.ipynb` -- Hopf bifurcation limit cycle analysis
- `02_double_gyre.ipynb` -- Double gyre flow field
- `03_mechanical_beam.ipynb` -- Structural dynamics (beam + spring-mass)
- `04_circuit.ipynb` -- RLC circuit (linear system, perfect recovery)

## Mathematical Background

KoopSim uses the Koopman operator framework: for a dynamical system x_{k+1} = F(x_k), there exists an infinite-dimensional linear operator K such that observable functions evolve linearly:

```
g(x_{k+1}) = K * g(x_k)
```

By choosing a finite dictionary of observables g = [psi_1, psi_2, ...], we approximate K as a matrix and predict:

```
state(t) = unlift(expm(log(K) * t / dt) * lift(x0))
```

## License

MIT
