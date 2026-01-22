# ContLearn: Continual Learning with Hamiltonian Gradients and AWB

A JAX/Equinox framework for continual learning using Hamiltonian-based gradient computation and Adaptive Weight Basis (AWB) for architecture adaptation.

## Overview

This framework implements a novel approach to continual learning that combines:

- **Hamiltonian Gradients**: Balances learning on current tasks with experience replay and regularization
- **Adaptive Weight Basis (AWB)**: Dynamically adapts network architecture and transfers knowledge via learned transformation matrices

### Key Features

- Support for MLP, CNN, and GCN architectures
- Unified training pipeline for regression and classification
- Experience replay with configurable buffer management
- Multiple learning rate schedules (constant, step, exponential, cosine)
- Comprehensive metrics tracking and checkpointing

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Install from source

```bash
git clone https://github.com/krm9c/ContLearn.git
cd ContLearn
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

### Install with CUDA support

```bash
pip install -e ".[cuda]"
```

## Quick Start

### Run an experiment

```bash
# Baseline training on sine regression
python run.py examples/configs/sine_baseline.json

# AWB-enabled training
python run.py examples/configs/sine_awb_full.json

# Multiple seeds
python run.py examples/configs/synthetic_graph_awb_full.json --runs 3
```

### Example notebooks

Interactive Jupyter notebooks are available in `examples/notebooks/`:

| Notebook | Dataset | Model | Description |
|----------|---------|-------|-------------|
| `01_sine_example.ipynb` | Sine regression | MLP | 5-task frequency/amplitude drift |
| `02_mnist_example.ipynb` | MNIST | CNN | 5-task rotation/scaling transforms |
| `03_synthetic_graph_example.ipynb` | Synthetic graphs | GCN | 10-task domain shift |

Each notebook demonstrates:
1. Quick training demo (optional, requires compute)
2. Loading and exploring pre-computed results
3. Creating comparison plots (Baseline vs AWB)

## Project Structure

```
ContLearn/
├── src/cl/                    # Core source code
│   ├── arch_search/           # Architecture search (MLP, CNN, GCN)
│   ├── config/                # Configuration and constants
│   ├── core/                  # Trainer mixins (losses, hamiltonian, loops, recording, awb)
│   ├── datasets/              # Dataset implementations
│   ├── models/                # Model architectures
│   └── runners/               # Training runners
├── examples/                  # Example notebooks and configs
│   ├── configs/               # 6 essential configs
│   ├── data/                  # Pre-computed results
│   ├── notebooks/             # Jupyter notebooks
│   └── utils.py               # Plotting utilities
├── tests/                     # Test suite
├── run.py                     # Main entry point
└── pyproject.toml             # Project configuration
```

## Core Concepts

### Hamiltonian Gradient

The framework uses a weighted combination of gradients:

```
grad = α·L_current + β·L_experience + γ·dV
```

Where:
- `L_current`: Loss on current task data
- `L_experience`: Loss on experience replay buffer
- `dV`: Regularization term (∂V/∂x, ∂V/∂θ)

Default weights: `[0.4, 0.4, 0.1]`

### AWB Pipeline

When AWB is enabled, the training follows a 5-step process for each new task:

1. **Preliminary training**: Initial training on new task
2. **Architecture decision**: Check if loss ratio exceeds threshold
3. **Architecture search**: Find optimal layer sizes
4. **A/B matrix training**: Learn transformation `V = A @ W @ B.T`
5. **V training**: Continue training with transferred weights

### Supported Datasets

| Dataset | Type | Network | Tasks |
|---------|------|---------|-------|
| `sine` | Regression | MLP | Frequency/amplitude drift |
| `mnist` | Classification | CNN | Rotation/scaling transforms |
| `cifar10/cifar100` | Classification | CNN | Class incremental |
| `synthetic_taskshift` | Graph Classification | GCN | Domain shift with perturbations |

## Configuration

Example configuration file:

```json
{
    "data": "sine",
    "network": "fcnn",
    "n_task": 5,
    "epochs_per_task": 200,
    "batch_size": 128,
    "lr": 0.0001,
    "lr_schedule": "cosine",
    "optimizer": "adam",
    "grad_weights": [0.4, 0.4, 0.1],
    "awb_enabled": true,
    "task_warmup_enabled": true,
    "task_warmup_epochs": 20
}
```

See `src/cl/config/constants.py` for all available parameters and defaults.

## Testing

```bash
# Run fast unit tests
pytest -m unit -v

# Run full pipeline tests
pytest -m training -v

# Run specific test file
pytest tests/test_awb.py -v
```

## Results

### Experimental Conditions

1. **Baseline (C1)**: Fixed architecture, constant learning rate
2. **Heuristics (C2)**: Task warmup, adaptive learning rate
3. **Arch Search (C3)**: Architecture search, no weight transfer
4. **AWB Full (C4)**: Architecture search + A/B weight transfer

### Summary

| Dataset | Tasks | Network | Best Method | Key Finding |
|---------|-------|---------|-------------|-------------|
| Sine | 5 | MLP | AWB Full | Lowest MSE, smooth training |
| MNIST | 5-10 | CNN | AWB Full | Handles transform difficulty |
| Synthetic Graph | 10 | GCN | AWB Full | 89.3% vs 79.4% baseline |

### Synthetic Graph Results (10-Task)

| Method | Avg Accuracy | Forgetting | Notes |
|--------|-------------|------------|-------|
| Baseline | 79.4% | 0.5% | Limited learning capacity |
| AWB Full | 89.3% | 3.6% | +12.5% accuracy gain |

## Metrics

The framework tracks standard continual learning metrics:

- **Average Accuracy/MSE**: Mean performance across all tasks
- **Backward Transfer (BWT)**: Performance change on old tasks after learning new ones
- **Forgetting**: Maximum accuracy drop per task

## License

This project is licensed under the MIT License - see the LICENSE file for details.
