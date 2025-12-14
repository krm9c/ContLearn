# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX/Equinox-based continual learning framework implementing Hamiltonian-based gradient computation with optional AWB (Adaptive Weight Basis) for architecture morphing during lifelong learning.

## Commands

### Running Experiments
```bash
# Basic run
python scripts/run.py config/sine.json

# Multiple runs with plots
python scripts/run.py config/sine.json --runs 3

# Skip plot generation
python scripts/run.py config/sine.json --no-plots

# Custom figures output directory
python scripts/run.py config/sine.json --figures-dir outputs/figures
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src/cl

# Verbose output
pytest -v --tb=short
```

### Development
```bash
# Install in editable mode
pip install -e ".[dev]"

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/
```

## Architecture

### Core Training Pipeline

The framework uses a mixin-based `Trainer` class (`src/cl/core/trainer.py`) that combines:
- **LossMixin** (`losses.py`): Loss and metric computation (MSE, cross-entropy)
- **HamiltonianMixin** (`hamiltonian.py`): Hamiltonian-based gradient computation
- **TrainingLoopsMixin** (`loops.py`): Unified training loop for all problem types
- **RecordingMixin** (`recording.py`): Metric recording with eigenvalue tracking

### Hamiltonian Gradient Computation

The core CL algorithm computes gradients as a weighted combination:
```
grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
```
Where:
- `delta_theta`: Current task loss gradient
- `grad_V`: Experience replay gradient
- `grad_dV`: Regularization term (change in loss due to perturbations)

Configurable via `grad_weights: [alpha, beta, gamma]` in config JSON.

### AWB (Adaptive Weight Basis) Pipeline

When `awb_enabled: true`, tasks 1+ follow a 5-step algorithm:
1. **STEP 1**: Preliminary training on new task
2. **STEP 2**: Decide if architecture change needed (loss ratio thresholds)
3. **STEP 3a**: Architecture search for optimal dimensions
4. **STEP 3b**: Train A/B matrices with W frozen (notABTrain=False)
5. **STEP 4**: Compute V = A @ W @ B.T (weight transformation)
6. **STEP 5**: Train V with A/B frozen (notABTrain=True)

AWB utilities in `src/cl/core/awb.py` handle matrix operations and model partitioning.

### Model Partitioning Pattern

Equinox models are partitioned into trainable/frozen components:
```python
params, static = eqx.partition(model, eqx.is_array)
# For AWB: move A/B matrices to static (frozen)
static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))
```

### Runners

Problem-specific orchestration in `src/cl/runners/`:
- `regression.py`: Sine wave regression (MLP)
- `classification.py`: MNIST/CIFAR classification (CNN, CNN3D)

Both support AWB pipeline with architecture search.

## Configuration

JSON config files in `config/` control all hyperparameters. Key fields:

```json
{
    "prob": "regression|classification",
    "data": "sine|mnist|permuted_mnist|cifar10|cifar100",
    "network": "fcnn|cnn",
    "awb_enabled": false,
    "grad_weights": [0.01, 0.98, 0.1],
    "lr_schedule": "constant|step|exponential|cosine|linear",
    "flag": [1.0, 1.0]
}
```

AWB-specific config fields prefixed with `awb_` (see `src/cl/config/constants.py` for defaults).

## Code Patterns

### Dataset Interface
All datasets implement `generate_dataset(task_id, batch_size, phase)` returning `(current_loader, experience_loader)` tuples. Experience replay managed via `append_to_experience(task_id)`.

### Training Loop Signature
```python
params, static, opt_state, record_dict = trainer.train__CL(
    train__=(trainloader, exploader, valloader, testloader),
    params=params, static=static, opt_state=opt_state, optim=optim,
    n_iter=epochs, task_id=i, config=config, record_dict=record_dict,
    notABTrain=True,  # False for AWB A/B training
    problem_type='vectors',  # or 'graph'
    loss_type='regression'   # or 'classification'
)
```

### Test Fixtures
`tests/conftest.py` provides fixtures for configs, dummy batches, and model sizes. Use `debug_mode: true` and `debug_limit: N` in configs for fast testing.