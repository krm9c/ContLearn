# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX/Equinox-based continual learning framework implementing Hamiltonian-based gradient computation with optional AWB (Adaptive Weight Basis) for architecture morphing during lifelong learning.

## Directory Structure

```
ContLearn/
├── .claude/                      # Claude Code project configuration
│   ├── CLAUDE.md                 # This file - project guide
│   ├── profiling_context.md     # Performance optimization reference (READ BEFORE OPTIMIZING)
│   ├── awb_refactoring_context.md
│   ├── comprehensive_context.md
│   └── session_context.md
├── src/cl/                       # Core framework source code
│   ├── arch_search/              # Architecture search modules (MLP, CNN, GCN)
│   ├── config/                   # Configuration parameters and constants
│   │   ├── constants.py          # All default hyperparameters
│   │   └── params.py             # Config loading and defaults
│   ├── core/                     # Core training components (mixins, trainer)
│   │   ├── trainer.py            # Main Trainer class
│   │   ├── losses.py             # LossMixin (DO NOT OPTIMIZE)
│   │   ├── hamiltonian.py        # HamiltonianMixin (DO NOT OPTIMIZE)
│   │   ├── loops.py              # TrainingLoopsMixin
│   │   ├── recording.py          # RecordingMixin
│   │   ├── awb.py                # AWB utilities (DO NOT OPTIMIZE)
│   │   └── profiling.py          # GPU/memory profiling
│   ├── datasets/                 # Dataset implementations
│   │   ├── jax_dataloader.py     # JAX async data pipeline (NEW - for GPU utilization)
│   │   ├── base.py               # Base dataset with experience replay
│   │   ├── sine.py               # Sine wave regression
│   │   ├── mnist.py              # MNIST/Permuted MNIST
│   │   ├── cifar.py              # CIFAR-10/100
│   │   └── synthetic_graph.py    # Graph classification
│   ├── models/                   # Neural network architectures
│   │   ├── mlp.py                # Fully connected networks
│   │   ├── cnn.py                # Convolutional networks (MNIST, CIFAR)
│   │   ├── gcn.py                # Graph convolutional networks
│   │   └── layers.py             # Custom layer implementations
│   └── runners/                  # Problem-specific orchestration
│       ├── generic_runner.py     # Base runner
│       ├── regression.py         # Sine wave regression
│       ├── classification.py     # MNIST/CIFAR classification
│       └── graph_classification.py # Graph classification
├── run_files/                    # Execution scripts (DEPRECATED - moved to kkt_run)
│   └── scripts/                  # Analysis scripts (moved to kkt_run/analysis)
├── kkt_run/                      # KKT cluster experiment infrastructure
│   ├── configs/                  # Production config files
│   │   ├── debug/                # Profiling configs (optimized, fast runs)
│   │   ├── *_condition1_baseline.json      # Condition 1: Baseline (20 configs)
│   │   ├── *_condition2_heuristics.json    # Condition 2: Heuristics
│   │   ├── *_condition3_arch_no_transfer.json # Condition 3: Arch search only
│   │   └── *_condition4_awb_full.json      # Condition 4: Full AWB
│   ├── experiments/              # Experiment management (JLSE cluster)
│   │   ├── slurm/                # SLURM submission scripts
│   │   ├── scripts/              # Execution scripts
│   │   └── results/              # Experiment outputs
│   ├── jlse/                     # JLSE cluster runs
│   │   ├── logs/                 # Training logs
│   │   ├── results/              # Training outputs
│   │   └── run_*.sh              # Per-dataset run scripts
│   ├── kkt/                      # KKT cluster runs
│   │   ├── logs/                 # Training logs (organized by config)
│   │   ├── results/              # Training outputs (organized by config)
│   │   ├── run_parallel.sh       # Run all datasets in parallel
│   │   ├── run_optimized_profiles.sh # Profile optimization performance
│   │   └── submit_all_datasets.sh # Submit all SLURM jobs
│   ├── analysis/                 # Analysis and plotting tools
│   │   ├── scripts/              # Analysis scripts
│   │   └── additional_python_scripts/ # Helper scripts
│   ├── logs/                     # Legacy logs (moved to kkt/logs/)
│   └── results/                  # Legacy results (organized by experiment)
├── tests/                        # Test suite
│   ├── training/                 # Full pipeline training tests (~5 min)
│   │   ├── configs/              # Debug configs (50 samples, 2 epochs)
│   │   └── test_all_configs.py   # Tests for all 10 configs
│   ├── awb_tests/                # AWB-specific tests
│   │   └── configs/              # AWB test configs
│   ├── configs/                  # Shared test configs
│   ├── gpu_reports/              # GPU profiling reports (JSON)
│   ├── test_*.py                 # Unit tests (~30 sec total)
│   └── conftest.py               # Pytest fixtures
├── data/                         # Dataset storage (auto-downloaded)
│   ├── MNIST/                    # MNIST dataset
│   ├── cifar-10-batches-py/      # CIFAR-10 dataset
│   └── cifar-100-python/         # CIFAR-100 dataset
├── docs/                         # Documentation
├── figures/                      # Generated plots
├── outputs/                      # Default output directory
└── run.py                        # Main entry point (root-level)
```

## Commands

### Running Experiments
```bash
# Basic run (run.py is now at project root)
python run.py kkt_run/configs/sine_condition1_baseline.json

# Multiple runs with plots
python run.py kkt_run/configs/sine_condition1_baseline.json --runs 3

# Skip plot generation
python run.py kkt_run/configs/sine_condition1_baseline.json --no-plots

# Custom figures output directory
python run.py kkt_run/configs/sine_condition1_baseline.json --figures-dir outputs/figures

# Run with JAX async prefetching disabled (for debugging)
python run.py kkt_run/configs/mnist_condition1_baseline.json
# Note: use_jax_prefetch=true by default (see profiling_context.md)

# Monitor GPU utilization in real-time
watch -n 0.5 nvidia-smi
```

### KKT Cluster Runs
```bash
# Parallel runs on KKT cluster
cd kkt_run/
./run_parallel_standard.sh    # Run all standard configs in parallel
./run_parallel_awb.sh          # Run all AWB configs in parallel
./run_single.sh <config_name>  # Run single config

# Slurm submission
sbatch submit_kkt.slurm        # Submit standard jobs
sbatch submit_kkt_awb_gpu.slurm # Submit AWB GPU jobs
```

### Testing
```bash
# Using run_tests.sh (recommended)
./run_tests.sh --unit             # Fast unit tests (~30 sec)
./run_tests.sh --training         # Full training tests (~5 min)
./run_tests.sh --all              # All tests (~5-10 min)
./run_tests.sh --fast             # Alias for --unit
./run_tests.sh --models           # Run model tests only
./run_tests.sh --datasets         # Run dataset tests only
./run_tests.sh --layers           # Run layer tests only
./run_tests.sh --losses           # Run loss function tests only
./run_tests.sh --awb              # Run AWB utility tests only
./run_tests.sh --recording        # Run recording tests only
./run_tests.sh --integration      # Run integration tests only
./run_tests.sh --verbose          # Verbose output
./run_tests.sh -k regression      # Run tests matching pattern
./run_tests.sh --all --cov        # Run with coverage report

# Using pytest directly
pytest -m unit                    # Unit tests only
pytest -m training                # Training tests only
pytest tests/test_models.py       # Run specific test file
pytest --cov=src/cl               # Run with coverage
pytest -v --tb=short              # Verbose output
```

## Test Organization

Tests are split into two tiers for efficient development workflow:

**Unit Tests** (`tests/*.py`) - 195 tests, ~30 seconds
- Model architecture tests (`test_models.py`, `test_cnn.py`, `test_graph.py`)
- Layer implementation tests (`test_layers.py`)
- Dataset tests (`test_datasets.py`, `test_mnist.py`)
- Loss and metric tests (`test_losses.py`)
- AWB utility tests (`test_awb.py`)
- Recording tests (`test_recording.py`)
- Component integration tests (`test_integration.py`)

**Training Tests** (`tests/training/`) - 11 tests, ~5 minutes
- Full pipeline tests for all 10 configs (sine, mnist, cifar10, cifar100, synthetic_graph + AWB variants)
- Test configs in `tests/training/configs/` with debug settings baked in (50 samples, 2 epochs)
- Validates end-to-end training workflow
- Outputs logged to `SCRIPT_TEST_RESULTS.md`

### Pytest Markers

Tests are marked for filtering:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.training` - Slow training tests
- `@pytest.mark.scripts` - Legacy alias for training

Filter with: `pytest -m unit` or `pytest -m training`

### Development
```bash
# Install in editable mode
pip install -e ".[dev]"

# Format code
black src/ tests/ run_files/scripts/
isort src/ tests/ run_files/scripts/

# Type checking
mypy src/
```

### Plotting and Analysis
```bash
# Generate plots from saved results
python run_files/scripts/plot_results.py <records_file.pkl> --output-dir figures

# Compare multiple runs
python run_files/scripts/compare_runs.py <results_dir>

# Profile training performance
python run_files/scripts/profile_training.py kkt_run/configs/sine.json
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
- `graph_classification.py`: Synthetic graph classification (GCN)

All runners support AWB pipeline with architecture search.

## Configuration

JSON config files in `kkt_run/configs/` control all hyperparameters. Available configs:
- `sine.json`, `sine_awb.json` - Sine wave regression
- `mnist.json`, `mnist_awb.json` - MNIST digit classification
- `cifar10.json`, `cifar10_awb.json` - CIFAR-10 classification
- `cifar100.json`, `cifar100_awb.json` - CIFAR-100 classification
- `synthetic_graph.json`, `synthetic_graph_awb.json` - Graph classification

Key config fields:
```json
{
    "prob": "regression|classification",
    "problem": "vectors|graph",
    "data": "sine|mnist|permuted_mnist|cifar10|cifar100|synthetic",
    "network": "fcnn|cnn|gcn",
    "awb_enabled": false,
    "grad_weights": [0.01, 0.98, 0.1],
    "lr_schedule": "constant|step|exponential|cosine|linear",
    "optimizer": "adam|adamw|sgd|rmsprop",
    "flag": [1.0, 1.0],
    "debug_mode": false,
    "debug_limit": 100
}
```

AWB-specific config fields prefixed with `awb_` (see `src/cl/config/constants.py` for full documentation).

**Test configs** in `tests/training/configs/` have debug settings (50 samples, 2 epochs) for fast validation.

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

## Plot Generation

After training, the framework automatically generates four types of plots:

1. **Losses** (`*_losses.png`): All loss components (H, V, dV, dV/dx, dV/dtheta, gradient norm)
2. **Metrics** (`*_metrics.png`): Train and test metrics over time
3. **Eigenvalues** (`*_eigenvalues.png`):
   - Standard mode: Weight matrix eigenvalues
   - AWB mode: A and B matrix eigenvalues
4. **Overview** (`*_overview.png`): Combined visualization of all metrics

Plots can also be generated manually:
```bash
python run_files/scripts/plot_results.py <records_file.pkl> --output-dir figures
```

## Key Implementation Details

### AWB 5-Step Algorithm Details

**Task 0**: Standard Hamiltonian CL training (no AWB)

**Tasks 1+** (when `awb_enabled: true`):

1. **STEP 1 - Preliminary Training** (`awb_preliminary_epochs`)
   - Train on new task with current architecture
   - Record preliminary loss for decision making

2. **STEP 2 - Architecture Change Decision**
   - Check if `loss_ratio > threshold` AND `loss increased`
   - If YES: proceed to architecture search (Steps 3-5)
   - If NO: continue standard training

3. **STEP 3a - Architecture Search**
   - Search for optimal hidden layer dimensions
   - Uses `src/cl/arch_search/` modules (MLP, CNN, GCN)
   - Evaluates candidate architectures on validation data

4. **STEP 3b - A/B Matrix Training** (`awb_ab_training_epochs`)
   - Initialize A/B matrices for new architecture
   - Freeze W (old weights), train A/B with `notABTrain=False`
   - A/B learn to transform old features to new architecture

5. **STEP 4 - Weight Transformation**
   - Compute `V = A @ W @ B.T`
   - V becomes new weight matrix in expanded architecture

6. **STEP 5 - Final Training** (remaining epochs)
   - Freeze A/B matrices, train V with `notABTrain=True`
   - V now trainable in new architecture space

**Key AWB functions** (`src/cl/core/awb.py`):
- `should_change_arch()`: Decision logic
- `set_new_AB_matrices()`: Initialize A/B
- `compute_V_from_AWB()`: V = A @ W @ B.T
- `partition_for_AB_training()`: Freeze W, train A/B
- `partition_for_standard_training()`: Freeze A/B, train V

### Loss Components Explained

During training, multiple loss values are recorded:
- **H**: Total Hamiltonian = V + dV
- **V**: Experience replay loss (loss on past data)
- **dV**: Regularization term (change in loss due to perturbations)
- **dV/dx**: Sensitivity to input perturbations
- **dV/dtheta**: Sensitivity to parameter perturbations
- **grad_norm**: L2 norm of total gradient

## Source Code Organization

### Core Framework (`src/cl/`)

**`core/`** - Training pipeline components
- `trainer.py`: Main Trainer class combining all mixins
- `losses.py`: LossMixin - loss and metric computation
- `hamiltonian.py`: HamiltonianMixin - Hamiltonian gradient computation
- `loops.py`: TrainingLoopsMixin - unified training loop
- `recording.py`: RecordingMixin - metric recording with eigenvalues
- `awb.py`: AWB utility functions (partitioning, matrix ops)
- `arch_search.py`: Architecture search orchestration

**`models/`** - Neural network architectures
- `mlp.py`: Fully connected networks for regression
- `cnn.py`: Convolutional networks for image classification
- `gcn.py`: Graph convolutional networks
- `layers.py`: Custom layer implementations (Linear, Conv, etc.)

**`datasets/`** - Data loading and preprocessing
- `base.py`: Base dataset class with experience replay
- `sine.py`: Sine wave regression dataset
- `mnist.py`: MNIST digit classification
- `cifar.py`: CIFAR-10/100 image classification
- `synthetic_graph.py`: Synthetic graph classification

**`runners/`** - Problem-specific orchestration
- `generic_runner.py`: Base runner with common training logic
- `regression.py`: Sine wave regression runner
- `classification.py`: MNIST/CIFAR classification runner
- `graph_classification.py`: Graph classification runner

**`arch_search/`** - Architecture search implementations
- `mlp_search.py`: Search for optimal MLP dimensions
- `cnn_search.py`: Search for optimal CNN dimensions
- `gcn_search.py`: Search for optimal GCN dimensions

**`config/`** - Configuration management
- `constants.py`: Default hyperparameters and AWB settings
- `params.py`: Config parsing and validation

### Execution Scripts (`run_files/scripts/`)

- `run.py`: Main training script
- `plot_results.py`: Plot generation from saved results
- `compare_runs.py`: Multi-run comparison and analysis
- `profile_training.py`: GPU/CPU profiling utilities
- `run_*.sh`: Convenience scripts for each config