# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

JAX/Equinox continual learning framework with Hamiltonian gradients and AWB (Adaptive Weight Basis).

**🚨 IMPORTANT**: Read `.claude/profiling_context.md` before performance optimizations.

---

## Directory Structure

```
ContLearn/
├── .claude/                  # Project configuration
│   ├── CLAUDE.md             # This file
│   └── profiling_context.md # Performance optimization guide (READ FIRST)
├── src/cl/                   # Core source code
│   ├── arch_search/          # Architecture search (MLP, CNN, GCN)
│   ├── config/               # constants.py, params.py
│   ├── core/                 # Trainer mixins (losses, hamiltonian, loops, recording, awb, profiling)
│   ├── datasets/             # jax_dataloader.py (NEW), base.py, sine.py, mnist.py, cifar.py, synthetic_graph.py
│   ├── models/               # mlp.py, cnn.py, gcn.py, layers.py
│   └── runners/              # generic_runner.py (unified), legacy: regression.py, classification.py, graph_classification.py
├── kkt_run/                  # Cluster experiments
│   ├── configs/              # 20 production configs (5 datasets × 4 conditions)
│   │   └── debug/            # Profiling configs (fast validation)
│   ├── kkt/                  # KKT cluster (logs/, results/, run scripts)
│   ├── jlse/                 # JLSE cluster (logs/, results/, run scripts)
│   ├── experiments/          # SLURM scripts and analysis
│   └── analysis/             # Plotting and analysis tools
├── tests/                    # Test suite
│   ├── training/configs/     # Debug configs (50 samples, 2 epochs)
│   ├── awb_tests/configs/    # AWB-specific tests
│   ├── gpu_reports/          # Profiling reports
│   ├── test_*.py             # Unit tests (~30 sec)
│   └── conftest.py           # Pytest fixtures
├── data/                     # MNIST, CIFAR-10, CIFAR-100 (auto-downloaded)
└── run.py                    # Main entry point
```

**DO NOT OPTIMIZE** (breaks accuracy): `losses.py`, `hamiltonian.py`, `awb.py`

---

## Quick Start

```bash
# Run experiment
python run.py kkt_run/configs/sine_condition1_baseline.json

# Run experiment with multiple runs and plots
python run.py kkt_run/configs/sine_condition1_baseline.json --runs 3

# Monitor GPU
watch -n 0.5 nvidia-smi

# Run tests (using pytest directly)
pytest -m unit -v           # Fast unit tests (~30 sec, ~216 tests)
pytest -m training -v       # Full pipeline tests (~5 min, ~11 tests)
pytest tests/ -v            # All tests
pytest tests/test_awb.py -v # Specific test file
pytest -k "mlp" -v          # Pattern matching

# AWB-specific tests
cd tests/awb_tests && ./run_all_tests.sh

# KKT cluster (parallel execution)
cd kkt_run/kkt
./run_parallel.sh                  # All datasets in parallel
./run_optimized_profiles.sh        # Profile optimizations
```

---

## Experiment Configs

**5 Datasets**: sine, mnist, cifar10, cifar100, synthetic_graph
**4 Conditions** (20 total configs):
1. **Baseline**: Fixed arch, constant LR, no warmup
2. **Heuristics**: Task warmup, adaptive LR/grad weights
3. **Arch search only**: Architecture search, no transfer
4. **AWB full**: Architecture search + A/B transfer

Config naming: `{dataset}_condition{1-4}_{baseline|heuristics|arch_no_transfer|awb_full}.json`

---

## Core Architecture

### Entry Point Flow
1. `run.py` → loads config, calls `train_model()` from `runners/generic_runner.py`
2. `generic_runner.py` → creates dataset, model, optimizer, and Trainer instance
3. Trainer runs task loop → either standard CL or AWB pipeline (via `awb_pipeline.py`)

### Trainer (Mixin-based - see `src/cl/core/trainer.py`)
The Trainer class inherits from 4 mixins, each in its own file:
- **LossMixin** (`losses.py`): MSE, cross-entropy, metrics - **DO NOT OPTIMIZE**
- **HamiltonianMixin** (`hamiltonian.py`): `grad = alpha*current + beta*experience + gamma*dV` - **DO NOT OPTIMIZE**
- **TrainingLoopsMixin** (`loops.py`): Unified training loop (vectors/graphs)
- **RecordingMixin** (`recording.py`): Metrics, eigenvalues, checkpointing

### AWB Pipeline (5 steps when `awb_enabled: true`)
1. Preliminary training → 2. Decide arch change → 3. Search + train A/B matrices → 4. Compute V=A@W@B.T → 5. Train V

### Model Architecture Pattern
All models (MLP, CNN, CNN3D, GCN) follow the same pattern:
- Defined as Equinox modules (`eqx.Module`)
- Each has an `AWBOps` class (e.g., `MLPAWBOps`, `CNNAWBOps`) for A/B matrix operations
- AWB matrices (A, B) are stored as PyTree leaves, enabling layer-level weight transfer
- Models support both standard training and AWB pipeline via partition functions

### Datasets
- All implement: `generate_dataset(task_id, batch_size, phase)` → `(current_loader, experience_loader)`
- Experience replay: `append_to_experience(task_id)`
- JAX async prefetch: `use_jax_prefetch=true` (default, see profiling_context.md)
- Base class: `BaseDataset` in `datasets/base.py`

---

## Key Config Parameters

```json
{
  "data": "sine|mnist|cifar10|cifar100|synthetic",
  "network": "fcnn|cnn|gcn",
  "awb_enabled": false,
  "grad_weights": [0.3, 0.6, 0.1],  // [current, experience, regularization]
  "lr_schedule": "constant|step|exponential|cosine",
  "optimizer": "adam|adamw|sgd|rmsprop",
  "use_jax_prefetch": true,  // JAX async data loading (NEW)
  "prefetch_size": 3,        // Batches to prefetch
  "eval_interval": 50,       // Test eval frequency (vs log_interval=1)
  "debug_mode": false,
  "debug_limit": 100
}
```

Defaults: `src/cl/config/constants.py`

---

## Testing

```bash
# Run by marker
pytest -m unit -v              # Fast unit tests (~216 tests, ~30 sec)
pytest -m training -v          # Full pipeline tests (~11 tests, ~5 min)

# Run specific test files
pytest tests/test_models.py -v       # Model architecture tests
pytest tests/test_datasets.py -v     # Dataset loading tests
pytest tests/test_awb.py -v          # AWB utility tests
pytest tests/test_recording.py -v    # Recording mixin tests
pytest tests/test_lr_schedules.py -v # LR schedules and adaptive features

# Run AWB pipeline tests (5-step verification)
pytest tests/awb_tests/ -v
cd tests/awb_tests && ./run_all_tests.sh

# Pattern matching
pytest -k "mlp" -v             # All MLP-related tests
pytest -k "awb" -v             # All AWB-related tests

# With coverage
pytest tests/ --cov=src/cl --cov-report=term-missing
```

**Test configs**: `tests/configs/` and `tests/training/configs/` (debug_mode=true, 50 samples, 2 epochs)
**Fixtures**: See `tests/conftest.py` for config loading fixtures (e.g., `test_sine_config`, `test_mnist_awb_config`)

---

## Development

```bash
pip install -e ".[dev]"
black src/ tests/
isort src/ tests/
mypy src/
```

---

## Loss Components

- **H**: Total Hamiltonian (V + dV)
- **V**: Experience replay loss
- **dV**: Regularization (∂V/∂x, ∂V/∂θ)
- **grad_norm**: L2 norm of gradient

---

## Plot Generation

Auto-generated after training:
1. `*_losses.png` - All loss components
2. `*_metrics.png` - Train/test metrics
3. `*_eigenvalues.png` - Weight/A/B eigenvalues
4. `*_overview.png` - Combined visualization

---

## References

- **Architecture details**: See inline comments in `src/cl/core/`
- **AWB algorithm**: `src/cl/core/awb.py`
- **Performance optimization**: `.claude/profiling_context.md` (READ BEFORE OPTIMIZING)
- **Config defaults**: `src/cl/config/constants.py`
