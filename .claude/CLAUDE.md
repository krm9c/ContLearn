# CLAUDE.md

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
│   └── runners/              # regression.py, classification.py, graph_classification.py
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

# Monitor GPU
watch -n 0.5 nvidia-smi

# Run tests
./run_tests.sh --unit      # Fast (~30 sec)
./run_tests.sh --training  # Full pipeline (~5 min)
./run_tests.sh --all       # Everything

# KKT cluster
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

### Trainer (Mixin-based)
- **LossMixin**: MSE, cross-entropy, metrics
- **HamiltonianMixin**: `grad = alpha*current + beta*experience + gamma*dV`
- **TrainingLoopsMixin**: Unified loop (vectors/graphs)
- **RecordingMixin**: Metrics, eigenvalues, checkpointing

### AWB Pipeline (5 steps when `awb_enabled: true`)
1. Preliminary training → 2. Decide arch change → 3. Search + train A/B matrices → 4. Compute V=A@W@B.T → 5. Train V

### Datasets
- All implement: `generate_dataset(task_id, batch_size, phase)` → `(current_loader, experience_loader)`
- Experience replay: `append_to_experience(task_id)`
- JAX async prefetch: `use_jax_prefetch=true` (default, see profiling_context.md)

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
# Pytest markers
pytest -m unit      # Fast unit tests
pytest -m training  # Full pipeline tests

# Specific suites
./run_tests.sh --models
./run_tests.sh --datasets
./run_tests.sh --awb
./run_tests.sh --recording

# With coverage
./run_tests.sh --all --cov
```

**Test configs**: `tests/training/configs/` (50 samples, 2 epochs for speed)

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
