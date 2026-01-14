# ContLearn - Continual Learning with Hamiltonian Gradients

JAX/Equinox framework for continual learning with Hamiltonian gradient computation and Adaptive Weight Basis (AWB) transfer.

**Branch**: `profiling` - GPU profiling and performance optimization

---

## Quick Start

```bash
# Install
pip install -e .

# Run experiment
python run.py runs__/configs/mnist_condition1_baseline.json

# Run with multiple seeds
python run.py runs__/configs/mnist_condition1_baseline.json --runs 3

# Run tests
pytest -m unit -v           # Fast (~30 sec)
pytest -m training -v       # Full pipeline (~5 min)

# Monitor GPU
watch -n 0.5 nvidia-smi
```

---

## Source Code Structure

```
src/cl/
├── core/                        # Training infrastructure
│   ├── trainer.py               # Main Trainer class (mixin-based)
│   ├── losses.py                # Loss functions (MSE, cross-entropy)
│   ├── hamiltonian.py           # Hamiltonian gradient: α*current + β*experience + γ*dV
│   ├── loops.py                 # Training loops (vectors, graphs)
│   ├── recording.py             # Metrics, checkpointing, plotting
│   ├── awb.py                   # AWB utilities and operations
│   ├── awb_pipeline.py          # 7-phase AWB training pipeline
│   ├── arch_search.py           # Architecture search
│   └── profiling.py             # GPU profiling, XLA flags, timing
│
├── models/                      # Neural network architectures
│   ├── mlp.py                   # Multi-layer perceptron + MLPAWBOps
│   ├── cnn.py                   # CNN/CNN3D + CNNAWBOps
│   ├── gcn.py                   # Graph convolutional network + GCNAWBOps
│   └── layers.py                # Shared layer utilities
│
├── datasets/                    # Data loading
│   ├── base.py                  # BaseDataset class
│   ├── sine.py                  # Sine wave regression
│   ├── mnist.py                 # MNIST classification
│   ├── cifar.py                 # CIFAR-10/100 classification
│   ├── synthetic_graph.py       # Graph classification
│   └── jax_dataloader.py        # JAX async prefetch data loader
│
├── config/                      # Configuration
│   ├── constants.py             # Default values, XLA flags
│   └── params.py                # Config loading and validation
│
├── runners/                     # Entry points
│   └── generic_runner.py        # Unified runner for all datasets
│
└── arch_search/                 # Architecture search modules
    ├── mlp_search.py
    ├── cnn_search.py
    └── gcn_search.py
```

### Core Components

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `losses.py` | Loss computation | **NO** - breaks accuracy |
| `hamiltonian.py` | Gradient computation | **NO** - breaks accuracy |
| `awb.py` | Weight transformations | **NO** - breaks accuracy |
| `loops.py` | Training loops | Yes - I/O, logging |
| `recording.py` | Metrics, checkpoints | Yes - logging only |
| `profiling.py` | GPU profiling | Yes |

---

## Profiling Infrastructure

### Built-in Profiling (`src/cl/core/profiling.py`)

```python
# Enable XLA optimization (BEFORE importing JAX)
from cl.core.profiling import set_xla_flags, configure_jax_for_gpu
set_xla_flags(enable=True, verbose=True)

import jax
configure_jax_for_gpu(verbose=True)

# Use timing decorators
from cl.core.profiling import profile, timed_section

@profile("Training Step")
def train_step():
    ...

with timed_section("Forward Pass"):
    ...
```

### External Profiling Toolkit (`jax-profiler/`)

```bash
# Quick GPU monitoring
python jax-profiler/run_awb_profile.py --quick

# Full AWB profiling
python jax-profiler/run_awb_profile.py --config runs__/configs/mnist_condition4_awb_full.json

# Output: JSON report with GPU utilization, timing breakdown
```

### Profiling Configs

| Config | Purpose |
|--------|---------|
| `mnist_condition1_profiling.json` | Baseline profiling |
| `mnist_condition4_profiling.json` | AWB profiling |
| `mnist_condition1_profiling_quick.json` | Fast validation |

### SLURM Profiling (KKT Cluster)

```bash
# Submit H200 benchmark
cd ContLearn
sbatch runs__/experiments/profiling/submit_profiling_benchmark.slurm

# Runs Condition 1 and 4 in parallel on 2 GPUs
# Results: runs__/profiling/results/
```

---

## `runs__/` Folder Structure

The `runs__/` directory contains all experiment configurations, SLURM scripts, and cluster-specific files.

```
runs__/
├── configs/                     # Experiment configurations (24 total)
│   ├── mnist_condition1_baseline.json
│   ├── mnist_condition2_heuristics.json
│   ├── mnist_condition3_arch_no_transfer.json
│   ├── mnist_condition4_awb_full.json
│   ├── mnist_condition1_profiling.json      # Profiling configs
│   ├── mnist_condition4_profiling.json
│   ├── sine_condition{1-4}_*.json           # Sine experiments
│   ├── sine_noise_condition{1-4}_*.json     # Noisy sine
│   └── synthetic_graph_*_condition{1-4}_*.json  # Graph experiments
│
├── experiments/                 # SLURM scripts by dataset
│   ├── mnist/
│   │   ├── submit_mnist_multiple_runs.slurm
│   │   ├── run_parallel_multiple_runs.sh
│   │   └── run_single.sh
│   ├── sine/
│   ├── sine_noise/
│   ├── synthetic_graph/
│   ├── profiling/               # Profiling experiments
│   │   └── submit_profiling_benchmark.slurm  # H200 benchmark
│   └── analysis/
│       └── compute_metrics.py
│
├── kkt/                         # KKT cluster scripts
│   ├── submit_sine_mnist_cond12.slurm
│   ├── submit_sine_mnist_cond34.slurm
│   ├── submit_synthetic_graph_cond12.slurm
│   ├── submit_synthetic_graph_cond34.slurm
│   ├── submit_cifar100.slurm
│   ├── run_parallel.sh
│   └── run_single.sh
│
├── profiling/                   # Profiling results
│   └── results/                 # Output from profiling runs
│
├── analysis/                    # Analysis and plotting tools
│
└── tests/                       # Test configurations
```

### Config Naming Convention

```
{dataset}_condition{1-4}_{variant}.json

Datasets: mnist, sine, sine_noise, synthetic_graph
Conditions:
  1 = baseline (fixed arch, constant LR)
  2 = heuristics (warmup, adaptive LR)
  3 = arch_no_transfer (arch search, no AWB)
  4 = awb_full (arch search + AWB transfer)
```

### Running Experiments

```bash
# Single experiment
python run.py runs__/configs/mnist_condition1_baseline.json

# Multiple runs with plots
python run.py runs__/configs/mnist_condition1_baseline.json --runs 5

# KKT cluster - all conditions
cd runs__/kkt
sbatch submit_sine_mnist_cond12.slurm  # Conditions 1&2
sbatch submit_sine_mnist_cond34.slurm  # Conditions 3&4

# Parallel execution (local)
bash runs__/kkt/run_parallel.sh
```

---

## AWB Pipeline (7 Phases)

When `awb_enabled: true`, training follows this pipeline:

| Phase | Description | Compute Cost |
|-------|-------------|--------------|
| 1. preliminary | Initial training | Low |
| 2. arch_decision | Compare losses | Low |
| 3. arch_search | Evaluate architectures | Medium |
| 4. ab_training | Train A/B matrices | **HIGH** (bottleneck) |
| 5. v_transform | Compute V = A @ W @ B.T | Low |
| 6. v_warmup | Warmup with new weights | Low |
| 7. v_training | Final training | Low |

**Key insight**: Phase 4 (ab_training) causes 58x slowdown because `A @ W @ B.T` is computed inside gradient.

---

## Benchmarks

### MNIST (A40 GPU, batch_size=1024)

| Condition | Time | GPU Util | Throughput |
|-----------|------|----------|------------|
| 1 (Baseline) | 7.9s | 71.4% | 9,076 samples/sec |
| 4 (AWB Full) | 460s | 5.5% | ~170 samples/sec |

### Full AWB Run (10 tasks, 200 epochs)
- Total time: 2.9 hours
- GPU utilization: 67.8% mean
- Memory: 34.4 GB

---

## Key Config Parameters

```json
{
  "data": "mnist",
  "network": "cnn",
  "n_task": 10,
  "epochs_per_task": 200,
  "batch_size": 1024,

  "awb_enabled": true,
  "awb_preliminary_epochs": 1,
  "awb_ab_training_epochs": 200,

  "grad_weights": [0.4, 0.4, 0.1],
  "lr_schedule": "cosine",
  "lr": 0.0001,

  "use_jax_prefetch": true,
  "prefetch_size": 3,
  "eval_interval": 50,

  "profiling_enabled": false,
  "detailed_profiling": false
}
```

---

## Testing

```bash
# Unit tests (fast)
pytest -m unit -v

# Training pipeline tests
pytest -m training -v

# Specific test file
pytest tests/test_awb.py -v

# Pattern matching
pytest -k "mnist" -v

# With coverage
pytest tests/ --cov=src/cl --cov-report=term-missing
```

---

## Claude Context Files

New Claude sessions should start with `START_HERE.md`, which points to:

| File | Purpose |
|------|---------|
| `.claude/CLAUDE.md` | Main project guide |
| `.claude/profiling_context.md` | Performance optimization |
| `jax-profiler/.claude/CLAUDE.md` | Profiling toolkit |

---

## References

- JAX: https://jax.readthedocs.io/
- Equinox: https://docs.kidger.site/equinox/
- JAX Profiling: https://jax.readthedocs.io/en/latest/profiling.html
