# ContLearn - JAX/Equinox Continual Learning Framework

## ⚠️ RECENT CHANGES (Dec 24, 2025)

**AWB Pipeline Refactored** - See `.claude/awb_refactoring_context.md` for full details.

**What Changed:**
- AWB logic consolidated from 4 files into strategy pattern implementation
- New files: `awb_operations.py`, `awb_pipeline.py`
- `generic_runner.py`: 1523 → 1083 lines (-440 lines)
- Deprecated: `classification.py`, `regression.py`, `graph_classification.py` (→ `.deprecated`)
- **Only `generic_runner.py` is used** - confirmed by user

## 🔧 DEBUGGING SESSION (Dec 25, 2025)

**Issues Fixed:**

1. **AWB Import Errors** - Fixed 3 incorrect imports in `awb_pipeline.py`:
   - `from ..utils.optimizer import create_optimizer` → `from ..runners.generic_runner import create_optimizer`

2. **AWB Decision Logic** - Simplified `should_change_arch()` in `awb.py`:
   - Removed `min_delta` requirement (was blocking architecture changes when loss decreased)
   - Now uses only ratio threshold: `return ratio > threshold_high`

3. **AWB Metadata Tracking** - Added task metadata:
   - `awb_pipeline.py`: Store preliminary_loss, architecture_changed, change_reason, etc.
   - `generic_runner.py`: Extract architecture history from AWB tasks

4. **Model Initializations** - Fixed `generic_runner.py` to match old working code:
   - **CNN (MNIST)**: Calculate flatten_size from conv/pool output, use `input_size=28` (not 784), `filter_size=4` (integer not list)
   - **CNN3D (CIFAR)**: Calculate flatten_size from two conv/pool layers
   - **GCN (Graphs)**: Get node_num from sample batch, use correct parameter names

**Testing Status:**
- ✅ Standard CL + MLP (sine regression) - WORKING
- ✅ Standard CL + CNN (MNIST) - WORKING
- ⚠️ AWB architecture search - needs batch format debugging (experience replay batches during search)

**Key Lesson:** Always check deprecated/old working code before adding new logic - refactoring introduced several parameter mismatches.

**Quick Test:**
```bash
# Standard CL (working)
python run_files/scripts/run.py kkt_run/configs/sine.json

# AWB (in progress)
./kkt_run/test_awb_forced.sh
```

## Directory Structure

```
ContLearn/
├── src/cl/              # Core framework source code
│   ├── arch_search/     # Architecture search modules (MLP, CNN, GCN)
│   ├── config/          # Configuration parameters and constants
│   ├── core/            # Core training components (mixins, trainer)
│   ├── datasets/        # Dataset implementations
│   ├── models/          # Neural network architectures
│   └── runners/         # Problem-specific orchestration
├── run_files/           # Execution scripts and utilities
│   └── scripts/         # Main execution scripts (run.py, plot_results.py, etc.)
├── kkt_run/             # KKT cluster-specific runs
│   ├── configs/         # Production config files (.json)
│   ├── logs/            # Training logs
│   ├── results/         # Training outputs
│   └── *.sh             # Slurm/parallel execution scripts
├── tests/               # Test suite
│   ├── training/        # Full pipeline training tests
│   │   └── configs/     # Debug configs for testing
│   ├── awb_tests/       # AWB-specific tests
│   └── *.py             # Unit tests
├── data/                # Dataset storage (MNIST, CIFAR, etc.)
└── docs/                # Documentation
```

## Core Components

- **Trainer**: Mixin-based class combining Loss, Hamiltonian, TrainingLoops, Recording
- **Hamiltonian**: Gradient computation using weighted combination of task/experience/regularization gradients
- **AWB (Adaptive Weight Basis)**: 5-step algorithm for architecture morphing during training
- **Datasets**: Sine regression, MNIST, CIFAR-10/100, synthetic graphs
- **Models**: MLP, CNN, CNN3D, GCN with optional AWB support

## Quick Commands

```bash
# Run experiments
python run_files/scripts/run.py kkt_run/configs/sine.json

# Run tests
./run_tests.sh --unit      # Fast unit tests (~30 sec)
./run_tests.sh --training  # Full training tests (~5 min)
./run_tests.sh --all       # All tests

# Using pytest directly
pytest -m unit
pytest -m training
```

## Key Files

- `src/cl/core/trainer.py` - Main trainer class
- `src/cl/core/hamiltonian.py` - Hamiltonian gradient computation
- `src/cl/core/loops.py` - Training loops
- `src/cl/core/awb.py` - AWB utilities
- `src/cl/models/mlp.py` - MLP architecture
- `src/cl/models/cnn.py` - CNN architectures
- `src/cl/runners/` - Problem-specific runners

