# ContLearn - JAX/Equinox Continual Learning Framework

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

