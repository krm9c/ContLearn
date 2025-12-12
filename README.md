# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContLearn is a JAX-based framework for continual learning research. It implements universal continual learning methods supporting multiple neural network architectures (MLP, CNN, GCN, GAT) with optional AWB (Adaptive Weight Basis) transformations for architecture search during lifelong learning.

## Commands

### Running Training
```bash
# Run via script.sh (recommended - edit to select dataset)
bash script.sh

# Direct execution with JSON config
python run.py train <num_runs> "<config_file>.json"

# Examples:
python run.py train 1 "param_sine.json"           # Sine regression
python run.py train 1 "paramgraph_enzymes.json"   # Graph classification (ENZYMES)
python run.py train 5 "paramgraph_mutag.json"     # Graph classification (MUTAG)
python run.py train 1 "paramomni10.json"          # MNIST classification
python run.py train 1 "param_cifar10.json"        # CIFAR-10
python run.py train 1 "param_permuted_mnist.json" # Permuted MNIST
```

Configuration files are stored in [config/jsons/](config/jsons/) directory.

### Running Tests
```bash
# Using test runner scripts (recommended)
./run_tests.sh --all              # Run all tests (bash)
python run_tests.py --all         # Run all tests (python)

# Quick test runs
./run_tests.sh --fast             # Skip slow integration tests
./run_tests.sh --models           # Run model tests only

# Using pytest directly
pytest tests/                     # Run all tests
pytest tests/test_models.py       # Run specific test file
pytest tests/ -k regression       # Run tests matching pattern

# With coverage
./run_tests.sh --all --cov        # Coverage report
./run_tests.sh --all --cov-html   # HTML coverage report

# Quick validation across all datasets (uses minimal epochs)
bash test_datasets.sh

# See TESTING.md for comprehensive testing guide
```

## Architecture

### Core Components

**[run.py](run.py)** - Main entry point:
- Argument parsing for training runs
- Dispatches to problem-specific training functions based on `prob` field

**[config/params.py](config/params.py)** - Configuration management:
- `Params` class for loading/saving JSON configurations
- Provides dict-like access to hyperparameters

**[training/](training/)** - Training orchestration:
- [training/runners.py](training/runners.py): `train_model_graph`, `train_model_reg`, `train_model_class`
- [training/checkpoint.py](training/checkpoint.py): `load_checkpoint` - initializes model, optimizer, trainer, and dataset

**[data/loaders.py](data/loaders.py)** - Data loading:
- `load_return_dataset`: Creates dataset objects for regression/classification
- `continuum_Graph_classification`: Splits graph datasets into continual learning tasks

**[utils/model.py](utils/model.py)** - Neural network architectures using Equinox:
- `MLP` / `MLPorig` - Feedforward networks with optional AWB transformation via `getAWB()`
- `CNN` / `CNN3D` - Convolutional networks (CNN for MNIST/Omniglot, CNN3D for CIFAR)
- `myNN` - Combined GCN + MLP for graph classification with `get_AWBT()` support
- `SingleHeadGAT`, `MultiHeadGAT` - Graph Attention Networks
- `Linear`, `Linear2`, `Linear3` - Custom linear layers with different bias shapes

**[utils/trainer.py](utils/trainer.py)** - Training logic:
- `Trainer` class with loss functions for vectors and graphs
- JIT-compiled loss/metric functions (`loss_fn_class`, `loss_fn_mse`, `accuracy_graphs`)
- Training methods: `train__CL__graph`, `train__CL__reg`, `train__CL__class`
- TensorBoard integration via `SummaryWriter`

**[utils/data.py](utils/data.py)** - Data handling:
- `data_return` class managing datasets (MNIST, Omniglot, CIFAR, sine, synthetic)
- Experience replay buffer management (`append_to_experience`)
- `Continual_Dataset` for PyTorch DataLoader compatibility

**[arch_search/](arch_search/)** - Architecture search modules:
- [arch_search/mlp_search.py](arch_search/mlp_search.py): MLP architecture search
- [arch_search/cnn_search.py](arch_search/cnn_search.py): CNN architecture search
- [arch_search/gcn_search.py](arch_search/gcn_search.py): GCN architecture search with `arch_search_GCN`

### Continual Learning Training Flow

1. **Initialization** ([training/checkpoint.py](training/checkpoint.py:14)):
   - Load dataset via `load_return_dataset` or graph data
   - Initialize model based on `prob` and `network` config
   - Create optimizer (Adam/AdamW) and Trainer

2. **Task Loop** (in [training/runners.py](training/runners.py)):
   - For each task `i` in `range(config['n_task'])`:
     - Generate task-specific data (continual learning splits)
     - Train on current task using `trainer.train__CL__*` methods
     - Optionally: Architecture search after task `arch_start_task`
     - Append task data to experience replay buffer

3. **Equinox Partition Pattern**:
   - Models are split into `params` (trainable arrays) and `static` (non-trainable)
   - AWB matrices (`A`, `B`) are moved to `static` to freeze them during regular training
   - `eqx.partition(model, eqx.is_array)` splits model
   - `eqx.combine(params, static)` reconstructs model

4. **Model Serialization**:
   - Saved using Equinox: `eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)`
   - Results saved as pickle: `pickle.dump(record_dict, f)`

### AWB (Adaptive Weight Basis) System

Models support optional `A` and `B` transformation matrices for architecture search:
- Standard forward: `model(x)` uses weights `W` directly
- AWB forward: `model.getAWB(x)` or `model.get_AWBT(x)` uses `A @ W @ B.T`
- This enables continuous architecture morphing without discrete changes

**Training phases** (currently in development, see [arch_search/](arch_search/)):
1. Train with fixed architecture (W trainable, A/B frozen)
2. Architecture search: evaluate neighboring architectures
3. If new architecture found: initialize new A/B, train A/B with W frozen
4. Continue continual learning with new architecture

### Configuration Keys

Key JSON config parameters (see [config/jsons/](config/jsons/) for examples):
- `prob`: Problem type (`regression`, `classification`, `graphclassification`)
- `problem`: Data structure (`vectors`, `graph`)
- `network`: Architecture (`fcnn`, `cnn`, `gnn`)
- `data`: Dataset (`sine`, `mnist`, `omni`, `cifar10`, `cifar100`, `permuted_mnist`, `ENZYMES`, `MUTAG`, `PROTEINS`, `synthetic`)
- `flag`: Regularization coefficients `[lambda1, lambda2]`
- `n_task`: Number of continual learning tasks
- `epochs_per_task`: Training epochs per task
- `class_per_task`: Number of classes per task (for classification)
- `lr`: Learning rate
- `batch_size` / `batch`: Batch size
- `hln`: Hidden layer neurons
- `n_layers`: Number of layers
- `delta`: Used in data/trainer configuration
- `save_iter`: Frequency of saving metrics to TensorBoard
- `tensorfile`: TensorBoard log directory
- `model_path`: Path to save model weights
- `arch_search`: Boolean to enable architecture search (optional)
- `arch_start_task`: Task index to start architecture search (optional)

### Supported Datasets

**Regression**: `sine` (synthetic sine waves with varying frequency/amplitude)

**Classification**:
- `mnist`, `omni` (Omniglot), `cifar10`, `cifar100`, `permuted_mnist`

**Graph Classification**:
- `ENZYMES`, `MUTAG`, `PROTEINS`, `synthetic` (from TUDataset/PyG)

## Dependencies

Core: jax, equinox, diffrax, optax, jaxopt
Data: torch, torchvision, torch_geometric, pandas, numpy
Visualization: matplotlib, tensorboard
Testing: pytest
