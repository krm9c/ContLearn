# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContLearn is a JAX-based framework for continual learning research. It implements universal continual learning methods supporting multiple neural network architectures (MLP, CNN, GCN, GAT) with optional AWB (Adaptive Weight Basis) transformations for architecture search during lifelong learning.

## Commands

### Running Training
```bash
# Run via script.sh (recommended)
bash script.sh

# Direct execution with JSON config
python run.py train <num_runs> "<config_file>.json"

# Examples:
python run.py train 1 "param2.json"           # Sine regression
python run.py train 1 "paramgraph1.json"      # Graph classification (ENZYMES)
python run.py train 5 "paramgraph2.json"      # Graph classification (MUTAG)
python run.py train 1 "paramomni10.json"      # MNIST classification
```

Configuration files are stored in `jsons/` directory.

## Architecture

### Core Components

**`run.py`** - Main entry point containing:
- `Params` class for loading JSON configurations
- Data loading functions (`load_graph_data`, `load_return_dataset`, `continuum_Graph_classification`)
- Training orchestration (`train_model_graph`, `train_model_reg`, `train_model_class`)
- Architecture search functions (`arch_search_GCN`, `arch_search_MLP`)

**`utils/model.py`** - Neural network architectures using Equinox:
- `MLP` / `MLPorig` - Feedforward networks with optional AWB transformation via `getAWB()`
- `CNN` / `CNNorig` - Convolutional networks with `get_AWBT()` for AWB mode
- `GCN` / `GCNorig` - Graph Convolutional Networks
- `myNN` - Combined GCN + MLP for graph classification with `get_AWBT()` support
- `SingleHeadGAT`, `MultiHeadGAT` - Graph Attention Networks
- `Linear`, `Linear2`, `Linear3` - Custom linear layers with different bias shapes
- `Pool`, `GraphPooling` - Graph pooling operations

**`utils/trainer.py`** - Training logic:
- `Trainer` class with loss functions for vectors and graphs
- JIT-compiled loss/metric functions (`loss_fn_class`, `loss_fn_mse`, `accuracy_graphs`)
- Hamiltonian-based training (`return_Hamiltonian_graph`)
- TensorBoard integration via `SummaryWriter`

**`utils/data.py`** - Data handling:
- `data_return` class managing datasets (MNIST, Omniglot, sine, synthetic)
- Experience replay buffer management (`append_to_experience`)
- `Continual_Dataset` for PyTorch DataLoader compatibility

**`utils/utils.py`** - Utilities:
- Sparse matrix operations (`sp_matmul`)
- Graph preprocessing (`normalize_adj`, `preprocess_features`)
- Gradient visualization

### AWB (Adaptive Weight Basis) System

Models support optional `A` and `B` transformation matrices for architecture search:
- Standard forward: `model(x)`
- AWB forward: `model.getAWB(x)` or `model.get_AWBT(x)`

The AWB mode applies transformations `A @ W @ B.T` to weight matrices, enabling continuous architecture search without discrete architecture decisions.

### Configuration Keys

Key JSON config parameters:
- `prob`: Problem type (`regression`, `classification`, `graphclassification`)
- `problem`: Data structure (`vectors`, `graph`)
- `network`: Architecture (`fcnn`, `cnn`, `gnn`)
- `data`: Dataset (`sine`, `mnist`, `omni`, `ENZYMES`, `MUTAG`, `PROTEINS`, `synthetic`)
- `flag`: Regularization coefficients `[lambda1, lambda2]`
- `n_task`: Number of continual learning tasks
- `epochs_per_task`: Training epochs per task

### Model Serialization

Models are saved using Equinox's serialization:
```python
eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)
```

## Dependencies

Core: jax, equinox, diffrax, optax, jaxopt
Data: torch, torchvision, torch_geometric, pandas, numpy
Visualization: matplotlib, tensorboard
