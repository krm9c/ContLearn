# ContLearn Session Summary

## Overview
ContLearn is a JAX/Equinox-based continual learning framework supporting regression, classification, and graph classification tasks with Adaptive Weight Basis (AWB) transformations for architecture search.

## Changes Made This Session

### 1. New Datasets Added (`utils/data.py`)
- **CIFAR-10**: 3-channel 32x32 images, 10 classes
- **CIFAR-100**: 3-channel 32x32 images, 100 classes
- **Permuted MNIST**: Task-specific pixel permutations for continual learning

Each dataset includes:
- Loading in `__init__()` using torchvision
- Task generation methods with random affine augmentation
- Integration in `generate_dataset()` dispatcher

### 2. CNN3D Model Added (`utils/model.py`)
New CNN architecture for 3-channel images:
- 2 convolutional layers (32 → 64 channels)
- MaxPool2d with `stride=2` (critical fix)
- AWB transformation support via `get_AWBT()`
- Feed-forward layers: 2304 → 512 → 256 → num_classes

### 3. Model Selection Updated (`run.py`)
`load_checkpoint()` modified to select CNN3D for CIFAR datasets:
```python
if config['data'] in ['cifar10', 'cifar100']:
    model = CNN3D(...)
else:
    model = CNN(...)  # For MNIST/Omniglot
```

### 4. Configuration Files Created (`jsons/`)
- `param_cifar10.json`
- `param_cifar100.json`
- `param_permuted_mnist.json`
- Test configs with 2 tasks, 1 epoch each

### 5. Documentation Created
- `CLAUDE.md`: Codebase guide for Claude Code
- `TODO_MODULARITY.md`: 13 refactoring tasks identified

## Key Bug Fixes
1. **IndentationError** in `model.py` line 63-64 (extra leading space)
2. **MaxPool2d stride** defaulted to 1 instead of 2, causing dimension mismatch (43264 vs 2304)

## Pending Modularity Tasks (TODO_MODULARITY.md)

### Priority 1 (High Impact)
1. Consolidate 4 Linear classes into 1 parameterized class
2. Extract AWB matrix initialization utility
3. Create data augmentation pipeline
4. Create train/test split utility
5. Create model factory function

### Priority 2 (Medium Impact)
1. Create constants module
2. Create trainer config builder
3. Create optimizer factory
4. Extract weight/bias initialization

### Priority 3 (Technical Debt)
1. Implement dataset registry pattern
2. Create abstract base model class
3. Remove duplicate `train__CL__graph` method
4. Standardize dataset info printing

## File Structure
```
ContLearn/
├── run.py                 # Main entry point
├── utils/
│   ├── model.py          # MLP, CNN, CNN3D, myNN, Linear classes
│   ├── data.py           # Dataset loading and task generation
│   ├── trainer.py        # Training loops (CL regression/classification/graph)
│   └── utils.py          # Utility functions
├── jsons/                # Configuration files
├── CLAUDE.md             # Codebase documentation
├── TODO_MODULARITY.md    # Refactoring tasks
└── test_datasets.sh      # Test script for new datasets
```

## Running the Code
```bash
# Train with specific config
python run.py train 1 "param_cifar10.json"

# Test all datasets (2 tasks, 1 epoch)
bash test_datasets.sh
```
