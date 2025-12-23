# Configuration Parameters

This document describes the configuration system for the ContLearn framework.

**Updated by Claude: Config system with automatic defaults from `constants.py` (layer-level AWB refactor)**

## Overview

The framework uses a **smart defaults system** where:
1. All tunable parameters have defaults defined in `src/cl/config/constants.py`
2. Config files only need to specify **non-default values**
3. The `load_config()` function automatically applies defaults based on problem type, dataset, and network

This means config files can be **minimal** - you only specify what's different from the defaults!

## How It Works

```python
from cl.config import load_config

# Load config with automatic defaults
config = load_config('config/sine.json')  # Defaults applied automatically

# Disable automatic defaults (raw JSON)
config = load_config('config/sine.json', apply_defaults_flag=False)
```

The `apply_defaults()` function (in `src/cl/config/params.py`) intelligently applies defaults based on:
- **Problem type** (`prob`): regression vs classification
- **Problem structure** (`problem`): vectors vs graph
- **Network architecture** (`network`): fcnn, cnn, cnn3d, gcn
- **Dataset** (`data`): sine, mnist, cifar10, synthetic, etc.

## Minimal Config Examples

### Sine Wave Regression (Minimal)
```json
{
    "data": "sine",
    "n_task": 2,
    "epochs_per_task": 10,
    "model_path": "outputs/sine_model",
    "debug_mode": true
}
```
All other parameters (lr, optimizer, batch_size, etc.) use smart defaults!

### MNIST Classification (Minimal)
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 40,
    "model_path": "outputs/mnist_model"
}
```
The system automatically applies MNIST-specific defaults for channel_in, input_size, feed_sizes, etc.

## Configuration Parameters

### Core Problem Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prob` | string | `"regression"` | Problem type: "regression", "classification" |
| `problem` | string | `"vectors"` | Problem structure: "vectors" or "graph" |
| `data` | string | `"sine"` | Dataset: "sine", "mnist", "permuted_mnist", "cifar10", "cifar100", "synthetic" |
| `network` | string | `"fcnn"` | Network: "fcnn" (MLP), "cnn", "cnn3d", "gcn" |
| `loss` | string | `"mse"` | Loss function: "mse" (regression) or "class" (classification) |
| `metric` | string | `"mse"` | Metric: "mse" or "class" (accuracy) |

### Training Loop Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_task` | int | `5` | Number of continual learning tasks |
| `epochs_per_task` | int | `100` | Training epochs per task |
| `batch_size` | int | Problem-specific | Batch size (64 for regression, 128 for classification, 20 for graph) |
| `save_iter` | int | `10` | Save metrics every N epochs |
| `model_path` | string | `"outputs/model"` | Path to save model and records |
| `len_exp_replay` | int | Problem-specific | Experience replay buffer (20k for vectors, 200k for graph) |

### Optimizer Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | string | `"adam"` | Optimizer: "adam", "adamw", "sgd", "rmsprop" |
| `lr` | float | Problem-specific | Learning rate (1e-4 for regression/graph, 1e-3 for classification) |
| `weight_decay` | float | `1e-4` | Weight decay for AdamW |
| `momentum` | float | `0.9` | Momentum for SGD/RMSprop |

### Learning Rate Schedule

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_schedule` | string | `"constant"` | Schedule: "constant", "step", "exponential", "cosine", "linear" |
| `lr_decay_factor` | float | `0.9` | Decay factor for step/exponential |
| `lr_decay_steps` | int | `1` | Steps between decay for step schedule |
| `lr_min` | float | `1e-6` | Minimum learning rate floor |

**Schedule Types:**
- **Constant:** No decay
- **Step:** `lr * (decay_factor ^ (task_id // decay_steps))`
- **Exponential:** `lr * (decay_factor ^ task_id)`
- **Cosine:** Smooth cosine annealing from lr to lr_min
- **Linear:** Linear interpolation from lr to lr_min

### Hamiltonian Gradient Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flag` | list | `[1.0, 1.0]` | Regularization weights [dV/dx, dV/dtheta] |
| `grad_weights` | list | `[0.01, 0.98, 0.1]` | Gradient combination [alpha, beta, gamma] |

**Gradient combination:**
```
grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
```
- `alpha`: Current task gradient weight (lower = less forgetting)
- `beta`: Experience replay gradient weight (higher = more retention)
- `gamma`: Hamiltonian regularization weight (stability)

### Debug Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug_mode` | bool | `false` | Enable debug mode with limited data |
| `debug_limit` | int | `100` | Number of samples when debug_mode=True |

## Model Architecture Defaults

### MLP/FCNN (Regression)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_layers` | int | `4` | Total number of layers |
| `hln` | int | `256` | Hidden layer size |

### CNN (MNIST-like)

| Parameter | Type | Default (MNIST) | Description |
|-----------|------|---------|-------------|
| `filter_size` | int | `4` | Convolution filter size |
| `channel_out` | int | `3` | Output channels from conv layers |
| `channel_in` | int | `1` | Input channels |
| `input_size` | int | `28` | Input image size |
| `feed_sizes` | list | `[1875, 512, 64, 10]` | Feed-forward layer sizes |

### CNN3D (CIFAR-like)

| Parameter | Type | Default (CIFAR) | Description |
|-----------|------|---------|-------------|
| `filter_size` | int | `4` | Convolution filter size |
| `channel_out` | int | `32` | Output channels from conv layers |
| `channel_in` | int | `3` | Input channels (RGB) |
| `input_size` | int | `32` | Input image size |
| `feed_sizes` | list | `[2304, 512, 256, 10]` | Feed-forward layer sizes |

### GCN (Graph Classification)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gcn_sizes` | list | `[None, 128]` | GCN layer sizes (first set to input size) |
| `feed_sizes` | list | `[128, 128, 128, 10]` | MLP layer sizes after GCN |

### Classification Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_class` | int | `10` | Number of output classes |
| `class_per_task` | int | `2` | Classes per task (incremental learning) |

## Dataset-Specific Defaults

### Sine Wave Regression

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | float | `0.001` | Task drift perturbation |
| `test_size` | float | `0.2` | Test set fraction |

### MNIST/CIFAR (Data Augmentation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rotation_range` | int | `180` | Random rotation range (degrees) |
| `scaling_range` | tuple | `(1, 2)` | Random scaling range |
| `permutation_seed_multiplier` | int | `1000` | Seed multiplier for permuted MNIST |

### Synthetic Graph Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_graphs` | int | `1000` | Number of graphs to generate |
| `num_channels` | int | `5` | Node feature channels |
| `avg_num_nodes` | int | `2` | Average nodes per graph |
| `num_classes` | int | `10` | Number of classes |

## AWB (Adaptive Weight Basis) Settings

AWB enables architecture morphing during lifelong learning using the 5-step algorithm.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `awb_enabled` | bool | `false` | Enable AWB pipeline |
| `awb_preliminary_epochs` | int | `10` | STEP 1: Preliminary training epochs |
| `awb_ab_training_epochs` | int | `50` | STEP 3b: A/B matrix training epochs |
| `awb_ab_warmup_epochs` | int | `2` | STEP 5: Warmup after V computation |
| `awb_ab_max_iterations` | int | `8` | Max iterations for A/B convergence |
| `awb_averaging_window` | int | `10` | Epochs to average for loss computation |
| `awb_arch` | list | Architecture-specific | Target architecture for AWB expansion |

**AWB 5-Step Algorithm:**
1. Preliminary training on new task
2. Decide if architecture change needed (loss ratio thresholds)
3a. Architecture search for optimal dimensions
3b. Train A/B matrices with W frozen
4. Compute V = A @ W @ B.T
5. Train V with A/B frozen

## Architecture Search Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch_search_enabled` | bool | `false` | Enable architecture search |
| `arch_search_epochs` | int | `100` | Epochs per search iteration |
| `arch_search_lr` | float | `1e-3` | Learning rate for search |
| `arch_search_batch_size` | int | `20` | Batch size for search |
| `arch_search_max_iter` | int | `10` | Max search iterations |
| `arch_search_loss_threshold` | float | `0.6` | Loss ratio threshold for change |

**CNN-specific:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch_search_hidden_range` | int | `3` | Range for hidden layer search |
| `arch_search_filter_min` | int | `2` | Minimum filter size |
| `arch_search_filter_max` | int | `5` | Maximum filter size (exclusive) |

**MLP/GCN-specific:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch_search_step_size_mlp` | int | `10` | Step size for MLP layer search |
| `arch_search_step_size_gcn` | int | `10` | Step size for GCN layer search |
| `arch_search_range` | int | `5` | Range for layer size search |

## Command Line Usage

```bash
python scripts/run.py <config_file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--runs N` | `1` | Number of experiment runs |
| `--no-plots` | `false` | Skip automatic plot generation |
| `--figures-dir DIR` | `"figures"` | Output directory for plots |

### Examples

```bash
# Single run with plots
python scripts/run.py config/sine.json

# Multiple runs
python scripts/run.py config/sine.json --runs 3

# Skip plot generation
python scripts/run.py config/sine.json --no-plots

# Custom figures directory
python scripts/run.py config/sine.json --figures-dir outputs/figures
```

## Plot Generation

After training, the framework automatically generates four types of plots:

1. **Losses** (`*_losses.png`): All loss components (H, V, dV, dV/dx, dV/dtheta, grad norm)
2. **Metrics** (`*_metrics.png`): Train and test metrics over time
3. **Eigenvalues** (`*_eigenvalues.png`):
   - Standard mode: Weight matrix eigenvalues
   - AWB mode: A and B matrix eigenvalues
4. **Overview** (`*_overview.png`): Combined visualization

Manual plot generation:
```bash
python scripts/plot_results.py <records_file.pkl> --output-dir figures
```

## Complete Config Examples

### Minimal Sine Regression
```json
{
    "data": "sine",
    "n_task": 2,
    "epochs_per_task": 10,
    "model_path": "outputs/sine_model",
    "debug_mode": true
}
```

### Sine with AWB
```json
{
    "data": "sine",
    "n_task": 2,
    "epochs_per_task": 10,
    "model_path": "outputs/sine_awb_model",
    "debug_mode": true,
    "awb_enabled": true,
    "awb_preliminary_epochs": 5,
    "awb_ab_warmup_epochs": 20
}
```

### MNIST Classification
```json
{
    "data": "mnist",
    "prob": "classification",
    "network": "cnn",
    "n_task": 10,
    "epochs_per_task": 40,
    "model_path": "outputs/mnist_model"
}
```

### CIFAR-10 with Custom LR Schedule
```json
{
    "data": "cifar10",
    "prob": "classification",
    "network": "cnn",
    "n_task": 3,
    "epochs_per_task": 100,
    "model_path": "outputs/cifar10_model",
    "lr_schedule": "exponential",
    "lr_decay_factor": 0.95
}
```

### Synthetic Graph with AWB
```json
{
    "data": "synthetic",
    "prob": "classification",
    "problem": "graph",
    "network": "gcn",
    "n_task": 2,
    "epochs_per_task": 100,
    "model_path": "outputs/graph_awb_model",
    "optimizer": "adamw",
    "awb_enabled": true
}
```

## Adding New Defaults

To add new default parameters:

1. **Add constant** to `src/cl/config/constants.py`:
   ```python
   DEFAULT_MY_PARAMETER = value
   ```

2. **Update `apply_defaults()`** in `src/cl/config/params.py`:
   ```python
   set_default('my_parameter', constants.DEFAULT_MY_PARAMETER)
   ```

3. **Use in code** with fallback:
   ```python
   my_param = config.get('my_parameter', constants.DEFAULT_MY_PARAMETER)
   ```

This ensures consistency across the codebase and reduces redundancy in config files!
