# Configuration Parameters

This document describes all configuration parameters for the CL Framework.

## Basic Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | string | - | Dataset name (e.g., "sine") |
| `prob` | string | - | Problem type: "regression", "classification", "graphclassification" |
| `problem` | string | "vectors" | Problem structure: "vectors" or "graph" |
| `network` | string | "fcnn" | Network architecture type |

## Training Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_task` | int | 40 | Number of continual learning tasks |
| `epochs_per_task` | int | - | Training epochs per task |
| `batch_size` | int | 64 | Batch size for training |
| `save_iter` | int | 10 | Save metrics every N epochs |
| `model_path` | string | "outputs/model" | Path to save model and records |

## Optimizer Settings

The framework supports multiple optimizers via the `optimizer` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | string | "adam" | Optimizer type: "adam", "adamw", "sgd", "rmsprop" |
| `lr` | float | 1e-4 | Initial learning rate (used by all optimizers) |
| `weight_decay` | float | 1e-4 | Weight decay (only for "adamw") |
| `momentum` | float | 0.9 | Momentum (only for "sgd") |

### Optimizer Examples

**Adam (default):**
```json
{
    "optimizer": "adam",
    "lr": 0.01
}
```

**AdamW with weight decay:**
```json
{
    "optimizer": "adamw",
    "lr": 0.001,
    "weight_decay": 1e-4
}
```

**SGD with momentum:**
```json
{
    "optimizer": "sgd",
    "lr": 0.01,
    "momentum": 0.9
}
```

**RMSprop:**
```json
{
    "optimizer": "rmsprop",
    "lr": 0.001
}
```

## Learning Rate Scheduling

The framework supports adaptive learning rate decay across tasks. The learning rate is updated at the beginning of each task while **preserving the optimizer state** (momentum, adaptive learning rates).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_schedule` | string | "constant" | Schedule type: "constant", "step", "exponential", "cosine", "linear" |
| `lr_decay_factor` | float | 0.9 | Decay factor for step/exponential schedules |
| `lr_decay_steps` | int | 1 | Tasks between decays for step schedule |
| `lr_min` | float | 1e-6 | Minimum learning rate floor |

### Schedule Types

**Constant (default):** No decay, learning rate stays at `lr` for all tasks.

**Step Decay:** Reduces LR by `lr_decay_factor` every `lr_decay_steps` tasks.
```
lr_t = lr * (lr_decay_factor ^ (task_id // lr_decay_steps))
```

**Exponential Decay:** Continuous decay at each task.
```
lr_t = lr * (lr_decay_factor ^ task_id)
```

**Cosine Annealing:** Smooth decay following cosine curve from `lr` to `lr_min`.
```
lr_t = lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * task_id / (n_task - 1)))
```

**Linear Decay:** Linear interpolation from `lr` to `lr_min` over all tasks.
```
lr_t = lr - (task_id / (n_task - 1)) * (lr - lr_min)
```

### LR Schedule Examples

**Exponential decay (recommended for CL):**
```json
{
    "lr": 0.01,
    "lr_schedule": "exponential",
    "lr_decay_factor": 0.9,
    "lr_min": 1e-6
}
```
Task 0: 0.01, Task 1: 0.009, Task 2: 0.0081, Task 3: 0.0073, ...

**Step decay every 2 tasks:**
```json
{
    "lr": 0.01,
    "lr_schedule": "step",
    "lr_decay_factor": 0.5,
    "lr_decay_steps": 2,
    "lr_min": 1e-6
}
```
Task 0-1: 0.01, Task 2-3: 0.005, Task 4-5: 0.0025, ...

**Cosine annealing:**
```json
{
    "lr": 0.01,
    "lr_schedule": "cosine",
    "lr_min": 1e-4
}
```

**Linear decay:**
```json
{
    "lr": 0.01,
    "lr_schedule": "linear",
    "lr_min": 1e-4
}
```

## Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_layers` | int | 4 | Total number of layers (including input/output) |
| `hln` | int | 128 | Hidden layer size |

## Loss and Metrics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | string | "mse" | Loss function: "mse" (regression) or "class" (classification) |
| `metric` | string | "mse" | Metric function: "mse" or "class" (accuracy) |
| `flag` | list | [1.0, 1.0] | Regularization flags [current_weight, experience_weight] |

## Gradient Combination Weights

The Hamiltonian gradient is a convex combination of three gradient components:

```
grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
```

Where:
- `delta_theta`: Gradient from current task loss
- `grad_V`: Gradient from experience replay loss
- `grad_dV`: Gradient from Hamiltonian regularization (perturbation sensitivity)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grad_weights` | list | [0.01, 0.98, 0.1] | [alpha, beta, gamma] weights for gradient combination |

### Gradient Weight Guidelines

- **alpha** (current task): Controls learning on new task. Lower values prevent catastrophic forgetting.
- **beta** (experience replay): Controls retention of old knowledge. Higher values preserve past learning.
- **gamma** (Hamiltonian): Controls regularization strength. Affects stability under perturbations.

### Examples

**Default (balanced CL):**
```json
{
    "grad_weights": [0.01, 0.98, 0.1]
}
```

**Focus on new task (faster adaptation, more forgetting):**
```json
{
    "grad_weights": [0.5, 0.4, 0.1]
}
```

**Strong retention (slower adaptation, less forgetting):**
```json
{
    "grad_weights": [0.01, 0.99, 0.0]
}
```

**High regularization (more stable, slower):**
```json
{
    "grad_weights": [0.01, 0.89, 0.1]
}
```

## Dataset-Specific Settings

### Sine Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | float | 0.001 | Task drift perturbation value |
| `len_exp_replay` | int | 20000 | Experience replay buffer size |
| `data_path` | string | "Incremental_Sine1e^4.p" | Path to sine data file |

## AWB (Adaptive Weight Basis) Settings

AWB enables architecture morphing during lifelong learning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `awb_enabled` | bool | false | Enable AWB pipeline |
| `awb_preliminary_epochs` | int | 250 | Preliminary training epochs before arch decision |
| `awb_ab_training_epochs` | int | 2000 | Epochs for A/B matrix training |
| `awb_ab_warmup_epochs` | int | 50 | Warmup epochs after architecture change |
| `awb_ab_max_iterations` | int | 8 | Max iterations for A/B convergence |
| `awb_averaging_window` | int | 10 | Window size for loss averaging |

## Debug Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug_mode` | bool | false | Enable debug mode with limited data |
| `debug_limit` | int | 100 | Number of samples in debug mode |

## Command Line Options

The `run.py` script supports additional command line options:

```bash
python scripts/run.py <config_file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--runs N` | 1 | Number of experiment runs |
| `--no-plots` | false | Skip automatic plot generation |
| `--figures-dir DIR` | "figures" | Output directory for plots |

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

1. **Losses** (`*_losses.png`): All loss components (H, V, dV, dV/dx, dV/dtheta, gradient norm)
2. **Metrics** (`*_metrics.png`): Train and test metrics over time
3. **Eigenvalues** (`*_eigenvalues.png`):
   - Standard mode: Weight matrix eigenvalues
   - AWB mode: A and B matrix eigenvalues
4. **Overview** (`*_overview.png`): Combined visualization of all metrics

Plots can also be generated manually:
```bash
python scripts/plot_results.py <records_file.pkl> --output-dir figures
```

## Complete Example Config

```json
{
    "data": "sine",
    "prob": "regression",
    "problem": "vectors",
    "network": "fcnn",
    "delta": 0.001,
    "n_task": 5,
    "epochs_per_task": 2000,
    "batch_size": 64,
    "n_layers": 4,
    "hln": 128,
    "flag": [1.0, 1.0],
    "loss": "mse",
    "metric": "mse",
    "save_iter": 10,
    "model_path": "outputs/sine_model",
    "len_exp_replay": 20000,
    "awb_enabled": false,
    "debug_mode": false,
    "debug_limit": 100,

    "optimizer": "adam",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "momentum": 0.9,

    "lr_schedule": "exponential",
    "lr_decay_factor": 0.9,
    "lr_decay_steps": 1,
    "lr_min": 1e-6,

    "grad_weights": [0.01, 0.98, 0.1]
}
```
