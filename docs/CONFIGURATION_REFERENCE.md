# Configuration Parameter Reference

This document provides a comprehensive reference for all configuration parameters used in the ContLearn framework. Use this as a basis for generating new experiment configurations.

---

## Table of Contents

1. [Condition Definitions](#condition-definitions)
2. [Parameter Categories](#parameter-categories)
3. [Complete Parameter List](#complete-parameter-list)
4. [Condition-Parameter Matrix](#condition-parameter-matrix)
5. [AWB Pipeline Flow](#awb-pipeline-flow)

---

## Condition Definitions

| Condition | Name | Description | Key Features |
|-----------|------|-------------|--------------|
| **C1** | Baseline | Fixed architecture, no smoothness | Constant LR, no warmup, no AWB |
| **C2** | Heuristics | Fixed architecture with smoothness heuristics | Cosine LR, task warmup, no AWB |
| **C3** | Arch Search Only | Architecture search without weight transfer | AWB enabled, skip_transfer=true, random reinit |
| **C4** | AWB Full | Full AWB pipeline with A/B transfer | AWB enabled, A/B training, task warmup |

---

## Parameter Categories

### Core Training Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `data` | string | 'sine' | Dataset name | `generic_runner.py:678` |
| `problem` | string | 'vectors' | Problem type: 'vectors' or 'graph' | `generic_runner.py:677` |
| `prob` | string | 'regression' | Task type: 'regression' or 'classification' | `generic_runner.py:902` |
| `network` | string | 'fcnn' | Network architecture: 'fcnn', 'cnn', 'cnn3d', 'gcn' | `generic_runner.py:679` |
| `n_task` | int | 5-40 | Number of tasks | `generic_runner.py:898` |
| `epochs_per_task` | int | 100 | Training epochs per task | `generic_runner.py:899` |
| `batch_size` | int | varies | Batch size (64 for graphs, 512 for images) | `generic_runner.py:684` |
| `seed` | int | 42 | Random seed for reproducibility | `loops.py:377` |

### Learning Rate Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `lr` | float | varies | Base learning rate | `generic_runner.py:225` |
| `lr_schedule` | string | 'constant' | LR schedule: 'constant', 'step', 'exponential', 'cosine' | `generic_runner.py:406` |
| `lr_decay_factor` | float | 0.5-0.99 | Decay factor for step/exponential schedules | `generic_runner.py:412` |
| `lr_min` | float | 1e-6 | Minimum learning rate | `generic_runner.py:327` |

### Optimizer Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `optimizer` | string | 'adam' | Optimizer: 'adam', 'adamw', 'sgd', 'rmsprop' | `generic_runner.py:224` |
| `weight_decay` | float | 0.0 | Weight decay (L2 regularization) | `generic_runner.py:226` |
| `momentum` | float | 0.9-0.99 | Momentum for SGD/RMSprop | `generic_runner.py:227` |

### Gradient Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `grad_weights` | list[3] | [0.4, 0.4, 0.1] | Hamiltonian gradient weights [current, experience, dV] | `loops.py:334` |
| `normalize_dV` | bool | true | Normalize dV component | `loops.py:337` |
| `dV_scale_factor` | float | 1.0 | Scale factor for dV | `loops.py:338` |
| `gradient_clip_norm` | float | None | Gradient clipping norm | `loops.py:339` |

### Task Warmup Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `task_warmup_enabled` | bool | false | Enable task warmup phase | `generic_runner.py:987` |
| `task_warmup_epochs` | int | 25 | Epochs for warmup phase | `generic_runner.py:988` |
| `task_warmup_lr_factor` | float | 0.1 | LR multiplier during warmup | `generic_runner.py:991` |
| `warmup_grad_weights` | list[3] | varies | Gradient weights during warmup | `generic_runner.py:992` |

### AWB Pipeline Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `awb_enabled` | bool | false | Enable AWB pipeline | `generic_runner.py:900` |
| `awb_skip_transfer` | bool | false | Skip A/B training (random init) | `awb_pipeline.py:318` |
| `awb_preliminary_epochs` | int | varies | STEP 1: Preliminary training epochs | `awb_pipeline.py:200` |
| `awb_ab_training_epochs` | int | 50 | STEP 3b: A/B matrix training epochs | `awb_pipeline.py:201` |
| `awb_ab_warmup_epochs` | int | 2 | STEP 5: V warmup epochs after transfer | `awb_pipeline.py:202` |
| `awb_ab_max_iterations` | int | varies | Max A/B training iterations | `awb_pipeline.py:203` |
| `awb_ab_lr` | float | 0.001 | Learning rate for A/B training | `awb_pipeline.py:350` |
| `awb_averaging_window` | int | varies | Window for loss averaging | `awb_pipeline.py:204` |
| `awb_loss_ratio_threshold` | float | 0.9 | Threshold for architecture change decision | `awb_pipeline.py:270` |
| `awb_validation_ratio` | float | 0.2 | Validation set ratio for arch search | `awb_pipeline.py:291` |
| `force_arch_change` | bool | false | Force architecture change (debug) | `awb_pipeline.py:276` |

### Architecture Search Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `arch_search_enabled` | bool | false | Enable architecture search | `params.py:244` |
| `arch_search_method` | string | 'grid' | Search method: 'grid', 'bayesian' | `arch_search.py:890` |
| `arch_search_epochs` | int | varies | Epochs per candidate evaluation | `arch_search.py:92` |
| `arch_search_lr` | float | varies | LR for architecture search | `arch_search.py:93` |
| `arch_search_batch_size` | int | varies | Batch size for arch search | `arch_search.py:94` |
| `arch_search_mlp_increment` | int | 15 | MLP layer size increment | `mlp.py:329` |
| `arch_search_range` | int | 2 | Search range (candidates per direction) | `mlp.py:331` |
| `arch_search_step_size_gcn` | int | varies | GCN layer size step | `gcn.py:475` |
| `arch_search_step_size_mlp` | int | varies | MLP layer size step | `gcn.py:476` |
| `arch_search_early_stop_patience` | int | 3 | Early stopping patience | `arch_search.py:828` |

### Logging and Saving Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `model_path` | string | None | Path for saving results | `generic_runner.py:1303` |
| `save_iter` | int | 50 | Epochs between saves | `generic_runner.py:206` |
| `log_interval` | int | 1 | Progress bar update frequency | `loops.py:342` |
| `eval_interval` | int | save_iter | Test evaluation frequency | `loops.py:343` |
| `debug_mode` | bool | false | Enable debug mode (limit samples) | `generic_runner.py:686` |
| `debug_limit` | int | 100 | Sample limit in debug mode | `generic_runner.py:687` |
| `per_task_eval_enabled` | bool | false | Evaluate on all previous tasks | `generic_runner.py:1146` |

### Data Loading Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `use_jax_prefetch` | bool | true | Enable JAX async prefetch | `loops.py:323` |
| `prefetch_size` | int | 3 | Batches to prefetch | `loops.py:324` |
| `len_exp_replay` | int | varies | Experience replay buffer size | `generic_runner.py:685` |
| `delta` | float | 0.001 | Small constant for numerical stability | `generic_runner.py:683` |
| `flag` | list[2] | [1.0, 1.0] | Data flags | `loops.py:318` |

### Experience Replay Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `balanced_replay_enabled` | bool | true | Enable balanced replay sampling | `base.py:105` |
| `recent_task_weight` | float | 0.1 | Weight for recent task samples | `base.py:106` |
| `older_tasks_weight` | float | 0.8 | Weight for older task samples | `base.py:107` |

### Checkpointing Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `checkpoint_interval` | int | 0 | Epochs between checkpoints (0=disabled) | `loops.py:346` |
| `max_checkpoints` | int | 3 | Maximum checkpoints to keep | `loops.py:352` |
| `checkpoint_memory_limit_gb` | float | 8.0 | Memory limit for checkpoints | `loops.py:353` |
| `async_checkpointing` | bool | true | Enable async checkpoint writes | `loops.py:354` |

### Profiling Parameters

| Parameter | Type | Default | Description | Source File |
|-----------|------|---------|-------------|-------------|
| `profiling_enabled` | bool | false | Enable performance profiling | `generic_runner.py:884` |

### Dataset-Specific Parameters

#### MNIST/CIFAR
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rotation_range` | float | varies | Rotation augmentation range |
| `scaling_range` | float | varies | Scaling augmentation range |
| `train_test_split` | float | 0.8 | Train/test split ratio |
| `image_size` | int | 28 | Image dimension |
| `permutation_seed_multiplier` | int | varies | Seed multiplier for permutations |

#### Synthetic Graph
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_graphs` | int | 2000 | Number of graphs to generate |
| `num_channels` | int | 10 | Node feature channels |
| `avg_num_nodes` | int | 20 | Average nodes per graph |
| `num_classes` | int | 5 | Number of classes |
| `task_shift_enabled` | bool | true | Enable task shift perturbations |
| `feature_noise_base` | float | 0.1 | Base feature noise |
| `edge_dropout_base` | float | 0.05 | Base edge dropout rate |
| `feature_shift_base` | float | 0.05 | Base feature shift |

#### Sine
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | string | 'data/...' | Path to sine data |
| `test_size` | float | 0.2 | Test split ratio |
| `noise_enabled` | bool | false | Enable noise augmentation |
| `noise_scale` | float | 0.1 | Noise scale |
| `noise_increment` | float | 0.05 | Noise increment per task |

### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_layers` | int | 4 | Number of MLP layers |
| `hln` | int | 128 | Hidden layer neurons |
| `input_size` | int | varies | Input dimension |
| `output_size` | int | varies | Output dimension (num_classes) |
| `channel_in` | int | 1-3 | Input channels |
| `channel_out` | int | 3 | Conv output channels |
| `filter_size` | int | 4 | Conv filter size |
| `feed_sizes` | list | varies | MLP layer sizes |
| `gcn_sizes` | list | varies | GCN layer sizes |
| `mlp_sizes` | list | varies | Post-GCN MLP sizes |

### Adaptive Feature Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_lr_min_enabled` | bool | false | Enable adaptive LR minimum |
| `lr_min_base` | float | varies | Base LR minimum |
| `lr_min_max` | float | varies | Max LR minimum |
| `lr_min_loss_ratio_threshold` | float | varies | Threshold for adaptive LR min |
| `adaptive_grad_weights_enabled` | bool | false | Enable adaptive gradient weights |
| `grad_weights_base` | list | varies | Base gradient weights |
| `grad_weights_max_current` | float | varies | Max current task weight |
| `grad_weights_min_experience` | float | varies | Min experience weight |
| `grad_weights_loss_ratio_threshold` | float | varies | Threshold for adaptive weights |

---

## Condition-Parameter Matrix

### Features Enabled per Condition

| Feature | C1 Baseline | C2 Heuristics | C3 Arch Only | C4 AWB Full |
|---------|-------------|---------------|--------------|-------------|
| **LR Schedule** |
| Constant LR | YES | no | YES | no |
| Cosine LR | no | YES | no | YES |
| **Task Warmup** |
| `task_warmup_enabled` | false | true | false | true |
| `task_warmup_epochs` | - | 25 | - | 50 |
| `task_warmup_lr_factor` | - | 0.001 | - | 0.001 |
| **AWB Pipeline** |
| `awb_enabled` | false | false | true | true |
| `awb_skip_transfer` | - | - | true | false |
| `awb_preliminary_epochs` | - | - | 1 | 1 |
| `awb_ab_training_epochs` | - | - | (skipped) | 200 |
| `awb_ab_warmup_epochs` | - | - | 0 | 20 |
| `awb_ab_lr` | - | - | - | 0.001 |
| `awb_loss_ratio_threshold` | - | - | 1.1 | 1.1 |
| `force_arch_change` | - | - | true | false |

### Training Phases per Condition

| Phase | C1 | C2 | C3 | C4 |
|-------|----|----|----|----|
| Standard training | YES | YES | - | - |
| Task warmup | - | YES (25 ep) | - | YES (50 ep) |
| AWB preliminary | - | - | YES (1 ep) | YES (1 ep) |
| Architecture search | - | - | YES | YES (if ratio > threshold) |
| A/B matrix training | - | - | SKIP | YES (200 ep) |
| V warmup | - | - | SKIP | YES (20 ep) |
| V main training | - | - | YES | YES |

---

## AWB Pipeline Flow

```
STEP 1: Preliminary Training
    └─> awb_preliminary_epochs (default: varies)

STEP 2: Architecture Change Decision
    └─> Compare: current_loss / previous_loss > awb_loss_ratio_threshold?
    └─> OR force_arch_change=true

    If NO change needed:
        └─> Continue standard training (epochs_per_task)

    If change needed:
        ↓
STEP 3a: Architecture Search
    └─> Search for optimal architecture
    └─> Uses awb_validation_ratio of data

STEP 3b: A/B Matrix Setup
    ├─> If awb_skip_transfer=true (Condition 3):
    │       └─> Random initialization (no training)
    │       └─> SKIP awb_ab_training_epochs
    │
    └─> If awb_skip_transfer=false (Condition 4):
            └─> Train A/B matrices (awb_ab_training_epochs)
            └─> Uses awb_ab_lr

STEP 4: Compute V = A @ W @ B^T
    └─> Transfer weights via A/B matrices

STEP 5: Train V
    ├─> Warmup phase: awb_ab_warmup_epochs (set to 0 for C3)
    └─> Main training: epochs_per_task
```

---

## Example Configurations

### Condition 1: Baseline
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 150,
    "batch_size": 512,
    "lr": 0.0001,
    "lr_schedule": "constant",
    "grad_weights": [0.4, 0.4, 0.1],
    "task_warmup_enabled": false,
    "awb_enabled": false
}
```

### Condition 2: Heuristics
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 150,
    "batch_size": 512,
    "lr": 0.0001,
    "lr_schedule": "cosine",
    "grad_weights": [0.4, 0.4, 0.1],
    "task_warmup_enabled": true,
    "task_warmup_epochs": 25,
    "task_warmup_lr_factor": 0.001,
    "awb_enabled": false
}
```

### Condition 3: Arch Search Only
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 200,
    "batch_size": 512,
    "lr": 0.0001,
    "lr_schedule": "constant",
    "grad_weights": [0.4, 0.4, 0.1],
    "task_warmup_enabled": false,
    "awb_enabled": true,
    "awb_skip_transfer": true,
    "awb_preliminary_epochs": 1,
    "awb_ab_warmup_epochs": 0,
    "awb_loss_ratio_threshold": 1.1,
    "force_arch_change": true
}
```

### Condition 4: AWB Full
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 200,
    "batch_size": 512,
    "lr": 0.0001,
    "lr_schedule": "cosine",
    "grad_weights": [0.4, 0.4, 0.1],
    "task_warmup_enabled": true,
    "task_warmup_epochs": 50,
    "task_warmup_lr_factor": 0.001,
    "awb_enabled": true,
    "awb_skip_transfer": false,
    "awb_preliminary_epochs": 1,
    "awb_ab_training_epochs": 200,
    "awb_ab_warmup_epochs": 20,
    "awb_ab_lr": 0.001,
    "awb_loss_ratio_threshold": 1.1,
    "force_arch_change": false
}
```

---

## Notes

1. **Parameter Defaults**: All defaults are defined in `src/cl/config/constants.py`
2. **Dataset-specific defaults**: Some parameters have different defaults based on `data` and `problem` type
3. **Deprecated Parameters**: `random_seed` (use `seed`), `batch` (use `batch_size`), `n_tasks` (use `n_task`)
4. **Model paths**: Should follow pattern `runs__/{cluster}/results/{dataset}_condition{N}`
