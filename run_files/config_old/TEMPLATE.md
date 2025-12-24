# ContLearn Configuration Template

**Complete reference for all configuration parameters.**

Use this guide to understand available options and defaults. Your config files only need to specify non-default values!

---

## Quick Start

**Minimal config example:**
```json
{
    "data": "sine",
    "n_task": 2,
    "epochs_per_task": 10,
    "model_path": "outputs/sine_model"
}
```

Everything else uses smart defaults!

---

## REQUIRED PARAMETERS

### `data` (REQUIRED)

**Primary selector - automatically sets prob, problem, network, loss, metric**

| Dataset | Problem Type | Problem Structure | Network | Description |
|---------|--------------|-------------------|---------|-------------|
| `"sine"` | regression | vectors | fcnn (MLP) | Sine wave regression with task drift |
| `"mnist"` | classification | vectors | cnn | MNIST handwritten digits (28x28, 1-channel) |
| `"permuted_mnist"` | classification | vectors | cnn | Permuted pixel MNIST variant |
| `"cifar10"` | classification | vectors | cnn3d | CIFAR-10 color images (32x32, 3-channel, 10 classes) |
| `"cifar100"` | classification | vectors | cnn3d | CIFAR-100 color images (32x32, 3-channel, 100 classes) |
| `"synthetic"` | classification | graph | gcn | Synthetic graph classification |

**Auto-selected based on dataset:**
- `prob` - "regression" or "classification"
- `problem` - "vectors" or "graph"
- `network` - "fcnn", "cnn", "cnn3d", or "gcn"
- `loss` - "mse" or "class"
- `metric` - "mse" or "class"

**Example:**
```json
{
    "data": "mnist"
    // Automatically sets: prob=classification, problem=vectors, network=cnn, loss=class, metric=class
}
```

---

## EXPERIMENT PARAMETERS (Configurable)

### `n_task`
**Type:** int
**Default:** 5
**Description:** Number of continual learning tasks/experiences
**Typical values:** 2-40 depending on dataset

**Example:**
```json
{
    "n_task": 10  // 10 sequential tasks
}
```

---

### `epochs_per_task`
**Type:** int
**Default:** 100
**Description:** Training epochs per task
**Typical values:** 10 (debug), 100 (standard), 500-2000 (full training)

**Example:**
```json
{
    "epochs_per_task": 500  // Longer training
}
```

---

### `save_iter`
**Type:** int
**Default:** 10
**Description:** Save metrics every N epochs
**When to override:** Set to 1 for detailed logging, higher for faster training

**Example:**
```json
{
    "save_iter": 5  // Save metrics every 5 epochs
}
```

---

## SMART DEFAULTS (Override when needed)

### Training Loop

#### `batch_size`
**Type:** int
**Auto-selected defaults:**
- 64 for regression (sine)
- 128 for classification (mnist, cifar)
- 20 for graph problems (synthetic)

**When to override:** GPU memory constraints, convergence tuning

**Example:**
```json
{
    "batch_size": 32  // Reduce for memory constraints
}
```

---

#### `len_exp_replay`
**Type:** int
**Auto-selected defaults:**
- 20,000 for vector problems (sine, mnist, cifar)
- 200,000 for graph problems (synthetic)

**Description:** Experience replay buffer size
**When to override:** Memory constraints or testing replay strategies

**Example:**
```json
{
    "len_exp_replay": 10000  // Smaller buffer
}
```

---

### Optimizer Settings

#### `optimizer`
**Type:** string
**Options:** `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`
**Default:** `"adam"`

**When to use each:**
- `"adam"` - Default, works well for most problems
- `"adamw"` - Better for graph problems, includes weight decay
- `"sgd"` - Classic, for research comparisons
- `"rmsprop"` - Alternative adaptive optimizer

**Example:**
```json
{
    "optimizer": "adamw",
    "weight_decay": 1e-4  // Used by adamw
}
```

---

#### `lr` (Learning Rate)
**Type:** float
**Auto-selected defaults:**
- 1e-4 for regression (sine)
- 1e-3 for classification (mnist, cifar)
- 1e-4 for graph (synthetic)

**When to override:** Fine-tuning convergence, different optimizer

**Example:**
```json
{
    "lr": 5e-4  // Custom learning rate
}
```

---

#### `weight_decay`
**Type:** float
**Default:** 1e-4
**Description:** L2 regularization (used by AdamW)
**When to override:** Regularization tuning

---

#### `momentum`
**Type:** float
**Default:** 0.9
**Description:** Momentum for SGD/RMSprop
**When to override:** Using SGD with different momentum

---

### Learning Rate Schedule

#### `lr_schedule`
**Type:** string
**Options:** `"constant"`, `"step"`, `"exponential"`, `"cosine"`, `"linear"`
**Default:** `"constant"`

**Schedule descriptions:**
- `"constant"` - No decay, LR stays fixed
- `"step"` - Reduce by `lr_decay_factor` every `lr_decay_steps` tasks
- `"exponential"` - Continuous exponential decay: `lr * (decay_factor ^ task_id)`
- `"cosine"` - Smooth cosine annealing from lr to lr_min
- `"linear"` - Linear decay from lr to lr_min

**When to override:** Long training runs with many tasks

**Example:**
```json
{
    "lr_schedule": "exponential",
    "lr_decay_factor": 0.95,  // 5% decay per task
    "lr_min": 1e-6  // Floor
}
```

---

#### `lr_decay_factor`
**Type:** float
**Default:** 0.9
**Description:** Decay factor for step/exponential schedules

---

#### `lr_decay_steps`
**Type:** int
**Default:** 1
**Description:** Tasks between decay for step schedule

---

#### `lr_min`
**Type:** float
**Default:** 1e-6
**Description:** Minimum learning rate floor

---

### Hamiltonian Gradient Weights

#### `flag`
**Type:** list of 2 floats
**Default:** `[1.0, 1.0]`
**Description:** Regularization weights `[dV/dx, dV/dtheta]`
**When to override:** Research on Hamiltonian regularization

**Example:**
```json
{
    "flag": [0.5, 1.0]  // Reduce input perturbation sensitivity
}
```

---

#### `grad_weights`
**Type:** list of 3 floats
**Default:** `[0.01, 0.98, 0.1]`
**Description:** Gradient combination `[alpha, beta, gamma]`

**Gradient formula:**
```
grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
```

**Components:**
- `alpha` - Current task gradient weight (↓ = less forgetting)
- `beta` - Experience replay gradient weight (↑ = more retention)
- `gamma` - Hamiltonian regularization weight (stability)

**When to override:** Studying forgetting/retention tradeoffs

**Example:**
```json
{
    "grad_weights": [0.05, 0.90, 0.05]  // More current task focus
}
```

---

### Model Architecture

Architecture parameters have **dataset-specific smart defaults** but can be overridden.

#### MLP/FCNN (for sine)

##### `n_layers`
**Type:** int
**Default:** 4
**Description:** Total number of MLP layers (including input/output)

##### `hln`
**Type:** int
**Default:** 256
**Description:** Hidden layer size

**Example:**
```json
{
    "n_layers": 3,
    "hln": 128  // Smaller network
}
```

---

#### CNN (for mnist)

##### `filter_size`
**Type:** int
**Default:** 4
**Description:** Convolution filter size (square)

##### `channel_out`
**Type:** int
**Default:** 3
**Description:** Output channels from convolutional layers

##### `feed_sizes`
**Type:** list of ints
**Default:** `[1875, 512, 64, 10]` (MNIST)
**Description:** Feed-forward layer sizes after convolution

**Auto-set dataset properties (DO NOT override):**
- `channel_in` - 1 for MNIST (grayscale)
- `input_size` - 28 for MNIST

**Example:**
```json
{
    "filter_size": 3,
    "channel_out": 5,
    "feed_sizes": [2000, 256, 10]  // Custom architecture
}
```

---

#### CNN3D (for cifar10, cifar100)

##### `filter_size`
**Type:** int
**Default:** 4
**Description:** Convolution filter size (square)

##### `channel_out`
**Type:** int
**Default:** 32
**Description:** Output channels from convolutional layers

##### `feed_sizes`
**Type:** list of ints
**Default:** `[2304, 512, 256, 10]` (CIFAR-10) or `[2304, 512, 256, 100]` (CIFAR-100)
**Description:** Feed-forward layer sizes after convolution

**Auto-set dataset properties (DO NOT override):**
- `channel_in` - 3 for CIFAR (RGB)
- `input_size` - 32 for CIFAR

**Example:**
```json
{
    "filter_size": 5,
    "feed_sizes": [3000, 1024, 512, 10]  // Larger network
}
```

---

#### GCN (for synthetic graphs)

##### `gcn_sizes`
**Type:** list of ints
**Default:** `[None, 128]` (first element auto-set to input features)
**Description:** GCN layer sizes

##### `feed_sizes`
**Type:** list of ints
**Default:** `[128, 128, 128, 10]`
**Description:** MLP layer sizes after GCN

**Example:**
```json
{
    "gcn_sizes": [5, 256],  // 5 input features, 256 hidden
    "feed_sizes": [256, 256, 10]  // Match GCN output
}
```

---

### Dataset-Specific Parameters

#### Sine Dataset

##### `delta`
**Type:** float
**Default:** 0.001
**Description:** Task drift perturbation magnitude

##### `test_size`
**Type:** float
**Default:** 0.2
**Description:** Test set fraction (0.0 to 1.0)

##### `data_path`
**Type:** string
**Default:** `"data/Incremental_Sine1e^4.p"`
**Description:** Path to sine dataset file

**Example:**
```json
{
    "delta": 0.005,  // Larger task drift
    "data_path": "data/custom_sine.p"
}
```

---

#### MNIST/CIFAR (Data Augmentation)

##### `rotation_range`
**Type:** int
**Default:** 180
**Description:** Random rotation range in degrees

##### `scaling_range`
**Type:** tuple of 2 floats
**Default:** `(1, 2)`
**Description:** Random scaling range (min, max)

**Example:**
```json
{
    "rotation_range": 90,  // Less rotation
    "scaling_range": [0.8, 1.2]  // Different scaling
}
```

---

#### Synthetic Graph Dataset

##### `num_graphs`
**Type:** int
**Default:** 1000
**Description:** Number of synthetic graphs to generate

##### `num_channels`
**Type:** int
**Default:** 5
**Description:** Node feature channels

##### `avg_num_nodes`
**Type:** int
**Default:** 2
**Description:** Average number of nodes per graph

##### `num_classes`
**Type:** int
**Default:** 10
**Description:** Number of graph classes

##### `class_per_task`
**Type:** int
**Default:** 2
**Description:** Classes per incremental task

**Example:**
```json
{
    "num_graphs": 2000,  // More data
    "num_channels": 10,  // Richer features
    "avg_num_nodes": 5   // Larger graphs
}
```

---

### AWB (Adaptive Weight Basis) Pipeline

#### `awb_enabled`
**Type:** bool
**Default:** false
**Description:** Enable AWB architecture morphing pipeline

**When to enable:** Experiments with adaptive architecture expansion

**Example:**
```json
{
    "awb_enabled": true
}
```

---

#### `awb_preliminary_epochs`
**Type:** int
**Default:** 10
**Description:** STEP 1 - Preliminary training epochs before architecture decision

---

#### `awb_ab_training_epochs`
**Type:** int
**Default:** 50
**Description:** STEP 3b - Epochs to train A/B matrices with W frozen

---

#### `awb_ab_warmup_epochs`
**Type:** int
**Default:** 2
**Description:** STEP 5 - Warmup epochs after V = A @ W @ B.T computation

---

#### `awb_ab_max_iterations`
**Type:** int
**Default:** 8
**Description:** Maximum iterations for A/B convergence loop

---

#### `awb_averaging_window`
**Type:** int
**Default:** 10
**Description:** Window size (epochs) for loss averaging

---

#### `awb_arch`
**Type:** list of ints
**Default:** Architecture-specific
**Description:** Target architecture for AWB expansion

**Example AWB config:**
```json
{
    "awb_enabled": true,
    "awb_preliminary_epochs": 5,
    "awb_ab_training_epochs": 20,
    "awb_arch": [1728, 700, 100, 10]  // Target architecture
}
```

---

### Architecture Search

#### `arch_search_enabled`
**Type:** bool
**Default:** false
**Description:** Enable architecture search (only exposed parameter)

**Note:** All other architecture search parameters (epochs, lr, thresholds, ranges) are internal constants and not configurable.

**Example:**
```json
{
    "arch_search_enabled": true  // Enable search
}
```

---

### Debug & Output

#### `debug_mode`
**Type:** bool
**Default:** false
**Description:** Enable debug mode with limited data samples

**When to enable:** Quick testing, development

---

#### `debug_limit`
**Type:** int
**Default:** 100
**Description:** Number of samples when debug_mode=True

---

#### `model_path`
**Type:** string
**Default:** `"outputs/model"`
**Description:** Path to save model checkpoints and training records

**Example:**
```json
{
    "debug_mode": true,
    "debug_limit": 50,
    "model_path": "experiments/exp_001/model"
}
```

---

## Complete Examples

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

### MNIST with Custom LR Schedule
```json
{
    "data": "mnist",
    "n_task": 10,
    "epochs_per_task": 40,
    "model_path": "outputs/mnist_exp",
    "lr_schedule": "exponential",
    "lr_decay_factor": 0.95
}
```

### CIFAR-10 with AWB
```json
{
    "data": "cifar10",
    "n_task": 5,
    "epochs_per_task": 100,
    "model_path": "outputs/cifar10_awb",
    "awb_enabled": true,
    "awb_preliminary_epochs": 10,
    "awb_ab_training_epochs": 30
}
```

### Synthetic Graphs with Custom Architecture
```json
{
    "data": "synthetic",
    "n_task": 5,
    "epochs_per_task": 50,
    "model_path": "outputs/graph_exp",
    "optimizer": "adamw",
    "gcn_sizes": [5, 256],
    "feed_sizes": [256, 256, 128, 10],
    "num_graphs": 2000
}
```

---

## Tips

1. **Start minimal** - Only specify `data`, `n_task`, `epochs_per_task`, `model_path`
2. **Check defaults** - Review `src/cl/config/constants.py` for all default values
3. **Override selectively** - Only add parameters you want to change
4. **Use comments** - Add `"__comment_*"` fields to document your choices
5. **Test with debug** - Use `"debug_mode": true` for quick validation

---

## Master Parameter Reference Table

Complete list of all parameters with defaults, configurability, and code locations:

### Core Configuration

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `data` | string | `"sine"` | **Required** - User specifies | Config file |
| `prob` | string | Auto-selected | Auto (from dataset) | `constants.py:DATASET_CONFIG_MAP` |
| `problem` | string | Auto-selected | Auto (from dataset) | `constants.py:DATASET_CONFIG_MAP` |
| `network` | string | Auto-selected | Auto (from dataset) | `constants.py:DATASET_CONFIG_MAP` |
| `loss` | string | Auto-selected | Auto (from dataset) | `constants.py:DATASET_CONFIG_MAP` |
| `metric` | string | Auto-selected | Auto (from dataset) | `constants.py:DATASET_CONFIG_MAP` |

### Training Loop

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `n_task` | int | `5` | **Yes** - Experimental variable | Config file |
| `epochs_per_task` | int | `100` | **Yes** - Experimental variable | Config file |
| `save_iter` | int | `10` | **Yes** - Override if needed | Config file |
| `batch_size` | int | Problem-specific (20/64/128) | Smart default + override | `constants.py:102-108, params.py:100-108` |
| `len_exp_replay` | int | Problem-specific (20k/200k) | Smart default + override | `constants.py:110-116, params.py:110-116` |

### Optimizer

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `optimizer` | string | `"adam"` | Smart default + override | `constants.py:119, params.py:119` |
| `lr` | float | Problem-specific (1e-4/1e-3) | Smart default + override | `constants.py:120-122, params.py:124-132` |
| `weight_decay` | float | `1e-4` | Smart default + override | `constants.py:123, params.py:120` |
| `momentum` | float | `0.9` | Smart default + override | `constants.py:124, params.py:121` |

### Learning Rate Schedule

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `lr_schedule` | string | `"constant"` | Smart default + override | `constants.py:129, params.py:135` |
| `lr_decay_factor` | float | `0.9` | Smart default + override | `constants.py:130, params.py:136` |
| `lr_decay_steps` | int | `1` | Smart default + override | `constants.py:131, params.py:137` |
| `lr_min` | float | `1e-6` | Smart default + override | `constants.py:132, params.py:138` |

### Hamiltonian Gradients

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `flag` | list[float, float] | `[1.0, 1.0]` | Smart default + override | `constants.py:137, params.py:141` |
| `grad_weights` | list[float, float, float] | `[0.01, 0.98, 0.1]` | Smart default + override | `constants.py:138, params.py:142` |

### Debug & Output

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `debug_mode` | bool | `false` | Smart default + override | `constants.py:143, params.py:145` |
| `debug_limit` | int | `100` | Smart default + override | `constants.py:144, params.py:146` |
| `model_path` | string | `"outputs/model"` | Smart default + override | `constants.py:105, params.py:97` |

### MLP Architecture (for sine)

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `n_layers` | int | `4` | Smart default + override | `constants.py:149, params.py:153` |
| `hln` | int | `256` | Smart default + override | `constants.py:150, params.py:154` |

### CNN Architecture (for mnist)

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `filter_size` | int | `4` | Smart default + override | `constants.py:155, params.py:158` |
| `channel_out` | int | `3` | Smart default + override | `constants.py:156, params.py:163` |
| `channel_in` | int | `1` (MNIST) | **Auto-constant** (dataset property) | `constants.py:158, params.py:164` |
| `input_size` | int | `28` (MNIST) | **Auto-constant** (dataset property) | `constants.py:159, params.py:165` |
| `feed_sizes` | list[int] | `[1875, 512, 64, 10]` | Smart default + override | `constants.py:168, params.py:166` |

### CNN3D Architecture (for cifar10, cifar100)

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `filter_size` | int | `4` | Smart default + override | `constants.py:155, params.py:158` |
| `channel_out` | int | `32` | Smart default + override | `constants.py:157, params.py:168` |
| `channel_in` | int | `3` (CIFAR) | **Auto-constant** (dataset property) | `constants.py:159, params.py:169` |
| `input_size` | int | `32` (CIFAR) | **Auto-constant** (dataset property) | `constants.py:160, params.py:170` |
| `feed_sizes` | list[int] | `[2304, 512, 256, 10]` or `[..., 100]` | Smart default + override | `constants.py:169, params.py:171` |

### GCN Architecture (for synthetic)

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `gcn_sizes` | list[int] | `[None, 128]` | Smart default + override | `constants.py:174, params.py:175` |
| `feed_sizes` | list[int] | `[128, 128, 128, 10]` | Smart default + override | `constants.py:175, params.py:176` |

### Dataset-Specific: Sine

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `delta` | float | `0.001` | Smart default + override | `constants.py:188, params.py:186` |
| `test_size` | float | `0.2` | Smart default + override | `constants.py:191, params.py:187` |
| `data_path` | string | `"data/Incremental_Sine1e^4.p"` | Smart default + override | `constants.py:190` |

### Dataset-Specific: MNIST/CIFAR

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `rotation_range` | int | `180` | Smart default + override | `constants.py:194, params.py:190` |
| `scaling_range` | tuple | `(1, 2)` | Smart default + override | `constants.py:195, params.py:191` |
| `permutation_seed_multiplier` | int | `1000` | **Hard constant** | `constants.py:196` |

### Dataset-Specific: Synthetic Graphs

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `num_graphs` | int | `1000` | Smart default + override | `constants.py:199, params.py:196` |
| `num_channels` | int | `5` | Smart default + override | `constants.py:200, params.py:197` |
| `avg_num_nodes` | int | `2` | Smart default + override | `constants.py:201, params.py:198` |
| `num_classes` | int | `10` | Smart default + override | `constants.py:202, params.py:199` |
| `class_per_task` | int | `2` | Smart default + override | `constants.py:181, params.py:200` |

### AWB Pipeline

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `awb_enabled` | bool | `false` | Smart default + override | `constants.py:209, params.py:203` |
| `awb_preliminary_epochs` | int | `10` | Smart default + override | `constants.py:212, params.py:206` |
| `awb_ab_training_epochs` | int | `50` | Smart default + override | `constants.py:213, params.py:207` |
| `awb_ab_warmup_epochs` | int | `2` | Smart default + override | `constants.py:214, params.py:208` |
| `awb_ab_max_iterations` | int | `8` | Smart default + override | `constants.py:215, params.py:209` |
| `awb_averaging_window` | int | `10` | Smart default + override | `constants.py:216, params.py:210` |
| `awb_arch` | list[int] | Architecture-specific | Smart default + override | Config file |

### Architecture Search

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `arch_search_enabled` | bool | `false` | Smart default + override | `constants.py:235, params.py:213` |
| All other arch_search_* | various | See constants.py | **Hard constants** | `constants.py:236-265` |

### Internal Framework Constants (Not Exposed)

| Parameter | Type | Default | Configurable | Code Location |
|-----------|------|---------|--------------|---------------|
| `DEFAULT_SEED` | int | `5678` | **Hard constant** | `constants.py:13` |
| `DEFAULT_GRAPH_SEED` | int | `10` | **Hard constant** | `constants.py:14` |
| `DEFAULT_PADDING` | int | `0` | **Hard constant** | `constants.py:162` |
| `DEFAULT_STRIDE` | int | `1` | **Hard constant** | `constants.py:163` |
| `DEFAULT_POOL_SIZE` | int | `2` | **Hard constant** | `constants.py:164` |
| `DEFAULT_POOL_STRIDE` | int | `2` | **Hard constant** | `constants.py:165` |
| Random key offsets | int | Various | **Hard constants** | `constants.py:268-270` |

---

## Code Locations Summary

### Where to Add New Defaults

1. **Add constant to `src/cl/config/constants.py`:**
   ```python
   DEFAULT_MY_PARAMETER = value
   ```

2. **Add to dataset mapping (if dataset-specific):**
   ```python
   # In constants.py:DATASET_CONFIG_MAP
   "my_dataset": {
       "prob": "...",
       "my_param": value
   }
   ```

3. **Add to `apply_defaults()` in `src/cl/config/params.py`:**
   ```python
   set_default('my_parameter', constants.DEFAULT_MY_PARAMETER)
   ```

4. **Use in code:**
   ```python
   my_param = config.get('my_parameter', constants.DEFAULT_MY_PARAMETER)
   ```

### Key Files

- **`src/cl/config/constants.py`** - All default values (203 lines)
- **`src/cl/config/params.py`** - Config loading with auto-defaults (apply_defaults function, ~180 lines)
- **`config/TEMPLATE.md`** - This file - Parameter documentation
- **`config/*.json`** - Minimal user config files

---

## See Also

- `config/0_config_readme.md` - Configuration system overview
- `src/cl/config/constants.py` - All default constants
- `src/cl/config/params.py` - Config loading logic
- `LAYER_AWB_ARCHITECTURE.md` - Plugin-and-play architecture guide
