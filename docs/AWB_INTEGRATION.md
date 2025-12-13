# AWB (Adaptive Weight Basis) Integration Guide

This document describes the changes made to integrate the 5-step AWB continual learning algorithm into the codebase, and provides a roadmap for extending this to CNN, CNN3D, and GNN models.

---

## Table of Contents

1. [Overview of AWB Algorithm](#overview-of-awb-algorithm)
2. [Files Modified/Created](#files-modifiedcreated)
3. [Detailed Changes](#detailed-changes)
4. [Configuration Options](#configuration-options)
5. [Extending to CNN Models](#extending-to-cnn-models)
6. [Extending to CNN3D Models](#extending-to-cnn3d-models)
7. [Extending to GNN Models](#extending-to-gnn-models)
8. [Testing](#testing)

---

## Overview of AWB Algorithm

The AWB (Adaptive Weight Basis) algorithm enables neural network architecture morphing during continual learning. It uses transformation matrices A and B to smoothly transition between architectures.

### The 5-Step Algorithm

```
Task 0: Standard continual learning training

Tasks 1+:
  STEP 1: Train for preliminary epochs on new task (assess forgetting)
  STEP 2: Decide if architecture change is needed (loss ratio thresholds)

  If architecture change needed:
    STEP 3a: Search for optimal new architecture
    STEP 3b: Initialize A/B matrices, train A/B with W frozen
    STEP 4:  Compute new weights V = A @ W @ B.T
    STEP 5:  Train V with A/B frozen

  Else:
    Continue standard training
```

### Key Concepts

- **W**: Original weight matrices (frozen during A/B training)
- **A**: Output transformation matrix (maps old output dim → new output dim)
- **B**: Input transformation matrix (maps old input dim → new input dim)
- **V**: New weight matrices computed as V = A @ W @ B.T
- **notABTrain**: Flag to switch between standard forward pass and AWB forward pass

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `config/constants.py` | MODIFIED | Added AWB pipeline constants |
| `training/awb_utils.py` | CREATED | AWB helper functions |
| `training/runners.py` | MODIFIED | Added AWB pipeline to `train_model_reg()` |
| `config/jsons/param_sine.json` | MODIFIED | Added AWB configuration options |
| `tests/test_awb_utils.py` | CREATED | Unit tests for AWB utilities |
| `tests/test_awb_training.py` | CREATED | Integration tests for AWB training |

---

## Detailed Changes

### 1. `config/constants.py`

Added AWB-specific constants (lines 47-56):

```python
# AWB Training Pipeline Defaults (5-Step Algorithm)
DEFAULT_AWB_ENABLED = False  # Master switch for AWB pipeline
DEFAULT_AWB_PRELIMINARY_EPOCHS = 250  # STEP 1: Epochs before checking arch change
DEFAULT_AWB_AB_TRAINING_EPOCHS = 2000  # STEP 3b: Epochs to train A/B matrices
DEFAULT_AWB_AB_WARMUP_EPOCHS = 50  # STEP 5: Warmup epochs after V computation
DEFAULT_AWB_CHANGE_THRESHOLD_HIGH = 0.45  # Ratio threshold to trigger arch change
DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA = 0.01  # Minimum loss increase for change
DEFAULT_AWB_AB_THRESHOLD_BASE = 0.6  # Base threshold for AB training convergence
DEFAULT_AWB_AB_MAX_ITERATIONS = 8  # Maximum iterations for AB training loop
DEFAULT_AWB_AVERAGING_WINDOW = 10  # Number of epochs to average for loss
```

### 2. `training/awb_utils.py` (New File)

Created helper functions to keep the main training code clean:

| Function | Purpose |
|----------|---------|
| `compute_avg_loss()` | Compute average loss over last N epochs from record dict |
| `should_change_arch()` | Decision logic for architecture change based on loss ratios |
| `compute_ab_threshold()` | Dynamic threshold for A/B training convergence |
| `set_new_AB_matrices()` | Initialize A/B matrices for architecture transition |
| `compute_V_from_AWB()` | Compute V = A @ W @ B.T for all layers |
| `partition_for_AB_training()` | Partition model to train only A/B (freeze W) |
| `partition_for_standard_training()` | Partition model to freeze A/B (train V) |
| `save_layer_weights()` | Save layer weights before architecture change |
| `restore_layer_weights()` | Restore layer weights if needed |
| `create_optimizer_for_phase()` | Create optimizer for each training phase |

### 3. `training/runners.py`

Modified `train_model_reg()` function (lines 85-330) to implement the full AWB pipeline:

**Key Changes:**

1. **AWB config extraction** (lines 100-108):
```python
awb_enabled = config.get('awb_enabled', DEFAULT_AWB_ENABLED)
awb_preliminary_epochs = config.get('awb_preliminary_epochs', DEFAULT_AWB_PRELIMINARY_EPOCHS)
awb_ab_training_epochs = config.get('awb_ab_training_epochs', DEFAULT_AWB_AB_TRAINING_EPOCHS)
# ... etc
```

2. **Task 0 baseline** (lines 140-160):
```python
if i == 0:
    # Standard training for task 0
    params, static, optim, record_dict[str(i)] = trainer.train__CL__reg(...)
    end_last0 = compute_avg_loss(record_dict[str(i)], i, epochs, awb_averaging_window)
```

3. **AWB pipeline for tasks 1+** (lines 165-320):
```python
if awb_enabled:
    # STEP 1: Preliminary training
    # STEP 2: Check if architecture change needed
    if should_change_arch(trainWLoss, end_last0, end_last):
        # STEP 3a: Architecture search
        # STEP 3b: Train A/B with W frozen
        # STEP 4: Compute V = A @ W @ B.T
        # STEP 5: Train V with A/B frozen
```

4. **Backward compatibility** (lines 325-330):
```python
else:
    # AWB disabled - existing behavior unchanged
    params, static, optim, record_dict[str(i)] = trainer.train__CL__reg(...)
```

### 4. `config/jsons/param_sine.json`

Added AWB configuration options:

```json
{
    "__comment_awb": "AWB Pipeline Settings (set awb_enabled=true to activate)",
    "awb_enabled": false,
    "awb_preliminary_epochs": 250,
    "awb_ab_training_epochs": 2000,
    "awb_ab_warmup_epochs": 50,
    "awb_ab_max_iterations": 8,
    "awb_averaging_window": 10
}
```

---

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `awb_enabled` | bool | `false` | Master switch for AWB pipeline |
| `awb_preliminary_epochs` | int | 250 | Epochs for preliminary training (Step 1) |
| `awb_ab_training_epochs` | int | 2000 | Max epochs for A/B training (Step 3b) |
| `awb_ab_warmup_epochs` | int | 50 | Warmup epochs after V computation (Step 5) |
| `awb_ab_max_iterations` | int | 8 | Max iterations of A/B training loop |
| `awb_averaging_window` | int | 10 | Window size for loss averaging |

---

## Extending to CNN Models

To extend AWB to CNN models (`utils/model.py:CNN` class), follow these steps:

### Step 1: Verify CNN has A/B matrices

The CNN class already has A/B matrices defined (lines 286-289):
```python
class CNN(eqx.Module):
    A_conv: jax.Array
    B_conv: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
```

And already has `get_AWBT()` method (lines 380-420).

### Step 2: Create `training/awb_utils_cnn.py`

Create CNN-specific AWB utilities:

```python
"""AWB utilities for CNN models."""

import jax
import jax.numpy as jnp
import equinox as eqx

def set_new_AB_matrices_cnn(model, original_arch, new_arch):
    """Initialize A/B matrices for CNN architecture transition.

    For CNN, we need to handle:
    - A_conv/B_conv for convolutional layers
    - A_feed/B_feed for feed-forward layers

    Args:
        model: CNN model
        original_arch: Dict with 'conv_sizes' and 'feed_sizes'
        new_arch: Dict with new 'conv_sizes' and 'feed_sizes'

    Returns:
        Updated model with new A/B matrices
    """
    initializer = jax.nn.initializers.glorot_uniform()
    key = jax.random.PRNGKey(5)

    # For conv layers: A_conv maps channel_out, B_conv maps channel_in
    # Shape depends on filter dimensions
    old_channel_out = original_arch['channel_out']
    new_channel_out = new_arch['channel_out']
    old_channel_in = original_arch['channel_in']
    new_channel_in = new_arch['channel_in']
    filter_size = model.filter_size

    # A_conv: list of [channel_out][channel_in] matrices of shape (filter, filter)
    A_conv = [[initializer(key, (filter_size, filter_size))
               for _ in range(new_channel_in)]
              for _ in range(new_channel_out)]

    # B_conv: similar structure
    B_conv = [[initializer(key, (filter_size, filter_size))
               for _ in range(old_channel_in)]
              for _ in range(old_channel_out)]

    # For feed layers: same as MLP
    old_feed = original_arch['feed_sizes']
    new_feed = new_arch['feed_sizes']

    A_feed = [initializer(key, (y, x))
              for x, y in zip(old_feed[1:], new_feed[1:])]
    B_feed = [initializer(key, (y, x))
              for x, y in zip(old_feed[:-1], new_feed[:-1])]

    model = eqx.tree_at(lambda x: x.A_conv, model, A_conv)
    model = eqx.tree_at(lambda x: x.B_conv, model, B_conv)
    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed)

    return model


def compute_V_from_AWB_cnn(model):
    """Compute V = A @ W @ B.T for CNN layers.

    For convolutional layers, the transformation is applied per-filter.
    For feed-forward layers, same as MLP.
    """
    # Conv layer transformation
    # V_conv[i][c] = A_conv[i][c] @ W_conv[i][c] @ B_conv[i][c].T
    # This is already implemented in CNN.get_AWBT()

    # Feed layer transformation (same as MLP)
    for j in range(len(model.feed_sizes) - 1):
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        Vb = model.feed_layers[j].bias @ model.A_feed[j].T
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)

    return model


def partition_for_AB_training_cnn(model):
    """Partition CNN for A/B training (freeze conv/feed weights, train A/B)."""
    import jax.tree_util as jtu

    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        filter_spec,
        replace=(True, True, True, True)
    )
    return eqx.partition(model, filter_spec)
```

### Step 3: Create `train_model_class_cnn()` in `training/runners.py`

Add a new function or modify `train_model_class()` to handle AWB for CNN:

```python
def train_model_class_cnn(config):
    """Train CNN classifier with optional AWB pipeline."""
    from training.awb_utils_cnn import (
        set_new_AB_matrices_cnn,
        compute_V_from_AWB_cnn,
        partition_for_AB_training_cnn
    )

    awb_enabled = config.get('awb_enabled', False)

    # ... similar structure to train_model_reg() ...

    for i in range(n_task):
        if i == 0:
            # Standard training
            pass
        elif awb_enabled:
            # AWB pipeline
            # STEP 1: Preliminary training
            # STEP 2: Check architecture change
            if should_change_arch(...):
                # STEP 3a: Architecture search for CNN
                new_arch = arch_search_CNN(...)  # Need to implement

                # STEP 3b: Set A/B and train
                model = set_new_AB_matrices_cnn(model, original_arch, new_arch)
                diff_model, static_model = partition_for_AB_training_cnn(model)
                # Train A/B...

                # STEP 4: Compute V
                model = compute_V_from_AWB_cnn(model)

                # STEP 5: Train V
                # ...
```

### Step 4: Create architecture search for CNN

Create `arch_search/cnn_search.py`:

```python
def arch_search_CNN(model, data, config):
    """Search for optimal CNN architecture.

    Search space:
    - Number of filters per conv layer
    - Hidden layer sizes in feed-forward part

    Returns:
        Dict with 'channel_out', 'channel_in', 'feed_sizes'
    """
    # Implement architecture search logic
    # Can use similar approach to MLP search
    pass
```

---

## Extending to CNN3D Models

The CNN3D model (`utils/model.py:CNN3D` class) has similar structure to CNN but with 3D convolutions for CIFAR.

### Step 1: Verify CNN3D A/B matrices

CNN3D already has A/B matrices (lines 468-475):
```python
class CNN3D(eqx.Module):
    A_conv1: jax.Array  # For first conv layer
    B_conv1: jax.Array
    A_conv2: jax.Array  # For second conv layer
    B_conv2: jax.Array
    A_feed: jax.Array   # For feed-forward layers
    B_feed: jax.Array
```

And has `get_AWBT()` method (lines 529-570).

### Step 2: Create `training/awb_utils_cnn3d.py`

```python
"""AWB utilities for CNN3D models."""

def set_new_AB_matrices_cnn3d(model, original_arch, new_arch):
    """Initialize A/B matrices for CNN3D architecture transition.

    CNN3D has two conv layers, so we need:
    - A_conv1/B_conv1 for first conv layer
    - A_conv2/B_conv2 for second conv layer
    - A_feed/B_feed for feed-forward layers
    """
    # Similar to CNN but with two conv layer pairs
    pass


def compute_V_from_AWB_cnn3d(model):
    """Compute V = A @ W @ B.T for CNN3D layers."""
    # Handle both conv layers and feed layers
    pass


def partition_for_AB_training_cnn3d(model):
    """Partition CNN3D for A/B training."""
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
        filter_spec,
        replace=(True, True, True, True, True, True)
    )
    return eqx.partition(model, filter_spec)
```

### Step 3: Modify training runner

Add AWB support to `train_model_class()` for CNN3D models:

```python
if config['network'] == 'cnn3d' and awb_enabled:
    from training.awb_utils_cnn3d import (
        set_new_AB_matrices_cnn3d,
        compute_V_from_AWB_cnn3d,
        partition_for_AB_training_cnn3d
    )
    # Use CNN3D-specific functions
```

---

## Extending to GNN Models

GNN models (`utils/model.py:myNN` class) require special handling due to graph structure.

### Step 1: Verify GNN A/B matrices

The myNN class already has A/B matrices for feed layers (lines 770-777):
```python
class myNN(eqx.Module):
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
```

And has `get_AWBT()` method (lines 854-880).

**Note**: GCN layers don't currently have A/B matrices. You may need to add them.

### Step 2: Add A/B to GCN layers (if needed)

Modify `utils/model.py:GCN` class:

```python
class GCN(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    A: jax.Array  # ADD: Output transformation
    B: jax.Array  # ADD: Input transformation
    # ...

    def __init__(self, in_size, out_size, key, bias=True, sparse=False):
        # ... existing init ...
        # Initialize A/B as identity
        self.A = jnp.eye(out_size)
        self.B = jnp.eye(in_size)

    def get_AWBT(self, x, adj):
        """Forward pass with AWB transformation."""
        # V = A @ W @ B.T
        V = self.A @ self.weight @ self.B.T
        support = x @ V
        x = self.matmul(adj, support, support.shape)
        if self.bias_flag:
            x += self.A @ self.bias  # Transform bias too
        return x
```

### Step 3: Create `training/awb_utils_gnn.py`

```python
"""AWB utilities for GNN models."""

def set_new_AB_matrices_gnn(model, original_arch, new_arch):
    """Initialize A/B matrices for GNN architecture transition.

    GNN has:
    - GCN layers with their own A/B (if added)
    - Feed-forward layers with A_feed/B_feed

    Args:
        original_arch: Dict with 'gcn_sizes' and 'feed_sizes'
        new_arch: Dict with new sizes
    """
    initializer = jax.nn.initializers.glorot_uniform()
    key = jax.random.PRNGKey(5)

    # For GCN layers (if A/B added to GCN)
    old_gcn = original_arch.get('gcn_sizes', [])
    new_gcn = new_arch.get('gcn_sizes', [])

    # Update each GCN layer's A/B
    new_gcn_layers = []
    for j, layer in enumerate(model.gcn_layers):
        A = initializer(key, (new_gcn[j+1], old_gcn[j+1]))
        B = initializer(key, (new_gcn[j], old_gcn[j]))
        # Create new layer with updated A/B
        new_layer = eqx.tree_at(lambda x: x.A, layer, A)
        new_layer = eqx.tree_at(lambda x: x.B, new_layer, B)
        new_gcn_layers.append(new_layer)

    model = eqx.tree_at(lambda x: x.gcn_layers, model, new_gcn_layers)

    # For feed layers (same as MLP)
    old_feed = original_arch['feed_sizes']
    new_feed = new_arch['feed_sizes']

    A_feed = [initializer(key, (y, x))
              for x, y in zip(old_feed[1:], new_feed[1:])]
    B_feed = [initializer(key, (y, x))
              for x, y in zip(old_feed[:-1], new_feed[:-1])]

    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed)

    return model


def compute_V_from_AWB_gnn(model):
    """Compute V = A @ W @ B.T for GNN layers."""
    # GCN layers
    for j, layer in enumerate(model.gcn_layers):
        Vw = layer.A @ layer.weight @ jnp.transpose(layer.B)
        model = eqx.tree_at(
            lambda x, idx=j: x.gcn_layers[idx].weight,
            model, Vw
        )

    # Feed layers
    for j in range(len(model.feed_sizes) - 1):
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)

    return model


def partition_for_AB_training_gnn(model):
    """Partition GNN for A/B training."""
    # Need to make A/B in GCN layers and feed layers trainable
    # while keeping weights frozen
    pass
```

### Step 4: Create architecture search for GNN

Create `arch_search/gnn_search.py`:

```python
def arch_search_GNN(model, data, config):
    """Search for optimal GNN architecture.

    Search space:
    - Hidden sizes in GCN layers
    - Hidden sizes in feed-forward layers
    - Number of GCN layers (optional)

    Constraints:
    - Input size fixed by node features
    - Output size fixed by number of classes
    """
    pass
```

### Step 5: Modify training runner for GNN

Add AWB support to `train_model_graph()`:

```python
def train_model_graph(config):
    """Train GNN with optional AWB pipeline."""
    awb_enabled = config.get('awb_enabled', False)

    if awb_enabled:
        from training.awb_utils_gnn import (
            set_new_AB_matrices_gnn,
            compute_V_from_AWB_gnn,
            partition_for_AB_training_gnn
        )

    for i in range(n_task):
        # Similar AWB pipeline structure
        pass
```

---

## Testing

### Existing Tests

- `tests/test_awb_utils.py`: 23 unit tests for AWB utilities
- `tests/test_awb_training.py`: 11 integration tests for AWB pipeline

### Tests to Add for CNN/CNN3D/GNN

Create the following test files:

1. `tests/test_awb_utils_cnn.py`:
```python
def test_set_new_AB_matrices_cnn():
    """Test A/B matrix initialization for CNN."""
    pass

def test_compute_V_from_AWB_cnn():
    """Test V computation for CNN."""
    pass

def test_partition_for_AB_training_cnn():
    """Test model partitioning for CNN."""
    pass
```

2. `tests/test_awb_utils_cnn3d.py`: Similar tests for CNN3D

3. `tests/test_awb_utils_gnn.py`: Similar tests for GNN

4. Integration tests:
```python
def test_train_model_class_with_awb_cnn():
    """Test CNN classification training with AWB enabled."""
    pass

def test_train_model_graph_with_awb():
    """Test GNN training with AWB enabled."""
    pass
```

---

## Summary Checklist

### For Each Model Type (CNN, CNN3D, GNN):

- [ ] Verify A/B matrices exist in model class (add if needed)
- [ ] Verify `get_AWBT()` method exists (add if needed)
- [ ] Create `training/awb_utils_{model}.py` with:
  - [ ] `set_new_AB_matrices_{model}()`
  - [ ] `compute_V_from_AWB_{model}()`
  - [ ] `partition_for_AB_training_{model}()`
  - [ ] `partition_for_standard_training_{model}()`
- [ ] Create `arch_search/{model}_search.py` for architecture search
- [ ] Modify training runner to use AWB pipeline
- [ ] Add configuration options to JSON configs
- [ ] Create unit tests
- [ ] Create integration tests
- [ ] Update this documentation

---

## Implementation TODO for CNN/CNN3D/GNN

This section contains the detailed implementation tasks and existing code context needed to extend AWB to the remaining model types.

### Current Status

| Model | AWB Implemented | Architecture Search | Tests |
|-------|-----------------|---------------------|-------|
| MLP (regression) | ✅ Complete | ✅ `arch_search/mlp_search.py` | ✅ Complete |
| CNN (MNIST) | ❌ Pending | ✅ `arch_search/cnn_search.py:arch_search_CNN()` | ❌ Pending |
| CNN3D (CIFAR) | ❌ Pending | ✅ `arch_search/cnn_search.py:arch_search_CNN3D()` | ❌ Pending |
| GNN (Graph) | ❌ Pending | ✅ `arch_search/gcn_search.py:arch_search_GCN()` | ❌ Pending |

### Existing Architecture Search Functions

The following architecture search functions already exist and should be reused:

#### 1. CNN Architecture Search (`arch_search/cnn_search.py`)

```python
# Line 19-177
def arch_search_CNN(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp):
    """
    Returns: opt_mlp (list), opt_filter (int)
    """

# Line 366-407: Existing prepABs() function for CNN
def prepABs(model, prev_feed_sizes, prev_filter_size):
    """Prepare A and B transformation matrices for CNN architecture search."""
    # Returns: A_feed, B_feed, A_conv, B_conv
```

#### 2. CNN3D Architecture Search (`arch_search/cnn_search.py`)

```python
# Line 180-363
def arch_search_CNN3D(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                      dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp):
    """
    Returns: opt_mlp (list), opt_filter (int)
    """

# Line 410-475: Existing prepABs_CNN3D() function
def prepABs_CNN3D(model, prev_feed_sizes, prev_filter_size):
    """Prepare A and B transformation matrices for CNN3D architecture search."""
    # Returns: A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2
```

#### 3. GNN Architecture Search (`arch_search/gcn_search.py`)

```python
# Line 14-157
def arch_search_GCN(original_gcn, original_mlp, task, trainW_loss, og_epochs, config,
                    train_loader, mem_train_loader, test):
    """
    Returns: opt_gcn (list), opt_mlp (list)
    """
```

### Implementation Tasks

#### Task 1: Create `training/awb_utils_cnn.py`

**File to create:** `training/awb_utils_cnn.py`

**Functions needed:**
```python
def set_new_AB_matrices_cnn(model, original_arch, new_arch):
    """Use prepABs() from arch_search/cnn_search.py as reference."""
    # A_conv: [channel_out] list of (filter_size, filter_size) matrices
    # B_conv: [channel_out] list of (filter_size, filter_size) matrices
    # A_feed: same as MLP
    # B_feed: same as MLP
    pass

def compute_V_from_AWB_cnn(model):
    """Compute V = A @ W @ B.T for CNN layers."""
    # For conv layers: reference CNN.get_AWBT() at utils/model.py:380-420
    # For feed layers: same as MLP
    pass

def partition_for_AB_training_cnn(model):
    """Partition to train only A_conv, B_conv, A_feed, B_feed."""
    pass

def partition_for_standard_training_cnn(model):
    """Partition to freeze A/B, train weights."""
    pass

def save_layer_weights_cnn(model):
    """Save conv and feed layer weights."""
    pass

def restore_layer_weights_cnn(model, weights):
    """Restore conv and feed layer weights."""
    pass
```

**Key model attributes (from `utils/model.py:CNN`):**
- `model.A_conv` - list of conv A matrices
- `model.B_conv` - list of conv B matrices
- `model.A_feed` - list of feed A matrices
- `model.B_feed` - list of feed B matrices
- `model.feed_sizes` - feed layer sizes
- `model.filter_size` - conv filter size
- `model.channel_out` - number of output channels
- `model.channel_in` - number of input channels (1 for MNIST)

#### Task 2: Create `training/awb_utils_cnn3d.py`

**File to create:** `training/awb_utils_cnn3d.py`

**Functions needed:**
```python
def set_new_AB_matrices_cnn3d(model, original_arch, new_arch):
    """Use prepABs_CNN3D() from arch_search/cnn_search.py as reference."""
    # A_conv1, B_conv1: for first conv layer
    # A_conv2, B_conv2: for second conv layer
    # A_feed, B_feed: for feed layers
    pass

def compute_V_from_AWB_cnn3d(model):
    """Compute V = A @ W @ B.T for CNN3D layers."""
    # Reference CNN3D.get_AWBT() at utils/model.py:529-570
    pass

def partition_for_AB_training_cnn3d(model):
    """Partition to train A_conv1, B_conv1, A_conv2, B_conv2, A_feed, B_feed."""
    pass

def partition_for_standard_training_cnn3d(model):
    """Partition to freeze A/B, train weights."""
    pass
```

**Key model attributes (from `utils/model.py:CNN3D`):**
- `model.A_conv1`, `model.B_conv1` - first conv layer A/B
- `model.A_conv2`, `model.B_conv2` - second conv layer A/B
- `model.A_feed`, `model.B_feed` - feed layer A/B
- `model.feed_sizes` - feed layer sizes
- `model.filter_size` - conv filter size
- `model.channel_out` - output channels after first conv
- `model.channel_in` - input channels (3 for CIFAR)

#### Task 3: Create `training/awb_utils_gnn.py`

**File to create:** `training/awb_utils_gnn.py`

**Functions needed:**
```python
def set_new_AB_matrices_gnn(model, original_arch, new_arch):
    """Initialize A/B for GNN architecture transition."""
    # A_gcn, B_gcn: for GCN layers (already exist in model)
    # A_feed, B_feed: for feed layers
    pass

def compute_V_from_AWB_gnn(model):
    """Compute V = A @ W @ B.T for GNN layers."""
    # Reference myNN.get_AWBT() at utils/model.py:854-880
    pass

def partition_for_AB_training_gnn(model):
    """Partition to train A_gcn, B_gcn, A_feed, B_feed."""
    pass

def partition_for_standard_training_gnn(model):
    """Partition to freeze A/B, train weights."""
    pass
```

**Key model attributes (from `utils/model.py:myNN`):**
- `model.A_gcn`, `model.B_gcn` - GCN layer A/B matrices
- `model.A_feed`, `model.B_feed` - feed layer A/B matrices
- `model.gcn_sizes` - GCN layer sizes
- `model.feed_sizes` - feed layer sizes

#### Task 4: Modify `train_model_class()` in `training/runners.py`

**Current location:** `training/runners.py:391-443`

**Changes needed:**
1. Add AWB imports for CNN/CNN3D
2. Extract AWB config parameters (same as `train_model_reg`)
3. Detect network type: `config['network']` is 'cnn' or 'cnn3d'
4. For task 0: standard training, compute baseline loss
5. For tasks 1+: implement 5-step AWB pipeline
   - STEP 1: Preliminary training
   - STEP 2: `should_change_arch()` decision
   - STEP 3a: Call `arch_search_CNN()` or `arch_search_CNN3D()`
   - STEP 3b: Use `prepABs()` or `prepABs_CNN3D()` to set A/B, train A/B
   - STEP 4: Compute V using model's `get_AWBT()` or new utility
   - STEP 5: Train V with A/B frozen

**Reference code pattern from `train_model_reg()`:** lines 187-347

#### Task 5: Modify `train_model_graph()` in `training/runners.py`

**Current location:** `training/runners.py:41-102`

**Changes needed:**
1. Add AWB imports for GNN
2. Extract AWB config parameters
3. For task 0: standard training, compute baseline loss
4. For tasks 1+: implement 5-step AWB pipeline
   - STEP 1: Preliminary training
   - STEP 2: `should_change_arch()` decision
   - STEP 3a: Call `arch_search_GCN()`
   - STEP 3b: Set A/B matrices, train A/B with notABTrain=False
   - STEP 4: Compute V
   - STEP 5: Train V with notABTrain=True

#### Task 6: Update Config JSON Files

**Files to update:**
- `config/jsons/param_mnist.json` - Add CNN AWB options
- `config/jsons/param_cifar.json` - Add CNN3D AWB options
- `config/jsons/param_graph.json` - Add GNN AWB options

**Options to add (same as param_sine.json):**
```json
{
    "__comment_awb": "AWB Pipeline Settings (set awb_enabled=true to activate)",
    "awb_enabled": false,
    "awb_preliminary_epochs": 250,
    "awb_ab_training_epochs": 2000,
    "awb_ab_warmup_epochs": 50,
    "awb_ab_max_iterations": 8,
    "awb_averaging_window": 10
}
```

#### Task 7: Create Unit Tests

**Test files to create:**
- `tests/test_awb_utils_cnn.py`
- `tests/test_awb_utils_cnn3d.py`
- `tests/test_awb_utils_gnn.py`
- `tests/test_awb_training_cnn.py`
- `tests/test_awb_training_gnn.py`

**Use `tests/test_awb_utils.py` and `tests/test_awb_training.py` as templates.**

### Key Code References

| Component | File | Line Numbers |
|-----------|------|--------------|
| MLP AWB utils | `training/awb_utils.py` | 1-308 |
| MLP training runner | `training/runners.py` | 105-373 |
| CNN model | `utils/model.py` | 283-377 |
| CNN get_AWBT | `utils/model.py` | 380-420 |
| CNN3D model | `utils/model.py` | 456-527 |
| CNN3D get_AWBT | `utils/model.py` | 529-570 |
| GNN model (myNN) | `utils/model.py` | 762-852 |
| GNN get_AWBT | `utils/model.py` | 854-880 |
| CNN arch search | `arch_search/cnn_search.py` | 19-177 |
| CNN prepABs | `arch_search/cnn_search.py` | 366-407 |
| CNN3D arch search | `arch_search/cnn_search.py` | 180-363 |
| CNN3D prepABs | `arch_search/cnn_search.py` | 410-475 |
| GCN arch search | `arch_search/gcn_search.py` | 14-157 |

### Implementation Order Recommendation

1. **CNN first** - Simpler with single conv layer
2. **CNN3D second** - Extends CNN pattern to two conv layers
3. **GNN third** - Different layer types (GCN + feed)

---

## References

- Original AWB implementation: `example.py`
- MLP AWB utilities: `training/awb_utils.py`
- MLP training runner: `training/runners.py:train_model_reg()`
- Model definitions: `utils/model.py`
- Architecture search: `arch_search/` directory
