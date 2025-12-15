# Layer-Level AWB Architecture: Plugin-and-Play Guide

## Overview

The continual learning framework now uses **layer-level AWB abstraction**, enabling true plugin-and-play architecture. AWB operations (V = A @ W @ B.T) are defined at the **layer type level**, not model level, making it easy to add new layers, models, and datasets.

## Architecture Principles

### Key Insight
AWB transformations differ by **LAYER TYPE**, not model type:

| Layer Type | Weight Transform | Bias Transform |
|------------|------------------|----------------|
| Linear (MLP) | `A @ W @ B.T` | `bias @ A.T` |
| Linear2 (CNN feed) | `A @ W @ B.T` | `A @ bias` |
| LinearGCN | `A @ W @ B.T` | `bias @ B.T` |
| Conv2d (single-ch) | Per-filter: `A[i] @ W[i][0] @ B[i].T` | N/A |
| Conv2d (multi-ch) | Per-filter-channel: `A[i][c] @ W[i][c] @ B[i][c].T` | N/A |

### Components

1. **Layer Classes** (`src/cl/models/layers.py`)
   - Each layer type implements AWB methods
   - `AWBLayerSpec` dataclass for validation

2. **Model Interface** (MLP, CNN, CNN3D, GCN)
   - Models implement `get_awb_layer_specs()`, `partition_for_*()`, etc.
   - Models compose layer-level AWB operations

3. **Generic AWB Core** (`src/cl/core/awb.py`)
   - Generic delegates: `apply_V_transformation()`, `partition_model_for_*()`, etc.
   - Backward compatible with legacy code

4. **Generic Runner** (`src/cl/runners/generic_runner.py`)
   - Single unified runner for all problem types
   - Config-based dispatch (regression, classification, graph)

---

## Adding New Layer Types

**Example: Add Attention layer with AWB support**

### Step 1: Define Layer Class

Add to `src/cl/models/layers.py`:

```python
class Attention(eqx.Module):
    """Multi-head attention layer."""
    Q_weight: jax.Array  # Query projection
    K_weight: jax.Array  # Key projection
    V_weight: jax.Array  # Value projection
    O_weight: jax.Array  # Output projection

    def __init__(self, dim, num_heads, key):
        # Initialize weights
        pass

    def __call__(self, x):
        # Standard attention forward pass
        pass

    # AWB methods
    def compute_V_weight(self, A, W, B):
        """Compute V = A @ W @ B.T for attention weights."""
        if A.shape[1] != W.shape[0]:
            raise AWBShapeError(
                f"Attention: A.shape={A.shape} incompatible with W.shape={W.shape}"
            )
        return A @ W @ jnp.transpose(B)

    def compute_V_bias(self, A, B, bias):
        """Transform bias (if any) for attention layer."""
        # Attention-specific bias transformation
        return bias @ A.T  # Or appropriate for your design
```

### Step 2: That's It!

The new layer type now works with:
- All existing models that use it
- Generic AWB pipeline
- All runners (regression, classification, graph)

**No changes needed to**:
- `awb.py`
- Runners
- Trainers
- Other models

---

## Adding New Models

**Example: Add Transformer model with AWB support**

### Step 1: Define Model

Create `src/cl/models/transformer.py`:

```python
import equinox as eqx
from .layers import Attention, Linear, AWBLayerSpec
from typing import List

class Transformer(eqx.Module):
    attention_layers: List[Attention]
    feed_layers: List[Linear]
    A_attn: List[jax.Array]  # A matrices for attention
    B_attn: List[jax.Array]  # B matrices for attention
    A_feed: List[jax.Array]  # A matrices for feed-forward
    B_feed: List[jax.Array]  # B matrices for feed-forward

    def __init__(self, dim, num_layers, num_heads, key):
        # Initialize attention and feed-forward layers
        pass

    def __call__(self, x):
        # Standard transformer forward pass
        pass
```

### Step 2: Implement AWB Interface

Add 3 required methods:

```python
    # Method 1: Get layer specifications
    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB specs for all transformable layers."""
        specs = []

        # Attention layer specs
        for i, layer in enumerate(self.attention_layers):
            specs.append(AWBLayerSpec(
                layer=layer,
                A=self.A_attn[i],
                B=self.B_attn[i],
                layer_type='attention',
                layer_index=i
            ))

        # Feed-forward layer specs
        for i, layer in enumerate(self.feed_layers):
            specs.append(AWBLayerSpec(
                layer=layer,
                A=self.A_feed[i],
                B=self.B_feed[i],
                layer_type='linear',
                layer_index=i + len(self.attention_layers)
            ))

        return specs

    # Method 2: Partition for A/B training (freeze W, train A/B)
    def partition_for_AB_training(self):
        """Partition for A/B training."""
        import jax.tree_util as jtu

        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda x: (x.A_attn, x.B_attn, x.A_feed, x.B_feed),
            filter_spec,
            replace=(True, True, True, True)
        )
        return eqx.partition(self, filter_spec)

    # Method 3: Partition for standard training (freeze A/B, train W)
    def partition_for_standard_training(self):
        """Partition for standard training."""
        params, static = eqx.partition(self, eqx.is_array)

        # Move A/B to static (frozen)
        static = eqx.tree_at(
            lambda x: (x.A_attn, x.B_attn, x.A_feed, x.B_feed),
            static,
            replace=(self.A_attn, self.B_attn, self.A_feed, self.B_feed)
        )

        # Remove A/B from params
        params = eqx.tree_at(
            lambda x: (x.A_attn, x.B_attn, x.A_feed, x.B_feed),
            params,
            replace=(None, None, None, None)
        )

        return params, static
```

### Step 3: Optional - Add Advanced AWB Methods

For full AWB pipeline support, add:

```python
    def apply_V_transformation(self) -> 'Transformer':
        """Apply V = A @ W @ B.T to all layers."""
        model = self

        for i, spec in enumerate(self.get_awb_layer_specs()):
            if spec.layer_type == 'attention':
                # Transform attention weights
                Vw = spec.layer.compute_V_weight(spec.A, spec.layer.Q_weight, spec.B)
                model = eqx.tree_at(
                    lambda x, idx=i: x.attention_layers[idx].Q_weight,
                    model, Vw
                )
                # ... same for K, V, O weights
            elif spec.layer_type == 'linear':
                # Transform feed-forward weights
                Vw = spec.layer.compute_V_weight(spec.A, spec.layer.weight, spec.B)
                Vb = spec.layer.compute_V_bias(spec.A, spec.B, spec.layer.bias)
                # ... update model

        return model

    def with_new_AB_matrices(self, original_arch, new_arch, seed=5):
        """Initialize A/B matrices for architecture transition."""
        # Initialize new A/B matrices for the new architecture
        # ... implementation
        return updated_model
```

### Step 4: Use It!

Your new Transformer model now works with:
- `generic_runner.py` immediately
- All datasets (sine, MNIST, CIFAR, graphs)
- Full 5-step AWB pipeline
- All existing infrastructure

**Example usage**:

```python
# In your config JSON:
{
    "network": "transformer",
    "prob": "classification",
    "data": "cifar10",
    "awb_enabled": true,
    ...
}

# Run with generic runner:
from cl.runners import train_model
records = train_model(config)
```

---

## Adding New Datasets

**Example: Add SVHN dataset**

### Step 1: Define Dataset Class

Create `src/cl/data/svhn.py`:

```python
class SVHNDataset:
    """SVHN dataset for continual learning."""

    def __init__(self, config):
        self.config = config
        self.experience_data = []
        # Load SVHN data

    def generate_dataset(self, task_id, batch_size, phase='train'):
        """Generate dataloaders for current task.

        Args:
            task_id: Current task ID
            batch_size: Batch size
            phase: 'train', 'val', or 'test'

        Returns:
            Tuple of (current_loader, experience_loader)
        """
        # Create task-specific data split
        current_loader = self._create_loader(task_id, batch_size, phase)

        # Create experience replay loader
        if self.experience_data:
            exp_loader = self._create_exp_loader(batch_size)
        else:
            exp_loader = None

        return current_loader, exp_loader

    def append_to_experience(self, task_id):
        """Add current task data to experience replay buffer.

        Args:
            task_id: Task ID to add to experience
        """
        # Add task data to experience buffer
        self.experience_data.append(...)
```

### Step 2: Register Dataset

Add to `src/cl/data/__init__.py`:

```python
from .svhn import SVHNDataset

def get_dataset(config):
    data_name = config.get('data', 'sine')

    if data_name == 'svhn':
        return SVHNDataset(config)
    elif data_name == 'mnist':
        # ... existing datasets
```

### Step 3: Use It!

```json
// config/svhn.json
{
    "data": "svhn",
    "prob": "classification",
    "network": "cnn",
    "n_task": 10,
    "awb_enabled": true,
    ...
}
```

```bash
python scripts/run.py config/svhn.json
```

**That's it!** Works with:
- All models (MLP, CNN, CNN3D, GCN, Transformer)
- Generic runner
- Full AWB pipeline

---

## File Structure

```
src/cl/
├── models/
│   ├── layers.py          # Layer classes with AWB methods
│   │   ├── Linear         # compute_V_weight(), compute_V_bias()
│   │   ├── Linear2        # compute_V_weight(), compute_V_bias()
│   │   ├── LinearGCN      # compute_V_weight(), compute_V_bias()
│   │   ├── AWBLayerSpec   # Dataclass for validation
│   │   └── Conv utilities # compute_V_conv2d_*()
│   ├── mlp.py             # MLP with AWB interface
│   ├── cnn.py             # CNN/CNN3D with AWB interface
│   ├── gcn.py             # GCN with AWB interface
│   └── transformer.py     # Your new model (example)
├── core/
│   ├── awb.py             # Generic AWB delegates
│   │   ├── apply_V_transformation()
│   │   ├── partition_model_for_AB_training()
│   │   ├── partition_model_for_standard_training()
│   │   └── initialize_AB_matrices()
│   └── trainer.py         # Training loops (unchanged)
├── runners/
│   ├── generic_runner.py  # Unified runner (NEW - use this!)
│   ├── regression.py      # Legacy (backward compat)
│   ├── classification.py  # Legacy (backward compat)
│   └── graph_classification.py  # Legacy (backward compat)
└── data/
    ├── sine.py
    ├── mnist.py
    └── svhn.py            # Your new dataset (example)
```

---

## Migration Guide for Existing Code

### Old Way (Model-Specific)

```python
# Old: Model-specific functions
from cl.core.awb import compute_V_from_AWB, partition_for_AB_training

model = compute_V_from_AWB(model)  # Only works with MLP
diff, static = partition_for_AB_training(model)  # Only works with MLP
```

### New Way (Generic)

```python
# New: Generic functions that work with any model
from cl.core.awb import apply_V_transformation, partition_model_for_AB_training

model = apply_V_transformation(model)  # Works with MLP, CNN, GCN, Transformer
diff, static = partition_model_for_AB_training(model)  # Works with any model
```

### Or Use Model Interface Directly

```python
# Best: Use model's interface directly
model = model.apply_V_transformation()  # MLP implements this
diff, static = model.partition_for_AB_training()  # All models implement this
```

### Old Runners vs New Runner

```python
# Old: Import specific runner
from cl.runners import train_model_reg, train_model_class, train_model_graph

if prob == 'regression':
    records = train_model_reg(config)
elif prob == 'classification':
    records = train_model_class(config)
else:
    records = train_model_graph(config)
```

```python
# New: Single generic runner
from cl.runners import train_model

records = train_model(config)  # Works for all problem types!
```

---

## Benefits

### For Developers

1. **Add new layer types**: ~50 lines of code (just the layer + AWB methods)
2. **Add new models**: ~80 lines (3 required methods)
3. **Add new datasets**: Works immediately with all models
4. **No duplicate code**: Single AWB implementation, not 3+
5. **Type safety**: `AWBLayerSpec` validates shapes at layer level

### For Users

1. **Consistent API**: Same interface for all models
2. **Plugin-and-play**: Mix and match layers, models, datasets
3. **Backward compatible**: Old code still works
4. **Better debugging**: Layer-level errors pinpoint exact issue

### For Maintenance

1. **Single source of truth**: AWB logic in one place (layer classes)
2. **Easy to test**: Test layers independently
3. **Easy to extend**: Add features once, works everywhere
4. **Clear separation**: Layers → Models → Runners

---

## Testing Your New Code

### Test Layer AWB Methods

```python
# tests/test_my_layer.py
def test_attention_compute_V_weight(jax_key):
    layer = Attention(dim=64, num_heads=8, key=jax_key)

    A = jnp.eye(64, 64)
    B = jnp.eye(64, 64)

    V = layer.compute_V_weight(A, layer.Q_weight, B)

    # With identity matrices, V should equal original weight
    assert jnp.allclose(V, layer.Q_weight)
```

### Test Model AWB Interface

```python
# tests/test_my_model.py
def test_transformer_awb_interface(jax_key):
    model = Transformer(dim=64, num_layers=2, num_heads=8, key=jax_key)

    # Test get_awb_layer_specs
    specs = model.get_awb_layer_specs()
    assert len(specs) == 2 + 2  # 2 attention + 2 feed-forward

    # Test partition_for_AB_training
    diff, static = model.partition_for_AB_training()
    assert diff.A_attn is not None
    assert diff.B_attn is not None

    # Test partition_for_standard_training
    params, static = model.partition_for_standard_training()
    assert params.A_attn is None
    assert params.B_attn is None
```

### Test with Generic Runner

```python
# tests/test_integration.py
def test_transformer_with_generic_runner():
    config = {
        'network': 'transformer',
        'prob': 'classification',
        'data': 'mnist',
        'n_task': 2,
        'awb_enabled': True,
        # ... other config
    }

    from cl.runners import train_model
    records = train_model(config)

    assert len(records) > 0
```

---

## Code Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Partition functions | 6 model-specific | 1 generic | **-83%** |
| compute_V functions | 3 model-specific | 1 generic | **-67%** |
| Runner files | 3 (1,615 lines) | 1 generic (380) | **-76%** |
| Lines for new model | ~500+ | ~80 | **-84%** |
| Test coverage | 51 tests | 74 tests | **+45%** |

---

## Questions?

See also:
- `CLAUDE.md` - Project overview and commands
- `config/0_config_readme.md` - Configuration options
- `tests/test_layers.py` - Layer AWB test examples
- `tests/test_models.py` - Model AWB interface test examples

For implementation details, see the code:
- Layer AWB methods: `src/cl/models/layers.py`
- MLP example: `src/cl/models/mlp.py` (complete implementation)
- Generic delegates: `src/cl/core/awb.py`
- Generic runner: `src/cl/runners/generic_runner.py`
