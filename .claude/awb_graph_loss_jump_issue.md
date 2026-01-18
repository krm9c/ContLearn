# AWB Graph Loss Jump Issue Investigation

## Problem Statement
After A/B training converges to loss ~0.56, the V training starts at loss ~1.27 instead of ~0.56.
This is specific to **GCN/graph** models - MNIST (MLP/CNN) works correctly.

## Key Observation (from user)
> "after awb, the starting point for the new V training should not be this different. 0.56 → 1.27 is higher than the preliminary cost (1.15)"

## What We've Verified Works Correctly

### 1. compute_V transformation is mathematically correct
```
V = A @ W @ B^T
```
- Tested with isolated model: `get_AWBT()` before compute_V equals `model()` after compute_V
- Max difference: 0.0000000000 (exactly equal)

### 2. Weight dimensions after transformation are correct
- gcn_layers[0].weight: (10, 32) → (10, 72)
- gcn_layers[1].weight: (32, 32) → (72, 72)
- feed_layers[0].weight: (32, 32) → (172, 72)

### 3. Bias dimensions after transformation are correct
- gcn_layers[0].bias: (1, 32) → (1, 72)
- gcn_layers[1].bias: (1, 32) → (1, 72)
- feed_layers[0].bias: (1, 32) → (1, 172)

### 4. set_AB_matrices preserves original weights
- Original W weights are unchanged after setting A/B matrices

### 5. Partitioning doesn't affect model behavior
- Loss before and after partition_for_standard_training is identical

## What Remains Unexplained

The loss jump from 0.56 to 1.27 during actual training, despite isolated tests showing mathematical equivalence.

## Potential Causes to Investigate

### 1. Data/Batching Differences
- AB loss recorded on training data at end of AB training (epoch 149)
- V training loss recorded at epoch 5 (eval_interval=5) on different batches
- Need to verify SAME batch gives same loss

### 2. JIT Compilation Caching
- `get_AWBT()` and `model()` are different JIT traces
- Model structure changes might cause stale JIT cache

### 3. GCN-Specific Forward Pass
- GCN uses: `support = x @ W; x = adj @ support + bias`
- get_AWBT uses: `V_weight = A @ W @ B.T; support = x @ V_weight; x = adj @ support + bias @ B.T`
- After compute_V: `support = x @ V; x = adj @ support + Vb`

### 4. Bias Handling in get_AWBT vs Standard Forward
In `get_AWBT`:
```python
x += (self.gcn_layers[i].bias @ self.B_gcn[i].T)
```
In standard forward after compute_V:
```python
x += self.bias  # where bias is now Vb = original_bias @ B.T
```
These SHOULD be equivalent, but need verification on actual data.

### 5. A/B Matrices After compute_V
After compute_V, the A/B matrices are NOT reset to identity. If any code path still uses them, it would compute:
```
A @ V @ B^T = A @ (A @ W @ B^T) @ B^T  (wrong!)
```
instead of just V.

## Files to Check

### Core AWB Pipeline
- `src/cl/core/awb_pipeline.py` - lines 391-412 (STEP 4 compute_V)
- `src/cl/core/awb.py` - `compute_V_from_AWB_gcn()` function (line 574)

### GCN Model
- `src/cl/models/gcn.py` - `get_AWBT()` method (line 367-415)
- `src/cl/models/gcn.py` - `__call__()` method (line 338-365)
- `src/cl/models/gcn.py` - GCNLayer `__call__()` (line 163-185)

### A/B Matrix Setup
- `src/cl/arch_search/gcn_search.py` - `prepABs_GCN()` function (line 95)

## Training Records Analysis

From `outputs/synthetic_graph_predetermined_arch_extended_ab_awb_run0`:

**Task 1 AB training:**
- H (Hamiltonian): [0.5555803932437074]
- V (Potential): [0.5576420602014543]

**Task 1 Main training (V phase):**
- H (first 5): [1.2704, 1.3710, 1.1440, 0.8712, 0.8359, 0.7040]
- First recorded is at epoch 5 (eval_interval=5)

## Experiment Configuration

Config file: `runs__/configs/synthetic_graph_predetermined_arch.json`
```json
{
  "awb_enabled": true,
  "awb_ab_training_epochs": 150,
  "awb_ab_warmup_epochs": 20,
  "batch_size": 128,
  "gcn_sizes": [10, 32, 32],
  "feed_sizes": [32, 32, 16, 5],
  "awb_predetermined_arch": {
    "1": {"gcn_sizes": [10, 72, 72], "feed_sizes": [72, 172, 196, 5]},
    "2": {"gcn_sizes": [10, 142, 142], "feed_sizes": [142, 332, 316, 5]}
  }
}
```

## Diagnostic Code to Run

```python
# Test if bias is the issue
import jax.numpy as jnp
import equinox as eqx
from cl.models.gcn import GCN
from cl.core.awb import compute_V_from_AWB_gcn, set_new_AB_matrices_gcn

# Create model and set A/B
model = GCN(in_size=10, gcn_sizes=[10, 32, 32], feed_sizes=[32, 32, 16, 5], SEED=42, graph=True)
model = set_new_AB_matrices_gcn(model, [10, 32, 32], [32, 32, 16, 5], [10, 72, 72], [72, 172, 196, 5])

# Before compute_V: check bias transformation manually
print("Before compute_V:")
print(f"gcn_layers[0].bias shape: {model.gcn_layers[0].bias.shape}")
print(f"B_gcn[0] shape: {model.B_gcn[0].shape}")
transformed_bias = model.gcn_layers[0].bias @ model.B_gcn[0].T
print(f"Transformed bias (bias @ B.T) shape: {transformed_bias.shape}")

# After compute_V: check if bias matches
model_v = compute_V_from_AWB_gcn(model)
print("\nAfter compute_V:")
print(f"gcn_layers[0].bias shape: {model_v.gcn_layers[0].bias.shape}")

# Compare values
print(f"\nBias values equal? {jnp.allclose(transformed_bias, model_v.gcn_layers[0].bias)}")
```

## Next Steps

1. Add debug logging immediately after compute_V to print loss on SAME batch
2. Compare loss with `model()` vs `get_AWBT()` on same batch after compute_V
3. Check if A/B matrices are inadvertently being used somewhere after compute_V
4. Verify GCN-specific bias handling in graph forward pass

## Related Files Created During Investigation
- `outputs/awb_vs_baseline_comparison.png` - Bar chart comparison
- `outputs/awb_vs_baseline_timeline.png` - Training progression
- `outputs/extended_ab_run.log` - Experiment output log
