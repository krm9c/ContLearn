# Plan: Implement Consistent Hamiltonian Component Normalization

## ⚠️ LAST KNOWN WORKING COMMIT

**If anything breaks, revert to this commit:**

```bash
git checkout 300cb01ae68f8f7bb49470ed2493d7196885094b
```

| Commit | Hash | Description |
|--------|------|-------------|
| **Last Working** | `300cb01ae68f8f7bb49470ed2493d7196885094b` | Before wdot fix - GCN uses `-1e-04` scaling, stable but dV regularization weak |
| wdot fix | `ae1821f` | GCN wdot changed to `-1` (matches MLP/CNN), dV_dθ now large |
| This plan | `cf56396` | Added this normalization plan |

**To fully revert all changes:**
```bash
git reset --hard 300cb01ae68f8f7bb49470ed2493d7196885094b
```

---

## Background and Motivation

### The Problem
After fixing GCN wdot scaling (commit ae1821f), the dV_dθ values are now large (~100-700) because:

```
dV_dθ = ∇V · wdot = ∇V · (-δθ)
```

When both gradients have magnitude ~200 and are aligned:
- Dot product ≈ 200 × 200 = 40,000
- After dividing by sqrt(param_count) ≈ 55 → dV_dθ ≈ 700

Meanwhile:
- CE loss: ~0.3 - 1.5
- dV_dx: ~1e-05 (tiny!)

This scale mismatch causes:
1. Hamiltonian components have incomparable magnitudes
2. dV_dθ dominates the total dV computation
3. If γ > 0, gradient updates would be unstable

### The Solution
Normalize all Hamiltonian velocity components to unit magnitude, then divide by V for dimensionless output.

---

## Implementation Plan

### Step 1: Create Normalization Helper Functions

**File**: `src/cl/core/hamiltonian.py`

Add at the top of the file (after imports):

```python
def _tree_norm(tree):
    """Compute L2 norm of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))

def _normalize_tree(tree, eps=1e-8):
    """Normalize a pytree to unit L2 norm."""
    norm = _tree_norm(tree)
    return jax.tree_util.tree_map(lambda x: x / (norm + eps), tree), norm
```

### Step 2: Modify MLP Hamiltonian Functions

**File**: `src/cl/core/hamiltonian.py`

#### Function: `_hamiltonian_core_mlp` (around line 60)

**Before**:
```python
delta_theta = jax.grad(loss_fn_curr)(params, x)
wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
```

**After**:
```python
delta_theta = jax.grad(loss_fn_curr)(params, x)

# Normalize wdot to unit magnitude (direction only)
wdot_unnorm = jax.tree_util.tree_map(lambda g: -g, delta_theta)
wdot, wdot_norm = _normalize_tree(wdot_unnorm)

# Normalize deltax to unit magnitude
deltax_norm = jnp.linalg.norm(deltax)
deltax_normalized = deltax / (deltax_norm + 1e-8)
```

Then update the dV computations:
```python
# Compute directional derivatives with normalized velocities
dV_dtheta_raw = f_jvp(wdot, jnp.zeros_like(deltax))
dV_dx_raw = f_jvp(zero_dtheta, deltax_normalized)

# Divide by V for dimensionless relative change
dV_dtheta = dV_dtheta_raw / (V + 1e-8)
dV_dx = dV_dx_raw / (V + 1e-8)
```

#### Function: `_hamiltonian_core_mlp_awb` (around line 110)
Apply same changes.

### Step 3: Modify CNN Hamiltonian Functions

**File**: `src/cl/core/hamiltonian.py`

#### Function: `_hamiltonian_core_cnn` (around line 160)
Same pattern as MLP.

#### Function: `_hamiltonian_core_cnn_awb` (around line 210)
Same pattern as MLP.

### Step 4: Modify GCN Hamiltonian Functions

**File**: `src/cl/core/hamiltonian.py`

#### Function: `_hamiltonian_core_graph_standard` (around line 350)

**Before** (current after wdot fix):
```python
delta_theta = jax.grad(loss_fn_curr)(params, x, adj)
wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
```

**After**:
```python
delta_theta = jax.grad(loss_fn_curr)(params, x, adj)

# Normalize wdot to unit magnitude
wdot_unnorm = jax.tree_util.tree_map(lambda g: -g, delta_theta)
wdot, wdot_norm = _normalize_tree(wdot_unnorm)

# Normalize deltax to unit magnitude
deltax_norm = jnp.linalg.norm(deltax)
deltax_normalized = deltax / (deltax_norm + 1e-8)

# Normalize delta_adj to unit magnitude
delta_adj_norm = jnp.linalg.norm(delta_adj)
delta_adj_normalized = delta_adj / (delta_adj_norm + 1e-8)
```

Update dV computations:
```python
# Compute directional derivatives with normalized velocities
dV_dtheta_raw = f_jvp(wdot, zero_dx, zero_dadj)
dV_dx_raw = f_jvp(zero_dtheta, deltax_normalized, zero_dadj)
dV_dadj_raw = f_jvp(zero_dtheta, zero_dx, delta_adj_normalized)

# Divide by V for dimensionless output
dV_dtheta = dV_dtheta_raw / (V + 1e-8)
dV_dx = dV_dx_raw / (V + 1e-8)
dV_dadj = dV_dadj_raw / (V + 1e-8)

# Total dV (remove old sqrt_param_count and dV_scale since we're normalizing differently)
dV = dV_dtheta + dV_dx + dV_dadj
```

#### Function: `_hamiltonian_core_graph_awb` (around line 400)
Same pattern as standard.

### Step 5: Update Return Values

The return tuple should include the normalized values:
```python
return grad, (V + dV, V, dV, dV_dtheta, dV_dx, dV_dadj)
```

Note: For MLP/CNN without adjacency:
```python
return grad, (V + dV, V, dV, dV_dtheta, dV_dx)
```

### Step 6: Remove Old Scaling Parameters

After normalization, the `sqrt_param_count` and `dV_scale` parameters may no longer be needed. Consider:
1. Keeping them for backward compatibility but setting to 1.0
2. Removing them entirely (breaking change)

**Recommendation**: Keep parameters but document that normalization makes them less critical.

---

## Expected Results After Implementation

| Component | Before | After |
|-----------|--------|-------|
| dV_dθ | ~100-700 | ~0.01-1.0 (dimensionless) |
| dV_dx | ~1e-05 | ~0.01-1.0 (dimensionless) |
| dV_dadj | varies | ~0.01-1.0 (dimensionless) |
| **Interpretation** | Absolute rates | Relative alignment |

**Physical meaning after normalization**:
- `dV_dθ ≈ cos(∇_θV, δθ)` — alignment between experience and current gradients
- `dV_dx ≈ cos(∇_xV, Δx)` — sensitivity to input distribution shift
- `dV_dadj ≈ cos(∇_adjV, Δadj)` — sensitivity to graph structure shift

---

## Testing Plan

### Test 1: Verify Normalization
```python
# After implementation, dV_dθ should be in [-1, 1] range
assert -1.5 <= dV_dtheta <= 1.5, f"dV_dθ out of expected range: {dV_dtheta}"
```

### Test 2: Compare Training Stability
Run with predetermined architecture config:
```bash
python run.py runs__/configs/synthetic_graph_predetermined_arch.json
```

Check:
1. A/B training loss oscillation reduced
2. V training remains stable
3. Final accuracy comparable or better

### Test 3: Verify MLP/CNN Not Broken
Run MLP and CNN experiments to ensure normalization doesn't break them:
```bash
python run.py runs__/configs/mnist_mlp.json
python run.py runs__/configs/cifar_cnn.json
```

---

## Files to Modify

1. **`src/cl/core/hamiltonian.py`**
   - Add `_tree_norm()` and `_normalize_tree()` helper functions
   - Modify all 6 Hamiltonian core functions:
     - `_hamiltonian_core_mlp` (~line 60)
     - `_hamiltonian_core_mlp_awb` (~line 110)
     - `_hamiltonian_core_cnn` (~line 160)
     - `_hamiltonian_core_cnn_awb` (~line 210)
     - `_hamiltonian_core_graph_standard` (~line 350)
     - `_hamiltonian_core_graph_awb` (~line 400)

2. **`src/cl/core/loops.py`** (if needed)
   - Update any code that interprets dV_dθ values
   - Update logging to show normalized values

---

## Rollback Plan

If normalization causes issues:

### Option 1: Revert only normalization (keep wdot fix)
```bash
# Revert specific commits but keep wdot fix
git revert <normalization-commit-hash>
```

### Option 2: Full revert to last known working state
```bash
# Go back to commit before ANY changes (GCN uses -1e-04 scaling)
git reset --hard 300cb01ae68f8f7bb49470ed2493d7196885094b
```

### Option 3: Keep wdot fix, use grad_weights instead
- Set `grad_weights = [0.5, 0.5, 0.0]` (γ=0) to disable dV contribution
- dV_dθ still computed for logging but doesn't affect training

---

## Context for Future Claude Sessions

### Current State (as of this commit)
- GCN wdot changed from `-1e-04` scaling to `-1` (simple negation)
- This matches MLP/CNN behavior
- dV_dθ values are now large (~100-700) - this is expected
- V training is stable with LR scaling (1/sqrt(expansion))
- A/B training shows some oscillation but converges

### Key Files
- `src/cl/core/hamiltonian.py` - Contains all Hamiltonian gradient computations
- `src/cl/core/awb_pipeline.py` - AWB training pipeline
- `src/cl/core/loops.py` - Training loops

### Key Insight
The Hamiltonian gradient is: `grad = α·δθ + β·∇V + γ·∇dV`

With `grad_weights = [0.5, 0.5, 0.0]` (γ=0), dV doesn't affect gradient directly, but:
1. dV_dθ is still computed and logged for diagnostics
2. Large dV_dθ indicates gradient alignment between tasks
3. Normalization makes dV_dθ interpretable as cosine similarity

### Commands to Run
```bash
# Test with predetermined architecture (faster, skips search)
python run.py runs__/configs/synthetic_graph_predetermined_arch.json

# Full AWB with architecture search
python run.py runs__/configs/synthetic_graph_minimal_awb.json

# Check current hamiltonian functions
grep -n "def _hamiltonian_core" src/cl/core/hamiltonian.py
```
