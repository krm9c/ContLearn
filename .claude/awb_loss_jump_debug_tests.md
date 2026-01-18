# AWB Loss Jump Debug Tests - Complete Investigation

## Problem Statement
After A/B training converges (loss ~0.56), V training starts at a much higher loss (~1.27 or higher).
This is specific to **GCN/graph** models.

---

## ROOT CAUSE IDENTIFIED

### The Smoking Gun: First Gradient Step Explosion
```
[EPOCH 0 DIAGNOSTIC] Phase=warmup
  Loss BEFORE first update: V=0.931337
  Loss AFTER first update: V=53.138731
  Loss change: +52.207394 (increased) ← 56x increase in ONE step!
```

### Why It Happens
1. **Massive parameter expansion**: 3,077 → 53,497 parameters (17.4x for Task 1)
2. **V weights in low-rank subspace**: After `compute_V`, V = A @ W @ B.T has low-rank structure
3. **Gradient points away from optimal subspace**: Full-rank gradient updates break the low-rank structure
4. **Same learning rate**: Using lr=0.001 for both 3K and 53K parameters

---

## SOLUTION IMPLEMENTED

### LR Reduction Based on Parameter Expansion
File: `src/cl/core/awb_pipeline.py` (lines 447-487)

```python
# Compute LR reduction factor based on parameter expansion
expansion_ratio = new_param_count / orig_param_count
v_lr_factor = 1.0 / math.sqrt(expansion_ratio)  # Xavier-inspired scaling
v_lr_factor = max(v_lr_factor, 0.01)  # Minimum 1% of original LR

# Create optimizer with reduced LR for V training
v_training_lr = base_lr * v_lr_factor
optim = create_optimizer_with_lr(config, v_training_lr)
```

### Results After Fix
| Task | Expansion | Before Fix | After Fix | Improvement |
|------|-----------|------------|-----------|-------------|
| Task 1 | 17.4x | 0.93 → 53.14 | 0.93 → 9.81 | **6x better** |
| Task 2 | 57.2x | 0.08 → 51.56 | 0.20 → 4.49 | **12x better** |

---

## Tests Performed

### Test 1: Isolated compute_V Verification
**File**: `debug_loss_jump.py`
**Purpose**: Verify that `compute_V` transformation is mathematically correct

**Method**:
1. Create GCN model with original architecture [10, 32, 32] / [32, 32, 16, 5]
2. Set up A/B matrices for expansion to [10, 72, 72] / [72, 172, 196, 5]
3. Compare `get_AWBT()` output BEFORE compute_V with `__call__()` output AFTER compute_V

**Result**: ✅ **PASS** - Losses match exactly
```
get_AWBT loss BEFORE compute_V: 1.458347
__call__ loss AFTER compute_V: 1.458347
Difference: 0.0000000000
```

**Conclusion**: `compute_V` transformation is mathematically correct.

---

### Test 2: In-Pipeline Debug Output
**File**: Modified `src/cl/core/awb_pipeline.py` (lines 394-436)
**Purpose**: Verify compute_V correctness within actual training pipeline

**Method**:
Added debug output to compare:
1. Loss BEFORE compute_V using `get_AWBT()`
2. Loss AFTER compute_V using `__call__()`
3. Experience loss after compute_V

**Result**: ✅ **PASS** - Losses match exactly
```
Task 1:
  [DEBUG] Current task - BEFORE (get_AWBT): 0.911460
  [DEBUG] Current task - AFTER (__call__): 0.911460
  [DEBUG] Current task - Difference: 0.0000000000
  [DEBUG] Experience - AFTER (__call__): 0.931337
```

**Conclusion**: `compute_V` works correctly in the actual pipeline.

---

### Test 3: Bias Transformation Investigation
**Purpose**: Check if bias handling differs between CNN and GCN

**Method**: Compared bias transformation in:
- `src/cl/core/awb.py`: `compute_V_from_AWB_gcn()` (lines 574-607)
- `src/cl/core/awb.py`: `compute_V_from_AWB_cnn()` (lines 702-739)
- `src/cl/models/gcn.py`: `get_AWBT()` (lines 386-415)

**Findings**:
- **CNN feed layers**: `Vb = A @ bias` (transforms output dimension with A)
- **GCN layers**: `Vb = bias @ B.T` (transforms output dimension with B)
- **GCN feed layers**: `Vb = bias @ B.T` (transforms output dimension with B)

The difference is due to different forward pass conventions:
- CNN: `W @ x + bias` (column vector) - A transforms output
- GCN: `x @ W + bias` (row vector) - B transforms output

**Result**: ✅ Both are mathematically correct for their respective conventions.

---

### Test 4: Epoch 0 Diagnostics
**File**: Modified `src/cl/core/loops.py` (lines 542-568)
**Purpose**: Capture exact loss values before and after first gradient update

**Added logging**:
```python
if epoch == 0 and batch_idx == 0 and phase in ['warmup', 'main']:
    print(f"[EPOCH 0 DIAGNOSTIC] Phase={phase}")
    print(f"  Loss BEFORE first update: V={V:.6f}")
    # ... after optimizer step ...
    print(f"  Loss AFTER first update: V={V_after:.6f}")
    print(f"  Loss change: {V_after - V:+.6f}")
```

**Critical Finding**:
```
[EPOCH 0 DIAGNOSTIC] Phase=warmup
  Loss BEFORE first update: V=0.931337, H=0.929266
  Gradient norm (pre-clip): 465.72, clipped: True
  Parameter count: 53,497
  Learning rate: 0.001
  Loss AFTER first update: V=53.138731
  Loss change: +52.207394 (increased)  ← CATASTROPHIC!
```

**Conclusion**: First gradient step causes 56x loss increase despite gradient clipping.

---

### Test 5: Parameter Count Analysis
**Purpose**: Understand magnitude of architecture expansion

```python
=== Parameter Count Analysis ===
Original GCN: 1408 (layers: [10, 32, 32])
Original Feed: 1669 (layers: [32, 32, 16, 5])
Original Total: 3,077

Expanded GCN: 6048 (layers: [10, 72, 72])
Expanded Feed: 47449 (layers: [72, 172, 196, 5])
Expanded Total: 53,497

Expansion factor: 17.39x
```

**Conclusion**: 17.4x parameter expansion with same LR causes instability.

---

## Key Findings Summary

### 1. compute_V is Mathematically Correct ✅
The transformation `V = A @ W @ B.T` produces identical outputs:
- `get_AWBT()` before compute_V = `__call__()` after compute_V

### 2. Problem: First Gradient Step Explosion
- Loss increases 56x in ONE step (0.93 → 53.14)
- Even with gradient clipping (465 → 1.0), direction is wrong
- V weights live in low-rank subspace, gradients break this structure

### 3. Root Cause: Parameter Expansion + Same LR
- 3,077 → 53,497 parameters (17.4x increase)
- Same lr=0.001 for vastly different parameter spaces
- Gradient direction optimal for full space, not low-rank subspace

### 4. Solution: LR Reduction by 1/sqrt(expansion)
- Auto-compute expansion ratio from parameter counts
- Reduce LR proportionally: `v_lr = base_lr / sqrt(expansion)`
- Task 1: lr=0.001 × 0.24 = 0.00024
- Task 2: lr=0.001 × 0.13 = 0.00013

---

## Files Modified

1. `src/cl/core/awb_pipeline.py` - Added V training LR reduction (lines 447-490)
2. `src/cl/core/loops.py` - Added epoch 0 diagnostics (lines 542-568)
3. `debug_loss_jump.py` - Standalone test script (created)

---

## Configuration Options

### awb_v_lr_factor (optional)
Override the auto-computed LR factor for V training:
```json
{
    "awb_v_lr_factor": 0.1  // Use 10% of base LR for V training
}
```

If not specified, automatically computed as `1/sqrt(expansion_ratio)`.

---

## Visual Flow: V Training with LR Fix

```
════════════════════════════════════════════════════════════════════════════════
            TASK 1: COMPLETE V TRAINING FLOW (WITH LR FIX)
            Parameter expansion: 3,077 → 53,497 (17.4x)
            LR reduced: 0.001 → 0.00024 (auto: 1/√17.4)
════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│  INITIAL STATE (after compute_V, before any training)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  Experience Loss (V): 0.931                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  WARMUP PHASE (20 epochs) - LR=0.00024, record_training=False               │
├──────────────────────────────────────────────────────────────────────────────┤
│  Batch 0:  BEFORE = 0.931  →  AFTER = 9.810  (jump +8.88)                   │
│            Still jumps but 6x better than before fix (was +52!)              │
│                                                                              │
│  Epoch │  CE (exp loss)  │  Trend                                           │
│  ───── │ ─────────────── │ ─────────────────────────────────────────────────│
│    5   │     0.988       │  ████████████████████████  (recovering)          │
│   10   │     0.863       │  █████████████████████  (decreasing)             │
│   15   │     0.896       │  ██████████████████████  (slight oscillation)    │
│   20   │     0.531       │  █████████████  (stabilized)                     │
│                                                                              │
│  → Warmup ends with loss ~0.53 (stabilized from initial 9.8 jump)           │
└──────────────────────────────────────────────────────────────────────────────┘

                            ════ TRANSITION ════
                         (optimizer state preserved, same LR)

┌──────────────────────────────────────────────────────────────────────────────┐
│  MAIN PHASE (30 epochs) - LR=0.00024, record_training=True                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Batch 0:  BEFORE = 0.487  →  AFTER = 0.388  (↓ DECREASED) ✓                │
│            First step now DECREASES loss - warmup worked!                    │
│                                                                              │
│  Epoch │  CE (exp loss)  │  Trend                                           │
│  ───── │ ─────────────── │ ─────────────────────────────────────────────────│
│    5   │     1.355       │  ██████████████████████████████████  (spike)     │
│   10   │     0.609       │  ███████████████  (recovered)                    │
│   15   │     1.565       │  ███████████████████████████████████████ (spike) │
│   20   │     0.481       │  ████████████  (good)                            │
│   25   │     0.511       │  █████████████  (stable)                         │
│   30   │     0.586       │  ███████████████  (final)                        │
│                                                                              │
│  Note: Some oscillation remains - could benefit from more warmup epochs     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  FINAL RESULT: Task 1 Loss = 0.549                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Observations from Flow

1. **First step still jumps** (0.93 → 9.81) but 6x better than before (was 0.93 → 53.14)
2. **Warmup does its job**: Loss drops from 9.8 → 0.53 during 20 warmup epochs
3. **Transition is smooth**: Main phase starts at 0.49, first step DECREASES to 0.39
4. **Some oscillation remains** in main phase (0.48 → 1.57 → 0.48)
5. **Overall downward trend**: Final loss 0.549 is reasonable

### Warmup → Main Transition Details

| Aspect | Warmup Phase | Main Phase |
|--------|--------------|------------|
| Optimizer | Same (`optim`) | Same |
| Optimizer state | Fresh at start | Preserved from warmup |
| Learning rate | 0.00024 | 0.00024 (same) |
| `record_training` | False | True |
| `n_iter` | 20 epochs | 30 epochs |

The transition is **seamless** - only `phase` label and `record_training` flag change.

---

## Future Improvements

1. **Stronger LR reduction**: Use `1/expansion` instead of `1/sqrt(expansion)` for more stability
2. **LR warmup within V training**: Start at 0.1x reduced LR, gradually increase
3. **Constrained updates**: Project gradients onto low-rank subspace
4. **Adam beta reset**: Reset momentum/velocity after compute_V

---

## ONGOING INVESTIGATION: Main Phase Oscillation

### Problem: Oscillation WITHIN Main Phase
Despite LR fix improving first-step explosion (56x → 6x), oscillation persists during main training:
```
MAIN PHASE (30 epochs)
  Batch 0:  BEFORE = 0.487  →  AFTER = 0.388  (↓ DECREASED) ✓
  Epoch 5:  CE = 0.374
  Epoch 10: CE = 0.385 (slight increase)
  Epoch 15: CE = 0.260 (better)
  Epoch 20: CE = 0.276 (slight increase)
  Epoch 25: CE = 0.398 (spike!)
  Epoch 30: CE = 0.196 (recovered)
```

### Key Finding: GCN vs MLP/CNN Difference

**This issue is GCN-specific** - MLP and CNN don't have this oscillation problem.

**Root Cause Found: GCN wdot scaling**

**MLP/CNN** (lines 187, 244, 282, 328 in hamiltonian.py):
```python
wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)  # Simple negation
```

**GCN** (lines 370-372, 418-420 in hamiltonian.py):
```python
def norm_param(g):
    return g * (-1e-04 / jnp.sqrt(jnp.linalg.norm(g**2) + 1e-8))  # Scaled by -1e-04!
wdot = jax.tree_util.tree_map(norm_param, delta_theta)
```

The `-1e-04` factor makes GCN's wdot **10,000x smaller**, effectively disabling dV regularization:
- dV = f_jvp(wdot, deltax) ≈ 0 because wdot ≈ 0
- grad_dV ≈ 0
- Gradient becomes: `grad = alpha * delta_theta + beta * grad_V + 0`
- No regularization term to stabilize training!

### Hypotheses Being Tested

1. **Momentum Carryover** (Testing):
   - Adam momentum from warmup carries into main phase
   - May cause overshooting when gradient directions change
   - Test: Reset `opt_state = optim.init(params)` before main phase

2. **GCN wdot Scaling** (Primary suspect):
   - `-1e-04` scaling disables dV regularization
   - MLP/CNN have effective dV, GCN does not
   - This explains why only GCN oscillates

### Test Status

- [ ] Test 1: Optimizer reset before main phase (in progress)
- [ ] Test 2: Remove GCN wdot scaling (if Test 1 doesn't fully fix it)
