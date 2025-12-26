# CIFAR-10 AWB Performance Issue: Profiling & Optimization Plan

**Created:** December 26, 2024
**Status:** Implementation In Progress
**Owner:** Research Team

---

## Executive Summary

**Problem:** CIFAR-10 experiments with AWB (Adaptive Weight Basis) enabled take ~2 hours for the **first epoch** of A/B training, with 0% GPU utilization. This makes Conditions 3 & 4 impractical to run (estimated 25-50 hours per condition).

**Impact:**
- Condition 3: Should take ~5 hours, currently taking ~25 hours
- Condition 4: Should take ~10-15 hours, currently taking ~50+ hours

**Root Causes Identified:**
1. **Missing `awb_skip_transfer` flag implementation** - Condition 3 runs A/B training when it should skip it
2. **JAX JIT compilation hang** - First A/B training epoch takes ~7200s (2 hours) instead of expected 30-180s

**Solution Approach:**
1. Implement `awb_skip_transfer` flag (fixes Condition 3)
2. Create progressive profiling system to identify bottleneck (fixes Condition 4)
3. Apply targeted optimization based on profiling data

---

## Problem Context

### Experimental Setup

The framework runs 4 experimental conditions across 5 datasets:

**Condition 1** - Baseline (No AWB, No Smoothing)
- Fixed architecture, constant LR, no task warmup
- Works fine, no performance issues

**Condition 2** - Heuristics/Warmup
- Task warmup enabled (5 epochs)
- Works fine, no performance issues

**Condition 3** - Architecture Search, No Transfer
- `awb_enabled: true`, `awb_skip_transfer: true`
- **BUG:** Flag not implemented, runs A/B training anyway
- Takes ~2.5 hours per task instead of ~30 minutes

**Condition 4** - Full AWB Pipeline
- `awb_enabled: true`, `awb_skip_transfer: false`
- A/B training first epoch hangs for 2 hours
- Takes ~5+ hours per task instead of ~1 hour

### AWB 5-Step Pipeline

When architecture changes are triggered (loss ratio > threshold):

1. **STEP 1:** Preliminary training (50 epochs)
2. **STEP 2:** Decide if architecture change needed
3. **STEP 3a:** Architecture search for optimal dimensions
4. **STEP 3b:** Train A/B matrices (100 epochs) ← **HANG OCCURS HERE**
5. **STEP 4:** Compute V = A @ W @ B.T
6. **STEP 5:** Train V with A/B frozen

### Evidence from User Logs

```
[STEP 3b] Train A/B matrices (100 epochs) - Recorded separately
  prepABs_CNN3D: filter 3→4, feed [2304, 512, 256, 10]→[1600, 552, 276, 10]
  Conv filter changed: A_conv/B_conv shape = (4, 3)

  0%|          | 0/100 [00:00<?, ?it/s]
  1%|          | 1/100 [1:57:21<193:38:42, 7041.64s/it]  ← 2 HOUR HANG
```

- **What it means:** 1 epoch out of 100 A/B training epochs completed in 1:57:21
- **GPU utilization:** 0% during this time
- **Expected time:** 30-180 seconds for JIT compilation, not 7200 seconds

---

## Implementation Plan

### PART 1: Implement `awb_skip_transfer` Flag

**Objective:** Fix Condition 3 to skip A/B training as intended.

**File:** `src/cl/core/awb_pipeline.py`
**Location:** After line 302, before existing A/B training code (line 303)

**Current Code (lines 298-305):**
```python
if new_arch != original_arch:
    # Added by Claude: Update metadata for architecture change
    task_metadata['architecture_changed'] = True
    task_metadata['change_reason'] = 'loss_ratio_threshold'

    # STEP 3b: Train A/B
    print(f"\n[STEP 3b] Train A/B matrices ({ab_training_epochs} epochs) - Recorded separately")
    model = awb_ops.set_AB_matrices(model, original_arch, new_arch)
    # ... continues with A/B training
```

**Required Change:**
```python
if new_arch != original_arch:
    task_metadata['architecture_changed'] = True
    task_metadata['change_reason'] = 'loss_ratio_threshold'

    # Added by Claude: Check if we should skip A/B transfer (Condition 3)
    skip_transfer = config.get('awb_skip_transfer', False)

    if skip_transfer:
        # CONDITION 3: Skip A/B training, use random init
        print(f"\n[STEP 3b] Skipping A/B training (awb_skip_transfer=True)")
        print(f"  Using random initialization for new architecture")

        # Re-initialize model with new architecture
        # Note: Models don't have expand_architecture() method yet
        # For now, we'll create new model and let it random init
        model = awb_ops.set_AB_matrices(model, original_arch, new_arch)
        model = awb_ops.compute_V(model)  # V = A @ W @ B.T (random A/B)
        params, static = awb_ops.partition_for_standard_training(model)

        # Reinitialize optimizer for new params
        from ..runners.generic_runner import create_optimizer
        optim = create_optimizer(config)
        opt_state = optim.init(params)

        # Jump to STEP 5 (main training) - skip STEP 3b and 4
        print(f"\n[STEP 5] Train new architecture ({epochs_per_task} epochs)")
        # Continue to warmup/main training below (lines 356+)

    else:
        # CONDITION 4: Run full A/B training pipeline
        # STEP 3b: Train A/B
        print(f"\n[STEP 3b] Train A/B matrices ({ab_training_epochs} epochs) - Recorded separately")
        # ... existing A/B training code (lines 305-343)
```

**Expected Outcome:**
- Condition 3 tasks finish in ~30 min (was ~2.5 hours)
- Total Condition 3: **5 hours** (was 25 hours)

**Testing:**
```bash
# Run Condition 3 debug config to verify skip works
python run_files/scripts/run.py kkt_run/configs/cifar10_condition3_arch_no_transfer.json
# Should see: "[STEP 3b] Skipping A/B training (awb_skip_transfer=True)"
# Should NOT see 100-epoch progress bar
```

---

### PART 2: Create Non-Intrusive Profiling System

**Objective:** Profile both normal training (31s/epoch) and abnormal A/B training (2hr first epoch) to identify bottleneck.

**Design Philosophy:**
- **No cumbersome code** added to source files
- Decorator-based profiling activated ONLY when `profiling_enabled: true` in config
- Progressive testing: Sine Wave → MNIST → CIFAR-10 (both standard and AWB)
- User runs debug scripts, reports timings, we analyze together

#### Step 2.1: Create Profiling Module

**File:** `src/cl/core/profiling.py` (NEW FILE)

```python
"""
Non-intrusive profiling decorators for continual learning framework.

Activated only when config['profiling_enabled'] = True.
Uses Python decorators and global flag to minimize overhead when disabled.

Added by Claude: December 26, 2024 - CIFAR-10 AWB performance debugging
"""
import time
import functools

_PROFILING_ENABLED = False

def enable_profiling(enabled: bool):
    """
    Enable/disable profiling globally.

    Args:
        enabled: True to enable profiling output, False to disable
    """
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enabled

def profile(phase_name: str):
    """
    Decorator to time a function if profiling is enabled.

    When profiling is disabled, this has zero overhead (simple boolean check).
    When enabled, prints timing information for the decorated function.

    Args:
        phase_name: Human-readable name for this profiling phase

    Example:
        @profile("Dataset Loading")
        def load_data():
            # ... code ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _PROFILING_ENABLED:
                return func(*args, **kwargs)

            print(f"\n[PROFILE] {phase_name} starting...")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

def profile_section(phase_name: str, enabled: bool = None):
    """
    Context manager for profiling a code section.

    Args:
        phase_name: Human-readable name for this profiling phase
        enabled: Optional override for profiling enabled status

    Example:
        with profile_section("JAX Pre-conversion"):
            # ... code to profile ...
    """
    class ProfileContext:
        def __enter__(self):
            if enabled is None:
                self.enabled = _PROFILING_ENABLED
            else:
                self.enabled = enabled

            if self.enabled:
                print(f"\n[PROFILE] {phase_name} starting...")
                self.start_time = time.time()
            return self

        def __exit__(self, *args):
            if self.enabled:
                elapsed = time.time() - self.start_time
                print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s")

    return ProfileContext()
```

**Lines of Code:** ~65 lines with docstrings

#### Step 2.2: Add Profiling Flag to Constants

**File:** `src/cl/config/constants.py`
**Location:** After line 14 (DEFAULT_GRAPH_SEED = 10)

```python
# ============================================================================
# PROFILING
# ============================================================================
# Added by Claude: Non-intrusive profiling for performance debugging
DEFAULT_PROFILING_ENABLED = False
```

#### Step 2.3: Create Debug Configs

**Directory:** `kkt_run/configs/debug/` (NEW DIRECTORY)

**File 1:** `kkt_run/configs/debug/sine_awb_profile_debug.json`
```json
{
    "__comment": "Debug config - Sine wave AWB profiling (baseline)",
    "__purpose": "Establishes baseline AWB metrics with simple MLP architecture",
    "data": "sine",
    "n_task": 2,
    "epochs_per_task": 10,
    "batch_size": 2048,
    "awb_enabled": true,
    "awb_skip_transfer": false,
    "awb_preliminary_epochs": 3,
    "awb_ab_training_epochs": 5,
    "awb_loss_ratio_threshold": 1.1,
    "debug_mode": true,
    "debug_limit": 1000,
    "profiling_enabled": true,
    "lr": 0.001,
    "grad_weights": [0.01, 0.98, 0.1]
}
```

**File 2:** `kkt_run/configs/debug/mnist_awb_profile_debug.json`
```json
{
    "__comment": "Debug config - MNIST/CNN profiling (medium complexity)",
    "__purpose": "Tests AWB profiling on CNN with 2D convolutions",
    "data": "mnist",
    "n_task": 2,
    "epochs_per_task": 10,
    "batch_size": 2048,
    "awb_enabled": true,
    "awb_skip_transfer": false,
    "awb_preliminary_epochs": 3,
    "awb_ab_training_epochs": 5,
    "awb_loss_ratio_threshold": 1.1,
    "debug_mode": true,
    "debug_limit": 1000,
    "profiling_enabled": true,
    "lr": 0.0001,
    "grad_weights": [0.01, 0.98, 0.1]
}
```

**File 3:** `kkt_run/configs/debug/cifar10_awb_profile_debug.json`
```json
{
    "__comment": "Debug config - CIFAR-10/CNN3D AWB profiling (high complexity)",
    "__purpose": "Reproduces 2-hour hang with A/B training on CNN3D",
    "data": "cifar10",
    "n_task": 2,
    "epochs_per_task": 10,
    "batch_size": 2048,
    "awb_enabled": true,
    "awb_skip_transfer": false,
    "awb_preliminary_epochs": 3,
    "awb_ab_training_epochs": 5,
    "awb_loss_ratio_threshold": 1.1,
    "debug_mode": true,
    "debug_limit": 1000,
    "profiling_enabled": true,
    "lr": 0.0001,
    "grad_weights": [0.01, 0.98, 0.1]
}
```

**File 4:** `kkt_run/configs/debug/cifar10_standard_profile_debug.json`
```json
{
    "__comment": "Debug config - CIFAR-10 NORMAL training (31s baseline)",
    "__purpose": "Captures normal CIFAR-10 iteration metrics without AWB",
    "data": "cifar10",
    "n_task": 2,
    "epochs_per_task": 10,
    "batch_size": 2048,
    "awb_enabled": false,
    "debug_mode": true,
    "debug_limit": 1000,
    "profiling_enabled": true,
    "lr": 0.0001,
    "grad_weights": [0.01, 0.98, 0.1]
}
```

#### Step 2.4: Add Profiling to AWB Pipeline

**File:** `src/cl/core/awb_pipeline.py`

**Location 1:** Top of file (after imports)
```python
# Added by Claude: Profiling support
from .profiling import profile
```

**Location 2:** Line ~153 (run_awb_task function definition)
```python
# Added by Claude: Profile entire AWB task pipeline
@profile("AWB Task Pipeline")
def run_awb_task(
    model, task_id, config, trainer, record_dict,
    trainloader, exploader, valloader, testloader,
    problem_type, loss_type
):
    # ... existing code ...
```

**Location 3:** Inside run_awb_task, before architecture search (line ~289)
```python
# STEP 3a: Architecture search
print(f"\n[STEP 3a] Architecture search")

# Added by Claude: Profile architecture search
from .profiling import profile_section
with profile_section("Architecture Search"):
    new_arch = awb_arch_search.search_architecture(
        model, task_id, trainWLoss, val_trainloader, val_exploader,
        test_curr, test_exp, config, trainer
    )
print(f"  Original: {original_arch}")
print(f"  Optimal: {new_arch}")
```

**Location 4:** Before A/B training (line ~313, inside else block for Condition 4)
```python
else:
    # CONDITION 4: Run full A/B training pipeline
    # STEP 3b: Train A/B
    print(f"\n[STEP 3b] Train A/B matrices ({ab_training_epochs} epochs) - Recorded separately")

    # Added by Claude: Profile A/B training
    from .profiling import profile_section
    import time
    ab_training_start = time.time()

    model = awb_ops.set_AB_matrices(model, original_arch, new_arch)
    # ... rest of A/B training code ...

    # After A/B training completes (line ~343)
    model = eqx.combine(diff_model, static_model)

    # Added by Claude: Report A/B training time
    if config.get('profiling_enabled'):
        ab_elapsed = time.time() - ab_training_start
        print(f"[PROFILE] A/B training total: {ab_elapsed:.2f}s")
```

#### Step 2.5: Add Profiling to Training Loop

**File:** `src/cl/core/loops.py`

**Location 1:** Top of file (after imports)
```python
# Added by Claude: Profiling support
from .profiling import profile, profile_section
import time
```

**Location 2:** train__CL function (line ~295) - ADD DECORATOR
```python
# Added by Claude: Profile training loop
@profile("Training Loop")
def train__CL(
    self, train__, params, static, opt_state, optim, n_iter, save_iter,
    task_id, config, record_dict, notABTrain=True, problem_type='vectors',
    loss_type='regression', phase='main', record_training=True,
    global_iteration_offset=0
):
    # ... existing code ...
```

**Location 3:** Inside train__CL, around JAX pre-conversion (lines ~335-356)
```python
# Added by Claude: Profile JAX pre-conversion (Phase 3 optimization)
profiling_enabled = config.get('profiling_enabled', False)

if profiling_enabled:
    print(f"\n[PROFILE] JAX pre-conversion starting...")
    preconv_start = time.time()

# Pre-convert training batches to JAX arrays (Phase 3 optimization)
if problem_type == 'vectors':
    # ... existing pre-conversion code ...

if profiling_enabled:
    preconv_elapsed = time.time() - preconv_start
    print(f"[PROFILE] JAX pre-conversion complete: {preconv_elapsed:.2f}s")
    print(f"[PROFILE]   Train batches: {len(train_batches_jax)}")
    print(f"[PROFILE]   Exp batches: {len(exp_batches_jax)}")
```

**Location 4:** Inside epoch loop, profile first batch (lines ~400+)
```python
# Training loop
for epoch in range(n_iter):
    # Added by Claude: Profile first epoch first batch
    if epoch == 0 and config.get('profiling_enabled'):
        print(f"\n[PROFILE] Starting first epoch...")
        first_batch_start = time.time()
        first_batch_done = False

    # Iterate over batches
    for batch_idx, (batch, batch_ex) in enumerate(zip(train_batches_jax, exp_batches_jax)):
        # ... existing training code ...

        # Added by Claude: Report first batch time
        if epoch == 0 and batch_idx == 0 and config.get('profiling_enabled') and not first_batch_done:
            first_batch_elapsed = time.time() - first_batch_start
            print(f"[PROFILE] First batch complete: {first_batch_elapsed:.2f}s")
            first_batch_done = True

    # ... rest of epoch loop ...
```

#### Step 2.6: Activate Profiling in Runner

**File:** `src/cl/runners/generic_runner.py`

**Location 1:** Top of file (after imports)
```python
# Added by Claude: Profiling support
from cl.core.profiling import enable_profiling
```

**Location 2:** Inside main runner function, before training starts
Search for where config is loaded and used. Add near the beginning:

```python
# Added by Claude: Enable profiling if requested
enable_profiling(config.get('profiling_enabled', False))
if config.get('profiling_enabled'):
    print(f"\n{'='*60}")
    print(f"PROFILING ENABLED - Detailed timing information will be shown")
    print(f"{'='*60}\n")
```

---

### PART 3: Progressive Testing Workflow

**Objective:** Run profiling configs in order of increasing complexity to isolate bottleneck.

#### Test 1: Sine Wave (Baseline)

```bash
cd /Users/kraghavan/Desktop/JMLR_paper/ContLearn
python run_files/scripts/run.py kkt_run/configs/debug/sine_awb_profile_debug.json > profile_sine.log 2>&1
```

**Expected Output:**
```
[PROFILE] AWB Task Pipeline starting...
[PROFILE] Training Loop starting...
[PROFILE] JAX pre-conversion starting...
[PROFILE] JAX pre-conversion complete: 0.5s
[PROFILE] Starting first epoch...
[PROFILE] First batch complete: 2.1s
[PROFILE] Training Loop complete: 45.2s
[PROFILE] Architecture Search starting...
[PROFILE] Architecture Search complete: 12.3s
[PROFILE] Training Loop starting...  (A/B training)
[PROFILE] JAX pre-conversion complete: 0.4s
[PROFILE] First batch complete: 3.2s
[PROFILE] Training Loop complete: 15.8s
[PROFILE] A/B training total: 15.8s
[PROFILE] AWB Task Pipeline complete: 120.5s
```

**Analysis:** Should be fast (simple MLP). Establishes baseline AWB timing.

#### Test 2: MNIST (Medium Complexity)

```bash
python run_files/scripts/run.py kkt_run/configs/debug/mnist_awb_profile_debug.json > profile_mnist.log 2>&1
```

**Expected:** Moderate compilation times for CNN. A/B training should still be reasonable (<5 min).

#### Test 3: CIFAR-10 Standard (Normal Case - 31s baseline)

```bash
python run_files/scripts/run.py kkt_run/configs/debug/cifar10_standard_profile_debug.json > profile_cifar10_standard.log 2>&1
```

**Expected Output:**
```
[PROFILE] Training Loop starting...
[PROFILE] JAX pre-conversion complete: 2.1s
[PROFILE] Starting first epoch...
[PROFILE] First batch complete: 31.2s  ← NORMAL CIFAR-10 TIME
[PROFILE] Training Loop complete: 310.0s (10 epochs × 31s)
```

**Analysis:** This captures the "normal" CIFAR-10 behavior without AWB. First batch ~31s is expected JIT compilation time.

#### Test 4: CIFAR-10 AWB (Problem Case)

```bash
python run_files/scripts/run.py kkt_run/configs/debug/cifar10_awb_profile_debug.json > profile_cifar10_awb.log 2>&1
```

**Expected Output (if bug still exists):**
```
[PROFILE] AWB Task Pipeline starting...
[PROFILE] Training Loop starting...  (preliminary training)
[PROFILE] First batch complete: 31.2s  ← NORMAL
[PROFILE] Architecture Search starting...
[PROFILE] Architecture Search complete: 45.0s
[PROFILE] Training Loop starting...  (A/B training)
[PROFILE] JAX pre-conversion complete: 3.5s
[PROFILE] Starting first epoch...
[PROFILE] First batch complete: 7234.1s  ← 2-HOUR HANG (ABNORMAL!)
```

**Analysis:** Compare this to Test 3. Same dataset, same network architecture, but A/B training first batch takes 200x longer than normal training first batch.

#### Collecting Results

After each test, user should extract `[PROFILE]` lines:
```bash
grep "\[PROFILE\]" profile_sine.log > profile_sine_summary.txt
grep "\[PROFILE\]" profile_mnist.log > profile_mnist_summary.txt
grep "\[PROFILE\]" profile_cifar10_standard.log > profile_cifar10_standard_summary.txt
grep "\[PROFILE\]" profile_cifar10_awb.log > profile_cifar10_awb_summary.txt
```

Share these summaries for analysis.

---

### PART 4: Optimization Based on Profiling Data

**Once profiling identifies the bottleneck, we'll implement one of these fixes:**

#### Hypothesis 1: Phase 3 Pre-Conversion Causing Issues

**Symptom:** Pre-conversion takes hours instead of seconds during A/B training.

**Fix:** Disable Phase 3 optimization for A/B training (use lazy conversion).

**Location:** `src/cl/core/loops.py` line ~337

```python
# Added by Claude: Skip pre-conversion during A/B training to avoid memory pressure
if notABTrain:  # False = A/B training phase
    # Use lazy conversion for A/B training
    train_batches_jax = trainloader
    exp_batches_jax = exploader
else:
    # Phase 3 optimization: pre-convert for standard training
    # ... existing pre-conversion code ...
```

#### Hypothesis 2: Experience Replay Too Large

**Symptom:** 200k experience samples overwhelming GPU during A/B training.

**Fix:** Reduce experience replay during A/B training.

**Location:** `src/cl/core/awb_pipeline.py` before A/B training call

```python
# Added by Claude: Use smaller experience replay for A/B training
# Full experience replay (200k samples) may overwhelm GPU during A/B JIT compilation
# Reduce to 50k samples for A/B training phase
from torch.utils.data import Subset
import random

exp_dataset = exploader.dataset
exp_size = len(exp_dataset)
reduced_size = min(exp_size, 50000)
indices = random.sample(range(exp_size), reduced_size)
reduced_exp_dataset = Subset(exp_dataset, indices)
reduced_exploader = DataLoader(reduced_exp_dataset, batch_size=config['batch_size'], shuffle=True)

# Use reduced loader for A/B training
diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
    train__=(trainloader, reduced_exploader, valloader, testloader),  # ← reduced
    # ... rest of args ...
)
```

#### Hypothesis 3: Complex Hamiltonian JIT Compilation

**Symptom:** JAX takes 2 hours to JIT compile AWB Hamiltonian.

**Fix:** Simplify Hamiltonian during A/B training (skip dV computation).

**Location:** `src/cl/core/hamiltonian.py`

Add new simplified function for A/B training:
```python
# Added by Claude: Simplified Hamiltonian for A/B training
def _hamiltonian_core_class_awb_simplified(params, static, x, y, exp_x, exp_y, alpha, beta):
    """
    Simplified Hamiltonian for A/B training - skips dV computation.

    During A/B training, we only need delta_theta and grad_V.
    Skip expensive dV perturbation computation to speed up JIT compilation.
    """
    # Same as full Hamiltonian but without deltax/dV computation
    # ... simplified implementation ...
```

Then modify AWB pipeline to use simplified version:
```python
# In awb_pipeline.py, pass flag to trainer
diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
    # ... args ...
    config={**config, 'use_simplified_hamiltonian': True},  # ← flag for A/B training
    notABTrain=False,
    # ... rest ...
)
```

#### Hypothesis 4: A/B Matrix Structure Issue

**Symptom:** A/B inline transformations prevent efficient JIT.

**Fix:** Restructure models to use A/B as separate layers.

**Location:** `src/cl/models/cnn.py`, `get_AWBT()` method

This would be a larger refactor - restructure A/B matrices as Equinox layers instead of inline transformations.

---

## Critical File Locations

### Files to Create
1. `src/cl/core/profiling.py` - Profiling decorator module (~65 lines)
2. `kkt_run/configs/debug/` - Debug configs directory
3. `kkt_run/configs/debug/sine_awb_profile_debug.json`
4. `kkt_run/configs/debug/mnist_awb_profile_debug.json`
5. `kkt_run/configs/debug/cifar10_awb_profile_debug.json`
6. `kkt_run/configs/debug/cifar10_standard_profile_debug.json`

### Files to Modify
1. `src/cl/core/awb_pipeline.py`
   - Line 1: Import profiling
   - Line 153: Add `@profile` decorator to `run_awb_task()`
   - Line 289: Add profiling to architecture search
   - Line 302: **CRITICAL** - Add `awb_skip_transfer` flag check
   - Line 313: Add profiling to A/B training

2. `src/cl/core/loops.py`
   - Line 1: Import profiling
   - Line 295: Add `@profile` decorator to `train__CL()`
   - Lines 335-356: Add profiling to JAX pre-conversion
   - Line 400+: Add profiling to first epoch/first batch

3. `src/cl/runners/generic_runner.py`
   - Line 1: Import profiling
   - Early in runner: Call `enable_profiling(config.get('profiling_enabled'))`

4. `src/cl/config/constants.py`
   - After line 14: Add `DEFAULT_PROFILING_ENABLED = False`

### Key Code Sections

**AWB Pipeline** (`src/cl/core/awb_pipeline.py`):
- Line 153: `run_awb_task()` - Main AWB orchestration
- Line 289: Architecture search
- Line 302-343: **CRITICAL** - A/B training decision and execution
- Line 356+: Warmup and main training

**Training Loop** (`src/cl/core/loops.py`):
- Line 295: `train__CL()` - Unified training loop
- Line 335-356: Phase 3 JAX pre-conversion
- Line 400+: Epoch loop with batch iteration

**Hamiltonian** (`src/cl/core/hamiltonian.py`):
- Line 305: `_hamiltonian_core_class_awb()` - AWB Hamiltonian computation
- This is where the 2-hour hang likely occurs during JIT compilation

---

## Testing Checklist

### After Part 1 Implementation
- [ ] Create `src/cl/core/profiling.py`
- [ ] Add profiling flag to `constants.py`
- [ ] Create `kkt_run/configs/debug/` directory
- [ ] Create 4 debug config files
- [ ] Add profiling imports to `awb_pipeline.py`, `loops.py`, `generic_runner.py`
- [ ] Add `@profile` decorators
- [ ] Add profiling sections with timing
- [ ] **Implement `awb_skip_transfer` flag check** in `awb_pipeline.py:302`

### After Part 1 Testing
- [ ] Run Condition 3 config, verify `awb_skip_transfer` works
- [ ] Should see: "[STEP 3b] Skipping A/B training"
- [ ] Should NOT see 100-epoch A/B progress bar
- [ ] Task should complete in ~30 minutes instead of ~2.5 hours

### Progressive Profiling Tests
- [ ] Test 1: Run `sine_awb_profile_debug.json`, collect profile summary
- [ ] Test 2: Run `mnist_awb_profile_debug.json`, collect profile summary
- [ ] Test 3: Run `cifar10_standard_profile_debug.json`, collect 31s baseline
- [ ] Test 4: Run `cifar10_awb_profile_debug.json`, collect 2hr hang data
- [ ] Compare all summaries, identify exact bottleneck

### After Part 2 Analysis
- [ ] Analyze profiling data to confirm bottleneck location
- [ ] Choose appropriate optimization (Hypothesis 1, 2, 3, or 4)
- [ ] Implement optimization
- [ ] Re-run `cifar10_awb_profile_debug.json` to verify fix
- [ ] First A/B epoch should complete in 30-180s instead of 2 hours

---

## Expected Outcomes

### Short Term (Part 1)
- ✅ Condition 3 goes from 25 hours → **5 hours**
- ✅ User can run Condition 3 experiments practically

### Medium Term (Part 2)
- ✅ Profiling data identifies exact bottleneck
- ✅ Clear understanding of why A/B training hangs

### Long Term (Part 3)
- ✅ Condition 4 goes from 50+ hours → **10-15 hours**
- ✅ Both Conditions 3 & 4 practical to run
- ✅ All 5 datasets × 4 conditions = 20 experiments completable in reasonable time

---

## Communication with User

### What to Ask User
1. "I've implemented Part 1 (profiling + `awb_skip_transfer`). Please run Test 1 (Sine) and share the `[PROFILE]` output."
2. After each test: "Please run Test N and share the profile summary. We're looking for where timing dramatically increases."
3. After Test 4: "Based on profiling data, the bottleneck appears to be [X]. I recommend implementing [Hypothesis Y]. Approve?"

### What User Will Provide
- Profile summaries from each test
- Confirmation that `awb_skip_transfer` fix works
- Approval for optimization approach

### What NOT to Assume
- Don't assume which hypothesis is correct without profiling data
- Don't implement all optimizations at once - wait for data
- Don't modify core Hamiltonian without user approval (high risk)

---

## Handoff Notes for Next Claude Instance

### Current Status
- **Plan Created:** December 26, 2024
- **Implementation Status:** Ready to begin
- **User Approval:** Plan approved, awaiting implementation

### First Steps for Next Session
1. Read this document fully
2. Read `session_context.md` for additional context
3. Check if any implementation has started (look for `profiling.py`)
4. If starting fresh: Begin with Part 1, Step 2.1 (create `profiling.py`)
5. If partially done: Check todo list, continue where previous instance left off

### Important Context
- This is a **long-term debugging effort** - may take multiple sessions
- User is on H200 GPU with JAX/Equinox framework
- User values **clean, non-intrusive code** - no cumbersome additions
- User wants **progressive testing** - start simple (Sine), build to complex (CIFAR-10)
- User will **run tests and report back** - we analyze together

### Red Flags to Watch For
- If profiling shows pre-conversion taking hours → Hypothesis 1
- If GPU memory errors during A/B training → Hypothesis 2
- If first batch time >> 180s → Likely Hypothesis 3 (JIT issue)
- If vmap/jit errors in logs → Hypothesis 4 (structural issue)

### Communication Style
- Be direct and technical
- User is experienced researcher, no hand-holding needed
- Focus on data-driven decisions
- Ask for approval before major changes (especially Hamiltonian modifications)

---

## References

- **Main Documentation:** `session_context.md`
- **Project Structure:** `.claude/CLAUDE.md`
- **Code Preferences:** `CLAUDE.md`
- **Plan File:** `/Users/kraghavan/.claude/plans/synthetic-stargazing-lighthouse.md`
- **This Document:** `PROFILING_PLAN.md`

---

## Revision History

- **v1.0** (Dec 26, 2024): Initial plan created
- User requested comprehensive handoff document for long-term debugging effort
- Includes full implementation details, testing workflow, and optimization hypotheses
