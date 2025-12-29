# MNIST Conditions: Configuration & Code Execution Map

**Purpose:** Reference guide for understanding how each experimental condition maps to code execution paths.

**Last Updated:** 2025-12-29

---

## Overview of 4 Conditions

| Condition | Name | Philosophy | Key Features |
|-----------|------|------------|--------------|
| 1 | Baseline | No smoothness | Constant LR, no warmup, fixed arch, static grad weights |
| 2 | Heuristics | Smoothness via heuristics | Cosine LR, task warmup, adaptive grad weights |
| 3 | Arch Search | Capacity without transfer | Architecture adaptation, random reinit |
| 4 | AWB Full | Architecture + transfer | Architecture adaptation + A/B knowledge transfer |

---

## Condition 1: Baseline (No Smoothness)

### Configuration
```json
{
    "lr_schedule": "constant",
    "lr": 0.0001,
    "task_warmup_enabled": false,
    "awb_enabled": false,
    "adaptive_grad_weights_enabled": false,
    "grad_weights": [0.4, 0.4, 0.1]
}
```

### Code Execution Path

**Entry:** `train_model()` → generic_runner.py:857

**For ALL tasks (0-9):**
```
Line 961: if task_id == 0 or not awb_enabled:  → TRUE
    ├─ Standard CL path (Lines 961-1157)
    │
    ├─ Line 955: compute_task_lr()
    │   └─ lr_schedule="constant" → Returns 0.0001 (Line 409)
    │
    ├─ Line 981: compute_adaptive_grad_weights()
    │   ├─ adaptive_enabled=False (from config)
    │   └─ Returns [0.4, 0.4, 0.1] (Line 366)
    │
    ├─ Line 985: if task_id > 0 and warmup_enabled and warmup_epochs > 0:  → FALSE
    │   └─ Warmup SKIPPED (Lines 985-1063)
    │
    └─ Line 1064-1094: Direct training (no warmup)
        └─ trainer.train__CL(n_iter=150, grad_weights=[0.4, 0.4, 0.1])
            └─ loops.py: 150 epochs with Hamiltonian gradient
```

### Key Characteristics
- **LR:** 0.0001 constant across all tasks
- **Grad weights:** [0.4, 0.4, 0.1] static (40% current, 40% exp, 10% reg)
- **No smoothing:** Sharp task transitions
- **Fixed architecture:** CNN architecture never changes
- **Total epochs/task:** 150

### Files Executed
- ✅ `generic_runner.py:961-1157` (Standard CL)
- ✅ `generic_runner.py:1064-1094` (Direct training)
- ✅ `generic_runner.py:409` (Constant LR)
- ✅ `loops.py:train__CL()` (Training loop)
- ✅ `hamiltonian.py` (Gradient computation)
- ❌ `awb_pipeline.py` (AWB disabled)

---

## Condition 2: Heuristics (Smoothness via Heuristics)

### Configuration
```json
{
    "lr_schedule": "cosine",
    "lr": 0.0001,
    "task_warmup_enabled": true,
    "task_warmup_epochs": 25,
    "task_warmup_lr_factor": 0.1,
    "warmup_grad_weights": [1.0, 0.0, 0.0],
    "awb_enabled": false,
    "adaptive_grad_weights_enabled": true,
    "grad_weights_base": [0.4, 0.4, 0.1],
    "grad_weights_max_current": 0.7,
    "grad_weights_min_experience": 0.2,
    "grad_weights_loss_ratio_threshold": 1.2
}
```

### Code Execution Path

**Task 0:**
```
Line 961: TRUE (task_id == 0)
    └─ Standard training (150 epochs, no warmup for task 0)
        └─ grad_weights = [0.4, 0.4, 0.1]
```

**Tasks 1-9:**
```
Line 955: compute_task_lr()
    ├─ lr_schedule="cosine" → Line 420-424
    └─ Returns: task_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + cos(π*progress))
        ├─ Task 1: ~0.000098
        ├─ Task 5: ~0.00005
        └─ Task 9: ~0.000001

Line 981: compute_adaptive_grad_weights(loss_ratio)
    ├─ adaptive_enabled=True
    ├─ base_weights=[0.4, 0.4, 0.1] (from config!)
    │
    ├─ If loss_ratio ≤ 1.2:
    │   └─ Return [0.4, 0.4, 0.1]
    │
    └─ If loss_ratio > 1.2:
        ├─ progress = min((loss_ratio - 1.2) / 1.2, 1.0)
        ├─ alpha = 0.4 + progress * (0.7 - 0.4)  → 0.4 to 0.7
        ├─ beta = 0.4 + progress * (0.2 - 0.4)   → 0.4 to 0.2
        └─ Return [alpha, beta, 0.1]

Line 985: if task_id > 0 and warmup_enabled and warmup_epochs > 0:  → TRUE
    │
    ├─ WARMUP PHASE (Lines 986-1018):
    │   ├─ warmup_lr = task_lr * 0.1  (10% of task LR)
    │   ├─ warmup_config['grad_weights'] = [1.0, 0.0, 0.0]
    │   └─ trainer.train__CL(
    │         n_iter=25,
    │         trainloader used for both current AND "experience",
    │         phase='warmup',
    │         record_training=False
    │       )
    │
    └─ MAIN PHASE (Lines 1020-1052):
        ├─ main_epochs = 150 - 25 = 125
        ├─ main_config['grad_weights'] = adaptive_grad_weights
        └─ trainer.train__CL(
              n_iter=125,
              full experience replay,
              phase='main',
              record_training=True
            )
```

### Key Characteristics
- **LR:** Cosine decay from 0.0001 → ~0.000001 across tasks
- **Warmup:** 25 epochs at 10% LR, focus on current task [1.0, 0.0, 0.0]
- **Main training:** 125 epochs with adaptive grad weights
- **Grad weights adaptation:**
  - Normal tasks: [0.4, 0.4, 0.1]
  - Struggling tasks: Shifts toward [0.7, 0.2, 0.1] (more current, less exp)
- **Fixed architecture:** No AWB
- **Total epochs/task:** 150 (25 warmup + 125 main)

### Files Executed
- ✅ `generic_runner.py:420-424` (Cosine LR schedule)
- ✅ `generic_runner.py:345-385` (Adaptive grad weights)
- ✅ `generic_runner.py:985-1063` (Warmup phase)
- ✅ `generic_runner.py:1020-1052` (Main phase)
- ✅ `loops.py:train__CL()` (Both warmup and main)
- ❌ `awb_pipeline.py` (AWB disabled)

---

## Condition 3: Architecture Search (No Transfer)

### Configuration
```json
{
    "lr_schedule": "constant",
    "lr": 0.0001,
    "awb_enabled": true,
    "awb_skip_transfer": true,
    "awb_preliminary_epochs": 30,
    "awb_ab_warmup_epochs": 2,
    "awb_loss_ratio_threshold": 1.1,
    "awb_validation_ratio": 1.0,
    "adaptive_grad_weights_enabled": false,
    "grad_weights": [0.4, 0.4, 0.1]
}
```

### Code Execution Path

**Task 0:**
```
Line 961: if task_id == 0 or not awb_enabled:  → TRUE (task_id == 0)
    └─ Standard CL (150 epochs, like Condition 1)
```

**Tasks 1-9:**
```
Line 961: if task_id == 0 or not awb_enabled:  → FALSE
    └─ AWB Pipeline (Lines 1158-1240)
        └─ Calls run_awb_task() in awb_pipeline.py:159

awb_pipeline.py execution:
    │
    ├─ STEP 1: Preliminary Training (Lines 220-256)
    │   └─ trainer.train__CL(n_iter=30, phase='preliminary')
    │       └─ Quick assessment with current architecture
    │
    ├─ STEP 2: Architecture Decision (Lines 258-283)
    │   ├─ loss_ratio = trainWLoss / previous_task_loss
    │   ├─ threshold = 1.1 (from config)
    │   └─ change_arch = (loss_ratio > 1.1)
    │
    ├─ IF change_arch == TRUE:
    │   │
    │   ├─ STEP 3a: Architecture Search (Lines 286-308)
    │   │   ├─ val_trainloader = create_balanced_validation_set(trainloader, 1.0)
    │   │   └─ new_arch = awb_ops.search_architecture(...)
    │   │       └─ arch_search/cnn_search.py: Test filter sizes, hidden dims
    │   │
    │   ├─ STEP 3b: Transfer Decision (Lines 317-338) ⭐ KEY
    │   │   ├─ skip_transfer = config['awb_skip_transfer']  → TRUE
    │   │   │
    │   │   └─ IF skip_transfer:  ✅ EXECUTED
    │   │       ├─ model = awb_ops.set_AB_matrices(model, orig, new)
    │   │       │   └─ Random initialization of A and B matrices
    │   │       │
    │   │       ├─ STEP 4: Compute V = A @ W @ B^T (Line 329)
    │   │       │   └─ V inherits structure from W but mostly random
    │   │       │
    │   │       └─ optim = create_optimizer(config)
    │   │           └─ Fresh optimizer, random init
    │   │
    │   └─ STEP 5: Train New Architecture (Lines 404-424)
    │       ├─ IF ab_warmup_epochs > 0:  (2 epochs)
    │       │   └─ trainer.train__CL(n_iter=2, phase='warmup')
    │       │
    │       └─ trainer.train__CL(n_iter=150, phase='main')
    │
    └─ IF change_arch == FALSE:
        └─ Standard training (Lines 449-471)
            └─ trainer.train__CL(n_iter=150, phase='main')
```

### Key Characteristics
- **LR:** 0.0001 constant
- **Preliminary:** 30 epochs to assess architecture adequacy
- **Architecture search:** Triggered when loss_ratio > 1.1
- **Transfer method:** Random A/B matrices (NO learned transfer)
- **Effect:** New capacity but loses old knowledge
- **Grad weights:** [0.4, 0.4, 0.1] static
- **Total epochs/task (if arch changes):** 30 prelim + 2 warmup + 150 main = 182

### Code Branching Decision Point
```python
# Line 318 in awb_pipeline.py
skip_transfer = config.get('awb_skip_transfer', False)

if skip_transfer:  # ← Condition 3 takes this branch
    # Random A/B initialization
    # Lines 320-338
else:  # ← Condition 4 takes this branch
    # Learned A/B training
    # Lines 340-402
```

### Files Executed
- ✅ `awb_pipeline.py:159-471` (Full AWB pipeline)
- ✅ `awb_pipeline.py:220-256` (Preliminary training)
- ✅ `awb_pipeline.py:286-308` (Architecture search)
- ✅ `awb_pipeline.py:320-338` (Random A/B init)
- ✅ `arch_search/cnn_search.py` (CNN search)
- ✅ `models/cnn.py:CNNAWBOps` (A/B operations)
- ❌ `awb_pipeline.py:340-402` (A/B training - skipped)

---

## Condition 4: AWB Full (Architecture + Transfer)

### Configuration
```json
{
    "lr_schedule": "cosine",
    "lr": 0.0001,
    "task_warmup_enabled": true,
    "task_warmup_epochs": 25,
    "task_warmup_lr_factor": 0.1,
    "warmup_grad_weights": [1.0, 0.0, 0.0],
    "awb_enabled": true,
    "awb_skip_transfer": false,  // ← KEY DIFFERENCE
    "awb_preliminary_epochs": 30,
    "awb_ab_training_epochs": 100,
    "awb_ab_warmup_epochs": 2,
    "awb_loss_ratio_threshold": 1.1,
    "adaptive_grad_weights_enabled": true,
    "grad_weights_base": [0.4, 0.4, 0.1],
    "grad_weights_max_current": 0.7,
    "grad_weights_min_experience": 0.2,
    "grad_weights_loss_ratio_threshold": 1.2
}
```

### Code Execution Path

**Task 0:**
```
Line 961: TRUE (task_id == 0)
    └─ Standard training (like Condition 2, with warmup disabled for task 0)
```

**Tasks 1-9:**
```
Line 961: if task_id == 0 or not awb_enabled:  → FALSE
    └─ AWB Pipeline (Lines 1158-1240)

awb_pipeline.py execution:
    │
    ├─ STEP 1: Preliminary Training (30 epochs)
    │   └─ Same as Condition 3
    │
    ├─ STEP 2: Architecture Decision
    │   └─ Same as Condition 3
    │
    ├─ IF change_arch == TRUE:
    │   │
    │   ├─ STEP 3a: Architecture Search
    │   │   └─ Same as Condition 3
    │   │
    │   ├─ STEP 3b: Transfer Decision (Lines 317-402) ⭐ KEY
    │   │   ├─ skip_transfer = config['awb_skip_transfer']  → FALSE
    │   │   │
    │   │   └─ ELSE branch:  ✅ EXECUTED (Lines 340-402)
    │   │       │
    │   │       ├─ Initialize A/B matrices (random first)
    │   │       ├─ model = awb_ops.set_AB_matrices(model, orig, new)
    │   │       │
    │   │       ├─ Partition for A/B training:
    │   │       │   └─ diff_model = A/B matrices (trainable)
    │   │       │   └─ static_model = W weights (frozen)
    │   │       │
    │   │       ├─ Train A/B (100 epochs):
    │   │       │   └─ trainer.train__CL(
    │   │       │         params=diff_model,  # Only A/B trainable
    │   │       │         static=static_model,  # W frozen
    │   │       │         n_iter=100,
    │   │       │         notABTrain=False,  # ← Signals A/B mode
    │   │       │         phase='ab'
    │   │       │       )
    │   │       │
    │   │       ├─ Check convergence:
    │   │       │   └─ If ab_loss still high, continue training (up to 2 iterations)
    │   │       │
    │   │       └─ STEP 4: Compute V = A @ W @ B^T
    │   │           └─ V now contains LEARNED compression of W
    │   │
    │   └─ STEP 5: Train V (Lines 404-424)
    │       ├─ Warmup (2 epochs) - optional
    │       └─ Main training (150 epochs)
    │           └─ Could also have task warmup if enabled
    │
    └─ IF change_arch == FALSE:
        └─ Standard training with task warmup (if enabled)
            └─ Uses Condition 2 smoothing (cosine + warmup + adaptive)
```

### Key Characteristics
- **Combines:** Condition 2 (smoothing) + Condition 3 (arch search) + A/B transfer
- **LR:** Cosine decay
- **Task warmup:** 25 epochs (like Condition 2)
- **Preliminary:** 30 epochs
- **A/B training:** 100 epochs (learns optimal compression)
- **Transfer method:** Learned A/B matrices via gradient descent
- **Grad weights:** Adaptive (like Condition 2)
- **Total epochs/task (if arch changes):** 30 prelim + 100 A/B + 2 warmup + 150 main = 282!

### A/B Training Details (Lines 340-402)

**What gets trained:**
```python
# Partition model
diff_model, static_model = awb_ops.partition_for_AB_training(model)
# diff_model contains: A and B matrices (trainable)
# static_model contains: W weights (FROZEN)

# Training minimizes loss while W is frozen
# Learns: A, B such that A @ W @ B^T ≈ optimal for both current and old tasks
```

**Objective:**
- Find A and B that compress W while preserving performance
- A/B matrices learn to extract/project relevant features
- V = A @ W @ B^T becomes the transferred knowledge

### Files Executed
- ✅ `awb_pipeline.py:159-471` (Full AWB pipeline)
- ✅ `awb_pipeline.py:340-402` (A/B training - KEY DIFFERENCE)
- ✅ `awb_pipeline.py:355-362` (A/B training loop)
- ✅ `loops.py:train__CL(notABTrain=False)` (A/B training mode)
- ✅ `generic_runner.py:420-424` (Cosine LR)
- ✅ `generic_runner.py:345-385` (Adaptive grad weights)
- ✅ All architecture search code (like Condition 3)

---

## Key Code Branching Points

### Branch Point 1: AWB Enabled? (generic_runner.py:961)
```python
if task_id == 0 or not awb_enabled:
    # → Conditions 1 & 2 take this branch
    # Standard CL path (Lines 961-1157)
else:
    # → Conditions 3 & 4 take this branch
    # AWB Pipeline (Lines 1158-1240)
```

### Branch Point 2: Task Warmup? (generic_runner.py:985)
```python
# Only in Standard CL path (Conditions 1 & 2)
if task_id > 0 and warmup_enabled and warmup_epochs > 0:
    # → Condition 2 takes this branch
    # Warmup + Main (Lines 985-1063)
else:
    # → Condition 1 takes this branch
    # Direct training (Lines 1064-1094)
```

### Branch Point 3: LR Schedule (generic_runner.py:406-431)
```python
schedule = config.get('lr_schedule', 'constant')

if schedule == 'constant':
    # → Conditions 1 & 3
    return base_lr  # 0.0001
elif schedule == 'cosine':
    # → Conditions 2 & 4
    progress = task_id / max(n_tasks - 1, 1)
    lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + cos(π * progress))
    return lr
```

### Branch Point 4: Skip A/B Transfer? (awb_pipeline.py:318)
```python
# Only in AWB Pipeline (Conditions 3 & 4)
skip_transfer = config.get('awb_skip_transfer', False)

if skip_transfer:
    # → Condition 3 takes this branch (Lines 320-338)
    # Random A/B initialization
    model = awb_ops.set_AB_matrices(model, orig_arch, new_arch)
    model = awb_ops.compute_V(model)  # V = A @ W @ B^T (random)
else:
    # → Condition 4 takes this branch (Lines 340-402)
    # Learned A/B training (100 epochs)
    # Train A/B matrices with W frozen
    # Then: V = A @ W @ B^T (learned compression)
```

### Branch Point 5: Adaptive Grad Weights? (generic_runner.py:363-385)
```python
adaptive_enabled = config.get('adaptive_grad_weights_enabled', True)

if not adaptive_enabled:
    # → Conditions 1 & 3
    return config.get('grad_weights', [0.4, 0.4, 0.1])
else:
    # → Conditions 2 & 4
    base_weights = config.get('grad_weights_base', [0.4, 0.4, 0.1])
    if loss_ratio <= threshold:
        return base_weights
    else:
        # Shift toward [max_current, min_experience, base_reg]
        # More focus on current task when struggling
```

---

## Comparison Matrix

| Feature | Cond 1 | Cond 2 | Cond 3 | Cond 4 |
|---------|--------|--------|--------|--------|
| **LR Schedule** | Constant | Cosine | Constant | Cosine |
| **Task Warmup** | ❌ No | ✅ 25 epochs | ❌ No | ✅ 25 epochs |
| **Architecture Search** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Preliminary Training** | ❌ No | ❌ No | ✅ 30 epochs | ✅ 30 epochs |
| **A/B Transfer** | N/A | N/A | ❌ Random init | ✅ 100 epochs |
| **Adaptive Grad Weights** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Total Epochs (no arch change)** | 150 | 150 | 180 | 180 |
| **Total Epochs (arch change)** | N/A | N/A | 182 | 282 |
| **Computational Cost** | Low | Low | Medium | High |

---

## Expected Behaviors

### Condition 1 (Baseline)
- **Expected**: Moderate forgetting, negative FWT
- **Why**: No smoothing, sharp transitions, fixed capacity
- **Use case**: Baseline for comparison

### Condition 2 (Heuristics)
- **Expected**: Low forgetting, positive BWT, good FWT
- **Why**: Warmup prevents disruption, cosine LR smoother, adaptive weights
- **Use case**: Best for preventing catastrophic forgetting

### Condition 3 (Arch Search)
- **Expected**: Good FWT, moderate-high forgetting
- **Why**: More capacity helps new tasks, but random reinit loses old knowledge
- **Use case**: When capacity is bottleneck, can afford some forgetting

### Condition 4 (AWB Full)
- **Expected**: Best FWT, variable BWT (depends on A/B quality)
- **Why**: Maximum capacity + smoothing + learned transfer
- **Use case**: When computational cost is acceptable, want best of all worlds
- **Risk**: A/B training may fail (information loss in compression)

---

## Debugging Guide

### If Condition X not behaving as expected:

**Check 1: Config keys**
- Ensure correct parameter names (e.g., `task_warmup_epochs` not `warmup_epochs`)
- Verify boolean flags are set correctly

**Check 2: Adaptive features**
- If grad_weights seems wrong, check `adaptive_grad_weights_enabled`
- Default is `True`, explicitly set to `False` to use static weights

**Check 3: Loss ratio**
- Architecture search only triggers if `loss_ratio > threshold`
- Check preliminary loss vs previous task loss
- Adjust `awb_loss_ratio_threshold` if needed

**Check 4: Warmup activation**
- Task warmup requires: `task_warmup_enabled: true` AND `task_warmup_epochs > 0`
- Task 0 never gets warmup (no previous knowledge to protect)

**Check 5: A/B training**
- Only runs if: `awb_enabled: true` AND `awb_skip_transfer: false` AND architecture changed
- Check logs for "STEP 3b" to confirm which path taken

---

## File Reference Map

### Core Training
- `generic_runner.py:857` - Entry point `train_model()`
- `loops.py` - `TrainingLoopsMixin.train__CL()`
- `hamiltonian.py` - `HamiltonianMixin` (DO NOT OPTIMIZE)
- `recording.py` - `RecordingMixin`

### Branching Logic
- `generic_runner.py:961` - AWB vs Standard CL
- `generic_runner.py:985` - Warmup vs Direct
- `generic_runner.py:406-431` - LR schedule
- `generic_runner.py:345-385` - Adaptive grad weights
- `awb_pipeline.py:318` - A/B transfer vs random

### AWB Pipeline
- `awb_pipeline.py:159` - `run_awb_task()` orchestrator
- `awb_pipeline.py:220-256` - Preliminary training
- `awb_pipeline.py:258-283` - Architecture decision
- `awb_pipeline.py:286-308` - Architecture search
- `awb_pipeline.py:320-338` - Random A/B (Condition 3)
- `awb_pipeline.py:340-402` - Learned A/B (Condition 4)

### Architecture Search
- `arch_search/cnn_search.py` - CNN architecture search
- `arch_search/mlp_search.py` - MLP architecture search
- `arch_search/gcn_search.py` - GCN architecture search

### Model Operations
- `models/cnn.py:CNNAWBOps` - CNN-specific AWB operations
- `models/mlp.py:MLPAWBOps` - MLP-specific AWB operations
- `models/gcn.py:GCNAWBOps` - GCN-specific AWB operations

---

## Notes

**Important:** This document reflects configurations as of 2025-12-29. If configs are modified, update this document accordingly.

**Code Version:** ContLearn framework with layer-level AWB abstraction

**Verification:** Trace execution with debug prints to confirm branching paths.
