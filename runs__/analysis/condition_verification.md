# Experimental Condition Verification

**Date**: 2026-01-03
**Purpose**: Systematically verify that all 4 experimental conditions are implemented correctly across all datasets

---

## Expected Condition Definitions

### Condition 1: Baseline (No Smoothness)
**Config naming**: `*_condition1_baseline.json`

**Expected behavior**:
- Fixed architecture (no architecture search)
- Constant learning rate (no schedule)
- No warmup (neither LR warmup nor task warmup)
- No AWB enabled
- Standard Hamiltonian gradient training

**Config parameters**:
```json
{
  "awb_enabled": false,
  "lr_schedule": "constant",
  "lr": 0.0001,
  "warmup_epochs": 0,
  "task_warmup_enabled": false
}
```

---

### Condition 2: Smoothness via Heuristics
**Config naming**: `*_condition2_heuristics.json`

**Expected behavior**:
- Fixed architecture (no architecture search)
- Cosine LR schedule (smoothing across tasks)
- LR warmup from low value at start of training
- Task warmup at task transitions (optional)
- Adaptive gradient weights based on loss ratio

**Config parameters**:
```json
{
  "awb_enabled": false,
  "lr_schedule": "cosine",
  "lr": 0.0001,
  "warmup_epochs": 25,           // LR warmup at training start
  "lr_warmup_factor": 0.1,       // Start at 10% of base LR
  "task_warmup_enabled": true,   // Warmup at task transitions
  "task_warmup_epochs": ???,     // How many epochs per task?
  "task_warmup_lr_factor": ???   // LR reduction during task warmup
}
```

---

### Condition 3: Architecture Search, No Transfer
**Config naming**: `*_condition3_arch_no_transfer.json`

**Expected behavior**:
- Architecture search enabled via AWB pipeline
- Preliminary training (30 epochs) to evaluate need for arch change
- If arch changes: Random reinitialization (no transfer learning)
- No A/B matrix training
- Constant LR (no smoothing)
- No warmup

**Config parameters**:
```json
{
  "awb_enabled": true,
  "awb_skip_transfer": true,
  "awb_preliminary_epochs": 30,
  "awb_loss_ratio_threshold": 1.1,  // Threshold to trigger arch search
  "lr_schedule": "constant",
  "lr": 0.0001,
  "warmup_epochs": 0
}
```

**Critical questions**:
- Is architecture search actually being triggered?
- Are architectures changing across tasks?
- What criterion determines when to change architecture?

---

### Condition 4: AWB Full (Architecture + Transfer)
**Config naming**: `*_condition4_awb_full.json`

**Expected behavior**:
- Architecture search enabled
- Preliminary training (30 epochs)
- A/B matrix training (100 epochs) for knowledge transfer
- Transfer learning via V = A @ W @ B^T
- Cosine LR schedule (smoothing)
- LR warmup at training start
- May have task warmup

**Config parameters**:
```json
{
  "awb_enabled": true,
  "awb_skip_transfer": false,
  "awb_preliminary_epochs": 30,
  "awb_ab_training_epochs": 100,
  "awb_loss_ratio_threshold": 1.1,
  "force_arch_change": true,        // Force arch change for testing?
  "lr_schedule": "cosine",
  "lr": 0.0001,
  "warmup_epochs": 25,
  "lr_warmup_factor": 0.1
}
```

**Critical questions**:
- Is A/B training happening?
- Is V transformation being applied?
- Are architectures changing?

---

## Issues Discovered (Sine Dataset)

### 1. ❌ LR Warmup Not Implemented
**Issue**: The `warmup_epochs` parameter in configs is **NOT connected to any code**

**Evidence**:
- Configs C2 and C4 have `warmup_epochs: 25`
- Code only looks for `task_warmup_epochs` (different parameter!)
- No LR warmup from low value observed in logs
- Task 0 in C2 shows `lr=0.000100` immediately (should start at `0.00001` if `lr_warmup_factor=0.1`)

**Impact**: C2 and C4 are missing a key smoothing mechanism

**Fix needed**: Implement per-task LR warmup or global LR warmup at training start

---

### 2. ❓ Architecture Search Not Triggering (C3, C4)
**Issue**: No architecture changes detected across 10 tasks in Sine dataset

**Evidence**:
- PKL data shows all tasks have `arch_changed: False`
- All tasks use same architecture: `[3, 256, 256, 10]`
- C3 has 49 checkpoints vs C1's 40 (preliminary training IS happening)
- But no actual architecture modifications occur

**Possible explanations**:
1. Loss ratio threshold never exceeded (task too easy)
2. Architecture search is running but deciding not to change
3. `force_arch_change: true` only in C4, not C3
4. Architecture search implementation has bugs

**Investigation needed**:
- Check logs for "Architecture search" messages
- Check `awb_loss_ratio_threshold: 1.1` vs actual loss ratios
- Verify architecture search is being called

---

### 3. ✅ Cosine LR Schedule Working (Partial)
**Issue**: Cosine schedule IS working, but effect is subtle

**Evidence** (from logs):
- C1: `lr=0.000100` constant across all tasks
- C2: `lr=0.000100 → 0.000098 → 0.000093 → 0.000085 → 0.000075 → ...`
- Decays ~25% over 10 tasks

**Status**: WORKING but gradual effect makes early tasks look identical

---

### 4. ❓ Task Warmup Not Enabled
**Issue**: `task_warmup_enabled` not set in C2/C4 configs

**Evidence**:
- C2 config has no `task_warmup_enabled` field
- Default is likely `false`
- No task warmup messages in logs

**Investigation needed**:
- Should C2/C4 have task warmup?
- Is this intentional or missing?

---

## Verification Plan

### Phase 1: Code Review
For each condition, verify in `src/cl/runners/generic_runner.py`:
- [ ] C1: Constant LR path is correct
- [ ] C2: Cosine LR + LR warmup implementation
- [ ] C3: AWB preliminary training + arch search without transfer
- [ ] C4: Full AWB pipeline + cosine LR + warmup

### Phase 2: Dataset-by-Dataset Verification
For each dataset (sine, mnist, cifar10, cifar100, synthetic_graph):

#### Checklist per condition:
1. **Config file review**
   - [ ] All required parameters present
   - [ ] Parameters match condition definition

2. **Log file analysis**
   - [ ] Learning rates match expected schedule
   - [ ] Architecture changes logged (C3, C4)
   - [ ] AWB phases logged (C3, C4)
   - [ ] Warmup phases logged (C2, C4)

3. **PKL data analysis**
   - [ ] Metadata matches config
   - [ ] Architecture history shows changes (C3, C4)
   - [ ] Task phase info is correct
   - [ ] Checkpoint counts make sense

4. **Performance analysis**
   - [ ] Conditions show expected differentiation
   - [ ] C2 should be smoother than C1
   - [ ] C3 should adapt architecture
   - [ ] C4 should outperform others

---

## Investigation Status

### Completed:
- ✅ Sine dataset initial investigation
- ✅ Identified LR warmup missing
- ✅ Confirmed cosine LR working
- ✅ Identified architecture search not changing architectures

### In Progress:
- 🔄 Systematic condition verification (starting with C1)

### Pending:
- ⏳ MNIST verification
- ⏳ CIFAR-10 verification
- ⏳ CIFAR-100 verification
- ⏳ Synthetic graph verification
- ⏳ Fix implementation issues
- ⏳ Re-run experiments if needed

---

## Datasets to Verify

1. **sine** (regression, MLP)
   - n_tasks: 10
   - epochs_per_task: 200
   - Status: Initial investigation complete

2. **mnist** (classification, MLP)
   - n_tasks: 5
   - epochs_per_task: 150
   - Status: Not started

3. **cifar10** (classification, CNN)
   - n_tasks: 5
   - epochs_per_task: 150
   - Status: Not started

4. **cifar100** (classification, CNN)
   - n_tasks: 10
   - epochs_per_task: 150
   - Status: Not started

5. **synthetic_graph** (graph classification, GCN)
   - n_tasks: 5
   - epochs_per_task: 150
   - Status: Not started

---

## Notes

### Parameter Naming Confusion
- `warmup_epochs`: Supposed to be for LR warmup (NOT IMPLEMENTED)
- `task_warmup_epochs`: For task transition warmup (EXISTS)
- `lr_warmup_factor`: Factor for LR warmup start value (NOT USED)
- Need to clarify which warmup mechanism should be used

### Code Locations
- Main training loop: `src/cl/runners/generic_runner.py:857` (train_model)
- LR schedule: `src/cl/runners/generic_runner.py:388` (compute_task_lr)
- Optimizer creation: `src/cl/runners/generic_runner.py:209` (create_optimizer)
- AWB pipeline: `src/cl/core/awb.py`
- Architecture search: `src/cl/arch_search/`

---

## Next Steps

1. **Start with Condition 1 verification** across all datasets
2. Verify baseline is truly baseline (no hidden smoothing)
3. Document exact behavior in logs and PKL files
4. Move to Condition 2, 3, 4 systematically
5. Create summary table of findings
6. Propose fixes for any issues found
