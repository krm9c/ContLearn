# Validation Experiments - Setup Summary

**Date**: 2025-12-24
**Status**: ✅ Infrastructure ready for deployment

---

## What Has Been Generated

###  1. Configuration Files (24 files)

```
experiments/configs/
├── sine/ (4 configs)
├── mnist/ (4 configs)
├── permuted_mnist/ (4 configs)
├── cifar10/ (4 configs)
├── cifar100/ (4 configs)
└── synthetic_graph/ (4 configs)
```

Each dataset has 4 condition configs:
- `condition1_baseline.json` - No smoothness
- `condition2_heuristics.json` - LR schedule + warm start
- `condition3_arch_no_transfer.json` - Arch search, no A/B transfer
- `condition4_awb_full.json` - Full AWB

### 2. SLURM Submission Scripts (8 files)

**Per-Dataset Scripts** (for parallel submission):
- `submit_sine.slurm` - 20 experiments (4 conditions × 5 runs)
- `submit_mnist.slurm` - 20 experiments
- `submit_permuted_mnist.slurm` - 20 experiments
- `submit_cifar10.slurm` - 20 experiments
- `submit_cifar100.slurm` - 20 experiments
- `submit_synthetic_graph.slurm` - 20 experiments

**Phase Scripts**:
- `submit_phase1_quick.slurm` - Quick validation (Sine + MNIST, 24 exp)
- `submit_validation_full.slurm` - Full validation (all datasets, 120 exp)

### 3. Helper Scripts (3 files)

- `run_single_validation.sh` - Run a single experiment
- `run_parallel_validation.sh` - Distribute experiments across GPUs
- `run_dataset_validation.sh` - Run all conditions for one dataset

### 4. Generation Scripts (1 file)

- `scripts/generate_all_configs.py` - Regenerate all 24 configs (if needed)

### 5. Documentation (2 files)

- `EXPERIMENT_MANIFEST.md` - Complete list of all 120 experiments
- `SETUP_SUMMARY.md` - This file

---

## What Needs to Be Done

### ⚠️ Code Modifications Required

#### 1. Add `awb_skip_transfer` Flag Support

**Files to modify**:
- `src/cl/config/constants.py` - Add default value
- `src/cl/runners/generic_runner.py` - Add skip logic in AWB pipeline

**Implementation**:
```python
# In AWB decision logic (after architecture search)
if config.get('awb_skip_transfer', False):
    # Skip A/B training (Steps 3b-4)
    # Reinitialize model with new architecture (random weights)
    model = initialize_model_new_arch(new_dims, key)
else:
    # Full AWB pipeline (A/B training + V computation)
    model = awb_transfer_pipeline(model, new_dims, ...)
```

#### 2. Add Warm Start Support

**Files to modify**:
- `src/cl/config/constants.py` - Add `warmup_epochs`, `lr_warmup_factor`
- `src/cl/core/loops.py` - Implement warmup LR reduction

**Implementation**:
```python
# At task boundaries, if warmup_epochs > 0:
if is_new_task and config.get('warmup_epochs', 0) > 0:
    warmup_lr = base_lr * config.get('lr_warmup_factor', 0.1)
    # Train for warmup_epochs with reduced LR
    # Then restore base_lr for remaining epochs
```

#### 3. Per-Task Performance Tracking (for CL Metrics)

**Files to modify**:
- `src/cl/core/recording.py` - Add per-task accuracy matrix
- `src/cl/runners/generic_runner.py` - Test on all previous tasks after each task

**Implementation**:
```python
# After training task j, test on ALL tasks i ∈ [0, j]
record_dict['task_performance_matrix'][j] = {
    'task_0': test_accuracy_on_task_0,
    'task_1': test_accuracy_on_task_1,
    ...
    'task_j': test_accuracy_on_task_j
}
```

This enables computing:
- ACC = (1/T) Σ A_{T,i}
- BWT = (1/(T-1)) Σ (A_{T,i} - A_{i,i})
- F = (1/(T-1)) Σ max(A_{i,i} - A_{i,j})

### 📊 Analysis Scripts Needed

**Files to create**:
- `experiments/analysis/compute_metrics.py` - Compute CL metrics from results
- `experiments/analysis/generate_plots.py` - Create comparison plots
- `experiments/analysis/generate_tables.py` - Create LaTeX tables
- `experiments/scripts/check_completion.py` - Monitor experiment progress

---

## How to Run Experiments

### Step 1: Test Phase 1 (Quick Validation)

```bash
# On H200 cluster
cd ~/ContLearn
cd experiments/slurm

# Test with Sine + MNIST (24 experiments)
sbatch submit_phase1_quick.slurm

# Monitor
squeue -u $USER
tail -f ../logs/phase1_quick_*.out
```

### Step 2: Run Phase 2 (Full Validation)

**Option A: Submit all datasets in parallel** (Recommended)
```bash
sbatch submit_sine.slurm
sbatch submit_mnist.slurm
sbatch submit_permuted_mnist.slurm
sbatch submit_cifar10.slurm
sbatch submit_cifar100.slurm
sbatch submit_synthetic_graph.slurm
```

**Option B: Submit full validation** (Single job, sequential)
```bash
sbatch submit_validation_full.slurm
```

### Step 3: Transfer Results to Local Machine

```bash
# On local machine
rsync -avz --progress \
    user@h200cluster:~/ContLearn/experiments/results/ \
    ~/Desktop/JMLR_paper/ContLearn/experiments/results/
```

### Step 4: Analyze Results

```bash
# On local machine
cd ~/Desktop/JMLR_paper/ContLearn/experiments/analysis

# Compute metrics
python compute_metrics.py --results-dir ../results

# Generate plots
python generate_plots.py --results-dir ../results --output-dir figures

# Generate tables
python generate_tables.py --results-dir ../results --output-dir tables
```

---

## File Structure Summary

```
experiments/
├── configs/                      # ✅ 24 config files
│   ├── sine/                     # ✅ 4 files
│   ├── mnist/                    # ✅ 4 files
│   ├── permuted_mnist/           # ✅ 4 files
│   ├── cifar10/                  # ✅ 4 files
│   ├── cifar100/                 # ✅ 4 files
│   └── synthetic_graph/          # ✅ 4 files
├── slurm/                        # ✅ 11 scripts
│   ├── submit_*.slurm            # ✅ 8 slurm scripts
│   ├── run_single_validation.sh  # ✅
│   ├── run_parallel_validation.sh # ✅
│   └── run_dataset_validation.sh # ✅
├── scripts/                      # ✅ 1 file
│   └── generate_all_configs.py   # ✅
├── analysis/                     # ⚠️ TODO
│   ├── compute_metrics.py        # ⚠️ TODO
│   ├── generate_plots.py         # ⚠️ TODO
│   └── generate_tables.py        # ⚠️ TODO
├── results/                      # (empty, filled after runs)
├── logs/                         # (empty, filled after runs)
├── EXPERIMENT_MANIFEST.md        # ✅
└── SETUP_SUMMARY.md              # ✅
```

---

## Next Steps

### Priority 1: Code Modifications (Required before running)

1. ✅ Review configs - Verify all 24 configs are correct
2. ⚠️ Implement `awb_skip_transfer` flag in runners
3. ⚠️ Implement warm start (`warmup_epochs`, `lr_warmup_factor`)
4. ⚠️ Add per-task performance tracking to recording.py

### Priority 2: Analysis Infrastructure (Before or during runs)

1. ⚠️ Create `compute_metrics.py`
2. ⚠️ Create `generate_plots.py`
3. ⚠️ Create `generate_tables.py`
4. ⚠️ Create `check_completion.py`

### Priority 3: Execution

1. Test Phase 1 on H200 cluster
2. Fix any issues found
3. Run Phase 2 (all datasets)
4. Transfer results
5. Analyze and generate report

---

## Estimated Timeline

- **Code modifications**: 1-2 days
- **Phase 1 (quick test)**: 6-8 hours
- **Phase 2 (full validation)**: 48-60 hours (distributed across datasets)
- **Analysis**: 2-3 days
- **Total**: ~1 week

---

## Questions?

- Config issues: Check `experiments/configs/`
- Script issues: Check `experiments/slurm/`
- Experiment list: Check `EXPERIMENT_MANIFEST.md`
- Recording structure: See notes above about per-task tracking

**Status**: Infrastructure ready, code modifications needed before execution.
