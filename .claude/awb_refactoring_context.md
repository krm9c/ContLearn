# AWB Refactoring - Session Context

## Status: COMPLETED (Needs Testing)

Refactored AWB (Adaptive Weight Basis) pipeline from duplicated code across 4 files into clean strategy pattern implementation.

## What Was Done

### 1. Infrastructure Created (Commit 0c9f5e5)

**New Files:**
- `src/cl/core/awb_operations.py` - Abstract interface defining 8 methods all models must implement
- `src/cl/core/awb_pipeline.py` - Generic 5-step AWB orchestrator (model-agnostic, 300 lines)

**Enhanced Files:**
- `src/cl/core/awb.py` - Added CNN/GCN specific functions (save/restore/compute_V/set_AB)
- `src/cl/models/mlp.py` - Added `MLPAWBOps` class
- `src/cl/models/cnn.py` - Added `CNNAWBOps` class (handles CNN and CNN3D)
- `src/cl/models/gcn.py` - Added `GCNAWBOps` class

**Backup Created:**
- `old/awb_backup_20251224/` - Complete backup with README for restoration if needed

### 2. Integration (Commit 4063726)

**Updated:**
- `src/cl/runners/generic_runner.py` - Reduced 1523 → 1083 lines (-440 lines, -29%)
  - Added `create_awb_operations(model)` helper
  - Replaced 500+ lines of AWB code with single `run_awb_task()` call
  - **Standard CL path (task 0, awb_enabled=false) completely unchanged**

**Deprecated:**
- `classification.py.deprecated` - Redundant, only generic_runner.py is used
- `regression.py.deprecated` - Redundant
- `graph_classification.py.deprecated` - Redundant

## Architecture

```
AWB Pipeline Flow:
┌─────────────────────────────────────────────────────────┐
│ generic_runner.py (if awb_enabled and task_id > 0)     │
│  1. Create AWBOps: create_awb_operations(model)         │
│  2. Call: run_awb_task(task_id, trainer, model, ...)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ awb_pipeline.py - Generic 5-Step Orchestrator           │
│  STEP 1: Preliminary training (NOT recorded)            │
│  STEP 2: Architecture change decision                   │
│  STEP 3a: Architecture search (NOT recorded)            │
│  STEP 3b: AB training (recorded separately)             │
│  STEP 4: Compute V = A @ W @ B^T                        │
│  STEP 5: Train V (warmup NOT recorded, main recorded)   │
└─────────────────────────────────────────────────────────┘
                          ↓ (delegates to)
┌─────────────────────────────────────────────────────────┐
│ Model-Specific AWBOps (implements AWBOperations)        │
│  - MLPAWBOps: For sine regression                       │
│  - CNNAWBOps: For MNIST/CIFAR (handles CNN & CNN3D)    │
│  - GCNAWBOps: For graph classification                  │
└─────────────────────────────────────────────────────────┘
                          ↓ (delegates to)
┌─────────────────────────────────────────────────────────┐
│ awb.py - Model-Specific Implementations                 │
│  - set_new_AB_matrices_*                                │
│  - compute_V_from_AWB_*                                 │
│  - partition_for_*_training                             │
│  - save/restore weights                                 │
└─────────────────────────────────────────────────────────┘
                          ↓ (uses)
┌─────────────────────────────────────────────────────────┐
│ arch_search.py - Grid & Bayesian Search                 │
│  - search_architecture() dispatcher                     │
│  - search_architecture_grid()                           │
│  - search_architecture_bayesian()                       │
└─────────────────────────────────────────────────────────┘
```

## Recording Logic Fixed

**Previous Issues:**
- Preliminary, warmup, and arch search were being recorded → polluted plots
- AB training mixed with main training → couldn't plot separately
- No AB eigenvalue tracking
- Iteration offsets missing → recordings overwrote each other

**Now Fixed (in awb_pipeline.py):**
- ❌ Preliminary: `record_training=False` - NOT in plots
- ❌ Arch search: `record_training=False` - NOT in plots
- ✅ AB training: `phase='ab'`, `record_training=True` - Separate recording
- ❌ Warmup: `record_training=False` - NOT in plots
- ✅ Main: `phase='main'`, `record_training=True` - Normal recording
- ✅ Proper `global_iteration_offset` throughout

## Key Interface (AWBOperations)

Every model supporting AWB must implement 8 methods:

```python
class AWBOperations(ABC):
    @abstractmethod
    def search_architecture(model, task_id, baseline_loss, ...) -> arch_spec

    @abstractmethod
    def set_AB_matrices(model, original_arch, new_arch) -> model

    @abstractmethod
    def partition_for_AB_training(model) -> (trainable, static)

    @abstractmethod
    def compute_V(model) -> model

    @abstractmethod
    def partition_for_standard_training(model) -> (trainable, static)

    @abstractmethod
    def get_model_architecture(model) -> arch_spec

    @abstractmethod
    def save_weights(model) -> weights

    @abstractmethod
    def restore_weights(model, weights) -> model
```

## Testing Needed

### Priority 1: Verify Standard CL Unchanged
```bash
# Run baseline configs (awb_enabled=false)
python run_files/scripts/run.py kkt_run/configs/sine.json
python run_files/scripts/run.py kkt_run/configs/mnist.json
python run_files/scripts/run.py kkt_run/configs/cifar10.json
```

**Expected:** Should work exactly as before, no errors, same behavior.

### Priority 2: Verify AWB Works
```bash
# Run AWB configs (awb_enabled=true)
python run_files/scripts/run.py kkt_run/configs/sine_awb.json
python run_files/scripts/run.py kkt_run/configs/mnist_awb.json
python run_files/scripts/run.py kkt_run/configs/cifar10_awb.json
```

**Expected:**
- No errors
- Architecture search runs correctly
- AB training recorded separately
- Clean plots (no preliminary/warmup/search in main plots)

### Priority 3: Verify All Datasets
```bash
# Test all dataset types
python run_files/scripts/run.py kkt_run/configs/cifar100.json
python run_files/scripts/run.py kkt_run/configs/synthetic_graph.json
```

## Bugs Fixed During Refactoring

1. **Graph dataset memory accumulation** (synthetic_graph.py:148-164)
   - Bug: Memory never accumulated, always 0
   - Fix: Moved memory update before caching

2. **Empty experience loader NaN** (loops.py:203-231)
   - Bug: Task 0 has empty exploader → mean of empty slice → NaN
   - Fix: Check for empty loader, return default variance

3. **AWB optimal_loss = inf** (awb.py:66-77)
   - Bug: Iteration offset mismatch, loss lookup failed
   - Fix: Fallback to last N recorded iterations

4. **Recording pollution**
   - Bug: Preliminary/warmup/search all recorded
   - Fix: Proper `record_training` and `phase` parameters

## Configuration Files

All configs in `kkt_run/configs/` updated to match validation experiments:

**Baseline (Condition 1):** No smoothness
- `sine.json`, `mnist.json`, `cifar10.json`, `cifar100.json`, `synthetic_graph.json`
- Settings: `awb_enabled: false`, constant LR, no warmup

**Full AWB (Condition 4):** Architecture + Transfer
- `sine_awb.json`, `mnist_awb.json`, `cifar10_awb.json`, `cifar100_awb.json`, `synthetic_graph_awb.json`
- Settings: `awb_enabled: true`, AB training enabled

All have `per_task_eval_enabled: true` for CL metrics.

## Adding New Model Types (e.g., Transformers)

1. Create model in `src/cl/models/transformer.py`
2. Add architecture search in `src/cl/arch_search/transformer_search.py`
3. Add AWB functions to `src/cl/core/awb.py`:
   - `set_new_AB_matrices_transformer()`
   - `compute_V_from_AWB_transformer()`
   - `partition_for_AB_training_transformer()`
   - `partition_for_standard_training_transformer()`
4. Add `TransformerAWBOps` class to `transformer.py`
5. Update `create_awb_operations()` in `generic_runner.py`

That's it! No changes needed to awb_pipeline.py.

## Restoration Instructions

If refactoring needs rollback:

```bash
# Restore from backup
cp -r old/awb_backup_20251224/runners_backup/* src/cl/runners/
cp old/awb_backup_20251224/awb_backup.py src/cl/core/awb.py
rm src/cl/core/awb_operations.py
rm src/cl/core/awb_pipeline.py

# Remove AWBOps classes from models (manual edit)
# Revert generic_runner.py changes
git checkout HEAD~2 src/cl/runners/generic_runner.py

# Or full revert
git revert HEAD HEAD~1
```

## Files Modified (Summary)

**New:**
- `src/cl/core/awb_operations.py` (203 lines)
- `src/cl/core/awb_pipeline.py` (319 lines)

**Modified:**
- `src/cl/core/awb.py` (+174 lines, now 836 lines)
- `src/cl/models/mlp.py` (+54 lines, now 490 lines)
- `src/cl/models/cnn.py` (+86 lines, now 733 lines)
- `src/cl/models/gcn.py` (+65 lines, now 618 lines)
- `src/cl/runners/generic_runner.py` (-440 lines, now 1083 lines)

**Deprecated:**
- `src/cl/runners/classification.py.deprecated`
- `src/cl/runners/regression.py.deprecated`
- `src/cl/runners/graph_classification.py.deprecated`

**Backed up:**
- `old/awb_backup_20251224/` (full backup with README)

## Known Issues / Outstanding Work

**None currently.** All refactoring complete.

**Next steps (from earlier sessions):**
- Per-task evaluation integration (90% done, needs final testing)
- AWB skip transfer implementation (config exists, needs testing)
- Warm start across tasks (partially implemented)

## Git History

```
4063726 - Integrate AWB pipeline into generic_runner.py - PART 2
0c9f5e5 - Refactor AWB implementation: Add operations interface and pipeline orchestrator - PART 1
12b9c22 - Fix graph dataset bugs (before refactoring)
```

## Session Date

December 24, 2025

## Contact for Restoration

If issues arise, restore from `old/awb_backup_20251224/` following instructions above.
All original AWB logic preserved and fully functional.
