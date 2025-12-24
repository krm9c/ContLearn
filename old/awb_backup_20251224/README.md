# AWB Implementation Backup - December 24, 2025

This backup was created before refactoring the AWB (Adaptive Weight Basis) pipeline implementation.

## Backed Up Files

### Core AWB Logic
- `awb_backup.py` - Original `src/cl/core/awb.py`
  - AWB utility functions (should_change_arch, compute_avg_loss, etc.)
  - Model partitioning functions
  - V computation helpers

- `arch_search_backup.py` - Original `src/cl/core/arch_search.py`
  - Generic architecture search framework
  - Bayesian optimization implementation
  - Grid search implementation

### Architecture Search Modules
- `arch_search_backup/` - Original `src/cl/arch_search/`
  - `mlp_search.py` - MLP architecture search
  - `cnn_search.py` - CNN/CNN3D architecture search
  - `gcn_search.py` - GCN architecture search
  - `__init__.py`

### Runners (ALL REDUNDANT - only generic_runner.py is used)
- `runners_backup/` - Original `src/cl/runners/`
  - `regression.py` - Sine wave regression runner (REDUNDANT)
  - `classification.py` - MNIST/CIFAR classification runner (REDUNDANT)
  - `graph_classification.py` - Graph classification runner (REDUNDANT)
  - `generic_runner.py` - **ONLY RUNNER IN USE** (will be refactored)
  - `__init__.py`

## Issues in Original Implementation

### Recording/Training Issues
1. **Preliminary training** - recorded when it shouldn't be
2. **Architecture search** - candidates recorded, polluting plots
3. **AB training** - no separate recording structure, no eigenvalue tracking
4. **Warmup** - partially recorded in some runners
5. **Iteration offsets** - missing or incorrect, causing overwrites

### Code Duplication
- AWB 5-step logic duplicated across:
  - `regression.py` (lines 312-442)
  - `classification.py` (lines 519-687)
  - `graph_classification.py` (similar pattern)
  - `generic_runner.py` (lines 920-1400)

### Bugs Fixed During Refactor
- `optimal_loss = inf` error (wrong task_id in compute_avg_loss)
- Graph dataset memory accumulation bug
- Empty experience loader NaN propagation
- Loss ratio calculation errors

## Refactoring Plan

### New Structure
1. **AWB Operations Interface** (`src/cl/core/awb_operations.py`)
   - Abstract base class defining model-specific operations
   - Implementations: MLPAWBOps, CNNAWBOps, GCNAWBOps

2. **AWB Pipeline Orchestrator** (`src/cl/core/awb_pipeline.py`)
   - Generic 5-step AWB control flow
   - Model-agnostic orchestration
   - Proper recording control (record_training, phase, offset)

3. **Simplified generic_runner.py**
   - Delegates AWB tasks to pipeline module
   - Standard CL code unchanged

4. **Remove Redundant Runners**
   - Delete regression.py, classification.py, graph_classification.py
   - Only generic_runner.py remains

## Restoration Instructions

If refactoring needs to be reverted:

```bash
# Restore original files
cp old/awb_backup_20251224/awb_backup.py src/cl/core/awb.py
cp old/awb_backup_20251224/arch_search_backup.py src/cl/core/arch_search.py
cp -r old/awb_backup_20251224/arch_search_backup/* src/cl/arch_search/
cp -r old/awb_backup_20251224/runners_backup/* src/cl/runners/

# Verify restoration
git diff src/cl/core/awb.py
git diff src/cl/core/arch_search.py
git diff src/cl/runners/
```

## Date
Created: December 24, 2025

## Context
This backup was created as part of a systematic refactoring to:
- Fix AWB recording issues across all datasets
- Eliminate code duplication
- Support future model types (Transformers)
- Maintain separation between standard CL and AWB logic
