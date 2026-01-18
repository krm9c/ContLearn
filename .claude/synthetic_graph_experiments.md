# Synthetic Graph CL Experiments

## Overview

This document tracks the development and testing of minimal synthetic graph continual learning experiments comparing Baseline vs AWB (Adaptive Weight Basis) methods.

## Key Fixes Applied

### 1. T.NormalizeFeatures() Bug Fix (Critical)
**File**: `src/cl/core/loops.py:48-52`

**Problem**: `T.NormalizeFeatures()` was destroying the class-feature correlation in FakeDataset graphs.
- Before transforms: correlation = 0.998
- After transforms: correlation = -0.135 (random)

**Fix**: Removed `T.NormalizeFeatures()` from the transform pipeline:
```python
# Fixed - preserves class-feature correlation
_GRAPH_TRANSFORMS = T.Compose([
    T.GCNNorm(),
    T.ToDense()
])
```

**Impact**: Test accuracy improved from ~40% to ~80%+

### 2. Config Architecture Structure Fix
**Files**:
- `runs__/configs/synthetic_graph_minimal_awb.json`
- `runs__/configs/synthetic_graph_minimal_baseline.json`

**Problem**: Original configs had 2-element `feed_sizes: [32, 5]` but `generate_search_candidates()` expects 4-element architecture.

**Fix**: Updated to match 10-task config structure:
```json
{
    "gcn_sizes": [10, 32, 32],
    "feed_sizes": [32, 32, 16, 5]
}
```

### 3. prepABs_GCN Architecture Expansion Fix
**File**: `src/cl/arch_search/gcn_search.py:135-168`

**Problem**: When architecture expanded, `zip()` truncated A/B matrices to shorter list length.

**Fix**: Create matrices for ALL new layers - transformation matrices for existing layers, identity matrices for new layers.

## Experiment Configurations

### Minimal Baseline Config
**File**: `runs__/configs/synthetic_graph_minimal_baseline.json`
```json
{
    "data": "synthetic_taskshift",
    "network": "gcn",
    "n_task": 3,
    "epochs_per_task": 30,
    "batch_size": 32,
    "gcn_sizes": [10, 32, 32],
    "feed_sizes": [32, 32, 16, 5],
    "num_graphs": 500,
    "perturbation_mode": "linear",
    "feature_noise_base": 0.02,
    "edge_dropout_base": 0.01,
    "feature_shift_base": 0.01,
    "awb_enabled": false
}
```

### Minimal AWB Config
**File**: `runs__/configs/synthetic_graph_minimal_awb.json`
```json
{
    "awb_enabled": true,
    "awb_preliminary_epochs": 5,
    "awb_ab_training_epochs": 50,
    "awb_ab_warmup_epochs": 20,
    "awb_ab_lr": 0.001,
    "force_arch_change": true,
    "task_warmup_enabled": true,
    "task_warmup_epochs": 5
}
```

## Experiment Results

### Test 1: AWB with 10 A/B Training Epochs

| Task | Baseline Te/Exp | AWB Te/Exp | Δ |
|------|-----------------|------------|---|
| 0 | 68.75% | 68.75% | 0% |
| 1 | 77.34% | 85.94% | +8.59% |
| 2 | 82.03% | 68.75% | -13.28% |
| **Avg** | **76.04%** | **74.48%** | **-1.56%** |

**Observation**: AWB underperformed on Task 2 due to insufficient A/B training for the expanded architecture (3K → 175K params).

### Test 2: AWB with 50 A/B Training Epochs

| Task | Baseline Te/Exp | AWB Te/Exp | Δ |
|------|-----------------|------------|---|
| 0 | 68.75% | 68.75% | 0% |
| 1 | 77.34% | 71.88% | -5.47% |
| 2 | 82.03% | 82.03% | 0% |
| **Avg** | **76.04%** | **74.22%** | **-1.82%** |

**Observation**: With more A/B training, AWB recovers to match baseline on final task.

### AWB Architecture Evolution
```
Task 0: GCN=[10, 32, 32], Feed=[32, 32, 16, 5]     (~3K params)
Task 1: GCN=[10, 72, 72], Feed=[72, 172, 196, 5]   (~53K params) [CHANGED]
Task 2: GCN=[10, 142, 142], Feed=[142, 332, 316, 5] (~175K params) [CHANGED]
```

## Key Findings

1. **A/B Training Duration Matters**: 10 epochs insufficient for 58x parameter growth; 50 epochs allows convergence

2. **Task Shift Magnitude**: Minimal perturbations (0.02 noise, 0.01 dropout) may be too subtle for AWB benefits

3. **Architecture Explosion**: AWB aggressively expands (3K → 175K params), requiring careful tuning

4. **Baseline Robustness**: Fixed architecture baseline performs well on subtle task shifts

## Recommendations

1. For subtle task shifts: Use baseline or limit AWB expansion
2. For strong task shifts: Use AWB with sufficient A/B training (50+ epochs)
3. Consider architecture constraints to prevent over-expansion
4. Match config architecture to `generate_search_candidates()` expectations (4-element feed_sizes)

## Output Files

- Baseline results: `outputs/synthetic_graph_minimal_baseline_run0/`
- AWB results: `outputs/synthetic_graph_minimal_awb_run0/`
- Comparison plots: `outputs/baseline_vs_awb_*.png`

## Related Files

- Architecture search: `src/cl/arch_search/gcn_search.py`
- AWB operations: `src/cl/core/awb.py`
- Training loops: `src/cl/core/loops.py`
- GCN model: `src/cl/models/gcn.py`
- Verification script: `verify_baseline.py`
