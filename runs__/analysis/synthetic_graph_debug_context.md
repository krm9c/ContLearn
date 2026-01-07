# Synthetic Graph Debug Context

## Problem Summary
Running 2-task synthetic graph classification with 4 conditions (C1-C4) shows suspicious results:
- C3 and C4 produce **identical results** (AWB transfer not working?)
- Large accuracy drop from Task 0 to Task 1 (98% → 26%)
- C1/C2 can't learn Task 1 at all (0-5% accuracy)
- Higher loss in C3/C4 but "higher average accuracy" - counterintuitive

## Experimental Results

### Task Performance Matrix (from pickle files)
```
C1 (Baseline):     Task 0: 98.1% → 97.1%,  Task 1: 4.8%
C2 (Heuristics):   Task 0: 98.1% → 97.9%,  Task 1: 0.0%
C3 (Arch Search):  Task 0: 98.1% → 85.2%,  Task 1: 26.2%
C4 (AWB Full):     Task 0: 98.1% → 85.2%,  Task 1: 26.2%  <-- IDENTICAL to C3!
```

### CL Metrics
| Condition | Avg Acc | BWT | FWT | Forgetting |
|-----------|---------|-----|-----|------------|
| C1: Baseline | 0.5097 | -0.0095 | 0.0000 | 0.0095 |
| C2: Heuristics | 0.4893 | -0.0024 | 0.0000 | 0.0024 |
| C3: Arch Search | 0.5568 | -0.1294 | 0.0000 | 0.1294 |
| C4: AWB Full | 0.5568 | -0.1294 | 0.0000 | 0.1294 |

### Loss Values (H) at Task 1 End
- C1: 0.39 (low)
- C2: 0.19 (very low)
- C3: 0.61 (high)
- C4: 0.61 (high) <-- IDENTICAL to C3!

## Config Files

### Location
`/Users/kraghavan/Desktop/JMLR_paper/ContLearn/runs__/configs/`

### C1 Baseline (`synthetic_graph_2task_condition1_baseline.json`)
```json
{
    "data": "synthetic",
    "problem": "graph",
    "network": "gcn",
    "n_task": 2,
    "epochs_per_task": 125,
    "batch_size": 20,
    "num_graphs": 10000,
    "num_channels": 5,
    "avg_num_nodes": 2,
    "num_classes": 10,
    "class_per_task": 5,
    "lr_schedule": "constant",
    "lr": 0.0001,
    "task_warmup_enabled": false,
    "awb_enabled": false
}
```

### C3 Arch Search (`synthetic_graph_2task_condition3_arch_no_transfer.json`)
```json
{
    "data": "synthetic",
    "problem": "graph",
    "network": "gcn",
    "n_task": 2,
    "epochs_per_task": 125,
    "batch_size": 20,
    "num_graphs": 10000,
    "num_channels": 5,
    "avg_num_nodes": 2,
    "num_classes": 10,
    "class_per_task": 5,
    "lr_schedule": "cosine",
    "lr": 0.0001,
    "task_warmup_enabled": true,
    "warmup_epochs": 15,
    "awb_enabled": true,
    "awb_transfer_enabled": false,  <-- KEY: Transfer disabled
    "arch_search_enabled": true,
    "force_arch_search": true,
    "awb_preliminary_epochs": 2,
    "awb_ab_training_epochs": 50,
    "awb_loss_ratio_threshold": 1.1
}
```

### C4 AWB Full (`synthetic_graph_2task_condition4_awb_full.json`)
```json
{
    "data": "synthetic",
    "problem": "graph",
    "network": "gcn",
    "n_task": 2,
    "epochs_per_task": 125,
    "batch_size": 20,
    "num_graphs": 10000,
    "num_channels": 5,
    "avg_num_nodes": 2,
    "num_classes": 10,
    "class_per_task": 5,
    "lr_schedule": "cosine",
    "lr": 0.0001,
    "task_warmup_enabled": true,
    "warmup_epochs": 15,
    "awb_enabled": true,
    "awb_transfer_enabled": true,   <-- KEY: Transfer enabled
    "arch_search_enabled": true,
    "force_arch_search": true,
    "awb_preliminary_epochs": 2,
    "awb_ab_training_epochs": 50,
    "awb_loss_ratio_threshold": 1.1
}
```

## Pickle File Locations
```
runs__/analysis/data_analysis/synthetic_graphs/
├── synthetic_graph_2task_condition1_run0/classification_synthetic_gcn_run0_records.pkl
├── synthetic_graph_2task_condition2_run0/classification_synthetic_gcn_run0_records.pkl
├── synthetic_graph_2task_condition3_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl
├── synthetic_graph_2task_condition4_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl
```

## Key Code Files to Investigate

### AWB Pipeline
- `src/cl/core/awb.py` - AWB transfer logic
- `src/cl/core/awb_pipeline.py` - AWB 5-step pipeline

### GCN Model
- `src/cl/models/gcn.py` - GCN model with AWB support

### Dataset
- `src/cl/datasets/synthetic_graph.py` - Synthetic graph dataset

### Runner
- `src/cl/runners/generic_runner.py` - Main training loop

### Constants (GCN defaults)
- `src/cl/config/constants.py`
  - `DEFAULT_BATCH_SIZE_GRAPH = 20`
  - `DEFAULT_LR_GRAPH = 1e-4`
  - `DEFAULT_GCN_SIZES = [None, 128]`
  - `DEFAULT_GCN_FEED_SIZES = [128, 128, 128, 10]`
  - `DEFAULT_AWB_GCN_ARCH = [100]`
  - `DEFAULT_AWB_FNN_ARCH = [100, 140, 140]`

## Suspected Issues

1. **C3 and C4 are identical** - `awb_transfer_enabled` flag may not be working for GCNs
   - Check if `awb_transfer_enabled` is being read/used in `awb_pipeline.py`
   - Check if GCN AWB ops (`GCNAWBOps`) properly handles transfer vs no-transfer

2. **Task 1 accuracy very low** - Possible issues:
   - Class selection may be overlapping or problematic
   - Experience replay may be overwhelming Task 1 learning
   - Architecture search may not be finding good architectures

3. **Loss higher but accuracy higher** - May indicate:
   - Loss is computed on different data than accuracy
   - Cross-entropy loss scaling issues with class imbalance

## Commands to Run Experiments

```bash
# Run single condition
python run.py runs__/configs/synthetic_graph_2task_condition4_awb_full.json

# Run all 4 conditions
./run_synthetic_graph.sh

# Generate comparison plots
python runs__/analysis/scripts/compare_synthetic_graph_from_pkl.py
```

## Plot Output Location
```
runs__/analysis/data_analysis/synthetic_graphs/synthetic_graph_comparison_plots/
├── accuracy_curves.pdf/png
├── cl_metrics_comparison.pdf/png
├── loss_components.pdf/png
├── performance_matrices.pdf/png
├── synthetic_graph_2task_results.pdf/png
```

## Quick Debug Script
```python
import pickle
from pathlib import Path

DATA_DIR = Path("runs__/analysis/data_analysis/synthetic_graphs")
pkl_path = DATA_DIR / "synthetic_graph_2task_condition4_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("Metadata:", data['metadata'])
print("Task Performance Matrix:", data['task_performance_matrix'])

# Check architecture history (if AWB is working, architectures should differ)
if 'architecture_history' in data:
    print("Architecture History:", data['architecture_history'])
```
