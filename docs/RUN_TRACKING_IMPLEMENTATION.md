# Run Tracking Implementation Summary

## Overview

Added outer key structure to the recording system to track multiple experiment repetitions. Each run is now uniquely identified by a `run_id` and can be saved individually or consolidated into a single file.

## Changes Made

### 1. Recording System (`utils/trainer_recording.py`)

#### Modified `initialize_record_dict()`:
- Added `run_id` parameter (default: 0)
- Stores `run_id` in metadata

```python
def initialize_record_dict(self, config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
    record_dict = {
        'metadata': {
            # ... other metadata fields
            'run_id': run_id,
        },
        'iterations': {}
    }
    return record_dict
```

#### Modified `save_record_dict()`:
- Changed filename pattern to include run ID: `{prob}_{dataset}_{network}_run{run_id}_records.pkl`

#### Added `save_all_runs()` static method:
- Saves all runs in a consolidated structure with outer 'runs' key
- Filename pattern: `{prob}_{dataset}_{network}_allruns.pkl`
- Structure:
```python
{
    'runs': {
        '0': {metadata, iterations},
        '1': {metadata, iterations},
        ...
    },
    'metadata': {
        'total_runs': int,
        'problem': str,
        'prob': str,
        'dataset': str,
        'network': str,
    }
}
```

### 2. Training Runners (`training/runners.py`)

Updated all three training functions to accept `run_id` parameter:
- `train_model_graph(config, run_id=0)`
- `train_model_reg(config, run_id=0)`
- `train_model_class(config, run_id=0)`

Each function passes `run_id` to `initialize_record_dict()`.

### 3. Main Script (`run.py`)

- Passes `run_id=j` to training functions in the main loop
- Calls `Trainer.save_all_runs()` to save consolidated file
- Maintains backward compatibility with legacy 'file' parameter

```python
for j in range(params['runs']):
    if params['prob'] == 'regression':
        all_runs_records[str(j)] = train_model_reg(params, run_id=j)
    # ... similar for other problem types

# Save all runs together
Trainer.save_all_runs(all_runs_records, params.get('model_path', ''), params)
```

### 4. Documentation (`docs/RECORDING_FORMAT.md`)

Updated with:
- New top-level structure showing both single run and all runs formats
- File naming conventions for both individual and consolidated files
- Example loading code for both formats
- Example analysis scripts for analyzing multiple runs

## File Structure

### Single Run Files
```
{prob}_{dataset}_{network}_run{run_id}_records.pkl
```

Contains:
```python
{
    'metadata': {
        'run_id': 0,
        'dataset': 'sine',
        # ... other metadata
    },
    'iterations': {
        0: {losses, gradients, metrics, eigenvalues, ...},
        10: {...},
        ...
    }
}
```

### Consolidated All Runs File
```
{prob}_{dataset}_{network}_allruns.pkl
```

Contains:
```python
{
    'runs': {
        '0': {
            'metadata': {'run_id': 0, ...},
            'iterations': {...}
        },
        '1': {
            'metadata': {'run_id': 1, ...},
            'iterations': {...}
        },
        ...
    },
    'metadata': {
        'total_runs': 2,
        'problem': 'vectors',
        'prob': 'regression',
        'dataset': 'sine',
        'network': 'fcnn'
    }
}
```

## Usage Examples

### Training with Multiple Runs
```bash
python run.py train 5 param_sine.json
```

Produces:
- `regression_sine_fcnn_run0_records.pkl`
- `regression_sine_fcnn_run1_records.pkl`
- `regression_sine_fcnn_run2_records.pkl`
- `regression_sine_fcnn_run3_records.pkl`
- `regression_sine_fcnn_run4_records.pkl`
- `regression_sine_fcnn_allruns.pkl` (consolidated)

### Loading Individual Run
```python
import pickle

with open('regression_sine_fcnn_run0_records.pkl', 'rb') as f:
    run_data = pickle.load(f)

print(f"Run ID: {run_data['metadata']['run_id']}")
print(f"Iterations: {len(run_data['iterations'])}")
```

### Loading All Runs
```python
import pickle

with open('regression_sine_fcnn_allruns.pkl', 'rb') as f:
    all_runs = pickle.load(f)

print(f"Total runs: {all_runs['metadata']['total_runs']}")

for run_id, run_data in all_runs['runs'].items():
    print(f"Run {run_id}: {len(run_data['iterations'])} iterations")
```

### Analyzing Across Runs
```python
import pickle
import numpy as np

# Load all runs
with open('regression_sine_fcnn_allruns.pkl', 'rb') as f:
    all_runs = pickle.load(f)

# Compute statistics across runs
final_losses = []
for run_id, run_data in all_runs['runs'].items():
    final_iter = max(run_data['iterations'].keys())
    final_H = run_data['iterations'][final_iter]['losses']['H']
    final_losses.append(final_H)

print(f"Final H across runs:")
print(f"  Mean: {np.mean(final_losses):.6e}")
print(f"  Std: {np.std(final_losses):.6e}")
print(f"  Min: {np.min(final_losses):.6e}")
print(f"  Max: {np.max(final_losses):.6e}")
```

## Testing

Verified with:
```bash
bash test_datasets.sh
```

Output confirmed:
- ✅ Individual run files created with correct naming
- ✅ Consolidated all runs file created
- ✅ `run_id` correctly stored in metadata
- ✅ All training pipelines (standard and AWB) work correctly
- ✅ Backward compatibility maintained

## Benefits

1. **Experiment Tracking**: Each repetition is uniquely identified
2. **Statistical Analysis**: Easy to compute statistics across multiple runs
3. **Reproducibility**: Complete metadata including run number
4. **Flexibility**: Can load individual runs or all runs together
5. **Backward Compatible**: Doesn't break existing code

## Files Modified

1. `/Users/kraghavan/Desktop/JMLR_paper/ContLearn/utils/trainer_recording.py`
2. `/Users/kraghavan/Desktop/JMLR_paper/ContLearn/training/runners.py`
3. `/Users/kraghavan/Desktop/JMLR_paper/ContLearn/run.py`
4. `/Users/kraghavan/Desktop/JMLR_paper/ContLearn/docs/RECORDING_FORMAT.md`

## Files Created

1. `/Users/kraghavan/Desktop/JMLR_paper/ContLearn/docs/RUN_TRACKING_IMPLEMENTATION.md` (this file)
