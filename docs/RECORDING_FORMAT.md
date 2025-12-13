# Unified Recording Format Documentation

## Overview

The ContLearn framework uses a unified, extensible recording system that replaces the previous TensorBoard + tuple-based approach. All metrics are recorded in a consistent nested dictionary structure indexed by iteration.

## Design Principles

1. **Consistent Format**: Same structure across all problem types (graphs, vectors)
2. **Indexed by Iteration**: Clean integer-based indexing for time-series analysis
3. **Self-Documenting**: Named fields instead of positional tuples
4. **Extensible**: Easy to add new metrics for different datasets
5. **Portable**: Standard Python dictionaries, easily saved as JSON/pickle
6. **Complete**: Captures all progress bar metrics + eigenvalues

## Record Dictionary Structure

### Top-Level Structure (Single Run)

Each training run produces a record dictionary with this structure:

```python
{
    'metadata': {
        # Configuration and dataset information
        'run_id': int,  # Experiment repetition number
        ...
    },
    'iterations': {
        iteration_number: {
            # Metrics for this iteration
        },
        ...
    }
}
```

### All Runs Structure

When multiple experiment runs are saved together:

```python
{
    'runs': {
        '0': {  # First run
            'metadata': {..., 'run_id': 0},
            'iterations': {...}
        },
        '1': {  # Second run
            'metadata': {..., 'run_id': 1},
            'iterations': {...}
        },
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

### Metadata Section

Records configuration and dataset information for reproducibility:

```python
'metadata': {
    'problem': str,              # 'vectors' or 'graph'
    'prob': str,                 # 'regression', 'classification', 'graphclassification'
    'dataset': str,              # 'sine', 'mnist', 'ENZYMES', 'MUTAG', etc.
    'network': str,              # 'fcnn', 'cnn', 'gnn'
    'loss_function': str,        # 'mse' or 'class'
    'metric_function': str,      # 'mse' or 'class'
    'n_tasks': int,              # Number of continual learning tasks
    'epochs_per_task': int,      # Training epochs per task
    'save_iter': int,            # Frequency of metric recording
    'learning_rate': float,      # Learning rate
    'batch_size': int,           # Batch size
    'awb_enabled': bool,         # Whether AWB is enabled
    'run_id': int,               # Experiment repetition/run number
}
```

### Iteration Records

Each iteration (saved every `save_iter` steps) contains:

```python
iteration_number: {
    'losses': {
        'H': float,                 # Hamiltonian (total loss)
        'V': float,                 # Value function / primary loss (MSE or Cross Entropy)
        'dV': float,                # dV term
        'dV_dx': float,             # Gradient of V w.r.t. input
        'dV_dtheta': float,         # Gradient of V w.r.t. parameters
        'dV_dadj': float,           # (Graphs only) Gradient of V w.r.t. adjacency
    },
    'gradients': {
        'grad_norm': float,         # Gradient norm ||dH/dθ||
    },
    'metrics': {
        'train': float,             # Training metric (accuracy or MSE)
        'test_current': float,      # Test metric on current task data
        'test_experience': float,   # Test metric on experience replay data
    },
    'eigenvalues': {
        'A': {
            'layer_0': ndarray,     # Eigenvalues of A_0 @ A_0^T
            'layer_1': ndarray,     # Eigenvalues of A_1 @ A_1^T
            'conv_0': ndarray,      # (CNN) Conv layer eigenvalues
            'feed_0': ndarray,      # (CNN) Feed layer eigenvalues
            'gcn_0': ndarray,       # (GNN) GCN layer eigenvalues
            ...
        },
        'B': {
            'layer_0': ndarray,     # Eigenvalues of B_0 @ B_0^T
            'layer_1': ndarray,     # Eigenvalues of B_1 @ B_1^T
            'conv_0': ndarray,      # (CNN) Conv layer eigenvalues
            'feed_0': ndarray,      # (CNN) Feed layer eigenvalues
            'gcn_0': ndarray,       # (GNN) GCN layer eigenvalues
            ...
        }
    },
    'step': int,                    # Step within current task
    'task_id': int,                 # Current task ID
    'global_step': int,             # Global step = step + task_id * n_iter
    'extra_metrics': {              # Optional dataset-specific metrics
        ...
    }
}
```

## Model-Specific Eigenvalue Structure

Different models have different A/B matrix organizations:

### MLP (Feedforward Neural Network)
```python
'eigenvalues': {
    'A': {
        'layer_0': ndarray,  # Input → Hidden1
        'layer_1': ndarray,  # Hidden1 → Hidden2
        ...
    },
    'B': {
        'layer_0': ndarray,
        'layer_1': ndarray,
        ...
    }
}
```

### CNN (Convolutional Neural Network)
```python
'eigenvalues': {
    'A': {
        'conv_0': ndarray,   # Convolutional layer 0
        'conv_1': ndarray,   # Convolutional layer 1
        'feed_0': ndarray,   # Feedforward layer 0
        'feed_1': ndarray,   # Feedforward layer 1
        ...
    },
    'B': { ... }  # Same structure
}
```

### GNN/myNN (Graph Neural Network)
```python
'eigenvalues': {
    'A': {
        'gcn_0': ndarray,    # GCN layer 0
        'gcn_1': ndarray,    # GCN layer 1
        'feed_0': ndarray,   # Feedforward layer 0
        'feed_1': ndarray,   # Feedforward layer 1
        ...
    },
    'B': { ... }  # Same structure
}
```

## File Naming Convention

Records are automatically saved with the naming pattern:

### Individual Run Files
```
{prob}_{dataset}_{network}_run{run_id}_records.pkl
```

Examples:
- `regression_sine_fcnn_run0_records.pkl`
- `classification_mnist_cnn_run0_records.pkl`
- `graphclassification_ENZYMES_gnn_run1_records.pkl`

### All Runs File
```
{prob}_{dataset}_{network}_allruns.pkl
```

Examples:
- `regression_sine_fcnn_allruns.pkl`
- `classification_mnist_cnn_allruns.pkl`
- `graphclassification_ENZYMES_gnn_allruns.pkl`

## Usage

### Initialization

At the start of training:

```python
record_dict = trainer.initialize_record_dict(config, run_id=0)
```

### Recording Metrics

During training (called automatically in training loops):

```python
record_dict['iterations'][iteration] = trainer.record_metrics(
    iteration=iteration,
    step=step,
    task_id=task_id,
    losses={
        'H': float(H),
        'V': float(V),
        'dV': float(dV),
        'dV_dx': float(dVstar_dx),
        'dV_dtheta': float(dVstar_dtheta),
    },
    gradients={
        'grad_norm': float(grad_norm),
    },
    metrics={
        'train': float(train_metric),
        'test_current': float(test_current_metric),
        'test_experience': float(test_exp_metric),
    },
    model=model,
    extra_metrics={'custom_metric': value}  # Optional
)
```

### Saving

At the end of training:

```python
trainer.save_record_dict(record_dict, config['model_path'])
```

### Loading

To analyze recorded data:

#### Loading All Runs
```python
import pickle

# Load consolidated runs file
with open('regression_sine_fcnn_allruns.pkl', 'rb') as f:
    all_runs = pickle.load(f)

# Access top-level metadata
print(f"Total runs: {all_runs['metadata']['total_runs']}")
print(f"Dataset: {all_runs['metadata']['dataset']}")

# Access specific run
run_0 = all_runs['runs']['0']
metadata = run_0['metadata']
print(f"Run {metadata['run_id']} - Loss function: {metadata['loss_function']}")

# Access iteration data from specific run
for iteration, data in run_0['iterations'].items():
    print(f"Iteration {iteration}:")
    print(f"  H = {data['losses']['H']}")
    print(f"  Train metric = {data['metrics']['train']}")
    print(f"  Test (current) = {data['metrics']['test_current']}")
    print(f"  Test (experience) = {data['metrics']['test_experience']}")

    # Access eigenvalues
    if 'layer_0' in data['eigenvalues']['A']:
        eigs_A0 = data['eigenvalues']['A']['layer_0']
        print(f"  Layer 0 A eigenvalues: {eigs_A0}")
```

#### Loading Single Run
```python
import pickle

# Load single run file
with open('regression_sine_fcnn_run0_records.pkl', 'rb') as f:
    record_dict = pickle.load(f)

# Access metadata
metadata = record_dict['metadata']
print(f"Run ID: {metadata['run_id']}")
print(f"Dataset: {metadata['dataset']}")
print(f"Loss function: {metadata['loss_function']}")

# Access iteration data
for iteration, data in record_dict['iterations'].items():
    print(f"Iteration {iteration}:")
    print(f"  H = {data['losses']['H']}")
    # ... rest same as above
```

## Adding Custom Metrics

To add dataset-specific metrics, pass them via `extra_metrics`:

```python
# In your training loop
custom_data = {
    'custom_loss': some_value,
    'special_metric': another_value,
}

record_dict['iterations'][iteration] = trainer.record_metrics(
    ...,
    extra_metrics=custom_data
)
```

## Migration from Old Format

### Old Format (TensorBoard + Tuples)

```python
# Old approach
self.writer.add_scalar('train/Loss/H', H.item(), step)
self.writer.add_scalar('train/Loss/MSE', V.item(), step)
...
dictum["train"+str(step)] = (V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm)
```

### New Format (Unified Dictionary)

```python
# New approach
record_dict['iterations'][iteration] = trainer.record_metrics(
    iteration=iteration,
    step=step,
    task_id=task_id,
    losses={'H': float(H), 'V': float(V), ...},
    gradients={'grad_norm': float(grad_norm)},
    metrics={...},
    model=model
)
```

## Benefits Over Previous System

| Feature | Old System | New System |
|---------|-----------|------------|
| Format | TensorBoard + Tuples | Nested Dict |
| Consistency | Different per problem type | Unified |
| Indexing | String keys `"train123"` | Integer iterations |
| Eigenvalues | Not recorded | Automatically computed |
| Metadata | Scattered | Centralized |
| Extensibility | Hard to extend | Easy via `extra_metrics` |
| Portability | TensorBoard dependency | Standard Python |
| Documentation | Tuple positions unclear | Self-documenting |

## Example Analysis Script

### Analyzing All Runs

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load all runs
with open('regression_sine_fcnn_allruns.pkl', 'rb') as f:
    all_runs = pickle.load(f)

# Analyze each run
fig, axes = plt.subplots(len(all_runs['runs']), 3, figsize=(15, 5*len(all_runs['runs'])))

for run_idx, (run_id, run_data) in enumerate(all_runs['runs'].items()):
    # Extract time series for this run
    iterations = sorted(run_data['iterations'].keys())
    H_values = [run_data['iterations'][i]['losses']['H'] for i in iterations]
    train_metrics = [run_data['iterations'][i]['metrics']['train'] for i in iterations]
    test_curr = [run_data['iterations'][i]['metrics']['test_current'] for i in iterations]
    test_exp = [run_data['iterations'][i]['metrics']['test_experience'] for i in iterations]

    # Plot Hamiltonian
    axes[run_idx, 0].plot(iterations, H_values)
    axes[run_idx, 0].set_xlabel('Iteration')
    axes[run_idx, 0].set_ylabel('Hamiltonian (H)')
    axes[run_idx, 0].set_title(f'Run {run_id}: Training Loss')

    # Plot metrics
    axes[run_idx, 1].plot(iterations, train_metrics, label='Train')
    axes[run_idx, 1].plot(iterations, test_curr, label='Test (Current)')
    axes[run_idx, 1].plot(iterations, test_exp, label='Test (Experience)')
    axes[run_idx, 1].set_xlabel('Iteration')
    axes[run_idx, 1].set_ylabel('Metric')
    axes[run_idx, 1].legend()
    axes[run_idx, 1].set_title(f'Run {run_id}: Metrics Over Time')

    # Plot eigenvalue evolution
    for i in iterations[::10]:  # Sample every 10 iterations
        if 'layer_0' in run_data['iterations'][i]['eigenvalues']['A']:
            eigs = run_data['iterations'][i]['eigenvalues']['A']['layer_0']
            axes[run_idx, 2].scatter([i]*len(eigs), eigs, alpha=0.5)
    axes[run_idx, 2].set_xlabel('Iteration')
    axes[run_idx, 2].set_ylabel('Eigenvalue Magnitude')
    axes[run_idx, 2].set_title(f'Run {run_id}: Layer 0 A-Matrix Eigenvalues')

plt.tight_layout()
plt.savefig('all_runs_analysis.png')

# Compute statistics across runs
print(f"Total runs: {all_runs['metadata']['total_runs']}")
for run_id, run_data in all_runs['runs'].items():
    final_iter = max(run_data['iterations'].keys())
    final_H = run_data['iterations'][final_iter]['losses']['H']
    final_test = run_data['iterations'][final_iter]['metrics']['test_current']
    print(f"Run {run_id}: Final H={final_H:.6e}, Final Test Metric={final_test:.4f}")
```

### Analyzing Single Run

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load single run
with open('regression_sine_fcnn_run0_records.pkl', 'rb') as f:
    records = pickle.load(f)

print(f"Analyzing Run {records['metadata']['run_id']}")

# Extract time series
iterations = sorted(records['iterations'].keys())
H_values = [records['iterations'][i]['losses']['H'] for i in iterations]
train_metrics = [records['iterations'][i]['metrics']['train'] for i in iterations]
test_curr = [records['iterations'][i]['metrics']['test_current'] for i in iterations]
test_exp = [records['iterations'][i]['metrics']['test_experience'] for i in iterations]

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(iterations, H_values)
plt.xlabel('Iteration')
plt.ylabel('Hamiltonian (H)')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(iterations, train_metrics, label='Train')
plt.plot(iterations, test_curr, label='Test (Current)')
plt.plot(iterations, test_exp, label='Test (Experience)')
plt.xlabel('Iteration')
plt.ylabel('Metric')
plt.legend()
plt.title('Metrics Over Time')

# Plot eigenvalue evolution
plt.subplot(1, 3, 3)
for i in iterations[::10]:  # Sample every 10 iterations
    if 'layer_0' in records['iterations'][i]['eigenvalues']['A']:
        eigs = records['iterations'][i]['eigenvalues']['A']['layer_0']
        plt.scatter([i]*len(eigs), eigs, alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Layer 0 A-Matrix Eigenvalue Evolution')

plt.tight_layout()
plt.savefig('training_analysis.png')
```

## Notes

1. **Eigenvalue Computation**: Eigenvalues are computed for `A @ A^T` and `B @ B^T` matrices to ensure positive semi-definiteness
2. **Float Conversion**: All JAX arrays are converted to Python floats for pickle compatibility
3. **Empty Eigenvalues**: If A/B matrices don't exist or computation fails, empty dicts are returned
4. **Save Frequency**: Metrics are only recorded at `save_iter` intervals to balance detail vs. file size
5. **Memory**: Eigenvalue arrays are stored as NumPy arrays to keep file sizes manageable

## Future Extensions

The format easily supports:

- Additional loss terms (e.g., regularization losses)
- Layer-wise metrics (e.g., activation statistics)
- Architecture search history
- Hyperparameter schedules
- Model checkpointing metadata
- Dataset statistics per task
