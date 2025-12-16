# Recording Structure Restructure Plan

## Problem Statement

Current recording structure has several issues for AWB training:

1. **All phases recorded together**: Preliminary training, warmup, AB training, and Step 5 are all recorded in `record_dict['iterations']` with a continuous global iteration counter
2. **Can't compare AWB vs non-AWB properly**: Different number of iterations, different phases make alignment impossible
3. **AB eigenvalues not tracked**: No tracking of how A/B matrices evolve during AB training
4. **No clear phase separation**: Hard to analyze each phase independently

## Current Structure

```python
record_dict = {
    'metadata': {
        'problem': 'regression',
        'dataset': 'sine',
        'network': 'fcnn',
        'awb_enabled': True,
        'n_tasks': 5,
        'epochs_per_task': 500,
        ...
    },
    'iterations': {
        0: {'losses': {...}, 'metrics': {...}, 'eigenvalues': {...}, 'task_id': 0, 'step': 0},
        10: {'losses': {...}, 'metrics': {...}, 'eigenvalues': {...}, 'task_id': 0, 'step': 10},
        # ... continuous numbering across all phases and tasks
    },
    'architecture_history': {...}
}
```

**Issues:**
- Iteration 500 in AWB run might be preliminary training, while iteration 500 in non-AWB run is main training
- AB training iterations mixed with V training iterations
- No way to extract just "main training" for comparison

---

## Proposed Structure

```python
record_dict = {
    'metadata': {
        'problem': 'regression',
        'prob': 'regression',
        'dataset': 'sine',
        'network': 'fcnn',
        'n_tasks': 5,
        'epochs_per_task': 500,
        'save_iter': 10,
        'awb_enabled': True,
        'run_id': 0,
    },

    'tasks': {
        0: {
            'main_training': {
                # Per-epoch data at save_iter intervals
                # Epoch numbers are WITHIN TASK (0 to epochs_per_task)
                'epochs': [0, 10, 20, 30, ...],
                'H': [0.5, 0.45, 0.4, ...],
                'V': [0.3, 0.28, 0.25, ...],
                'dV': [...],
                'dV_dx': [...],
                'dV_dtheta': [...],
                'grad_norm': [...],
                'train_metric': [...],
                'test_current': [...],
                'test_experience': [...],
                'eigenvalues': {
                    # Eigenvalues at each recorded epoch
                    'A': {
                        'layer_0': [array([...]), array([...]), ...],  # One per epoch
                        'layer_1': [array([...]), array([...]), ...],
                    },
                    'B': {
                        'layer_0': [array([...]), array([...]), ...],
                        'layer_1': [array([...]), array([...]), ...],
                    }
                }
            },
            'phase_info': {
                'type': 'standard',  # 'standard' for task 0, 'awb_step5' or 'continuation' for later
                'total_epochs': 500
            },
            'arch_changed': False,
            'architecture': {
                'sizes': [1, 128, 64, 1]  # or feed_sizes/filter_size for CNN
            }
        },

        1: {
            # PRELIMINARY - Not recorded in detail, just summary
            'preliminary': {
                'n_epochs': 50,
                'warmup_epochs': 10,
                'final_loss': 0.45,
                'decision': 'arch_change'  # or 'no_change'
            },

            # AB TRAINING - Recorded separately with AB eigenvalues
            'ab_training': {
                'epochs': [0, 10, 20, ...],  # Within AB training phase
                'H': [...],
                'V': [...],
                'iterations': 2,  # How many AB training iterations
                'ab_eigenvalues': {
                    # Track how A/B matrices evolve during AB training
                    'A': {
                        'layer_0': [array([...]), array([...]), ...],
                        'layer_1': [array([...]), array([...]), ...],
                    },
                    'B': {
                        'layer_0': [array([...]), array([...]), ...],
                        'layer_1': [array([...]), array([...]), ...],
                    }
                }
            },

            # MAIN TRAINING - Step 5 (V training) - THIS IS WHAT GETS COMPARED
            'main_training': {
                'epochs': [0, 10, 20, ...],  # Within main training phase
                'H': [...],
                'V': [...],
                'dV': [...],
                'dV_dx': [...],
                'dV_dtheta': [...],
                'grad_norm': [...],
                'train_metric': [...],
                'test_current': [...],
                'test_experience': [...],
                'eigenvalues': {
                    'A': {...},
                    'B': {...}
                }
            },

            'phase_info': {
                'type': 'awb_step5',
                'warmup_epochs': 100,  # V warmup
                'total_epochs': 500
            },
            'arch_changed': True,
            'architecture': {
                'original': {'sizes': [1, 128, 64, 1]},
                'new': {'sizes': [1, 256, 128, 1]}
            }
        },

        # Task where arch search found same architecture
        2: {
            'preliminary': {
                'n_epochs': 50,
                'warmup_epochs': 10,
                'final_loss': 0.35,
                'decision': 'no_change'
            },
            # No ab_training key when arch didn't change
            'main_training': {
                'epochs': [0, 10, 20, ...],
                'H': [...],
                ...
            },
            'phase_info': {
                'type': 'continuation',  # Continued training without arch change
                'total_epochs': 500
            },
            'arch_changed': False,
            'architecture': {
                'sizes': [1, 256, 128, 1]  # Same as previous
            }
        }
    },

    # Keep for backward compatibility
    'architecture_history': {
        0: {...},
        1: {...},
        ...
    }
}
```

---

## Recording Rules

| Scenario | Phase | Record to | Used for Comparison? |
|----------|-------|-----------|---------------------|
| Task 0 (any run) | Standard training | `tasks[0]['main_training']` | YES |
| Task 1+ (AWB disabled) | Standard training | `tasks[i]['main_training']` | YES |
| Task 1+ (AWB enabled) | Task warmup | `tasks[i]['preliminary']` (summary only) | NO |
| Task 1+ (AWB enabled) | Preliminary training | `tasks[i]['preliminary']` (summary only) | NO |
| Task 1+ (AWB enabled) | AB training | `tasks[i]['ab_training']` | NO (separate plots) |
| Task 1+ (AWB enabled) | V warmup (Step 5) | `tasks[i]['main_training']` | YES |
| Task 1+ (AWB enabled) | V training (Step 5) | `tasks[i]['main_training']` | YES |
| Task 1+ (AWB, no arch change) | Continuation | `tasks[i]['main_training']` | YES |

---

## Files to Modify

### 1. `src/cl/core/recording.py`

**Current methods:**
- `initialize_record_dict()` - Creates initial structure
- `record_metrics()` - Records a single iteration
- `_compute_eigenvalues()` - Extracts eigenvalues from model

**New/Modified methods:**

```python
def initialize_record_dict(self, config, run_id=0):
    """Initialize with new task-based structure."""
    return {
        'metadata': {...},
        'tasks': {},
        'architecture_history': {}
    }

def initialize_task(self, record_dict, task_id, arch_info):
    """Initialize a new task's recording structure."""
    record_dict['tasks'][task_id] = {
        'main_training': {
            'epochs': [],
            'H': [], 'V': [], 'dV': [], 'dV_dx': [], 'dV_dtheta': [],
            'grad_norm': [],
            'train_metric': [], 'test_current': [], 'test_experience': [],
            'eigenvalues': {'A': {}, 'B': {}}
        },
        'arch_changed': False,
        'architecture': arch_info
    }

def record_preliminary_summary(self, record_dict, task_id, n_epochs, warmup_epochs,
                                final_loss, decision):
    """Record summary of preliminary phase (not detailed metrics)."""
    record_dict['tasks'][task_id]['preliminary'] = {
        'n_epochs': n_epochs,
        'warmup_epochs': warmup_epochs,
        'final_loss': final_loss,
        'decision': decision
    }

def record_main_training_epoch(self, record_dict, task_id, epoch, losses,
                                gradients, metrics, model):
    """Record a single epoch of main training (Step 5 or standard)."""
    task = record_dict['tasks'][task_id]['main_training']
    task['epochs'].append(epoch)
    task['H'].append(float(losses['H']))
    task['V'].append(float(losses['V']))
    # ... other metrics

    # Record eigenvalues
    eigs = self._compute_eigenvalues(model)
    for matrix_type in ['A', 'B']:
        for layer_name, eig_values in eigs[matrix_type].items():
            if layer_name not in task['eigenvalues'][matrix_type]:
                task['eigenvalues'][matrix_type][layer_name] = []
            task['eigenvalues'][matrix_type][layer_name].append(eig_values)

def initialize_ab_training(self, record_dict, task_id):
    """Initialize AB training recording for a task."""
    record_dict['tasks'][task_id]['ab_training'] = {
        'epochs': [],
        'H': [], 'V': [],
        'iterations': 0,
        'ab_eigenvalues': {'A': {}, 'B': {}}
    }

def record_ab_training_epoch(self, record_dict, task_id, epoch, losses, model):
    """Record a single epoch of AB training with AB eigenvalues."""
    ab = record_dict['tasks'][task_id]['ab_training']
    ab['epochs'].append(epoch)
    ab['H'].append(float(losses['H']))
    ab['V'].append(float(losses['V']))

    # Record A/B eigenvalues
    eigs = self._compute_eigenvalues(model)
    for matrix_type in ['A', 'B']:
        for layer_name, eig_values in eigs[matrix_type].items():
            if layer_name not in ab['ab_eigenvalues'][matrix_type]:
                ab['ab_eigenvalues'][matrix_type][layer_name] = []
            ab['ab_eigenvalues'][matrix_type][layer_name].append(eig_values)
```

### 2. `src/cl/core/loops.py`

**Changes needed:**
- Add `phase` parameter to `train__CL`: `'main'`, `'ab'`, `'preliminary'`
- Add `record_training` parameter (default True, set False for preliminary/warmup)
- Conditional recording based on phase

```python
def train__CL(self, train__, params, static, opt_state, optim,
              n_iter, task_id, config, record_dict,
              notABTrain=True, problem_type='vectors', loss_type='regression',
              phase='main',  # NEW: 'main', 'ab', 'preliminary'
              record_training=True):  # NEW: Whether to record metrics
    """
    ...
    Args:
        phase: Training phase - 'main' (Step 5/standard), 'ab' (AB training),
               'preliminary' (not recorded in detail)
        record_training: Whether to record metrics (False for preliminary/warmup)
    """

    # In the epoch loop, at the recording point:
    if record_training and (epoch % save_iter == 0 or is_last_epoch):
        if phase == 'main':
            self.record_main_training_epoch(record_dict, task_id, epoch,
                                            losses_dict, gradients, metrics, model)
        elif phase == 'ab':
            self.record_ab_training_epoch(record_dict, task_id, epoch,
                                          losses_dict, model)
        # 'preliminary' phase: don't record individual epochs
```

### 3. `src/cl/runners/generic_runner.py`

**Changes needed in `train_model()` function:**

```python
def train_model(config, run_id=0):
    # ... setup code ...

    for task_id in range(n_tasks):
        # Initialize task recording
        arch_info = get_model_architecture(model)
        trainer.initialize_task(record_dict, task_id, arch_info)

        if task_id == 0 or not awb_enabled:
            # Standard training - record to main_training
            params, static, opt_state, record_dict = trainer.train__CL(
                ...,
                phase='main',
                record_training=True
            )
        else:
            # AWB PIPELINE

            # Task warmup (don't record)
            if task_warmup_epochs > 0:
                params, static, opt_state, record_dict = trainer.train__CL(
                    ...,
                    n_iter=task_warmup_epochs,
                    phase='preliminary',
                    record_training=False  # Don't record warmup
                )

            # Preliminary training (don't record in detail)
            remaining_prelim = awb_prelim_epochs - task_warmup_epochs
            params, static, opt_state, record_dict = trainer.train__CL(
                ...,
                n_iter=remaining_prelim,
                phase='preliminary',
                record_training=False  # Don't record preliminary
            )

            # Record preliminary summary
            trainWLoss = compute_avg_loss(...)
            trainer.record_preliminary_summary(
                record_dict, task_id,
                n_epochs=awb_prelim_epochs,
                warmup_epochs=task_warmup_epochs,
                final_loss=trainWLoss,
                decision='arch_change' if change_arch else 'no_change'
            )

            if change_arch and (opt_arch != original_arch):
                # Architecture changed
                record_dict['tasks'][task_id]['arch_changed'] = True
                record_dict['tasks'][task_id]['architecture'] = {
                    'original': original_arch,
                    'new': new_arch
                }

                # Initialize AB training recording
                trainer.initialize_ab_training(record_dict, task_id)

                # AB Training - record separately
                diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                    ...,
                    phase='ab',
                    record_training=True  # Record AB training
                )
                record_dict['tasks'][task_id]['ab_training']['iterations'] = ab_iter

                # V transformation (Step 4)
                model = compute_V_from_AWB(model)

                # Step 5: V warmup (record to main_training)
                if v_warmup_epochs > 0:
                    params, static, opt_state, record_dict = trainer.train__CL(
                        ...,
                        n_iter=v_warmup_epochs,
                        phase='main',
                        record_training=True  # This IS main training
                    )

                # Step 5: V training (record to main_training)
                params, static, opt_state, record_dict = trainer.train__CL(
                    ...,
                    n_iter=remaining_epochs,
                    phase='main',
                    record_training=True
                )

                record_dict['tasks'][task_id]['phase_info'] = {
                    'type': 'awb_step5',
                    'warmup_epochs': v_warmup_epochs,
                    'total_epochs': epochs_per_task
                }
            else:
                # No architecture change - continuation training
                params, static, opt_state, record_dict = trainer.train__CL(
                    ...,
                    phase='main',
                    record_training=True
                )

                record_dict['tasks'][task_id]['phase_info'] = {
                    'type': 'continuation',
                    'total_epochs': epochs_per_task
                }
```

### 4. `scripts/compare_runs.py`

**New helper function to extract task-based data:**

```python
def extract_task_series(record_dict):
    """Extract time series organized by task from new structure."""
    tasks = record_dict.get('tasks', {})
    metadata = record_dict.get('metadata', {})

    result = {
        'metadata': metadata,
        'tasks': {}
    }

    for task_id, task_data in tasks.items():
        main = task_data.get('main_training', {})
        result['tasks'][task_id] = {
            'epochs': main.get('epochs', []),
            'H': main.get('H', []),
            'V': main.get('V', []),
            'dV': main.get('dV', []),
            'train_metric': main.get('train_metric', []),
            'test_current': main.get('test_current', []),
            'test_experience': main.get('test_experience', []),
            'eigenvalues': main.get('eigenvalues', {}),
            'arch_changed': task_data.get('arch_changed', False),
        }

        # Include AB training data if present
        if 'ab_training' in task_data:
            result['tasks'][task_id]['ab_training'] = task_data['ab_training']

    return result


def plot_comparison_by_task(run1, run2, labels, output_dir, smooth_window=5):
    """Plot comparison aligned by task and within-task epoch."""
    series1 = extract_task_series(run1)
    series2 = extract_task_series(run2)

    n_tasks = max(len(series1['tasks']), len(series2['tasks']))

    fig, axes = plt.subplots(n_tasks, 3, figsize=(18, 4*n_tasks))

    for task_id in range(n_tasks):
        task1 = series1['tasks'].get(task_id, {})
        task2 = series2['tasks'].get(task_id, {})

        # Plot H loss
        ax = axes[task_id, 0]
        if task1.get('epochs'):
            ax.plot(task1['epochs'], smooth_data(task1['H'], smooth_window),
                   label=labels[0], color=COLOR_PAIRS['primary'][0])
        if task2.get('epochs'):
            ax.plot(task2['epochs'], smooth_data(task2['H'], smooth_window),
                   label=labels[1], color=COLOR_PAIRS['primary'][1])
        ax.set_title(f'Task {task_id}: Hamiltonian (H)')
        ax.legend()

        # Plot test metric
        ax = axes[task_id, 1]
        # ... similar plotting code

        # Mark arch change
        if task2.get('arch_changed'):
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5,
                      label='Arch Change')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_by_task.png', dpi=150)


def plot_ab_eigenvalues(run_with_awb, output_dir):
    """Plot AB eigenvalue evolution during AB training phases."""
    series = extract_task_series(run_with_awb)

    # Find tasks with AB training
    ab_tasks = [tid for tid, data in series['tasks'].items()
                if 'ab_training' in data]

    if not ab_tasks:
        print("No AB training phases found")
        return

    fig, axes = plt.subplots(len(ab_tasks), 2, figsize=(14, 5*len(ab_tasks)))
    if len(ab_tasks) == 1:
        axes = axes.reshape(1, -1)

    for idx, task_id in enumerate(ab_tasks):
        ab_data = series['tasks'][task_id]['ab_training']
        ab_eigs = ab_data.get('ab_eigenvalues', {})
        epochs = ab_data.get('epochs', [])

        # Plot A eigenvalues
        ax = axes[idx, 0]
        for layer_name, eig_list in ab_eigs.get('A', {}).items():
            # Plot mean eigenvalue per epoch
            means = [np.mean(np.abs(e)) for e in eig_list]
            ax.plot(epochs[:len(means)], means, label=layer_name)
        ax.set_title(f'Task {task_id}: A Matrix Eigenvalues (AB Training)')
        ax.set_xlabel('AB Training Epoch')
        ax.legend()

        # Plot B eigenvalues
        ax = axes[idx, 1]
        for layer_name, eig_list in ab_eigs.get('B', {}).items():
            means = [np.mean(np.abs(e)) for e in eig_list]
            ax.plot(epochs[:len(means)], means, label=layer_name)
        ax.set_title(f'Task {task_id}: B Matrix Eigenvalues (AB Training)')
        ax.set_xlabel('AB Training Epoch')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ab_eigenvalues.png', dpi=150)
```

### 5. `scripts/plot_results.py`

Similar updates to read from `tasks[i]['main_training']` structure.

---

## Backward Compatibility

To maintain backward compatibility with existing record files:

```python
def extract_task_series(record_dict):
    """Extract time series, supporting both old and new formats."""

    # New format: has 'tasks' key
    if 'tasks' in record_dict:
        return extract_task_series_new(record_dict)

    # Old format: has 'iterations' key
    if 'iterations' in record_dict:
        return convert_old_format_to_task_series(record_dict)

    raise ValueError("Unknown record format")


def convert_old_format_to_task_series(record_dict):
    """Convert old iteration-based format to task-based format."""
    iterations = record_dict['iterations']
    metadata = record_dict['metadata']
    n_tasks = metadata.get('n_tasks', 1)
    epochs_per_task = metadata.get('epochs_per_task', 100)

    result = {'metadata': metadata, 'tasks': {}}

    for task_id in range(n_tasks):
        task_data = {
            'epochs': [], 'H': [], 'V': [], 'dV': [],
            'train_metric': [], 'test_current': [], 'test_experience': [],
            'eigenvalues': {'A': {}, 'B': {}}
        }

        # Find iterations belonging to this task
        for iter_num, iter_data in iterations.items():
            if iter_data.get('task_id') == task_id:
                epoch_in_task = iter_data.get('step', iter_num % epochs_per_task)
                task_data['epochs'].append(epoch_in_task)
                task_data['H'].append(iter_data['losses'].get('H', 0))
                task_data['V'].append(iter_data['losses'].get('V', 0))
                # ... etc

        result['tasks'][task_id] = task_data

    return result
```

---

## Testing Plan

1. **Unit tests**: Add tests for new recording methods
2. **Integration test**: Run sine_awb with new recording, verify structure
3. **Comparison test**: Run sine and sine_awb, verify plots align properly
4. **Backward compat test**: Load old record files, verify conversion works

---

## Implementation Order

1. **Phase 1: Recording structure** (`recording.py`)
   - Add new methods
   - Keep old methods for backward compat

2. **Phase 2: Training loop** (`loops.py`)
   - Add phase parameter
   - Add record_training parameter
   - Conditional recording

3. **Phase 3: Runner** (`generic_runner.py`)
   - Restructure to use new recording
   - Proper phase labeling

4. **Phase 4: Plotting** (`compare_runs.py`, `plot_results.py`)
   - Update extraction functions
   - Add AB eigenvalue plots
   - Backward compat handling

5. **Phase 5: Testing**
   - Run experiments
   - Verify plots

---

## Config Changes

No config changes needed - the recording structure change is internal.

---

## Summary

| Before | After |
|--------|-------|
| Global iteration counter | Per-task epoch indexing |
| All phases mixed together | Clear phase separation |
| No AB eigenvalue tracking | Full AB eigenvalue evolution |
| Hard to compare runs | Easy task-by-task comparison |
| Preliminary recorded | Preliminary summary only |
