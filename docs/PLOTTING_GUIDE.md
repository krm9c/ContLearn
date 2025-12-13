# Plotting Guide for ContLearn Results

## Overview

The `plot_results.py` script generates comprehensive visualizations of training results from the ContLearn framework. It supports both single-run and multi-run analysis.

## Installation Requirements

```bash
pip install numpy matplotlib
```

## Basic Usage

### Plotting Single Run Results

```bash
python plot_results.py <path_to_single_run_file.pkl>
```

Example:
```bash
python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl
```

### Plotting All Runs Results

```bash
python plot_results.py <path_to_allruns_file.pkl>
```

Example:
```bash
python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl
```

### Specifying Output Directory

```bash
python plot_results.py <path_to_file.pkl> --output-dir <output_directory>
```

Example:
```bash
python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl --output-dir figures/sine_experiment
```

## Generated Plots

### For Single Run Files

The script generates 4 comprehensive plots:

#### 1. Losses Plot (`*_losses.png`)
Six subplots showing all loss components:
- **Hamiltonian (H)**: Total loss function
- **Primary Loss (V)**: MSE for regression or Cross Entropy for classification
- **dV Term**: Time derivative of value function
- **dV/dx**: Gradient with respect to input
- **dV/dθ**: Gradient with respect to parameters
- **Gradient Norm** or **dV/dAdj**: Overall gradient magnitude (or adjacency gradient for graphs)

**Features**:
- **Shaded regions** indicate different tasks (color-coded)
- Vertical dashed lines mark exact task boundaries
- All subplots share the same x-axis (iteration)
- High-resolution (300 DPI)

#### 2. Metrics Plot (`*_metrics.png`)
Two subplots showing performance metrics:
- **Training Metric**: Performance on training data
- **Test Metrics**: Separate lines for:
  - Current task test data (green)
  - Experience replay test data (red)

**Features**:
- **Shaded regions** indicate different tasks (color-coded)
- Markers on lines for easier tracking
- Vertical dashed lines mark task boundaries
- Legend for clear identification

#### 3. Eigenvalues Plot (`*_eigenvalues.png`)
Box plots showing eigenvalue distributions per layer:
- **Left column**: A matrix eigenvalues for each layer
- **Right column**: B matrix eigenvalues for each layer
- One row per layer in the network

**Features**:
- **Shaded regions** indicate different tasks (color-coded)
- Box plots show distribution at each save_iter checkpoint
- Red median line
- Vertical dashed lines mark task boundaries
- Shows eigenvalue evolution over training

#### 4. Overview Plot (`*_overview.png`)
Combined visualization in 3 rows:
- **Row 1**: H, V, and gradient norm
- **Row 2**: dV/dx, dV/dθ, and dV
- **Row 3**: All metrics (train, test current, test experience) in one plot

**Features**:
- **Shaded regions** indicate different tasks (color-coded)
- Comprehensive single-page overview
- Vertical dashed lines mark task boundaries
- Useful for presentations and reports
- All key information in one figure

### For Multi-Run Files

When plotting an all-runs file, the script generates:

1. **Individual plots for each run**: All 4 plot types above for each run
2. **Multi-run comparison** (`*_allruns_comparison.png`):
   - Overlaid plots showing all runs together
   - 4 subplots: H, V, test current, test experience
   - Each run shown in different color

3. **Multi-run statistics** (`*_allruns_statistics.png`):
   - Mean ± standard deviation across runs
   - Shaded regions show variability
   - 4 subplots: H, V, test current, test experience

## File Naming Convention

All generated plots follow this naming pattern:

```
{prob}_{dataset}_{network}_run{run_id}_{plot_type}.png
```

Examples:
- `regression_sine_fcnn_run0_losses.png`
- `classification_mnist_cnn_run1_metrics.png`
- `graphclassification_ENZYMES_gnn_run0_eigenvalues.png`
- `regression_sine_fcnn_allruns_comparison.png`

## Example Workflows

### Quick Analysis After Training

```bash
# Run training
python run.py train 1 param_sine.json

# Generate plots
python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl

# Plots are saved in figures/
ls figures/
```

### Multi-Run Experiment Analysis

```bash
# Run multiple experiments
python run.py train 5 param_sine.json

# Plot all runs with statistics
python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl --output-dir figures/sine_5runs

# View comparison plots
open figures/sine_5runs/*allruns*.png
```

### Comparing Different Configurations

```bash
# Run baseline
python run.py train 3 param_sine.json
python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl --output-dir figures/baseline

# Run with AWB
# (modify config to enable AWB)
python run.py train 3 param_sine_awb.json
python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl --output-dir figures/awb

# Compare figures side by side
```

## Visual Elements Explained

### Task Shading

All plots include **color-coded shaded regions** to visually separate different tasks:

- Each task is represented by a different colored background (blue, green, red, orange, purple, etc.)
- Tasks are identified from the `task_id` field in the iteration data
- Shading uses low transparency (alpha=0.08) to not obscure the data
- **Vertical dashed lines** mark the exact boundaries where tasks change

This makes it easy to see:
- How performance changes when a new task is introduced
- Task-specific behavior (e.g., loss jumps at task boundaries)
- Recovery from catastrophic forgetting within each task
- The extent of each task in the training timeline

**Example**: In a 3-task experiment, you'll see three distinct shaded regions (e.g., blue for task 0, green for task 1, red for task 2), with vertical dashed lines at the transition points.

## Plot Interpretation Guide

### Loss Plots

**Hamiltonian (H)**:
- Should generally decrease over training
- Jumps at task boundaries are normal (new task introduced)
- Convergence indicates stable training

**Primary Loss (V)**:
- MSE: Lower is better, measures prediction error
- Cross Entropy: Lower is better, measures classification loss
- Should decrease within each task

**Gradient Components**:
- dV/dx: Input sensitivity, important for robustness
- dV/dθ: Parameter updates, should stabilize over time
- Gradient Norm: Overall update magnitude

### Metrics Plots

**Training Metric**:
- Should improve (increase for accuracy, decrease for MSE)
- May show slight degradation when new tasks introduced

**Test Current vs Experience**:
- **Current**: Performance on current task's test data
- **Experience**: Performance on replay buffer (older tasks)
- Gap between them indicates catastrophic forgetting
- Ideally both should remain high/low (depending on metric)

### Eigenvalue Plots

**A and B Matrices**:
- Show the learned transformation basis
- Distribution spread indicates expressiveness
- Changes at task boundaries indicate architecture adaptation (if AWB enabled)
- Stable eigenvalues within task indicate convergence

**Layer-wise Analysis**:
- Compare early vs late layers
- Early layers: Often more stable
- Late layers: More task-specific adaptation

## Advanced Usage

### Batch Plotting

Create a script to plot multiple experiments:

```bash
#!/bin/bash
# plot_all.sh

for file in logdir/model/*_allruns.pkl; do
    echo "Plotting $file"
    python plot_results.py "$file" --output-dir figures/
done
```

### Custom Analysis

The script can be imported as a module:

```python
import plot_results

# Load data
data, is_multi = plot_results.load_records('path/to/file.pkl')

# Extract time series
series = plot_results.extract_time_series(data)

# Custom plotting
import matplotlib.pyplot as plt
plt.plot(series['iterations'], series['loss_H'])
plt.show()
```

## Troubleshooting

### No Eigenvalues in Plot

If eigenvalue plots are empty:
- Check if AWB is enabled in the config
- Standard models without A/B matrices won't have eigenvalues
- This is normal for baseline experiments

### Memory Issues with Large Files

For very large record files:
```python
# Modify plot_results.py to sample iterations
# In extract_time_series(), add:
iterations = sorted(run_data['iterations'].keys())[::10]  # Sample every 10th
```

### Plot Resolution

To change resolution, modify the `dpi` parameter:
```python
plt.savefig(filepath, dpi=600, bbox_inches='tight')  # Higher resolution
```

## Tips for Publication-Quality Figures

1. **High DPI**: Already set to 300 DPI by default
2. **Vector Format**: For publications, modify to save as PDF:
   ```python
   filepath = filepath.replace('.png', '.pdf')
   plt.savefig(filepath, format='pdf', bbox_inches='tight')
   ```
3. **Font Sizes**: Adjust in the plotting functions if needed
4. **Color Scheme**: Modify color schemes for colorblind-friendly plots

## Integration with Paper Writing

### LaTeX Integration

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/regression_sine_fcnn_run0_overview.png}
    \caption{Training overview for sine regression task showing loss components,
             gradients, and performance metrics across continual learning tasks.}
    \label{fig:sine_overview}
\end{figure}
```

### Multi-Panel Figures

The overview and eigenvalue plots are designed as multi-panel figures suitable for academic papers.

## See Also

- [RECORDING_FORMAT.md](RECORDING_FORMAT.md) - Understanding the data structure
- [RUN_TRACKING_IMPLEMENTATION.md](RUN_TRACKING_IMPLEMENTATION.md) - Run tracking system
- [README.md](../README.md) - Main documentation
