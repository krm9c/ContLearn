#!/usr/bin/env python
"""
Compare two continual learning experiments side-by-side.

Generates comparison plots for all recorded metrics between two runs
(e.g., sine vs sine_awb).

Usage:
    python scripts/compare_runs.py <path_to_run1_records.pkl> <path_to_run2_records.pkl> \
           --labels "Baseline" "AWB" --output-dir figures/comparison

Example:
    python scripts/compare_runs.py \
        outputs/sine_model/regression_sine_fcnn_run0_records.pkl \
        outputs/sine_awb_model/regression_sine_fcnn_awb_run0_records.pkl \
        --labels "Sine" "Sine+AWB" \
        --output-dir figures/comparison
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.ndimage import uniform_filter1d

# Added by Claude: Import helper functions from plot_results
import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_results import extract_time_series, add_task_shading

# Added by Claude: Contrasting color pairs for comparison plots
# Each pair has two distinct, easily distinguishable colors
COLOR_PAIRS = {
    'primary': ('#1f77b4', '#ff7f0e'),    # Blue vs Orange
    'secondary': ('#2ca02c', '#d62728'),  # Green vs Red
    'tertiary': ('#9467bd', '#8c564b'),   # Purple vs Brown
    'quaternary': ('#17becf', '#e377c2'), # Cyan vs Pink
}


def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply smoothing to time series data using a moving average.

    Args:
        data: Input array to smooth
        window_size: Size of smoothing window (default: 5, 0 or 1 = no smoothing)

    Returns:
        Smoothed array (same length as input)
    """
    if window_size <= 1 or len(data) < window_size:
        return np.array(data).astype(float)
    # Use uniform filter for moving average smoothing
    return uniform_filter1d(data.astype(float), size=window_size, mode='nearest')


def load_records(filepath: str) -> Dict[str, Any]:
    """Load records from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # If this is multi-run data, extract first run
    if 'runs' in data:
        first_run_id = list(data['runs'].keys())[0]
        return data['runs'][first_run_id]
    return data


def plot_loss_comparison(run1: Dict[str, Any], run2: Dict[str, Any],
                         labels: List[str], output_dir: str,
                         smooth_window: int = 5):
    """Compare all loss components between two runs.

    Args:
        run1, run2: Record dicts from two runs
        labels: Labels for the two runs
        output_dir: Directory to save plots
        smooth_window: Window size for smoothing (default: 5)
    """
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Loss Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    # Added by Claude: Use contrasting colors for each run
    color1, color2 = COLOR_PAIRS['primary']

    loss_components = [
        ('loss_H', 'Hamiltonian (H)'),
        ('loss_V', f'V ({metadata1["loss_function"]})'),
        ('loss_dV', 'dV'),
        ('loss_dV_dx', 'dV/dx'),
        ('loss_dV_dtheta', 'dV/dθ'),
        ('grad_norm', '||dH/dθ||'),
    ]

    for idx, (key, ylabel) in enumerate(loss_components):
        ax = axes[idx // 3, idx % 3]

        # Added by Claude: Apply smoothing and use contrasting colors
        data1_smooth = smooth_data(np.array(series1[key]), smooth_window)
        data2_smooth = smooth_data(np.array(series2[key]), smooth_window)

        # Plot both runs with contrasting colors
        ax.plot(series1['iterations'], data1_smooth, color=color1, linewidth=2.5,
                label=labels[0], alpha=0.9)
        ax.plot(series2['iterations'], data2_smooth, color=color2, linewidth=2.5,
                label=labels[1], alpha=0.9)

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add task shading
        add_task_shading(ax, series1, metadata1, alpha=0.06)

        # Add task boundaries
        task_changes = np.where(np.diff(series1['task_ids']) != 0)[0] + 1
        for tc in task_changes:
            ax.axvline(series1['iterations'][tc], color='k', linestyle=':',
                      alpha=0.4, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_comparison_losses.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_metrics_comparison(run1: Dict[str, Any], run2: Dict[str, Any],
                            labels: List[str], output_dir: str,
                            smooth_window: int = 5):
    """Compare training and test metrics between two runs.

    Args:
        run1, run2: Record dicts from two runs
        labels: Labels for the two runs
        output_dir: Directory to save plots
        smooth_window: Window size for smoothing (default: 5)
    """
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Metrics Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    # Added by Claude: Use contrasting colors for each run
    color1, color2 = COLOR_PAIRS['primary']

    metrics = [
        ('metric_train', 'Train'),
        ('metric_test_current', 'Test (Current Task)'),
        ('metric_test_experience', 'Test (Experience)'),
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]

        # Added by Claude: Apply smoothing and use contrasting colors
        data1_smooth = smooth_data(np.array(series1[key]), smooth_window)
        data2_smooth = smooth_data(np.array(series2[key]), smooth_window)

        # Plot both runs with contrasting colors
        ax.plot(series1['iterations'], data1_smooth, color=color1, linewidth=2.5,
                label=labels[0], marker='o', markersize=4,
                markevery=max(1, len(series1['iterations'])//20), alpha=0.9)
        ax.plot(series2['iterations'], data2_smooth, color=color2, linewidth=2.5,
                label=labels[1], marker='s', markersize=4,
                markevery=max(1, len(series2['iterations'])//20), alpha=0.9)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(f'{metadata1["metric_function"]}', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add task shading
        add_task_shading(ax, series1, metadata1, alpha=0.06)

        # Add task boundaries
        task_changes = np.where(np.diff(series1['task_ids']) != 0)[0] + 1
        for tc in task_changes:
            ax.axvline(series1['iterations'][tc], color='k', linestyle=':',
                      alpha=0.4, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_comparison_metrics.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_final_metrics_bar(run1: Dict[str, Any], run2: Dict[str, Any],
                           labels: List[str], output_dir: str):
    """Bar chart comparing final metrics between two runs."""
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    # Added by Claude: Use contrasting colors for bars
    color1, color2 = COLOR_PAIRS['primary']

    # Get final values (last iteration)
    final_idx1 = -1
    final_idx2 = -1

    metrics_data = {
        'Train': [series1['metric_train'][final_idx1], series2['metric_train'][final_idx2]],
        'Test\n(Current)': [series1['metric_test_current'][final_idx1],
                           series2['metric_test_current'][final_idx2]],
        'Test\n(Experience)': [series1['metric_test_experience'][final_idx1],
                              series2['metric_test_experience'][final_idx2]],
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Final Metrics Comparison - {metadata1["dataset"]}',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(metrics_data))
    width = 0.35

    vals1 = [metrics_data[k][0] for k in metrics_data]
    vals2 = [metrics_data[k][1] for k in metrics_data]

    # Added by Claude: Use contrasting colors for bars
    bars1 = ax.bar(x - width/2, vals1, width, label=labels[0], color=color1, alpha=0.9)
    bars2 = ax.bar(x + width/2, vals2, width, label=labels[1], color=color2, alpha=0.9)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(f'{metadata1["metric_function"]}', fontsize=12)
    ax.set_title('Final Performance After All Tasks', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data.keys())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_comparison_final_metrics_bar.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_combined_comparison(run1: Dict[str, Any], run2: Dict[str, Any],
                             labels: List[str], output_dir: str,
                             smooth_window: int = 5):
    """Combined overview comparison plot.

    Args:
        run1, run2: Record dicts from two runs
        labels: Labels for the two runs
        output_dir: Directory to save plots
        smooth_window: Window size for smoothing (default: 5)
    """
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    # Added by Claude: Use contrasting colors for each run
    color1, color2 = COLOR_PAIRS['primary']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Training Overview Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    # Added by Claude: Apply smoothing to all data
    loss_H_1 = smooth_data(np.array(series1['loss_H']), smooth_window)
    loss_H_2 = smooth_data(np.array(series2['loss_H']), smooth_window)
    loss_V_1 = smooth_data(np.array(series1['loss_V']), smooth_window)
    loss_V_2 = smooth_data(np.array(series2['loss_V']), smooth_window)
    test_curr_1 = smooth_data(np.array(series1['metric_test_current']), smooth_window)
    test_curr_2 = smooth_data(np.array(series2['metric_test_current']), smooth_window)
    test_exp_1 = smooth_data(np.array(series1['metric_test_experience']), smooth_window)
    test_exp_2 = smooth_data(np.array(series2['metric_test_experience']), smooth_window)

    # Plot 1: Hamiltonian - with contrasting colors
    axes[0, 0].plot(series1['iterations'], loss_H_1, color=color1, linewidth=2.5,
                   label=labels[0], alpha=0.9)
    axes[0, 0].plot(series2['iterations'], loss_H_2, color=color2, linewidth=2.5,
                   label=labels[1], alpha=0.9)
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Hamiltonian (H)', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # Plot 2: Primary Loss (V) - with contrasting colors
    axes[0, 1].plot(series1['iterations'], loss_V_1, color=color1, linewidth=2.5,
                   label=labels[0], alpha=0.9)
    axes[0, 1].plot(series2['iterations'], loss_V_2, color=color2, linewidth=2.5,
                   label=labels[1], alpha=0.9)
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel(f'V ({metadata1["loss_function"]})', fontsize=11)
    axes[0, 1].set_title('Primary Loss (V)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # Plot 3: Test Current Task - with contrasting colors
    axes[1, 0].plot(series1['iterations'], test_curr_1, color=color1,
                   linewidth=2.5, label=labels[0], marker='o', markersize=4,
                   markevery=max(1, len(series1['iterations'])//20), alpha=0.9)
    axes[1, 0].plot(series2['iterations'], test_curr_2, color=color2,
                   linewidth=2.5, label=labels[1], marker='s', markersize=4,
                   markevery=max(1, len(series2['iterations'])//20), alpha=0.9)
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel(f'{metadata1["metric_function"]}', fontsize=11)
    axes[1, 0].set_title('Test (Current Task)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # Plot 4: Test Experience - with contrasting colors
    axes[1, 1].plot(series1['iterations'], test_exp_1, color=color1,
                   linewidth=2.5, label=labels[0], marker='o', markersize=4,
                   markevery=max(1, len(series1['iterations'])//20), alpha=0.9)
    axes[1, 1].plot(series2['iterations'], test_exp_2, color=color2,
                   linewidth=2.5, label=labels[1], marker='s', markersize=4,
                   markevery=max(1, len(series2['iterations'])//20), alpha=0.9)
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel(f'{metadata1["metric_function"]}', fontsize=11)
    axes[1, 1].set_title('Test (Experience)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)

    # Add task shading to all subplots
    for ax in axes.flat:
        add_task_shading(ax, series1, metadata1, alpha=0.06)

        # Add task boundaries
        task_changes = np.where(np.diff(series1['task_ids']) != 0)[0] + 1
        for tc in task_changes:
            ax.axvline(series1['iterations'][tc], color='k', linestyle=':',
                      alpha=0.4, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_comparison_overview.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def print_summary_statistics(run1: Dict[str, Any], run2: Dict[str, Any],
                             labels: List[str]):
    """Print summary statistics comparing the two runs."""
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    print("\n" + "="*70)
    print("COMPARISON SUMMARY STATISTICS")
    print("="*70)
    print(f"\nDataset: {metadata1['dataset']}")
    print(f"Network: {metadata1['network']}")
    print(f"Number of tasks: {metadata1['n_tasks']}")
    print(f"\nRun 1: {labels[0]} (AWB: {metadata1.get('awb_enabled', False)})")
    print(f"Run 2: {labels[1]} (AWB: {run2['metadata'].get('awb_enabled', False)})")

    # Final metrics
    print("\n" + "-"*70)
    print("FINAL METRICS (Last Iteration)")
    print("-"*70)

    metrics = [
        ('metric_train', 'Train'),
        ('metric_test_current', 'Test (Current Task)'),
        ('metric_test_experience', 'Test (Experience)'),
    ]

    print(f"{'Metric':<25} {labels[0]:>15} {labels[1]:>15} {'Improvement':>12}")
    print("-"*70)

    for key, name in metrics:
        val1 = series1[key][-1]
        val2 = series2[key][-1]

        # For MSE, lower is better; for accuracy, higher is better
        if metadata1['metric_function'] == 'mse':
            improvement = ((val1 - val2) / val1) * 100  # Positive = improvement
            better = "✓" if val2 < val1 else "✗"
        else:
            improvement = ((val2 - val1) / val1) * 100  # Positive = improvement
            better = "✓" if val2 > val1 else "✗"

        print(f"{name:<25} {val1:>15.6f} {val2:>15.6f} {improvement:>10.2f}% {better}")

    # Loss components
    print("\n" + "-"*70)
    print("FINAL LOSSES (Last Iteration)")
    print("-"*70)

    losses = [
        ('loss_H', 'Hamiltonian (H)'),
        ('loss_V', f'Primary Loss (V)'),
        ('loss_dV', 'dV'),
        ('grad_norm', 'Gradient Norm'),
    ]

    print(f"{'Loss':<25} {labels[0]:>15} {labels[1]:>15} {'Improvement':>12}")
    print("-"*70)

    for key, name in losses:
        val1 = series1[key][-1]
        val2 = series2[key][-1]
        improvement = ((val1 - val2) / val1) * 100  # Positive = improvement (lower loss)
        better = "✓" if val2 < val1 else "✗"
        print(f"{name:<25} {val1:>15.6f} {val2:>15.6f} {improvement:>10.2f}% {better}")

    print("="*70 + "\n")


def plot_ab_eigenvalue_comparison(run1: Dict[str, Any], run2: Dict[str, Any],
                                   labels: List[str], output_dir: str):
    """Compare AB training eigenvalue evolution between two AWB runs.

    Added by Claude: Compares A and B eigenvalue evolution during AB training
    phases for tasks that underwent architecture changes.

    Args:
        run1, run2: Record dicts from two runs
        labels: Labels for the two runs
        output_dir: Directory to save plots
    """
    # Check if new structure exists
    if 'tasks' not in run1 or 'tasks' not in run2:
        print("Warning: Task-based structure not found. Skipping AB eigenvalue comparison.")
        return

    metadata1 = run1['metadata']
    awb_enabled1 = metadata1.get('awb_enabled', False)
    awb_enabled2 = run2['metadata'].get('awb_enabled', False)

    if not (awb_enabled1 and awb_enabled2):
        print("Warning: AWB not enabled in both runs. Skipping AB eigenvalue comparison.")
        return

    # Find tasks with AB training in both runs
    tasks_with_ab_1 = set()
    tasks_with_ab_2 = set()

    for task_id, task_data in run1['tasks'].items():
        if 'ab_training' in task_data and task_data['ab_training']:
            tasks_with_ab_1.add(task_id)

    for task_id, task_data in run2['tasks'].items():
        if 'ab_training' in task_data and task_data['ab_training']:
            tasks_with_ab_2.add(task_id)

    common_tasks = sorted(tasks_with_ab_1.intersection(tasks_with_ab_2))

    if not common_tasks:
        print("Warning: No common AB training tasks found. Skipping AB eigenvalue comparison.")
        return

    color1, color2 = COLOR_PAIRS['primary']

    # Create subplots: one row per task
    n_tasks = len(common_tasks)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(18, 6 * n_tasks))
    if n_tasks == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'AB Training Eigenvalue Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    for idx, task_id in enumerate(common_tasks):
        ab1 = run1['tasks'][task_id]['ab_training']
        ab2 = run2['tasks'][task_id]['ab_training']

        # Plot A matrices
        ax_a = axes[idx, 0]

        # Get layer names (use first run as reference)
        layer_names_a = sorted(ab1.get('ab_eigenvalues', {}).get('A', {}).keys())

        for layer_name in layer_names_a:
            if layer_name in ab1['ab_eigenvalues']['A']:
                eig_series1 = ab1['ab_eigenvalues']['A'][layer_name]
                iters1 = ab1.get('iterations', [])[:len(eig_series1)]
                mean_eigs1 = [np.mean(np.abs(np.real(eigs))) for eigs in eig_series1]
                ax_a.plot(iters1, mean_eigs1, '-o', color=color1, linewidth=2,
                         label=f'{labels[0]} - {layer_name}', markersize=6, alpha=0.8)

            if layer_name in ab2['ab_eigenvalues']['A']:
                eig_series2 = ab2['ab_eigenvalues']['A'][layer_name]
                iters2 = ab2.get('iterations', [])[:len(eig_series2)]
                mean_eigs2 = [np.mean(np.abs(np.real(eigs))) for eigs in eig_series2]
                ax_a.plot(iters2, mean_eigs2, '--s', color=color2, linewidth=2,
                         label=f'{labels[1]} - {layer_name}', markersize=6, alpha=0.8)

        ax_a.set_xlabel('AB Training Iteration', fontsize=12)
        ax_a.set_ylabel('Mean Eigenvalue Magnitude', fontsize=12)
        ax_a.set_title(f'Task {task_id} - A Matrix Eigenvalues', fontsize=13, fontweight='bold')
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(fontsize=9, loc='best')

        # Plot B matrices
        ax_b = axes[idx, 1]

        layer_names_b = sorted(ab1.get('ab_eigenvalues', {}).get('B', {}).keys())

        for layer_name in layer_names_b:
            if layer_name in ab1['ab_eigenvalues']['B']:
                eig_series1 = ab1['ab_eigenvalues']['B'][layer_name]
                iters1 = ab1.get('iterations', [])[:len(eig_series1)]
                mean_eigs1 = [np.mean(np.abs(np.real(eigs))) for eigs in eig_series1]
                ax_b.plot(iters1, mean_eigs1, '-o', color=color1, linewidth=2,
                         label=f'{labels[0]} - {layer_name}', markersize=6, alpha=0.8)

            if layer_name in ab2['ab_eigenvalues']['B']:
                eig_series2 = ab2['ab_eigenvalues']['B'][layer_name]
                iters2 = ab2.get('iterations', [])[:len(eig_series2)]
                mean_eigs2 = [np.mean(np.abs(np.real(eigs))) for eigs in eig_series2]
                ax_b.plot(iters2, mean_eigs2, '--s', color=color2, linewidth=2,
                         label=f'{labels[1]} - {layer_name}', markersize=6, alpha=0.8)

        ax_b.set_xlabel('AB Training Iteration', fontsize=12)
        ax_b.set_ylabel('Mean Eigenvalue Magnitude', fontsize=12)
        ax_b.set_title(f'Task {task_id} - B Matrix Eigenvalues', fontsize=13, fontweight='bold')
        ax_b.grid(True, alpha=0.3)
        ax_b.legend(fontsize=9, loc='best')

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_ab_eigenvalue_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_task_by_task_comparison(run1: Dict[str, Any], run2: Dict[str, Any],
                                  labels: List[str], output_dir: str):
    """Compare main training between AWB and non-AWB runs task-by-task.

    Added by Claude: Uses new task-based structure for fair comparison.
    Compares only the 'main_training' phase of each task, excluding
    preliminary and AB training phases.

    Args:
        run1, run2: Record dicts from two runs
        labels: Labels for the two runs
        output_dir: Directory to save plots
    """
    # Import at function level to avoid circular import
    from plot_results import extract_task_based_series

    # Check if new structure exists
    if 'tasks' not in run1 or 'tasks' not in run2:
        print("Warning: Task-based structure not found. Skipping task-by-task comparison.")
        return

    metadata1 = run1['metadata']
    n_tasks = metadata1.get('n_tasks', 5)
    color1, color2 = COLOR_PAIRS['primary']

    # Create subplots for each task
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Task-by-Task Main Training Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Plot each task's main training
    for task_id in range(min(n_tasks, 6)):  # Limit to 6 tasks for visualization
        if task_id >= len(axes):
            break

        ax = axes[task_id]

        # Extract main training data for this task
        series1 = extract_task_based_series(run1, task_id, 'main_training')
        series2 = extract_task_based_series(run2, task_id, 'main_training')

        if not series1 or not series2:
            ax.text(0.5, 0.5, f'Task {task_id}\nNo data', ha='center', va='center')
            ax.set_title(f'Task {task_id}')
            continue

        # Plot V loss (primary loss) using within-task epochs
        if 'loss_V' in series1 and 'loss_V' in series2:
            ax.plot(series1['epochs'], series1['loss_V'], color=color1,
                   linewidth=2.5, label=labels[0], alpha=0.9)
            ax.plot(series2['epochs'], series2['loss_V'], color=color2,
                   linewidth=2.5, label=labels[1], alpha=0.9)

        ax.set_xlabel('Epoch (within task)', fontsize=10)
        ax.set_ylabel(f'V ({metadata1["loss_function"]})', fontsize=10)
        ax.set_title(f'Task {task_id} - Main Training', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    filename = f'{metadata1["dataset"]}_task_by_task_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare two continual learning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare sine vs sine_awb
    python scripts/compare_runs.py \\
        outputs/sine_model/regression_sine_fcnn_run0_records.pkl \\
        outputs/sine_awb_model/regression_sine_fcnn_awb_run0_records.pkl \\
        --labels "Baseline" "AWB"

    # With custom output directory
    python scripts/compare_runs.py run1.pkl run2.pkl \\
        --labels "Method A" "Method B" \\
        --output-dir figures/comparison
        """
    )
    parser.add_argument('run1', type=str, help='Path to first run records.pkl')
    parser.add_argument('run2', type=str, help='Path to second run records.pkl')
    parser.add_argument('--labels', type=str, nargs=2, default=['Run 1', 'Run 2'],
                       help='Labels for the two runs (default: "Run 1" "Run 2")')
    parser.add_argument('--output-dir', type=str, default='figures/comparison',
                       help='Output directory for comparison plots')
    parser.add_argument('--smooth', type=int, default=5,
                       help='Smoothing window size (default: 5, 0 to disable)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')
    print(f'Smoothing window: {args.smooth}')

    # Load records
    print(f'\nLoading run 1: {args.run1}')
    run1_data = load_records(args.run1)

    print(f'Loading run 2: {args.run2}')
    run2_data = load_records(args.run2)

    # Generate comparison plots with smoothing
    print(f'\nGenerating comparison plots...')
    plot_loss_comparison(run1_data, run2_data, args.labels, args.output_dir, args.smooth)
    plot_metrics_comparison(run1_data, run2_data, args.labels, args.output_dir, args.smooth)
    plot_final_metrics_bar(run1_data, run2_data, args.labels, args.output_dir)
    plot_combined_comparison(run1_data, run2_data, args.labels, args.output_dir, args.smooth)

    # Added by Claude: Generate task-by-task comparison using new structure
    plot_task_by_task_comparison(run1_data, run2_data, args.labels, args.output_dir)

    # Added by Claude: Generate AB eigenvalue comparison if AWB is enabled
    plot_ab_eigenvalue_comparison(run1_data, run2_data, args.labels, args.output_dir)

    # Print summary statistics
    print_summary_statistics(run1_data, run2_data, args.labels)

    print(f'\n✓ All comparison plots generated in {args.output_dir}/')


if __name__ == '__main__':
    main()
