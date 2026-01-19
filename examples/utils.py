"""
Utility functions for example notebooks.

This module provides helper functions for:
- Loading experiment results from pickle files
- Computing continual learning metrics (Average Performance, BWT, Forgetting)
- Creating comparison plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def load_results(pkl_path: str) -> Dict:
    """Load experiment results from pickle file.

    Args:
        pkl_path: Path to the pickle file containing experiment results

    Returns:
        Dictionary containing:
        - 'metadata': Experiment configuration (n_tasks, epochs_per_task, etc.)
        - 'tasks': Per-task training metrics (H, V, grad_norm, metrics)
        - 'task_performance_matrix': Performance matrix for CL metrics
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_cl_metrics(data: Dict, is_regression: bool = False) -> Dict:
    """Compute continual learning metrics from experiment data.

    Args:
        data: Loaded experiment data dictionary
        is_regression: If True, compute MSE-based metrics; otherwise accuracy-based

    Returns:
        Dictionary containing:
        - 'avg_metric': Average final performance across all tasks
        - 'bwt': Backward Transfer (how much performance changed on old tasks)
        - 'forgetting': Maximum forgetting across tasks
    """
    matrix = data.get('task_performance_matrix', {})

    if not matrix:
        return {'avg_metric': 0.0, 'bwt': 0.0, 'forgetting': 0.0}

    # Convert to numpy array
    n_tasks = len(matrix)
    perf_matrix = np.zeros((n_tasks, n_tasks))

    for j in range(n_tasks):
        for i in range(n_tasks):
            i_str = str(i)
            if i_str in matrix.get(j, {}):
                perf_matrix[j, i] = matrix[j][i_str]

    # Average final performance (last row)
    avg_metric = np.mean(perf_matrix[-1, :])

    # Backward Transfer: average change in performance on old tasks
    # BWT = (1/(T-1)) * sum(A_{T,i} - A_{i,i}) for i = 1 to T-1
    if n_tasks > 1:
        bwt = np.mean([perf_matrix[-1, i] - perf_matrix[i, i] for i in range(n_tasks - 1)])
    else:
        bwt = 0.0

    # Forgetting: average maximum drop in performance
    # For each task i, forgetting = max(A_{j,i}) - A_{T,i} where j <= T
    if n_tasks > 1:
        forgetting_per_task = []
        for i in range(n_tasks - 1):
            max_perf = np.max(perf_matrix[:, i])
            final_perf = perf_matrix[-1, i]
            forgetting_per_task.append(max(0, max_perf - final_perf))
        forgetting = np.mean(forgetting_per_task)
    else:
        forgetting = 0.0

    return {
        'avg_metric': avg_metric,
        'bwt': bwt,
        'forgetting': forgetting
    }


def extract_training_curves(data: Dict, metric_key: str = 'test_experience') -> Tuple[np.ndarray, np.ndarray]:
    """Extract training curves from experiment data.

    Args:
        data: Loaded experiment data dictionary
        metric_key: Which metric to extract ('H', 'V', 'grad_norm', 'test_current', 'test_experience')

    Returns:
        Tuple of (iterations, values) arrays
    """
    tasks = data.get('tasks', {})

    all_iters = []
    all_values = []
    offset = 0

    for task_id in sorted(tasks.keys()):
        task_data = tasks[task_id]
        main_training = task_data.get('main_training', {})

        iters = main_training.get('iterations', [])
        values = main_training.get(metric_key, [])

        if len(iters) > 0 and len(values) > 0:
            all_iters.extend([i + offset for i in iters])
            all_values.extend(values)
            offset = all_iters[-1] + 1 if all_iters else 0

    return np.array(all_iters), np.array(all_values)


def plot_comparison(baseline_data: Dict, awb_data: Dict,
                   metric_type: str = 'accuracy',
                   title_prefix: str = '',
                   figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """Create a 4-panel comparison plot between Baseline and AWB methods.

    Args:
        baseline_data: Loaded data for baseline method
        awb_data: Loaded data for AWB method
        metric_type: 'accuracy' or 'mse' (affects scaling and labels)
        title_prefix: Prefix for the figure title (e.g., 'Sine', 'MNIST')
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Colors
    baseline_color = '#1f77b4'  # Blue
    awb_color = '#ff7f0e'  # Orange

    # Panel (a): Test metric over training
    ax = axes[0, 0]
    metric_key = 'test_experience'

    iters_b, vals_b = extract_training_curves(baseline_data, metric_key)
    iters_a, vals_a = extract_training_curves(awb_data, metric_key)

    ax.plot(iters_b, vals_b, color=baseline_color, label='Baseline', alpha=0.8)
    ax.plot(iters_a, vals_a, color=awb_color, label='AWB Full', alpha=0.8)

    ax.set_xlabel('Iteration')
    if metric_type == 'mse':
        ax.set_ylabel('Test MSE (Experience)')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Test Accuracy (Experience)')
        ax.set_ylim(0, 1.05)
    ax.set_title('(a) Test Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (b): Hamiltonian loss
    ax = axes[0, 1]
    iters_b, vals_b = extract_training_curves(baseline_data, 'H')
    iters_a, vals_a = extract_training_curves(awb_data, 'H')

    ax.plot(iters_b, vals_b, color=baseline_color, label='Baseline', alpha=0.8)
    ax.plot(iters_a, vals_a, color=awb_color, label='AWB Full', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hamiltonian Loss')
    ax.set_yscale('log')
    ax.set_title('(b) Hamiltonian Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (c): Gradient norm
    ax = axes[1, 0]
    iters_b, vals_b = extract_training_curves(baseline_data, 'grad_norm')
    iters_a, vals_a = extract_training_curves(awb_data, 'grad_norm')

    ax.plot(iters_b, vals_b, color=baseline_color, label='Baseline', alpha=0.8)
    ax.plot(iters_a, vals_a, color=awb_color, label='AWB Full', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.set_title('(c) Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (d): Bar chart of CL metrics
    ax = axes[1, 1]

    is_regression = (metric_type == 'mse')
    metrics_b = compute_cl_metrics(baseline_data, is_regression)
    metrics_a = compute_cl_metrics(awb_data, is_regression)

    x = np.arange(3)
    width = 0.35

    if metric_type == 'mse':
        labels = ['Avg MSE', 'BWT', 'Forgetting']
        vals_baseline = [metrics_b['avg_metric'], abs(metrics_b['bwt']), metrics_b['forgetting']]
        vals_awb = [metrics_a['avg_metric'], abs(metrics_a['bwt']), metrics_a['forgetting']]
    else:
        labels = ['Avg Accuracy', 'BWT', 'Forgetting']
        vals_baseline = [metrics_b['avg_metric'], metrics_b['bwt'], metrics_b['forgetting']]
        vals_awb = [metrics_a['avg_metric'], metrics_a['bwt'], metrics_a['forgetting']]

    bars1 = ax.bar(x - width/2, vals_baseline, width, label='Baseline', color=baseline_color)
    bars2 = ax.bar(x + width/2, vals_awb, width, label='AWB Full', color=awb_color)

    ax.set_ylabel('Value')
    ax.set_title('(d) Continual Learning Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'{title_prefix} Continual Learning: Baseline vs AWB', fontsize=14)
    plt.tight_layout()

    return fig


def add_task_boundaries(ax: plt.Axes, n_tasks: int, epochs_per_task: int,
                       iterations_per_epoch: int = 1, linestyle: str = '--',
                       color: str = 'gray', alpha: float = 0.5):
    """Add vertical lines at task boundaries.

    Args:
        ax: Matplotlib axes object
        n_tasks: Number of tasks
        epochs_per_task: Epochs per task
        iterations_per_epoch: Number of iterations per epoch
        linestyle: Line style for boundaries
        color: Line color
        alpha: Line transparency
    """
    for task in range(1, n_tasks):
        x = task * epochs_per_task * iterations_per_epoch
        ax.axvline(x=x, linestyle=linestyle, color=color, alpha=alpha)


def print_experiment_summary(data: Dict, name: str = 'Experiment'):
    """Print a summary of the experiment results.

    Args:
        data: Loaded experiment data
        name: Name to display in the summary
    """
    metadata = data.get('metadata', {})

    print(f"\n{'='*50}")
    print(f" {name} Summary")
    print(f"{'='*50}")
    print(f"Tasks: {metadata.get('n_tasks', 'N/A')}")
    print(f"Epochs per task: {metadata.get('epochs_per_task', 'N/A')}")
    print(f"Problem type: {metadata.get('problem', 'N/A')}")
    print(f"Network: {metadata.get('network', 'N/A')}")
    print(f"AWB enabled: {metadata.get('awb_enabled', False)}")

    # Compute and print CL metrics
    is_regression = metadata.get('problem', '') == 'regression'
    metrics = compute_cl_metrics(data, is_regression)

    print(f"\nContinual Learning Metrics:")
    if is_regression:
        print(f"  Average MSE: {metrics['avg_metric']:.6f}")
    else:
        print(f"  Average Accuracy: {metrics['avg_metric']:.4f}")
    print(f"  Backward Transfer: {metrics['bwt']:.4f}")
    print(f"  Forgetting: {metrics['forgetting']:.4f}")
    print(f"{'='*50}\n")
