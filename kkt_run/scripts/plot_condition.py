#!/usr/bin/env python
"""
Per-condition plotting for kkt_run tests.

Generates:
1. Standard training plots (losses, metrics, eigenvalues)
2. CL metrics visualization (ACC, BWT, Forgetting, FWT)
3. Task performance matrix heatmap

Usage:
    python plot_condition.py --records path/to/records.pkl --output-dir path/to/plots
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_records(filepath: str) -> Dict[str, Any]:
    """Load records from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_performance_matrix(record_dict: Dict) -> np.ndarray:
    """Extract performance matrix A[j][i] from record_dict.

    A[j][i] = performance on task i after training task j
    """
    task_perf = record_dict.get('task_performance_matrix', {})

    if not task_perf:
        print("Warning: No task_performance_matrix found")
        return None

    n_tasks = len(task_perf)
    matrix = np.zeros((n_tasks, n_tasks))

    for j_str, perf_dict in task_perf.items():
        j = int(j_str)
        for i_str, perf_value in perf_dict.items():
            i = int(i_str)
            matrix[j, i] = perf_value

    return matrix


def compute_cl_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """Compute all CL metrics from performance matrix."""
    T = matrix.shape[0]

    # Average Accuracy
    ACC = np.mean(matrix[T-1, :])

    # Backward Transfer
    if T == 1:
        BWT = 0.0
    else:
        BWT = np.mean([matrix[T-1, i] - matrix[i, i] for i in range(T-1)])

    # Forgetting
    if T == 1:
        Forgetting = 0.0
    else:
        forgetting_per_task = []
        for i in range(T-1):
            max_forget = max([matrix[i, i] - matrix[j, i] for j in range(i, T)])
            forgetting_per_task.append(max_forget)
        Forgetting = np.mean(forgetting_per_task)

    # Forward Transfer
    if T == 1:
        FWT = 0.0
    else:
        FWT = np.mean([matrix[i-1, i] for i in range(1, T)])

    return {
        'ACC': ACC,
        'BWT': BWT,
        'Forgetting': Forgetting,
        'FWT': FWT,
        'forgetting_per_task': forgetting_per_task if T > 1 else []
    }


def plot_cl_metrics(metrics: Dict[str, float], metadata: Dict, output_path: Path):
    """Plot CL metrics in 4-panel layout."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    condition = metadata.get('condition', 'unknown')
    dataset = metadata.get('dataset', 'unknown')

    fig.suptitle(f'Continual Learning Metrics - {dataset} ({condition})',
                 fontsize=14, fontweight='bold')

    # Panel 1: ACC
    axes[0, 0].bar([0], [metrics['ACC']], color='green', alpha=0.7, width=0.5)
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Average Accuracy (ACC): {metrics["ACC"]:.4f}')
    axes[0, 0].set_xticks([])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=metrics['ACC'], color='green', linestyle='--', alpha=0.5)

    # Panel 2: BWT
    color = 'green' if metrics['BWT'] >= 0 else 'red'
    axes[0, 1].bar([0], [metrics['BWT']], color=color, alpha=0.7, width=0.5)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].set_ylabel('Backward Transfer')
    axes[0, 1].set_title(f'Backward Transfer (BWT): {metrics["BWT"]:.4f}')
    axes[0, 1].set_xticks([])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Forgetting (per-task bars)
    if len(metrics['forgetting_per_task']) > 0:
        tasks = list(range(len(metrics['forgetting_per_task'])))
        axes[1, 0].bar(tasks, metrics['forgetting_per_task'], alpha=0.7)
        axes[1, 0].axhline(y=metrics['Forgetting'], color='red',
                          linestyle='--', linewidth=2, label=f'Avg: {metrics["Forgetting"]:.4f}')
        axes[1, 0].set_xlabel('Task')
        axes[1, 0].set_ylabel('Forgetting')
        axes[1, 0].set_title(f'Per-Task Forgetting (Avg: {metrics["Forgetting"]:.4f})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'N/A (single task)',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Per-Task Forgetting')

    # Panel 4: FWT
    color = 'green' if metrics['FWT'] >= 0 else 'orange'
    axes[1, 1].bar([0], [metrics['FWT']], color=color, alpha=0.7, width=0.5)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].set_ylabel('Forward Transfer')
    axes[1, 1].set_title(f'Forward Transfer (FWT): {metrics["FWT"]:.4f}')
    axes[1, 1].set_xticks([])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_performance_matrix(matrix: np.ndarray, metadata: Dict, output_path: Path):
    """Plot task performance matrix as heatmap."""
    T = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    condition = metadata.get('condition', 'unknown')
    dataset = metadata.get('dataset', 'unknown')

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1.0, cbar_kws={'label': 'Performance'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('Task i (tested on)', fontsize=12)
    ax.set_ylabel('After training task j', fontsize=12)
    ax.set_title(f'Task Performance Matrix - {dataset} ({condition})',
                 fontsize=14, fontweight='bold')

    # Highlight diagonal
    for i in range(T):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='blue', lw=3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_standard_metrics(records: Dict, output_dir: Path):
    """Generate standard training plots (losses, metrics, eigenvalues)."""
    # Extract time series
    iterations = sorted(records['iterations'].keys())

    series = {
        'iterations': np.array(iterations),
        'task_ids': np.array([records['iterations'][i]['task_id'] for i in iterations]),
    }

    # Extract losses
    for loss_key in ['H', 'V', 'dV', 'dV_dx', 'dV_dtheta']:
        series[f'loss_{loss_key}'] = np.array([
            records['iterations'][i]['losses'][loss_key] for i in iterations
        ])

    # Extract gradients
    series['grad_norm'] = np.array([
        records['iterations'][i]['gradients']['grad_norm'] for i in iterations
    ])

    # Extract metrics
    series['metric_train'] = np.array([
        records['iterations'][i]['metrics']['train'] for i in iterations
    ])
    series['metric_test_current'] = np.array([
        records['iterations'][i]['metrics']['test_current'] for i in iterations
    ])
    series['metric_test_experience'] = np.array([
        records['iterations'][i]['metrics']['test_experience'] for i in iterations
    ])

    metadata = records['metadata']
    condition = metadata.get('condition', 'unknown')
    dataset = metadata.get('dataset', 'unknown')

    # Plot 1: Losses
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Loss Components - {dataset} ({condition})', fontsize=14, fontweight='bold')

    axes[0, 0].plot(series['iterations'], series['loss_H'], 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('H (Hamiltonian)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(series['iterations'], series['loss_V'], 'r-', linewidth=1.5)
    axes[0, 1].set_ylabel('V')
    axes[0, 1].set_title('Experience Replay Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(series['iterations'], series['loss_dV'], 'g-', linewidth=1.5)
    axes[0, 2].set_ylabel('dV')
    axes[0, 2].set_title('Regularization Term')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(series['iterations'], series['loss_dV_dx'], 'c-', linewidth=1.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('dV/dx')
    axes[1, 0].set_title('Input Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(series['iterations'], series['loss_dV_dtheta'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('dV/dθ')
    axes[1, 1].set_title('Parameter Sensitivity')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(series['iterations'], series['grad_norm'], 'orange', linewidth=1.5)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Gradient Norm')
    axes[1, 2].set_title('Gradient Magnitude')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'losses.png'}")

    # Plot 2: Metrics
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(series['iterations'], series['metric_train'], label='Train', linewidth=2)
    ax.plot(series['iterations'], series['metric_test_current'], label='Test (Current)', linewidth=2)
    ax.plot(series['iterations'], series['metric_test_experience'], label='Test (Experience)', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title(f'Training Metrics - {dataset} ({condition})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots for a single condition')
    parser.add_argument('--records', required=True, help='Path to records.pkl file')
    parser.add_argument('--output-dir', required=True, help='Output directory for plots')
    args = parser.parse_args()

    # Load records
    records_path = Path(args.records)
    if not records_path.exists():
        print(f"Error: Records file not found: {records_path}")
        return 1

    print(f"Loading records from: {records_path}")
    records = load_records(records_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots in: {output_dir}")

    # Generate standard plots
    print("\n1. Standard training plots...")
    plot_standard_metrics(records, output_dir)

    # Extract performance matrix
    matrix = extract_performance_matrix(records)

    if matrix is not None:
        # Compute CL metrics
        print("\n2. CL metrics...")
        metrics = compute_cl_metrics(matrix)
        plot_cl_metrics(metrics, records['metadata'], output_dir / 'cl_metrics.png')

        # Plot performance matrix
        print("\n3. Performance matrix heatmap...")
        plot_performance_matrix(matrix, records['metadata'], output_dir / 'performance_matrix.png')
    else:
        print("\nWarning: Skipping CL plots (no performance matrix found)")

    print(f"\n✓ All plots generated successfully!")
    print(f"  Location: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
