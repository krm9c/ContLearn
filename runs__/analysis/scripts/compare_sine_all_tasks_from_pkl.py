#!/usr/bin/env python3
"""
Compare Sine ALL TASKS (10 tasks) loss curves across all 4 experimental conditions using pkl files.
Loads training metrics from pkl records and generates comparison plots.
Uses ITERATIONS on x-axis with task boundaries.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Data directory - UPDATE THIS TO POINT TO sine_all_Tasks
DATA_DIR = Path("runs__/analysis/data_analysis/sine_all_Tasks")

CONDITIONS = {
    'C1: Baseline': 'sine_condition1_run0/regression_sine_fcnn_run0_records.pkl',
    'C2: Heuristics': 'sine_condition2_run0/regression_sine_fcnn_run0_records.pkl',
    'C3: Arch Search': 'sine_condition3_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
    'C4: AWB Full': 'sine_condition4_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
}


def load_metrics_from_pkl(pkl_path):
    """Load and concatenate metrics from all tasks in pkl file with epoch information."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract metadata
    metadata = data['metadata']
    save_iter = metadata.get('save_iter', 50)
    epochs_per_task = metadata.get('epochs_per_task', 500)
    n_tasks = metadata['n_tasks']

    # Extract metrics across all tasks
    metrics = defaultdict(list)
    epochs = []
    task_boundaries_epochs = []

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            print(f"  Warning: Task {task_id} not found in data")
            continue

        task_data = data['tasks'][task_id]

        # Mark task boundary (at the START of each new task, except task 0)
        if task_id > 0:
            task_boundaries_epochs.append(task_id * epochs_per_task)

        # Extract main training data
        if 'main_training' in task_data:
            training = task_data['main_training']

            # Map pkl keys to our metric names
            metric_map = {
                'H': 'H',
                'V': 'V',
                'dV': 'dV',
                'grad_norm': 'grad_norm',
                'train_metric': 'train_mse',
                'test_current': 'test_cur',
                'test_experience': 'test_exp',
            }

            for pkl_key, metric_name in metric_map.items():
                if pkl_key in training:
                    values = training[pkl_key]
                    metrics[metric_name].extend(values)

            # Track epochs based on iterations (iterations are global, already absolute)
            if 'iterations' in training:
                iters = training['iterations']
                epochs.extend(iters)

    return metrics, np.array(epochs), task_boundaries_epochs, n_tasks, data


def compute_cl_metrics(data):
    """Compute ACC, BWT, FWT from task_performance_matrix.

    For regression (MSE), we flip signs so positive BWT = improvement.
    Standard CL metrics assume higher = better, but MSE has lower = better.
    """
    if 'task_performance_matrix' not in data:
        return None

    tpm = data['task_performance_matrix']

    # Handle dict-of-dicts format: {task_j: {task_i: value}}
    if isinstance(tpm, dict):
        n_tasks = len(tpm)
        matrix = np.zeros((n_tasks, n_tasks))
        for j in range(n_tasks):
            if j in tpm:
                for i_str, val in tpm[j].items():
                    i = int(i_str)
                    matrix[j, i] = val
    else:
        matrix = np.array(tpm)

    if matrix.size == 0:
        return None

    n_tasks = matrix.shape[0]

    # ACC: Average of final row (lower = better for MSE)
    acc = np.mean(matrix[-1, :])

    # BWT: Backward transfer (SIGN FLIPPED for MSE)
    # BWT = R_{i,i} - R_{T,i} so positive = MSE decreased = good
    bwt_values = []
    for i in range(n_tasks - 1):
        bwt_values.append(matrix[i, i] - matrix[-1, i])  # FLIPPED SIGN
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # FWT: Forward transfer (SIGN FLIPPED for MSE)
    # FWT = baseline - R_{i-1,i} so positive = better than baseline
    fwt_values = []
    baseline = matrix[0, 0]  # Task 0 performance as baseline
    for i in range(1, n_tasks):
        fwt_values.append(baseline - matrix[i-1, i])  # FLIPPED SIGN
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting: How much MSE increased from best to final (CORRECTED)
    # Forgetting = final_MSE - min_MSE (positive = bad)
    forgetting_values = []
    for i in range(n_tasks - 1):
        min_mse = np.min(matrix[:, i])  # Best (minimum) MSE achieved
        final_mse = matrix[-1, i]       # Final MSE
        forgetting_values.append(max(0, final_mse - min_mse))  # CORRECTED
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'ACC': acc,
        'BWT': bwt,
        'FWT': fwt,
        'Forgetting': forgetting,
        'matrix': matrix,
    }


def plot_comparison(all_metrics, all_epochs, all_boundaries, all_cl_metrics, output_dir):
    """Generate comparison plots across conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define color palette
    colors = sns.color_palette("husl", len(CONDITIONS))

    # 1. Loss Components Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sine (10 Tasks): Loss Components Across Conditions', fontsize=16, fontweight='bold')

    loss_metrics = [
        ('H', 'Hamiltonian (Total)'),
        ('V', 'Experience Replay Loss'),
        ('dV', 'Regularization (dV)'),
        ('grad_norm', 'Gradient Norm'),
    ]

    for idx, (metric, title) in enumerate(loss_metrics):
        ax = axes[idx // 2, idx % 2]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name] and len(all_metrics[cond_name][metric]) > 0:
                values = all_metrics[cond_name][metric]
                epochs = all_epochs[cond_name]
                ax.plot(epochs, values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries (using C1 as reference)
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Task Boundary' if boundary == ref_boundaries[0] else '')

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if metric not in ['dV']:  # dV can be negative
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_all_tasks_loss_comparison.png'}")
    plt.close()

    # 2. MSE Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sine (10 Tasks): MSE Metrics Across Conditions', fontsize=16, fontweight='bold')

    mse_metrics = [
        ('train_mse', 'Training MSE'),
        ('test_cur', 'Test MSE (Current Task)'),
        ('test_exp', 'Test MSE (Experience Replay)'),
    ]

    for idx, (metric, title) in enumerate(mse_metrics):
        ax = axes[idx]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name] and len(all_metrics[cond_name][metric]) > 0:
                values = all_metrics[cond_name][metric]
                epochs = all_epochs[cond_name]
                ax.plot(epochs, values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_mse_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_all_tasks_mse_comparison.png'}")
    plt.close()

    # 3. Gradient Norm Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle('Sine (10 Tasks): Gradient Norm Across Conditions', fontsize=16, fontweight='bold')

    for (cond_name, _), color in zip(CONDITIONS.items(), colors):
        if 'grad_norm' in all_metrics[cond_name] and len(all_metrics[cond_name]['grad_norm']) > 0:
            values = all_metrics[cond_name]['grad_norm']
            epochs = all_epochs[cond_name]
            ax.plot(epochs, values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

    # Add task boundaries
    ref_boundaries = all_boundaries['C1: Baseline']
    for boundary in ref_boundaries:
        ax.axvline(boundary, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm', fontsize=11)
    ax.set_title('Gradient Norm Evolution', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_gradient_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_all_tasks_gradient_comparison.png'}")
    plt.close()

    # 4. Combined Overview (for paper)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Sine (10 Tasks): Training Dynamics Comparison', fontsize=14, fontweight='bold')

    overview_metrics = [
        ('H', 'Hamiltonian Loss'),
        ('train_mse', 'Training MSE'),
        ('test_exp', 'Test MSE (Experience Replay)'),
        ('grad_norm', 'Gradient Norm'),
    ]

    for idx, (metric, title) in enumerate(overview_metrics):
        ax = axes[idx // 2, idx % 2]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name] and len(all_metrics[cond_name][metric]) > 0:
                values = all_metrics[cond_name][metric]
                epochs = all_epochs[cond_name]
                ax.plot(epochs, values, label=cond_name, alpha=0.8, linewidth=2, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if metric not in ['dV']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sine_all_tasks_overview_comparison.pdf', bbox_inches='tight')  # For paper
    print(f"Saved: {output_dir / 'sine_all_tasks_overview_comparison.png'}")
    print(f"Saved: {output_dir / 'sine_all_tasks_overview_comparison.pdf'}")
    plt.close()

    # 5. CL Metrics Bar Chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Sine (10 Tasks): Continual Learning Metrics Comparison', fontsize=14, fontweight='bold')

    metric_names = ['ACC', 'BWT', 'FWT', 'Forgetting']
    metric_labels = [
        'Avg MSE\n(lower better)',
        'BWT\n(positive = improvement)',
        'FWT\n(positive = helped)',
        'Forgetting\n(lower better)'
    ]
    cond_names_short = ['C1', 'C2', 'C3', 'C4']

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]

        values = []
        for cond_name in CONDITIONS.keys():
            if cond_name in all_cl_metrics and all_cl_metrics[cond_name] is not None:
                values.append(all_cl_metrics[cond_name][metric_name])
            else:
                values.append(0)

        bars = ax.bar(cond_names_short, values, color=colors)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_label, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_cl_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_all_tasks_cl_metrics_comparison.png'}")
    plt.close()

    # 6. Performance Matrix Heatmaps
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Sine (10 Tasks): Task Performance Matrix (Lower MSE = Better)', fontsize=14, fontweight='bold')

    for idx, cond_name in enumerate(CONDITIONS.keys()):
        ax = axes[idx]

        if cond_name in all_cl_metrics and all_cl_metrics[cond_name] is not None:
            matrix = all_cl_metrics[cond_name]['matrix']

            # For regression, lower is better, so we plot MSE directly
            # Mask upper triangle (future tasks not yet trained)
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

            im = ax.imshow(np.where(mask, np.nan, matrix), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.03)
            ax.set_title(f'{cond_name}\nAvg MSE: {all_cl_metrics[cond_name]["ACC"]:.4f}', fontsize=10)
            ax.set_xlabel('Task ID', fontsize=9)
            ax.set_ylabel('Trained Through Task', fontsize=9)

            # Set ticks
            n = matrix.shape[0]
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))

            # Add text annotations only for lower triangle
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if j <= i:  # Lower triangle + diagonal
                        text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=7)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_all_tasks_performance_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sine_all_tasks_performance_matrix.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_all_tasks_performance_matrix.png'}")
    print(f"Saved: {output_dir / 'sine_all_tasks_performance_matrix.pdf'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("Sine ALL TASKS (10 tasks) Condition Comparison Analysis")
    print("Updated with ITERATION x-axis and task boundaries")
    print("="*70)

    all_metrics = {}
    all_epochs = {}
    all_boundaries = {}
    all_num_tasks = {}
    all_data = {}
    all_cl_metrics = {}

    # Load metrics from all conditions
    for cond_name, pkl_file in CONDITIONS.items():
        pkl_path = DATA_DIR / pkl_file
        print(f"\nProcessing: {cond_name}")
        print(f"  File: {pkl_path}")

        if not pkl_path.exists():
            print(f"  ⚠️  PKL file not found!")
            continue

        metrics, epochs, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path)
        all_metrics[cond_name] = metrics
        all_epochs[cond_name] = epochs
        all_boundaries[cond_name] = boundaries
        all_num_tasks[cond_name] = num_tasks
        all_data[cond_name] = data

        # Compute CL metrics
        cl_metrics = compute_cl_metrics(data)
        all_cl_metrics[cond_name] = cl_metrics

        print(f"  Tasks: {num_tasks}")
        print(f"  Total checkpoints: {len(metrics.get('H', []))}")
        print(f"  Task boundaries (iterations): {boundaries}")
        if cl_metrics:
            print(f"  Avg MSE: {cl_metrics['ACC']:.4f}, BWT: {cl_metrics['BWT']:.4f} (+good), FWT: {cl_metrics['FWT']:.4f} (+good), Forgetting: {cl_metrics['Forgetting']:.4f} (lower good)")

        # Show available metrics
        available = [k for k, v in metrics.items() if len(v) > 0]
        print(f"  Metrics extracted: {', '.join(available)}")

    # Print summary table
    print("\n" + "="*70)
    print("CONTINUAL LEARNING METRICS SUMMARY (10 Tasks)")
    print("="*70)
    print(f"{'Condition':<20} {'Avg MSE (↓)':<15} {'BWT (↑)':<15} {'FWT (↑)':<15} {'Forgetting (↓)':<15}")
    print("-"*70)
    for cond_name in CONDITIONS.keys():
        if cond_name in all_cl_metrics and all_cl_metrics[cond_name] is not None:
            m = all_cl_metrics[cond_name]
            print(f"{cond_name:<20} {m['ACC']:<15.4f} {m['BWT']:<15.4f} {m['FWT']:<15.4f} {m['Forgetting']:<15.4f}")
    print("="*70)

    # Generate plots
    print("\n" + "="*70)
    print("Generating Comparison Plots")
    print("="*70)

    output_dir = DATA_DIR / "sine_all_tasks_comparison_plots"
    plot_comparison(all_metrics, all_epochs, all_boundaries, all_cl_metrics, output_dir)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. sine_all_tasks_loss_comparison.png - Loss components")
    print("  2. sine_all_tasks_mse_comparison.png - MSE metrics")
    print("  3. sine_all_tasks_gradient_comparison.png - Gradient norms")
    print("  4. sine_all_tasks_overview_comparison.png/pdf - Combined overview")
    print("  5. sine_all_tasks_cl_metrics_comparison.png - CL metrics bar chart")
    print("  6. sine_all_tasks_performance_matrix.png/pdf - Task performance matrices")


if __name__ == '__main__':
    main()
