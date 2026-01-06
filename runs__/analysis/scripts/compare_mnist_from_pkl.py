#!/usr/bin/env python3
"""
Compare MNIST accuracy curves across all 4 experimental conditions using pkl files.
Loads training metrics from pkl records and generates comparison plots.
Ignores warmup phases - only plots main_training data.
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

# Data directory - UPDATE TO MNIST FOLDER
DATA_DIR = Path("runs__/analysis/data_analysis/mnist")
OUTPUT_DIR = DATA_DIR / "mnist_comparison_plots"

COLORS = {
    'C1: Baseline': '#E74C3C',      # Red
    'C2: Heuristics': '#F39C12',    # Orange
    'C3: Arch Search': '#3498DB',   # Blue
    'C4: AWB Full': '#9B59B6',      # Purple
}

CONDITIONS = {
    'C1: Baseline': 'mnist_condition1_run0/classification_mnist_cnn_run0_records.pkl',
    'C2: Heuristics': 'mnist_condition2_run0/classification_mnist_cnn_run0_records.pkl',
    'C3: Arch Search': 'mnist_condition3_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
    'C4: AWB Full': 'mnist_condition4_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
}


def load_metrics_from_pkl(pkl_path):
    """Load and concatenate metrics from all tasks in pkl file with iteration information."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract metadata
    metadata = data['metadata']
    epochs_per_task = metadata.get('epochs_per_task', 200)
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
                'train_metric': 'train_acc',
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

    For classification (accuracy), higher is better.
    NO sign flipping needed.
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

    # ACC: Average of final row (higher = better for accuracy)
    acc = np.mean(matrix[-1, :])

    # BWT: Backward transfer (NO SIGN FLIP for accuracy)
    # BWT = R_{T,i} - R_{i,i} so positive = accuracy increased = good
    bwt_values = []
    for i in range(n_tasks - 1):
        bwt_values.append(matrix[-1, i] - matrix[i, i])  # NO FLIP
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # FWT: Forward transfer (NO SIGN FLIP for accuracy)
    # FWT = R_{i-1,i} - baseline so positive = better than baseline
    fwt_values = []
    baseline = matrix[0, 0]  # Task 0 performance as baseline
    for i in range(1, n_tasks):
        if str(i) in tpm.get(i-1, {}):
            fwt_values.append(matrix[i-1, i] - baseline)  # NO FLIP
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting: How much accuracy decreased from best to final
    # Forgetting = best_acc - final_acc (positive = bad)
    forgetting_values = []
    for i in range(n_tasks - 1):
        best_acc = np.max(matrix[:, i])  # Best (maximum) accuracy achieved
        final_acc = matrix[-1, i]        # Final accuracy
        forgetting_values.append(max(0, best_acc - final_acc))
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'ACC': acc,
        'BWT': bwt,
        'FWT': fwt,
        'Forgetting': forgetting,
        'matrix': matrix,
    }


def plot_comparison(all_metrics, all_boundaries, output_dir):
    """Generate comparison plots across conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define color palette
    colors = sns.color_palette("husl", len(CONDITIONS))

    # 1. Loss Components Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('MNIST: Loss Components Across Conditions', fontsize=16, fontweight='bold')

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
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries (using C1 as reference)
        ref_boundaries = all_boundaries.get('C1: Baseline', [])
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if metric not in ['dV']:  # dV can be negative
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'mnist_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mnist_loss_comparison.png'}")
    plt.close()

    # 2. Accuracy Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MNIST: Accuracy Metrics Across Conditions', fontsize=16, fontweight='bold')

    acc_metrics = [
        ('train_acc', 'Training Accuracy'),
        ('test_cur', 'Test Accuracy (Current Task)'),
        ('test_exp', 'Test Accuracy (Experience Replay)'),
    ]

    for idx, (metric, title) in enumerate(acc_metrics):
        ax = axes[idx]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name] and len(all_metrics[cond_name][metric]) > 0:
                values = all_metrics[cond_name][metric]
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries.get('C1: Baseline', [])
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'mnist_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mnist_accuracy_comparison.png'}")
    plt.close()

    # 3. Gradient Norm Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle('MNIST: Gradient Norm Across Conditions', fontsize=16, fontweight='bold')

    for (cond_name, _), color in zip(CONDITIONS.items(), colors):
        if 'grad_norm' in all_metrics[cond_name] and len(all_metrics[cond_name]['grad_norm']) > 0:
            values = all_metrics[cond_name]['grad_norm']
            ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

    # Add task boundaries
    ref_boundaries = all_boundaries.get('C1: Baseline', [])
    for boundary in ref_boundaries:
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'mnist_gradient_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mnist_gradient_comparison.png'}")
    plt.close()

    # 4. Combined Overview (for paper)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('MNIST: Training Dynamics Comparison', fontsize=14, fontweight='bold')

    overview_metrics = [
        ('H', 'Hamiltonian Loss'),
        ('train_acc', 'Training Accuracy'),
        ('test_exp', 'Test Acc (Experience Replay)'),
        ('grad_norm', 'Gradient Norm'),
    ]

    for idx, (metric, title) in enumerate(overview_metrics):
        ax = axes[idx // 2, idx % 2]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name] and len(all_metrics[cond_name][metric]) > 0:
                values = all_metrics[cond_name][metric]
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=2, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries.get('C1: Baseline', [])
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.2, linewidth=1)

        ax.set_xlabel('Checkpoint', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

        if metric in ['train_acc', 'test_exp', 'test_cur']:
            ax.set_ylim([0, 1.05])
        elif metric not in ['dV']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'mnist_overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mnist_overview_comparison.pdf', bbox_inches='tight')  # For paper
    print(f"Saved: {output_dir / 'mnist_overview_comparison.png'}")
    print(f"Saved: {output_dir / 'mnist_overview_comparison.pdf'}")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("MNIST CLASSIFICATION - CONTINUAL LEARNING ANALYSIS")
    print("="*80 + "\n")

    # Load all data
    print("Loading experimental data...")
    pkl_data = {}
    all_training_data = {}
    all_cl_metrics = {}

    for cond_name, pkl_file in CONDITIONS.items():
        pkl_path = DATA_DIR / pkl_file
        if not pkl_path.exists():
            print(f"⚠ Warning: {pkl_path} not found, skipping {cond_name}")
            continue

        print(f"  Loading {cond_name}...")
        metrics, epochs, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path)
        pkl_data[cond_name] = data

        # Store training curves
        all_training_data[cond_name] = (boundaries, {'iterations': epochs, **metrics})

        # Compute CL metrics
        cl_metrics = compute_cl_metrics(data)
        all_cl_metrics[cond_name] = cl_metrics

        print(f"    Tasks: {num_tasks}")
        print(f"    Total checkpoints: {len(metrics.get('H', []))}")
        if cl_metrics:
            print(f"    Avg Acc: {cl_metrics['ACC']:.4f}, BWT: {cl_metrics['BWT']:.4f}, FWT: {cl_metrics['FWT']:.4f}, Forgetting: {cl_metrics['Forgetting']:.4f}")

    print(f"\n✓ Loaded {len(pkl_data)} conditions\n")

    # Print metrics summary
    print_metrics_summary(all_cl_metrics)

    # Generate plots
    print("Generating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison_new(all_training_data, all_cl_metrics, OUTPUT_DIR)

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


def print_metrics_summary(all_metrics):
    """Print formatted metrics table."""
    print("="*80)
    print("CONTINUAL LEARNING METRICS SUMMARY - MNIST (2 Tasks)")
    print("="*80)
    print(f"{'Condition':<20} {'Avg Acc (↑)':<15} {'BWT (↑)':<15} {'FWT (↑)':<15} {'Forgetting (↓)':<15}")
    print("-"*80)

    for cond_name in CONDITIONS.keys():
        if cond_name in all_metrics and all_metrics[cond_name] is not None:
            m = all_metrics[cond_name]
            print(f"{cond_name:<20} {m['ACC']:<15.4f} {m['BWT']:<15.4f} {m['FWT']:<15.4f} {m['Forgetting']:<15.4f}")

    print("="*80 + "\n")


def plot_comparison_new(all_data, all_cl_metrics, output_dir):
    """Generate comparison plots - updated version."""
    colors = list(COLORS.values())

    # 1. Loss Components
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    components = [('H', 'Hamiltonian (Total Loss)'), ('V', 'Experience Replay Loss'),
                  ('dV', 'Total Regularization'), ('grad_norm', 'Gradient Norm')]

    for idx, (comp_key, comp_label) in enumerate(components):
        ax = axes[idx]
        for cond_name, (boundaries, curves) in all_data.items():
            ax.plot(curves['iterations'], curves.get(comp_key, []),
                   label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
        if boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(comp_label, fontsize=12)
        ax.set_title(comp_label, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_components.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_components.pdf', bbox_inches='tight')
    print(f"✓ Saved: loss_components.png/pdf")
    plt.close()

    # 2. Accuracy Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('train_acc', 'Training Accuracy'), ('test_cur', 'Test Accuracy (Current Task)'),
               ('test_exp', 'Test Accuracy (Experience Replay)')]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        for cond_name, (boundaries, curves) in all_data.items():
            ax.plot(curves['iterations'], curves.get(metric_key, []),
                   label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
        if boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(metric_label, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_curves.pdf', bbox_inches='tight')
    print(f"✓ Saved: accuracy_curves.png/pdf")
    plt.close()

    # 3. CL Metrics Bar Chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_names = ['ACC', 'BWT', 'FWT', 'Forgetting']
    metric_labels = ['Avg Accuracy\n(higher better)', 'BWT\n(positive = improvement)',
                     'FWT\n(positive = helped)', 'Forgetting\n(lower better)']

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = [all_cl_metrics[cond][metric_name] for cond in CONDITIONS.keys() if cond in all_cl_metrics]
        colors_list = [COLORS[cond] for cond in CONDITIONS.keys() if cond in all_cl_metrics]
        bars = ax.bar(range(len(values)), values, color=colors_list, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([c.split(':')[0] for c in CONDITIONS.keys() if c in all_cl_metrics], fontsize=10)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cl_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cl_metrics_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: cl_metrics_comparison.png/pdf")
    plt.close()

    # 4. Performance Matrices
    n_conditions = len(all_cl_metrics)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 4))
    if n_conditions == 1:
        axes = [axes]

    for idx, (cond_name, metrics) in enumerate(all_cl_metrics.items()):
        ax = axes[idx]
        matrix = metrics['matrix']
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   mask=mask, ax=ax, cbar_kws={'label': 'Accuracy'},
                   vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
        ax.set_xlabel('Task ID', fontsize=12)
        ax.set_ylabel('Trained Through Task', fontsize=12)
        ax.set_title(f'{cond_name}\nAvg Acc: {metrics["ACC"]:.4f}, BWT: {metrics["BWT"]:.4f}',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_matrices.pdf', bbox_inches='tight')
    print(f"✓ Saved: performance_matrices.png/pdf")
    plt.close()


if __name__ == '__main__':
    main()
