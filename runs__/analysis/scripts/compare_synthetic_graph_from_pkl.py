#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Synthetic Graph accuracy curves across all 4 experimental conditions using pkl files.
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

# Data directory
DATA_DIR = Path("runs__/analysis/data_analysis/synthetic_graphs")
OUTPUT_DIR = DATA_DIR / "synthetic_graph_comparison_plots"

COLORS = {
    'C1: Baseline': '#E74C3C',
    'C2: Heuristics': '#F39C12',
    'C3: Arch Search': '#3498DB',
    'C4: AWB Full': '#9B59B6',
}

CONDITIONS = {
    'C1: Baseline': 'synthetic_graph_2task_condition1_run0/classification_synthetic_gcn_run0_records.pkl',
    'C2: Heuristics': 'synthetic_graph_2task_condition2_run0/classification_synthetic_gcn_run0_records.pkl',
    'C3: Arch Search': 'synthetic_graph_2task_condition3_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl',
    'C4: AWB Full': 'synthetic_graph_2task_condition4_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl',
}


def load_metrics_from_pkl(pkl_path):
    """Load and concatenate metrics from all tasks in pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    metadata = data['metadata']
    epochs_per_task = metadata.get('epochs_per_task', 125)
    n_tasks = metadata['n_tasks']

    metrics = defaultdict(list)
    epochs = []
    task_boundaries_epochs = []

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            continue

        task_data = data['tasks'][task_id]

        if task_id > 0:
            task_boundaries_epochs.append(task_id * epochs_per_task)

        if 'main_training' in task_data:
            training = task_data['main_training']

            metric_map = {
                'H': 'H', 'V': 'V', 'dV': 'dV', 'grad_norm': 'grad_norm',
                'train_metric': 'train_acc', 'test_current': 'test_cur',
                'test_experience': 'test_exp',
            }

            for pkl_key, metric_name in metric_map.items():
                if pkl_key in training:
                    metrics[metric_name].extend(training[pkl_key])

            if 'iterations' in training:
                epochs.extend(training['iterations'])

    return metrics, np.array(epochs), task_boundaries_epochs, n_tasks, data


def compute_cl_metrics(data):
    """Compute ACC, BWT, FWT from task_performance_matrix."""
    if 'task_performance_matrix' not in data:
        return None

    tpm = data['task_performance_matrix']

    if isinstance(tpm, dict):
        n_tasks = len(tpm)
        matrix = np.zeros((n_tasks, n_tasks))
        for j in range(n_tasks):
            if j in tpm:
                for i_str, val in tpm[j].items():
                    matrix[j, int(i_str)] = val
    else:
        matrix = np.array(tpm)

    if matrix.size == 0:
        return None

    n_tasks = matrix.shape[0]
    acc = np.mean(matrix[-1, :])

    bwt_values = [matrix[-1, i] - matrix[i, i] for i in range(n_tasks - 1)]
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    fwt_values = []
    baseline = matrix[0, 0]
    for i in range(1, n_tasks):
        if str(i) in tpm.get(i-1, {}):
            fwt_values.append(matrix[i-1, i] - baseline)
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    forgetting_values = [max(0, np.max(matrix[:, i]) - matrix[-1, i]) for i in range(n_tasks - 1)]
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {'ACC': acc, 'BWT': bwt, 'FWT': fwt, 'Forgetting': forgetting, 'matrix': matrix}


def print_metrics_summary(all_metrics):
    """Print formatted metrics table."""
    print("="*80)
    print("CONTINUAL LEARNING METRICS SUMMARY - SYNTHETIC GRAPH (2 Tasks)")
    print("="*80)
    print(f"{'Condition':<20} {'Avg Acc':<15} {'BWT':<15} {'FWT':<15} {'Forgetting':<15}")
    print("-"*80)

    for cond_name in CONDITIONS.keys():
        if cond_name in all_metrics and all_metrics[cond_name] is not None:
            m = all_metrics[cond_name]
            print(f"{cond_name:<20} {m['ACC']:<15.4f} {m['BWT']:<15.4f} {m['FWT']:<15.4f} {m['Forgetting']:<15.4f}")

    print("="*80 + "\n")


def plot_comparison(all_data, all_cl_metrics, output_dir):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss Components
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    components = [('H', 'Hamiltonian (Total Loss)'), ('V', 'Experience Replay Loss'),
                  ('dV', 'Total Regularization'), ('grad_norm', 'Gradient Norm')]

    for idx, (comp_key, comp_label) in enumerate(components):
        ax = axes[idx]
        for cond_name, (boundaries, curves) in all_data.items():
            if comp_key in curves and len(curves[comp_key]) > 0:
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
    print(f"Saved: loss_components.png/pdf")
    plt.close()

    # 2. Accuracy Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('train_acc', 'Training Accuracy'), ('test_cur', 'Test Accuracy (Current Task)'),
               ('test_exp', 'Test Accuracy (Experience Replay)')]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        for cond_name, (boundaries, curves) in all_data.items():
            if metric_key in curves and len(curves[metric_key]) > 0:
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
    print(f"Saved: accuracy_curves.png/pdf")
    plt.close()

    # 3. CL Metrics Bar Chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_names = ['ACC', 'BWT', 'FWT', 'Forgetting']
    metric_labels = ['Avg Accuracy\n(higher better)', 'BWT\n(positive = improvement)',
                     'FWT\n(positive = helped)', 'Forgetting\n(lower better)']

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = []
        colors_list = []
        cond_labels = []
        for cond in CONDITIONS.keys():
            if cond in all_cl_metrics and all_cl_metrics[cond] is not None:
                values.append(all_cl_metrics[cond][metric_name])
                colors_list.append(COLORS[cond])
                cond_labels.append(cond.split(':')[0])

        if values:
            bars = ax.bar(range(len(values)), values, color=colors_list, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(cond_labels, fontsize=10)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cl_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cl_metrics_comparison.pdf', bbox_inches='tight')
    print(f"Saved: cl_metrics_comparison.png/pdf")
    plt.close()

    # 4. Performance Matrices
    valid_metrics = {k: v for k, v in all_cl_metrics.items() if v is not None}
    n_conditions = len(valid_metrics)
    if n_conditions > 0:
        fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 4))
        if n_conditions == 1:
            axes = [axes]

        for idx, (cond_name, mets) in enumerate(valid_metrics.items()):
            ax = axes[idx]
            matrix = mets['matrix']
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                       mask=mask, ax=ax, cbar_kws={'label': 'Accuracy'},
                       vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
            ax.set_xlabel('Task ID', fontsize=12)
            ax.set_ylabel('Trained Through Task', fontsize=12)
            ax.set_title(f'{cond_name}\nAvg Acc: {mets["ACC"]:.4f}, BWT: {mets["BWT"]:.4f}',
                        fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_matrices.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'performance_matrices.pdf', bbox_inches='tight')
        print(f"Saved: performance_matrices.png/pdf")
        plt.close()

    # 5. Combined Overview (3-panel for paper)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Synthetic Graph Classification (2 Tasks): Comparison of Conditions C1-C4',
                fontsize=14, fontweight='bold')

    overview_metrics = [
        ('test_exp', 'Test Accuracy (Experience Replay)'),
        ('H', 'Hamiltonian Loss'),
        ('grad_norm', 'Gradient Norm'),
    ]

    for idx, (metric_key, title) in enumerate(overview_metrics):
        ax = axes[idx]
        for cond_name, (boundaries, curves) in all_data.items():
            if metric_key in curves and len(curves[metric_key]) > 0:
                ax.plot(curves['iterations'], curves[metric_key],
                       label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=2)

        ref_cond = list(all_data.keys())[0]
        ref_boundaries = all_data[ref_cond][0]
        for boundary in ref_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_key if metric_key != 'test_exp' else 'Accuracy', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        if metric_key == 'test_exp':
            ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_graph_2task_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'synthetic_graph_2task_results.pdf', bbox_inches='tight')
    print(f"Saved: synthetic_graph_2task_results.png/pdf")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("SYNTHETIC GRAPH CLASSIFICATION - CONTINUAL LEARNING ANALYSIS")
    print("="*80 + "\n")

    print("Loading experimental data...")
    pkl_data = {}
    all_training_data = {}
    all_cl_metrics = {}

    for cond_name, pkl_file in CONDITIONS.items():
        pkl_path = DATA_DIR / pkl_file
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found, skipping {cond_name}")
            continue

        print(f"  Loading {cond_name}...")
        metrics, epochs, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path)
        pkl_data[cond_name] = data

        all_training_data[cond_name] = (boundaries, {'iterations': epochs, **metrics})

        cl_metrics = compute_cl_metrics(data)
        all_cl_metrics[cond_name] = cl_metrics

        print(f"    Tasks: {num_tasks}")
        print(f"    Total checkpoints: {len(metrics.get('H', []))}")
        if cl_metrics:
            print(f"    Avg Acc: {cl_metrics['ACC']:.4f}, BWT: {cl_metrics['BWT']:.4f}, FWT: {cl_metrics['FWT']:.4f}, Forgetting: {cl_metrics['Forgetting']:.4f}")

    print(f"\nLoaded {len(pkl_data)} conditions\n")

    if not pkl_data:
        print("No data found! Please check that pickle files exist in:")
        print(f"  {DATA_DIR}")
        return

    print_metrics_summary(all_cl_metrics)

    print("Generating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(all_training_data, all_cl_metrics, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
