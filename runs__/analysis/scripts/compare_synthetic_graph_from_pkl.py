#!/usr/bin/env python3
"""
Synthetic Graph Classification - Continual Learning Analysis
Compares 4 experimental conditions:
- C1: Baseline (no smoothness)
- C2: Heuristics (cosine LR, warmup)
- C3: Architecture search, no AWB transfer
- C4: AWB Full (complete theory)

Dataset: Synthetic graph classification with GCN
Metric: Classification accuracy (higher is better)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
DATA_DIR = Path("/Users/kraghavan/Desktop/JMLR_paper/ContLearn/runs__/analysis/data_analysis/synthetic_graphs")
OUTPUT_DIR = DATA_DIR / "synthetic_graph_comparison_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    'C1: Baseline': 'synthetic_graph_condition1_run0/classification_synthetic_gcn_run0_records.pkl',
    'C2: Heuristics': 'synthetic_graph_condition2_run0/classification_synthetic_gcn_run0_records.pkl',
    'C3: Arch Search': 'synthetic_graph_condition3_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl',
    'C4: AWB Full': 'synthetic_graph_condition4_awb_run0/classification_synthetic_gcn_awb_run0_records.pkl',
}

COLORS = {
    'C1: Baseline': '#E74C3C',      # Red
    'C2: Heuristics': '#F39C12',    # Orange
    'C3: Arch Search': '#3498DB',   # Blue
    'C4: AWB Full': '#9B59B6',      # Purple
}

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_pkl_data(pkl_path: Path) -> Dict:
    """Load pickle file and return data."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def extract_training_curves(data: Dict) -> Tuple[List, Dict]:
    """Extract training curves from all tasks."""
    n_tasks = data['metadata']['n_tasks']
    epochs_per_task = data['metadata']['epochs_per_task']

    curves = {
        'iterations': [],
        'H': [],
        'V': [],
        'dV': [],
        'grad_norm': [],
        'train_metric': [],
        'test_current': [],
        'test_experience': [],
    }

    task_boundaries = []

    for task_id in range(n_tasks):
        task_data = data['tasks'][task_id]
        training = task_data['main_training']

        # Add task boundary at start of each task
        if task_id > 0:
            task_boundaries.append(task_id * epochs_per_task)

        # Iterations are already global
        curves['iterations'].extend(training['iterations'])
        curves['H'].extend(training['H'])
        curves['V'].extend(training['V'])
        curves['dV'].extend(training['dV'])
        curves['grad_norm'].extend(training['grad_norm'])
        curves['train_metric'].extend(training['train_metric'])
        curves['test_current'].extend(training['test_current'])
        curves['test_experience'].extend(training['test_experience'])

    return task_boundaries, curves

def compute_cl_metrics(data: Dict) -> Dict[str, float]:
    """
    Compute continual learning metrics from task performance matrix.

    For classification accuracy (higher is better):
    - BWT: matrix[-1, i] - matrix[i, i] (positive = improvement, negative = forgetting)
    - FWT: matrix[i-1, i] - baseline (positive = transfer helped)
    - Forgetting: max drop from best performance (lower is better)
    """
    matrix_dict = data['task_performance_matrix']
    n_tasks = data['metadata']['n_tasks']

    # Convert dict of dicts to numpy array
    matrix = np.zeros((n_tasks, n_tasks))
    for j in range(n_tasks):
        for i_str, acc in matrix_dict[j].items():
            i = int(i_str)
            matrix[j, i] = acc

    # Average Accuracy (final row average)
    avg_acc = np.mean(matrix[-1, :])

    # Backward Transfer (BWT)
    # How much did training on new tasks affect old tasks?
    # For accuracy: positive = improvement, negative = forgetting
    bwt_values = []
    for i in range(n_tasks - 1):
        bwt_values.append(matrix[-1, i] - matrix[i, i])
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # Forward Transfer (FWT)
    # How much did prior knowledge help learn new tasks?
    # Compare to random initialization (use first task as baseline)
    fwt_values = []
    baseline = matrix[0, 0]  # First task initial performance
    for i in range(1, n_tasks):
        # Performance on task i after training on task i-1
        if i-1 in matrix_dict and str(i) in matrix_dict[i-1]:
            fwt_values.append(matrix[i-1, i] - baseline)
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting Measure
    # Maximum drop from best performance to final performance
    forgetting_values = []
    for i in range(n_tasks - 1):
        # Find best performance on task i across all training
        best_perf = max([matrix[j, i] for j in range(i, n_tasks) if str(i) in matrix_dict[j]])
        # Forgetting is drop from best to final
        forgetting_values.append(max(0, best_perf - matrix[-1, i]))
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'avg_acc': avg_acc,
        'bwt': bwt,
        'fwt': fwt,
        'forgetting': forgetting,
        'matrix': matrix
    }

def plot_loss_components(all_data: Dict[str, Dict]):
    """Plot Hamiltonian components for all conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    components = [
        ('H', 'Hamiltonian (Total Loss)', 0),
        ('V', 'Experience Replay Loss', 1),
        ('dV', 'Total Regularization', 2),
        ('grad_norm', 'Gradient Norm', 3)
    ]

    for comp_key, comp_label, ax_idx in components:
        ax = axes[ax_idx]

        for cond_name, cond_data in all_data.items():
            boundaries, curves = cond_data
            ax.plot(curves['iterations'], curves[comp_key],
                   label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)

        # Add task boundaries
        if boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(comp_label, fontsize=12)
        ax.set_title(comp_label, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_components.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'loss_components.pdf', bbox_inches='tight')
    print(f"✓ Saved: loss_components.png/pdf")
    plt.close()

def plot_accuracy_curves(all_data: Dict[str, Dict]):
    """Plot accuracy metrics for all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('train_metric', 'Training Accuracy', 0),
        ('test_current', 'Test Accuracy (Current Task)', 1),
        ('test_experience', 'Test Accuracy (Experience Replay)', 2)
    ]

    for metric_key, metric_label, ax_idx in metrics:
        ax = axes[ax_idx]

        for cond_name, cond_data in all_data.items():
            boundaries, curves = cond_data
            ax.plot(curves['iterations'], curves[metric_key],
                   label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)

        # Add task boundaries
        if boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(metric_label, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'accuracy_curves.pdf', bbox_inches='tight')
    print(f"✓ Saved: accuracy_curves.png/pdf")
    plt.close()

def plot_gradient_norm(all_data: Dict[str, Dict]):
    """Plot gradient norm separately for detailed view."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond_name, cond_data in all_data.items():
        boundaries, curves = cond_data
        ax.plot(curves['iterations'], curves['grad_norm'],
               label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)

    # Add task boundaries
    if boundaries:
        for boundary in boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gradient_norm.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'gradient_norm.pdf', bbox_inches='tight')
    print(f"✓ Saved: gradient_norm.png/pdf")
    plt.close()

def plot_overview_4panel(all_data: Dict[str, Dict]):
    """4-panel overview for paper figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Hamiltonian
    ax1 = fig.add_subplot(gs[0, 0])
    for cond_name, cond_data in all_data.items():
        boundaries, curves = cond_data
        ax1.plot(curves['iterations'], curves['H'],
                label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
    if boundaries:
        for boundary in boundaries:
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Hamiltonian', fontsize=12)
    ax1.set_title('(A) Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Experience Replay Loss
    ax2 = fig.add_subplot(gs[0, 1])
    for cond_name, cond_data in all_data.items():
        boundaries, curves = cond_data
        ax2.plot(curves['iterations'], curves['V'],
                label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
    if boundaries:
        for boundary in boundaries:
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Experience Replay Loss', fontsize=12)
    ax2.set_title('(B) Multi-Task Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Test Accuracy (Experience Replay)
    ax3 = fig.add_subplot(gs[1, 0])
    for cond_name, cond_data in all_data.items():
        boundaries, curves = cond_data
        ax3.plot(curves['iterations'], curves['test_experience'],
                label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
    if boundaries:
        for boundary in boundaries:
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('(C) Test Accuracy (All Tasks)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Gradient Norm
    ax4 = fig.add_subplot(gs[1, 1])
    for cond_name, cond_data in all_data.items():
        boundaries, curves = cond_data
        ax4.plot(curves['iterations'], curves['grad_norm'],
                label=cond_name, color=COLORS[cond_name], alpha=0.8, linewidth=1.5)
    if boundaries:
        for boundary in boundaries:
            ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Gradient Norm', fontsize=12)
    ax4.set_title('(D) Gradient Stability', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'overview_4panel.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'overview_4panel.pdf', bbox_inches='tight')
    print(f"✓ Saved: overview_4panel.png/pdf")
    plt.close()

def plot_cl_metrics_bar_chart(all_metrics: Dict[str, Dict]):
    """Bar chart comparing CL metrics across conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    conditions = list(all_metrics.keys())

    metrics_to_plot = [
        ('avg_acc', 'Average Accuracy', 0, True),      # Higher is better
        ('bwt', 'Backward Transfer', 1, True),          # Higher is better (positive = improvement)
        ('fwt', 'Forward Transfer', 2, True),           # Higher is better
        ('forgetting', 'Forgetting', 3, False)          # Lower is better
    ]

    for metric_key, metric_label, ax_idx, higher_better in metrics_to_plot:
        ax = axes[ax_idx]

        values = [all_metrics[cond][metric_key] for cond in conditions]
        colors_list = [COLORS[cond] for cond in conditions]

        bars = ax.bar(range(len(conditions)), values, color=colors_list, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label}\n({"Higher" if higher_better else "Lower"} is Better)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cl_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'cl_metrics_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: cl_metrics_comparison.png/pdf")
    plt.close()

def plot_performance_matrices(all_metrics: Dict[str, Dict], pkl_data: Dict[str, Dict]):
    """Heatmap of task performance matrices for each condition."""
    n_conditions = len(all_metrics)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 4))

    if n_conditions == 1:
        axes = [axes]

    for idx, (cond_name, metrics) in enumerate(all_metrics.items()):
        ax = axes[idx]
        matrix = metrics['matrix']
        n_tasks = matrix.shape[0]

        # Mask upper triangle (future tasks)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   mask=mask, ax=ax, cbar_kws={'label': 'Accuracy'},
                   vmin=0, vmax=1, linewidths=0.5, linecolor='gray')

        ax.set_xlabel('Task ID', fontsize=12)
        ax.set_ylabel('Trained Through Task', fontsize=12)
        ax.set_title(f'{cond_name}\nAvg Acc: {metrics["avg_acc"]:.4f}, BWT: {metrics["bwt"]:.4f}',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'performance_matrices.pdf', bbox_inches='tight')
    print(f"✓ Saved: performance_matrices.png/pdf")
    plt.close()

def print_metrics_summary(all_metrics: Dict[str, Dict]):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING METRICS SUMMARY")
    print("="*80)
    print(f"{'Condition':<20} {'Avg Acc (↑)':<15} {'BWT (↑)':<15} {'FWT (↑)':<15} {'Forgetting (↓)':<15}")
    print("-"*80)

    for cond_name, metrics in all_metrics.items():
        print(f"{cond_name:<20} {metrics['avg_acc']:<15.4f} {metrics['bwt']:<15.4f} "
              f"{metrics['fwt']:<15.4f} {metrics['forgetting']:<15.4f}")

    print("="*80)

    # Find best condition for each metric
    best_avg_acc = max(all_metrics.items(), key=lambda x: x[1]['avg_acc'])
    best_bwt = max(all_metrics.items(), key=lambda x: x[1]['bwt'])
    best_fwt = max(all_metrics.items(), key=lambda x: x[1]['fwt'])
    best_forgetting = min(all_metrics.items(), key=lambda x: x[1]['forgetting'])

    print("\nBEST PERFORMERS:")
    print(f"  Avg Accuracy:  {best_avg_acc[0]} ({best_avg_acc[1]['avg_acc']:.4f})")
    print(f"  BWT:           {best_bwt[0]} ({best_bwt[1]['bwt']:.4f})")
    print(f"  FWT:           {best_fwt[0]} ({best_fwt[1]['fwt']:.4f})")
    print(f"  Forgetting:    {best_forgetting[0]} ({best_forgetting[1]['forgetting']:.4f})")
    print("="*80 + "\n")

def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("SYNTHETIC GRAPH CLASSIFICATION - CONTINUAL LEARNING ANALYSIS")
    print("="*80 + "\n")

    # Load all data
    print("Loading experimental data...")
    pkl_data = {}
    all_training_data = {}
    all_metrics = {}

    for cond_name, pkl_file in CONDITIONS.items():
        pkl_path = DATA_DIR / pkl_file
        if not pkl_path.exists():
            print(f"⚠ Warning: {pkl_path} not found, skipping {cond_name}")
            continue

        print(f"  Loading {cond_name}...")
        data = load_pkl_data(pkl_path)
        pkl_data[cond_name] = data

        # Extract training curves
        boundaries, curves = extract_training_curves(data)
        all_training_data[cond_name] = (boundaries, curves)

        # Compute metrics
        metrics = compute_cl_metrics(data)
        all_metrics[cond_name] = metrics

    print(f"\n✓ Loaded {len(pkl_data)} conditions\n")

    # Print metrics summary
    print_metrics_summary(all_metrics)

    # Generate plots
    print("Generating plots...")
    plot_loss_components(all_training_data)
    plot_accuracy_curves(all_training_data)
    plot_gradient_norm(all_training_data)
    plot_overview_4panel(all_training_data)
    plot_cl_metrics_bar_chart(all_metrics)
    plot_performance_matrices(all_metrics, pkl_data)

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
