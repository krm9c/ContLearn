#!/usr/bin/env python3
"""
Compare Sine loss curves across all 4 experimental conditions using pkl files.
Loads training metrics from pkl records and generates comparison plots.
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
DATA_DIR = Path("runs__/analysis")

CONDITIONS = {
    'C1: Baseline': 'sine_condition1_run0/regression_sine_fcnn_run0_records.pkl',
    'C2: Heuristics': 'sine_condition2_run0/regression_sine_fcnn_run0_records.pkl',
    'C3: Arch Search': 'sine_condition3_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
    'C4: AWB Full': 'sine_condition4_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
}


def load_metrics_from_pkl(pkl_path):
    """Load and concatenate metrics from all tasks in pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract metrics across all tasks
    metrics = defaultdict(list)
    task_boundaries = []
    global_step = 0

    n_tasks = data['metadata']['n_tasks']

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            print(f"  Warning: Task {task_id} not found in data")
            continue

        task_data = data['tasks'][task_id]

        # Mark task boundary (before adding new task data)
        if task_id > 0:
            task_boundaries.append(global_step)

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

            # Track global step
            if 'iterations' in training:
                global_step += len(training['iterations'])

    return metrics, task_boundaries, n_tasks


def plot_comparison(all_metrics, all_boundaries, output_dir):
    """Generate comparison plots across conditions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define color palette
    colors = sns.color_palette("husl", len(CONDITIONS))

    # 1. Loss Components Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sine: Loss Components Across Conditions', fontsize=16, fontweight='bold')

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
        ref_boundaries = all_boundaries['C1: Baseline']
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
    plt.savefig(output_dir / 'sine_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_loss_comparison.png'}")
    plt.close()

    # 2. MSE Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sine: MSE Metrics Across Conditions', fontsize=16, fontweight='bold')

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
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_mse_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_mse_comparison.png'}")
    plt.close()

    # 3. Gradient Norm Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle('Sine: Gradient Norm Across Conditions', fontsize=16, fontweight='bold')

    for (cond_name, _), color in zip(CONDITIONS.items(), colors):
        if 'grad_norm' in all_metrics[cond_name] and len(all_metrics[cond_name]['grad_norm']) > 0:
            values = all_metrics[cond_name]['grad_norm']
            ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

    # Add task boundaries
    ref_boundaries = all_boundaries['C1: Baseline']
    for boundary in ref_boundaries:
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_gradient_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_gradient_comparison.png'}")
    plt.close()

    # 4. Combined Overview (for paper)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Sine: Training Dynamics Comparison', fontsize=14, fontweight='bold')

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
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=2, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.2, linewidth=1)

        ax.set_xlabel('Checkpoint', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if metric not in ['dV']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sine_overview_comparison.pdf', bbox_inches='tight')  # For paper
    print(f"Saved: {output_dir / 'sine_overview_comparison.png'}")
    print(f"Saved: {output_dir / 'sine_overview_comparison.pdf'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("Sine Condition Comparison Analysis (from PKL files)")
    print("="*70)

    all_metrics = {}
    all_boundaries = {}
    all_num_tasks = {}

    # Load metrics from all conditions
    for cond_name, pkl_file in CONDITIONS.items():
        pkl_path = DATA_DIR / pkl_file
        print(f"\nProcessing: {cond_name}")
        print(f"  File: {pkl_path}")

        if not pkl_path.exists():
            print(f"  ⚠️  PKL file not found!")
            continue

        metrics, boundaries, num_tasks = load_metrics_from_pkl(pkl_path)
        all_metrics[cond_name] = metrics
        all_boundaries[cond_name] = boundaries
        all_num_tasks[cond_name] = num_tasks

        print(f"  Tasks: {num_tasks}")
        print(f"  Total checkpoints: {len(metrics.get('H', []))}")
        print(f"  Task boundaries: {len(boundaries)}")

        # Show available metrics
        available = [k for k, v in metrics.items() if len(v) > 0]
        print(f"  Metrics extracted: {', '.join(available)}")

    # Generate plots
    print("\n" + "="*70)
    print("Generating Comparison Plots")
    print("="*70)

    output_dir = Path("runs__/analysis/sine_comparison_plots")
    plot_comparison(all_metrics, all_boundaries, output_dir)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. sine_loss_comparison.png - Loss components")
    print("  2. sine_mse_comparison.png - MSE metrics")
    print("  3. sine_gradient_comparison.png - Gradient norms")
    print("  4. sine_overview_comparison.png/pdf - Combined overview")


if __name__ == '__main__':
    main()
