#!/usr/bin/env python3
"""
Compare Sine loss curves across all 4 experimental conditions.
Extracts training metrics from logs and generates comparison plots.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Log files
LOG_DIR = Path("runs__/kkt/logs")
CONDITIONS = {
    'C1: Baseline': 'sine_condition1_baseline_20260103_154705.log',
    'C2: Heuristics': 'sine_condition2_heuristics_20260103_154705.log',
    'C3: Arch Search': 'sine_condition3_arch_no_transfer_20260103_162649.log',
    'C4: AWB Full': 'sine_condition4_awb_full_20260103_162649.log',
}

# Regex patterns for extracting metrics from progress bars
PATTERNS = {
    'MSE': r'MSE=([\d.e+-]+)',  # Sine uses MSE, not CE
    'H': r'H=([\d.e+-]+)',
    'V': r'V=([\d.e+-]+)',
    'dV': r'dV=([\d.e+-]+)',
    'grad_norm': r'\|\|∇\|\|=([\d.e+-]+)',
    'train_mse': r'Tr=([\d.e+-]+)',  # Training MSE (not accuracy)
    'test_cur': r'Te/Cur=([\d.e+-]+)',  # Test MSE current task
    'test_exp': r'Te/Exp=([\d.e+-]+)',  # Test MSE experience replay
}

def extract_metrics_from_log(log_path):
    """Extract training metrics from main training phase only (150 epoch loops)."""
    metrics = defaultdict(list)
    task_boundaries = []
    current_task = -1
    global_step = 0

    with open(log_path, 'r') as f:
        for line in f:
            # Detect task start
            if re.match(r'^Task \d+$', line.strip()):
                match = re.search(r'Task (\d+)', line)
                if match:
                    new_task = int(match.group(1))
                    if new_task != current_task:
                        if current_task >= 0:
                            task_boundaries.append(global_step)
                        current_task = new_task

            # Extract metrics ONLY from main training progress bars (X/200 for sine)
            # This filters out preliminary (/30), A/B training (/100), and warmup (/2) bars
            if '%|' in line and '/200 ' in line and 'MSE=' in line:  # Main training progress bar
                for metric_name, pattern in PATTERNS.items():
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        metrics[metric_name].append(value)

                # Track global step
                if 'MSE=' in line:  # Only count once per line
                    global_step += 1

    return metrics, task_boundaries, current_task + 1

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
        ('MSE', 'MSE Loss'),
        ('H', 'Hamiltonian (Total)'),
        ('V', 'Experience Replay Loss'),
        ('dV', 'Regularization (dV)'),
    ]

    for idx, (metric, title) in enumerate(loss_metrics):
        ax = axes[idx // 2, idx % 2]

        for (cond_name, _), color in zip(CONDITIONS.items(), colors):
            if metric in all_metrics[cond_name]:
                values = all_metrics[cond_name][metric]
                ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

        # Add task boundaries (using C1 as reference)
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric != 'dV':  # dV can be negative
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_loss_comparison.png'}")
    plt.close()

    # 2. MSE Comparison (skip accuracy for regression)
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

        ax.set_xlabel('Training Step')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)  # MSE starts at 0, no upper limit

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_mse_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_mse_comparison.png'}")
    plt.close()

    # 3. Gradient Norm Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle('Sine: Gradient Norm Across Conditions', fontsize=16, fontweight='bold')

    for (cond_name, _), color in zip(CONDITIONS.items(), colors):
        if 'grad_norm' in all_metrics[cond_name]:
            values = all_metrics[cond_name]['grad_norm']
            ax.plot(values, label=cond_name, alpha=0.8, linewidth=1.5, color=color)

    # Add task boundaries
    ref_boundaries = all_boundaries['C1: Baseline']
    for boundary in ref_boundaries:
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_gradient_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sine_gradient_comparison.png'}")
    plt.close()

    # 4. Combined Overview (smaller, for paper)
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
                # Smooth for cleaner visualization
                if len(values) > 10:
                    window = min(20, len(values) // 10)
                    values_smooth = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(values_smooth, label=cond_name, alpha=0.8, linewidth=2, color=color)
                else:
                    ax.plot(values, label=cond_name, alpha=0.8, linewidth=2, color=color)

        # Add task boundaries
        ref_boundaries = all_boundaries['C1: Baseline']
        for boundary in ref_boundaries:
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.2, linewidth=1)

        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric != 'dV':  # dV can be negative
            ax.set_ylim(bottom=0)
        # For sine MSE metrics, let y-axis scale automatically

    plt.tight_layout()
    plt.savefig(output_dir / 'sine_overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sine_overview_comparison.pdf', bbox_inches='tight')  # For paper
    print(f"Saved: {output_dir / 'sine_overview_comparison.png'}")
    print(f"Saved: {output_dir / 'sine_overview_comparison.pdf'}")
    plt.close()

def main():
    """Main execution."""
    print("="*70)
    print("Sine Condition Comparison Analysis")
    print("="*70)

    all_metrics = {}
    all_boundaries = {}
    all_num_tasks = {}

    # Extract metrics from all conditions
    for cond_name, log_file in CONDITIONS.items():
        log_path = LOG_DIR / log_file
        print(f"\nProcessing: {cond_name}")
        print(f"  Log: {log_path}")

        if not log_path.exists():
            print(f"  ⚠️  Log file not found!")
            continue

        metrics, boundaries, num_tasks = extract_metrics_from_log(log_path)
        all_metrics[cond_name] = metrics
        all_boundaries[cond_name] = boundaries
        all_num_tasks[cond_name] = num_tasks

        print(f"  Tasks: {num_tasks}")
        print(f"  Total steps: {len(metrics.get('MSE', []))}")
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
