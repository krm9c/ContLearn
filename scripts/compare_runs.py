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

# Added by Claude: Import helper functions from plot_results
import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_results import extract_time_series, add_task_shading


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
                         labels: List[str], output_dir: str):
    """Compare all loss components between two runs."""
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Loss Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    loss_components = [
        ('loss_H', 'Hamiltonian (H)', 'b'),
        ('loss_V', f'V ({metadata1["loss_function"]})', 'r'),
        ('loss_dV', 'dV', 'g'),
        ('loss_dV_dx', 'dV/dx', 'c'),
        ('loss_dV_dtheta', 'dV/dθ', 'm'),
        ('grad_norm', '||dH/dθ||', 'orange'),
    ]

    for idx, (key, ylabel, color) in enumerate(loss_components):
        ax = axes[idx // 3, idx % 3]

        # Plot both runs
        ax.plot(series1['iterations'], series1[key], color=color, linewidth=2.5,
                label=labels[0], alpha=0.8)
        ax.plot(series2['iterations'], series2[key], color=color, linewidth=2.5,
                label=labels[1], linestyle='--', alpha=0.8)

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
                            labels: List[str], output_dir: str):
    """Compare training and test metrics between two runs."""
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Metrics Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('metric_train', 'Train', 'b'),
        ('metric_test_current', 'Test (Current Task)', 'g'),
        ('metric_test_experience', 'Test (Experience)', 'r'),
    ]

    for idx, (key, title, color) in enumerate(metrics):
        ax = axes[idx]

        # Plot both runs
        ax.plot(series1['iterations'], series1[key], color=color, linewidth=2.5,
                label=labels[0], marker='o', markersize=4,
                markevery=max(1, len(series1['iterations'])//20), alpha=0.8)
        ax.plot(series2['iterations'], series2[key], color=color, linewidth=2.5,
                label=labels[1], marker='s', markersize=4,
                markevery=max(1, len(series2['iterations'])//20),
                linestyle='--', alpha=0.8)

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

    bars1 = ax.bar(x - width/2, vals1, width, label=labels[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, vals2, width, label=labels[1], alpha=0.8)

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
                             labels: List[str], output_dir: str):
    """Combined overview comparison plot."""
    series1 = extract_time_series(run1)
    series2 = extract_time_series(run2)
    metadata1 = run1['metadata']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Training Overview Comparison: {labels[0]} vs {labels[1]} - {metadata1["dataset"]}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Hamiltonian
    axes[0, 0].plot(series1['iterations'], series1['loss_H'], 'b-', linewidth=2.5,
                   label=labels[0], alpha=0.8)
    axes[0, 0].plot(series2['iterations'], series2['loss_H'], 'b--', linewidth=2.5,
                   label=labels[1], alpha=0.8)
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Hamiltonian (H)', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # Plot 2: Primary Loss (V)
    axes[0, 1].plot(series1['iterations'], series1['loss_V'], 'r-', linewidth=2.5,
                   label=labels[0], alpha=0.8)
    axes[0, 1].plot(series2['iterations'], series2['loss_V'], 'r--', linewidth=2.5,
                   label=labels[1], alpha=0.8)
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel(f'V ({metadata1["loss_function"]})', fontsize=11)
    axes[0, 1].set_title('Primary Loss (V)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # Plot 3: Test Current Task
    axes[1, 0].plot(series1['iterations'], series1['metric_test_current'], 'g-',
                   linewidth=2.5, label=labels[0], marker='o', markersize=4,
                   markevery=max(1, len(series1['iterations'])//20), alpha=0.8)
    axes[1, 0].plot(series2['iterations'], series2['metric_test_current'], 'g--',
                   linewidth=2.5, label=labels[1], marker='s', markersize=4,
                   markevery=max(1, len(series2['iterations'])//20), alpha=0.8)
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel(f'{metadata1["metric_function"]}', fontsize=11)
    axes[1, 0].set_title('Test (Current Task)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # Plot 4: Test Experience
    axes[1, 1].plot(series1['iterations'], series1['metric_test_experience'], 'purple',
                   linewidth=2.5, label=labels[0], marker='o', markersize=4,
                   markevery=max(1, len(series1['iterations'])//20), alpha=0.8)
    axes[1, 1].plot(series2['iterations'], series2['metric_test_experience'], 'purple',
                   linewidth=2.5, label=labels[1], marker='s', markersize=4,
                   markevery=max(1, len(series2['iterations'])//20),
                   linestyle='--', alpha=0.8)
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

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Load records
    print(f'\nLoading run 1: {args.run1}')
    run1_data = load_records(args.run1)

    print(f'Loading run 2: {args.run2}')
    run2_data = load_records(args.run2)

    # Generate comparison plots
    print(f'\nGenerating comparison plots...')
    plot_loss_comparison(run1_data, run2_data, args.labels, args.output_dir)
    plot_metrics_comparison(run1_data, run2_data, args.labels, args.output_dir)
    plot_final_metrics_bar(run1_data, run2_data, args.labels, args.output_dir)
    plot_combined_comparison(run1_data, run2_data, args.labels, args.output_dir)

    # Print summary statistics
    print_summary_statistics(run1_data, run2_data, args.labels)

    print(f'\n✓ All comparison plots generated in {args.output_dir}/')


if __name__ == '__main__':
    main()
