#!/usr/bin/env python3
"""
Compare Sine regression curves across all 4 experimental conditions with multiple seeds.
Generates plots with error bars (shaded regions) and computes mean ± std statistics.

Added by Claude: Multi-seed analysis for JMLR paper.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import interpolate

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Data directory
DATA_DIR = Path("runs__/analysis/data_analysis/sine_3seed/results")
OUTPUT_DIR = Path("runs__/analysis/data_analysis/sine_3seed/sine_comparison_plots")

# Pastel colors for main lines (darker) and shaded regions (lighter)
COLORS = {
    'C1: Baseline': {
        'line': '#E57373',      # Pastel red
        'fill': '#FFCDD2',      # Lighter pastel red
    },
    'C2: Heuristics': {
        'line': '#FFB74D',      # Pastel orange
        'fill': '#FFE0B2',      # Lighter pastel orange
    },
    'C3: Arch Search': {
        'line': '#64B5F6',      # Pastel blue
        'fill': '#BBDEFB',      # Lighter pastel blue
    },
    'C4: AWB Full': {
        'line': '#BA68C8',      # Pastel purple
        'fill': '#E1BEE7',      # Lighter pastel purple
    },
}

# PKL file patterns for each condition (3 seeds each)
CONDITIONS = {
    'C1: Baseline': [
        'sine_condition1_baseline_run0/sine_condition1_baseline_run0_run0/regression_sine_fcnn_run0_records.pkl',
        'sine_condition1_baseline_run1/sine_condition1_baseline_run1_run1/regression_sine_fcnn_run1_records.pkl',
        'sine_condition1_baseline_run2/sine_condition1_baseline_run2_run2/regression_sine_fcnn_run2_records.pkl',
    ],
    'C2: Heuristics': [
        'sine_condition2_heuristics_run0/sine_condition2_heuristics_run0_run0/regression_sine_fcnn_run0_records.pkl',
        'sine_condition2_heuristics_run1/sine_condition2_heuristics_run1_run1/regression_sine_fcnn_run1_records.pkl',
        'sine_condition2_heuristics_run2/sine_condition2_heuristics_run2_run2/regression_sine_fcnn_run2_records.pkl',
    ],
    'C3: Arch Search': [
        'sine_condition3_arch_no_transfer_run0/sine_condition3_arch_no_transfer_run0_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
        'sine_condition3_arch_no_transfer_run1/sine_condition3_arch_no_transfer_run1_awb_run1/regression_sine_fcnn_awb_run1_records.pkl',
        'sine_condition3_arch_no_transfer_run2/sine_condition3_arch_no_transfer_run2_awb_run2/regression_sine_fcnn_awb_run2_records.pkl',
    ],
    'C4: AWB Full': [
        'sine_condition4_awb_full_run0/sine_condition4_awb_full_run0_run0/regression_sine_fcnn_awb_run0_records.pkl',
        'sine_condition4_awb_full_run1/sine_condition4_awb_full_run1_run1/regression_sine_fcnn_awb_run1_records.pkl',
        'sine_condition4_awb_full_run2/sine_condition4_awb_full_run2_run2/regression_sine_fcnn_awb_run2_records.pkl',
    ],
}


def load_metrics_from_pkl(pkl_path):
    """Load and concatenate metrics from all tasks in pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    metadata = data['metadata']
    epochs_per_task = metadata.get('epochs_per_task', 500)
    n_tasks = metadata['n_tasks']

    metrics = defaultdict(list)
    iterations = []
    task_boundaries = []

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            continue

        task_data = data['tasks'][task_id]

        # Mark task boundary at start of each new task (except task 0)
        if task_id > 0:
            task_boundaries.append(task_id * epochs_per_task)

        # Extract main training data
        if 'main_training' in task_data:
            training = task_data['main_training']

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
                    metrics[metric_name].extend(training[pkl_key])

            if 'iterations' in training:
                iterations.extend(training['iterations'])

    return metrics, np.array(iterations), task_boundaries, n_tasks, data


def compute_cl_metrics(data):
    """Compute Avg MSE, BWT, FWT, Forgetting from task_performance_matrix.

    For regression (MSE), lower is better.
    Sign conventions are FLIPPED from accuracy.
    """
    if 'task_performance_matrix' not in data:
        return None

    tpm = data['task_performance_matrix']

    # Handle dict-of-dicts format
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

    # Avg MSE: Average of final row (lower = better)
    avg_mse = np.mean(matrix[-1, :])

    # BWT: Backward transfer (FLIPPED for MSE)
    # BWT = R_{i,i} - R_{T-1,i}, positive = MSE decreased = good
    bwt_values = []
    for i in range(n_tasks - 1):
        bwt_values.append(matrix[i, i] - matrix[-1, i])
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # FWT: Forward transfer (FLIPPED for MSE)
    # FWT = baseline - R_{i-1,i}, positive = new task learned better
    fwt_values = []
    baseline = matrix[0, 0]
    for i in range(1, n_tasks):
        if str(i) in tpm.get(i-1, {}):
            fwt_values.append(baseline - matrix[i-1, i])
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting: MSE increase from best (lowest) to final
    forgetting_values = []
    for i in range(n_tasks - 1):
        best_mse = np.min(matrix[:, i])  # Best = lowest MSE
        final_mse = matrix[-1, i]
        forgetting_values.append(max(0, final_mse - best_mse))
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'Avg_MSE': avg_mse,
        'BWT': bwt,
        'FWT': fwt,
        'Forgetting': forgetting,
        'matrix': matrix,
    }


def aggregate_curves(all_seed_data, metric_key):
    """Aggregate curves across seeds, handling different lengths via interpolation."""
    # Find the common iteration range
    all_iters = []
    all_values = []

    for seed_data in all_seed_data:
        iterations, metrics = seed_data
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            all_iters.append(iterations)
            all_values.append(np.array(metrics[metric_key]))

    if not all_iters:
        return None, None, None, None

    # Use the shortest common length for simplicity
    min_len = min(len(v) for v in all_values)

    # Truncate all to same length
    truncated_iters = all_iters[0][:min_len]
    truncated_values = np.array([v[:min_len] for v in all_values])

    mean_values = np.mean(truncated_values, axis=0)
    std_values = np.std(truncated_values, axis=0)

    return truncated_iters, mean_values, std_values, truncated_values


def plot_with_error_bands(ax, iterations, mean_vals, std_vals, color_dict, label, alpha_fill=0.3):
    """Plot mean curve with shaded error region."""
    ax.plot(iterations, mean_vals, label=label, color=color_dict['line'],
            linewidth=2, alpha=0.9)
    ax.fill_between(iterations, mean_vals - std_vals, mean_vals + std_vals,
                    color=color_dict['fill'], alpha=alpha_fill, linewidth=0)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("SINE REGRESSION - MULTI-SEED CONTINUAL LEARNING ANALYSIS (3 Seeds)")
    print("="*80 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("Loading experimental data...")
    all_condition_data = {}  # {cond_name: [(iters, metrics), ...]}
    all_cl_metrics = {}      # {cond_name: [cl_metrics_dict, ...]}
    task_boundaries = None
    n_tasks = None

    for cond_name, pkl_files in CONDITIONS.items():
        print(f"\n  {cond_name}:")
        all_condition_data[cond_name] = []
        all_cl_metrics[cond_name] = []

        for seed_idx, pkl_file in enumerate(pkl_files):
            pkl_path = DATA_DIR / pkl_file
            if not pkl_path.exists():
                print(f"    ⚠ Seed {seed_idx}: {pkl_path} not found, skipping")
                continue

            metrics, iters, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path)
            all_condition_data[cond_name].append((iters, dict(metrics)))

            if task_boundaries is None:
                task_boundaries = boundaries
                n_tasks = num_tasks

            cl_metrics = compute_cl_metrics(data)
            if cl_metrics:
                all_cl_metrics[cond_name].append(cl_metrics)
                print(f"    ✓ Seed {seed_idx}: {len(metrics.get('H', []))} checkpoints, "
                      f"Avg MSE={cl_metrics['Avg_MSE']:.4f}")

    print(f"\n✓ Loaded data for {len(all_condition_data)} conditions")
    print(f"  Tasks: {n_tasks}, Task boundaries: {task_boundaries[:3]}..." if task_boundaries else "")

    # Compute aggregate statistics
    print("\n" + "="*80)
    print("COMPUTING AGGREGATE STATISTICS")
    print("="*80)

    agg_cl_metrics = {}
    for cond_name, seed_metrics in all_cl_metrics.items():
        if not seed_metrics:
            continue

        agg_cl_metrics[cond_name] = {
            'Avg_MSE_mean': np.mean([m['Avg_MSE'] for m in seed_metrics]),
            'Avg_MSE_std': np.std([m['Avg_MSE'] for m in seed_metrics]),
            'BWT_mean': np.mean([m['BWT'] for m in seed_metrics]),
            'BWT_std': np.std([m['BWT'] for m in seed_metrics]),
            'FWT_mean': np.mean([m['FWT'] for m in seed_metrics]),
            'FWT_std': np.std([m['FWT'] for m in seed_metrics]),
            'Forgetting_mean': np.mean([m['Forgetting'] for m in seed_metrics]),
            'Forgetting_std': np.std([m['Forgetting'] for m in seed_metrics]),
            'n_seeds': len(seed_metrics),
        }

    # Print summary table
    print_metrics_table(agg_cl_metrics)

    # Generate plots
    print("\nGenerating plots with error bands...")
    plot_all_comparisons(all_condition_data, agg_cl_metrics, task_boundaries, OUTPUT_DIR)

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


def print_metrics_table(agg_metrics):
    """Print formatted metrics table with mean ± std."""
    print("\n" + "="*100)
    print("CONTINUAL LEARNING METRICS SUMMARY - SINE REGRESSION (10 Tasks, 3 Seeds)")
    print("="*100)
    print(f"{'Condition':<18} {'Avg MSE (↓)':<20} {'BWT (↑)':<20} {'FWT (↑)':<20} {'Forgetting (↓)':<20}")
    print("-"*100)

    for cond_name in CONDITIONS.keys():
        if cond_name in agg_metrics:
            m = agg_metrics[cond_name]
            avg_mse = f"{m['Avg_MSE_mean']:.4f} ± {m['Avg_MSE_std']:.4f}"
            bwt = f"{m['BWT_mean']:.4f} ± {m['BWT_std']:.4f}"
            fwt = f"{m['FWT_mean']:.4f} ± {m['FWT_std']:.4f}"
            forg = f"{m['Forgetting_mean']:.4f} ± {m['Forgetting_std']:.4f}"
            print(f"{cond_name:<18} {avg_mse:<20} {bwt:<20} {fwt:<20} {forg:<20}")

    print("="*100)
    print("\nNote: BWT positive = MSE decreased (improvement). FWT positive = new task learned better than baseline.")
    print("      Lower Avg MSE and Forgetting are better. Higher BWT and FWT are better.\n")


def plot_all_comparisons(all_condition_data, agg_cl_metrics, task_boundaries, output_dir):
    """Generate all comparison plots with error bands."""

    # 1. Loss Components (H, V, dV, grad_norm)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle('Sine Regression: Loss Components (Mean ± Std, 3 Seeds)',
                 fontsize=16, fontweight='bold', y=1.02)

    components = [
        ('H', 'Hamiltonian (Total Loss)'),
        ('V', 'Experience Replay Loss'),
        ('dV', 'Regularization (dV)'),
        ('grad_norm', 'Gradient Norm'),
    ]

    for idx, (comp_key, comp_label) in enumerate(components):
        ax = axes[idx]

        for cond_name, seed_data_list in all_condition_data.items():
            iters, mean_vals, std_vals, _ = aggregate_curves(seed_data_list, comp_key)
            if iters is not None:
                plot_with_error_bands(ax, iters, mean_vals, std_vals,
                                     COLORS[cond_name], cond_name)

        # Add task boundaries
        if task_boundaries:
            for boundary in task_boundaries[:5]:  # Show first 5 boundaries
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(comp_label)
        ax.set_title(comp_label, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        if comp_key != 'dV':  # dV can be negative
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_components.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_components.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: loss_components.png/pdf")
    plt.close()

    # 2. MSE Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sine Regression: MSE Metrics (Mean ± Std, 3 Seeds)',
                 fontsize=16, fontweight='bold', y=1.02)

    mse_metrics = [
        ('train_mse', 'Training MSE'),
        ('test_cur', 'Test MSE (Current Task)'),
        ('test_exp', 'Test MSE (Experience Replay)'),
    ]

    for idx, (metric_key, metric_label) in enumerate(mse_metrics):
        ax = axes[idx]

        for cond_name, seed_data_list in all_condition_data.items():
            iters, mean_vals, std_vals, _ = aggregate_curves(seed_data_list, metric_key)
            if iters is not None:
                plot_with_error_bands(ax, iters, mean_vals, std_vals,
                                     COLORS[cond_name], cond_name)

        if task_boundaries:
            for boundary in task_boundaries[:5]:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title(metric_label, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'mse_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mse_curves.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: mse_curves.png/pdf")
    plt.close()

    # 3. CL Metrics Bar Chart with Error Bars
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Sine Regression: Continual Learning Metrics (Mean ± Std, 3 Seeds)',
                 fontsize=16, fontweight='bold', y=1.02)

    metric_configs = [
        ('Avg_MSE', 'Avg MSE\n(lower = better)', False),
        ('BWT', 'Backward Transfer\n(positive = improvement)', True),
        ('FWT', 'Forward Transfer\n(positive = helped)', True),
        ('Forgetting', 'Forgetting\n(lower = better)', False),
    ]

    for idx, (metric_key, metric_label, higher_better) in enumerate(metric_configs):
        ax = axes[idx]

        cond_names = [c for c in CONDITIONS.keys() if c in agg_cl_metrics]
        means = [agg_cl_metrics[c][f'{metric_key}_mean'] for c in cond_names]
        stds = [agg_cl_metrics[c][f'{metric_key}_std'] for c in cond_names]
        colors_list = [COLORS[c]['line'] for c in cond_names]

        x_pos = np.arange(len(cond_names))
        bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.8,
                     edgecolor='black', linewidth=1.2, capsize=5,
                     error_kw={'elinewidth': 2, 'capthick': 2})

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            offset = std + 0.001 if height >= 0 else -std - 0.003
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.split(':')[0] for c in cond_names], fontsize=10)
        ax.set_ylabel(metric_label.split('\n')[0])
        ax.set_title(metric_label, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add zero line for BWT/FWT
        if metric_key in ['BWT', 'FWT']:
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'cl_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cl_metrics_comparison.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: cl_metrics_comparison.png/pdf")
    plt.close()

    # 4. Overview Plot (4-panel for paper)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Sine Regression: Training Dynamics Overview (3 Seeds)',
                 fontsize=16, fontweight='bold', y=1.02)

    overview_metrics = [
        ('H', 'Hamiltonian Loss', True),
        ('train_mse', 'Training MSE', False),
        ('test_exp', 'Test MSE (Experience Replay)', False),
        ('grad_norm', 'Gradient Norm', True),
    ]

    for idx, (metric_key, metric_label, use_log) in enumerate(overview_metrics):
        ax = axes[idx]

        for cond_name, seed_data_list in all_condition_data.items():
            iters, mean_vals, std_vals, _ = aggregate_curves(seed_data_list, metric_key)
            if iters is not None:
                plot_with_error_bands(ax, iters, mean_vals, std_vals,
                                     COLORS[cond_name], cond_name, alpha_fill=0.25)

        if task_boundaries:
            for boundary in task_boundaries[:5]:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'overview_comparison.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: overview_comparison.png/pdf")
    plt.close()

    # 5. Final Performance Summary (single figure for paper)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sine Regression: Final Performance Summary (10 Tasks, 3 Seeds)',
                 fontsize=16, fontweight='bold', y=1.02)

    # Left: Test Experience MSE over time
    ax = axes[0]
    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals, _ = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            plot_with_error_bands(ax, iters, mean_vals, std_vals,
                                 COLORS[cond_name], cond_name, alpha_fill=0.25)

    if task_boundaries:
        for i, boundary in enumerate(task_boundaries[:5]):
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(boundary, ax.get_ylim()[1]*0.95, f'T{i+1}', fontsize=8, ha='center')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Test MSE (Experience Replay)', fontsize=12)
    ax.set_title('Multi-Task Performance Over Training', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: CL Metrics Summary Bar Chart
    ax = axes[1]
    cond_names = [c for c in CONDITIONS.keys() if c in agg_cl_metrics]
    x = np.arange(len(cond_names))
    width = 0.35

    avg_mse_means = [agg_cl_metrics[c]['Avg_MSE_mean'] for c in cond_names]
    avg_mse_stds = [agg_cl_metrics[c]['Avg_MSE_std'] for c in cond_names]
    bwt_means = [agg_cl_metrics[c]['BWT_mean'] for c in cond_names]
    bwt_stds = [agg_cl_metrics[c]['BWT_std'] for c in cond_names]

    colors_list = [COLORS[c]['line'] for c in cond_names]

    bars1 = ax.bar(x - width/2, avg_mse_means, width, yerr=avg_mse_stds,
                   label='Avg MSE (↓)', color=colors_list, alpha=0.7,
                   edgecolor='black', capsize=4)

    # Scale BWT for visibility (multiply by 10)
    bwt_scaled = [b * 10 for b in bwt_means]
    bwt_stds_scaled = [s * 10 for s in bwt_stds]
    bars2 = ax.bar(x + width/2, bwt_scaled, width, yerr=bwt_stds_scaled,
                   label='BWT × 10 (↑)', color=colors_list, alpha=0.4,
                   edgecolor='black', hatch='//', capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([c.split(':')[0] + '\n' + c.split(': ')[1] for c in cond_names],
                       fontsize=9)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Final CL Metrics Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'final_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'final_summary.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: final_summary.png/pdf")
    plt.close()


if __name__ == '__main__':
    main()
