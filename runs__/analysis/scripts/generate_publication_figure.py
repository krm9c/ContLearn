#!/usr/bin/env python3
"""
Generate Nature Communications style publication figure for sine regression results.
4-panel figure with letter labels (a, b, c, d), minimal framing, pastel colors.

Added by Claude: Publication-quality figure for JMLR paper.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Nature Communications style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Data directory
DATA_DIR = Path("runs__/analysis/data_analysis/sine_3seed/results")
OUTPUT_DIR = Path("runs__/analysis/data_analysis/sine_3seed/sine_comparison_plots")

# Pastel color scheme
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

# Short labels for legend
SHORT_LABELS = {
    'C1: Baseline': 'Baseline',
    'C2: Heuristics': 'Heuristics',
    'C3: Arch Search': 'Arch Search',
    'C4: AWB Full': 'AWB Full',
}

# PKL file patterns
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
    """Load metrics from pkl file."""
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
        if task_id > 0:
            task_boundaries.append(task_id * epochs_per_task)

        if 'main_training' in task_data:
            training = task_data['main_training']
            metric_map = {
                'H': 'H',
                'V': 'V',
                'test_experience': 'test_exp',
            }
            for pkl_key, metric_name in metric_map.items():
                if pkl_key in training:
                    metrics[metric_name].extend(training[pkl_key])
            if 'iterations' in training:
                iterations.extend(training['iterations'])

    return metrics, np.array(iterations), task_boundaries, n_tasks, data


def compute_cl_metrics(data):
    """Compute CL metrics from task_performance_matrix."""
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
    avg_mse = np.mean(matrix[-1, :])

    bwt_values = [matrix[i, i] - matrix[-1, i] for i in range(n_tasks - 1)]
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    forgetting_values = [max(0, matrix[-1, i] - np.min(matrix[:, i])) for i in range(n_tasks - 1)]
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {'Avg_MSE': avg_mse, 'BWT': bwt, 'Forgetting': forgetting}


def aggregate_curves(all_seed_data, metric_key):
    """Aggregate curves across seeds."""
    all_values = []
    all_iters = []

    for iters, metrics in all_seed_data:
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            all_iters.append(iters)
            all_values.append(np.array(metrics[metric_key]))

    if not all_iters:
        return None, None, None

    min_len = min(len(v) for v in all_values)
    truncated_iters = all_iters[0][:min_len]
    truncated_values = np.array([v[:min_len] for v in all_values])

    return truncated_iters, np.mean(truncated_values, axis=0), np.std(truncated_values, axis=0)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label (a, b, c, d) in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')


def main():
    """Generate publication figure."""
    print("Loading data...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_condition_data = {}
    all_cl_metrics = {}
    task_boundaries = None

    for cond_name, pkl_files in CONDITIONS.items():
        all_condition_data[cond_name] = []
        all_cl_metrics[cond_name] = []

        for pkl_file in pkl_files:
            pkl_path = DATA_DIR / pkl_file
            if pkl_path.exists():
                metrics, iters, boundaries, _, data = load_metrics_from_pkl(pkl_path)
                all_condition_data[cond_name].append((iters, dict(metrics)))
                if task_boundaries is None:
                    task_boundaries = boundaries
                cl_metrics = compute_cl_metrics(data)
                if cl_metrics:
                    all_cl_metrics[cond_name].append(cl_metrics)

    # Aggregate CL metrics
    agg_metrics = {}
    for cond_name, seed_metrics in all_cl_metrics.items():
        if seed_metrics:
            agg_metrics[cond_name] = {
                'Avg_MSE_mean': np.mean([m['Avg_MSE'] for m in seed_metrics]),
                'Avg_MSE_std': np.std([m['Avg_MSE'] for m in seed_metrics]),
                'BWT_mean': np.mean([m['BWT'] for m in seed_metrics]),
                'BWT_std': np.std([m['BWT'] for m in seed_metrics]),
                'Forgetting_mean': np.mean([m['Forgetting'] for m in seed_metrics]),
                'Forgetting_std': np.std([m['Forgetting'] for m in seed_metrics]),
            }

    # Create figure - 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))  # Nature single column ~3.5", double ~7.2"

    # Panel (a): Test MSE (Experience Replay)
    ax = axes[0, 0]
    add_panel_label(ax, 'a')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                           color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    # Task boundaries (show first 3)
    if task_boundaries:
        for i, boundary in enumerate(task_boundaries[:3]):
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test MSE (Experience)')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (b): Hamiltonian Loss
    ax = axes[0, 1]
    add_panel_label(ax, 'b')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'H')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                           color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:3]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hamiltonian Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Avg MSE Bar Chart
    ax = axes[1, 0]
    add_panel_label(ax, 'c')

    cond_names = list(CONDITIONS.keys())
    x_pos = np.arange(len(cond_names))
    means = [agg_metrics[c]['Avg_MSE_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Avg_MSE_std'] for c in cond_names]
    colors_list = [COLORS[c]['line'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8)
    ax.set_ylabel('Average MSE')
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    # Panel (d): Forgetting Bar Chart
    ax = axes[1, 1]
    add_panel_label(ax, 'd')

    means = [agg_metrics[c]['Forgetting_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Forgetting_std'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8)
    ax.set_ylabel('Forgetting')
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(OUTPUT_DIR / 'publication_figure.png', dpi=300, facecolor='white')
    fig.savefig(OUTPUT_DIR / 'publication_figure.pdf', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'publication_figure.svg', facecolor='white')

    print(f"✓ Saved: publication_figure.png/pdf/svg")
    print(f"  Location: {OUTPUT_DIR}")

    # Also copy to paperFigures
    paper_fig_dir = Path("/Users/kraghavan/Desktop/JMLR_paper/Allyson-nonsmooth-dynamics/paperFigures")
    if paper_fig_dir.exists():
        fig.savefig(paper_fig_dir / 'sine_multiseed_results.png', dpi=300, facecolor='white')
        fig.savefig(paper_fig_dir / 'sine_multiseed_results.pdf', facecolor='white')
        print(f"✓ Also saved to: {paper_fig_dir}")

    plt.close()


if __name__ == '__main__':
    main()
