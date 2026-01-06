#!/usr/bin/env python3
"""
Generate Nature Communications style publication figures:
- 2-task figures: 3 panels (no forgetting)
- 10-task figure: 6 panels (add BWT and gradient norm)

Added by Claude: Updated publication-quality figures for JMLR paper.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
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

# Pastel color scheme
COLORS = {
    'C1: Baseline': {'line': '#E57373', 'fill': '#FFCDD2'},
    'C2: Heuristics': {'line': '#FFB74D', 'fill': '#FFE0B2'},
    'C3: Arch Search': {'line': '#64B5F6', 'fill': '#BBDEFB'},
    'C4: AWB Full': {'line': '#BA68C8', 'fill': '#E1BEE7'},
}

SHORT_LABELS = {
    'C1: Baseline': 'Baseline',
    'C2: Heuristics': 'Heuristics',
    'C3: Arch Search': 'Arch Search',
    'C4: AWB Full': 'AWB Full',
}

PAPER_FIG_DIR = Path("/Users/kraghavan/Desktop/JMLR_paper/Allyson-nonsmooth-dynamics/paperFigures")


def load_metrics_from_pkl(pkl_path, is_classification=False):
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
                'grad_norm': 'grad_norm',
                'train_metric': 'train_metric',
                'test_current': 'test_cur',
                'test_experience': 'test_exp',
            }
            for pkl_key, metric_name in metric_map.items():
                if pkl_key in training:
                    metrics[metric_name].extend(training[pkl_key])
            if 'iterations' in training:
                iterations.extend(training['iterations'])

    return metrics, np.array(iterations), task_boundaries, n_tasks, data


def compute_cl_metrics(data, is_classification=False):
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
    avg_metric = np.mean(matrix[-1, :])

    if is_classification:
        bwt_values = [matrix[-1, i] - matrix[i, i] for i in range(n_tasks - 1)]
        forgetting_values = [max(0, np.max(matrix[:, i]) - matrix[-1, i]) for i in range(n_tasks - 1)]
    else:
        bwt_values = [matrix[i, i] - matrix[-1, i] for i in range(n_tasks - 1)]
        forgetting_values = [max(0, matrix[-1, i] - np.min(matrix[:, i])) for i in range(n_tasks - 1)]

    bwt = np.mean(bwt_values) if bwt_values else 0.0
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {'Avg_Metric': avg_metric, 'BWT': bwt, 'Forgetting': forgetting}


def aggregate_curves(seed_data_list, metric_key):
    """Aggregate curves across seeds."""
    all_values = []
    all_iters = []
    for iters, metrics in seed_data_list:
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            all_iters.append(iters)
            all_values.append(np.array(metrics[metric_key]))
    if not all_iters:
        return None, None, None
    min_len = min(len(v) for v in all_values)
    truncated_iters = all_iters[0][:min_len]
    truncated_values = np.array([v[:min_len] for v in all_values])
    mean_vals = np.mean(truncated_values, axis=0)
    std_vals = np.std(truncated_values, axis=0) if len(truncated_values) > 1 else np.zeros_like(mean_vals)
    return truncated_iters, mean_vals, std_vals


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')


def generate_2task_figure(dataset_name, data_dir, conditions, output_name,
                          is_classification=False, metric_label="MSE"):
    """Generate 3-panel figure for 2-task experiments (no forgetting panel)."""
    print(f"\nGenerating 3-panel figure for {dataset_name} (2 tasks)...")

    # Load data
    all_condition_data = {}
    all_cl_metrics = {}
    task_boundaries = None
    n_tasks = None

    for cond_name, pkl_file in conditions.items():
        pkl_path = data_dir / pkl_file
        if pkl_path.exists():
            metrics, iters, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path, is_classification)
            all_condition_data[cond_name] = [(iters, dict(metrics))]
            if task_boundaries is None:
                task_boundaries = boundaries
                n_tasks = num_tasks
            cl_metrics = compute_cl_metrics(data, is_classification)
            if cl_metrics:
                all_cl_metrics[cond_name] = [cl_metrics]

    # Aggregate metrics
    agg_metrics = {}
    for cond_name, seed_metrics in all_cl_metrics.items():
        if seed_metrics:
            agg_metrics[cond_name] = {
                'Avg_Metric_mean': np.mean([m['Avg_Metric'] for m in seed_metrics]),
                'BWT_mean': np.mean([m['BWT'] for m in seed_metrics]),
            }

    # Create 3-panel figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Panel (a): Test metric (Experience Replay)
    ax = axes[0]
    add_panel_label(ax, 'a')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, _ = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)

    if task_boundaries:
        for boundary in task_boundaries:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Iteration')
    y_label = 'Test Accuracy (Exp.)' if is_classification else 'Test MSE (Exp.)'
    ax.set_ylabel(y_label)
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    if is_classification:
        ax.set_ylim(0, 1.05)

    # Panel (b): Hamiltonian Loss
    ax = axes[1]
    add_panel_label(ax, 'b')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, _ = aggregate_curves(seed_data_list, 'H')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)

    if task_boundaries:
        for boundary in task_boundaries:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hamiltonian Loss')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Average Metric Bar Chart
    ax = axes[2]
    add_panel_label(ax, 'c')

    cond_names = [c for c in conditions.keys() if c in agg_metrics]
    x_pos = np.arange(len(cond_names))
    means = [agg_metrics[c]['Avg_Metric_mean'] for c in cond_names]
    colors_list = [COLORS[c]['line'] for c in cond_names]

    bars = ax.bar(x_pos, means, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    for bar, mean in zip(bars, means):
        offset = 0.02 if is_classification else 0.002
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel(f'Avg {metric_label}')
    if is_classification:
        ax.set_ylim(0, 1.15)
    else:
        ax.set_ylim(0, max(means) * 1.3)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    # Save
    output_dir = data_dir.parent / f"{data_dir.name}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f'{output_name}.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / f'{output_name}.pdf', facecolor='white')
    print(f"  ✓ Saved: {output_dir / output_name}.png/pdf")

    if PAPER_FIG_DIR.exists():
        fig.savefig(PAPER_FIG_DIR / f'{output_name}.png', dpi=300, facecolor='white')
        fig.savefig(PAPER_FIG_DIR / f'{output_name}.pdf', facecolor='white')
        print(f"  ✓ Also saved to: {PAPER_FIG_DIR / output_name}.pdf")

    plt.close()


def generate_10task_figure(dataset_name, data_dir, conditions, output_name):
    """Generate 6-panel figure for 10-task experiment with BWT and gradient norm."""
    print(f"\nGenerating 6-panel figure for {dataset_name} (10 tasks, 3 seeds)...")

    # Load data
    all_condition_data = {}
    all_cl_metrics = {}
    task_boundaries = None
    n_tasks = None

    for cond_name, pkl_files in conditions.items():
        all_condition_data[cond_name] = []
        all_cl_metrics[cond_name] = []

        for pkl_file in pkl_files:
            pkl_path = data_dir / pkl_file
            if pkl_path.exists():
                metrics, iters, boundaries, num_tasks, data = load_metrics_from_pkl(pkl_path)
                all_condition_data[cond_name].append((iters, dict(metrics)))
                if task_boundaries is None:
                    task_boundaries = boundaries
                    n_tasks = num_tasks
                cl_metrics = compute_cl_metrics(data, is_classification=False)
                if cl_metrics:
                    all_cl_metrics[cond_name].append(cl_metrics)

    # Aggregate metrics
    agg_metrics = {}
    for cond_name, seed_metrics in all_cl_metrics.items():
        if seed_metrics:
            agg_metrics[cond_name] = {
                'Avg_Metric_mean': np.mean([m['Avg_Metric'] for m in seed_metrics]),
                'Avg_Metric_std': np.std([m['Avg_Metric'] for m in seed_metrics]),
                'BWT_mean': np.mean([m['BWT'] for m in seed_metrics]),
                'BWT_std': np.std([m['BWT'] for m in seed_metrics]),
                'Forgetting_mean': np.mean([m['Forgetting'] for m in seed_metrics]),
                'Forgetting_std': np.std([m['Forgetting'] for m in seed_metrics]),
            }

    # Create 6-panel figure (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Panel (a): Test MSE (Experience Replay)
    ax = axes[0, 0]
    add_panel_label(ax, 'a')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'test_exp')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:5]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test MSE (Exp.)')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (b): Hamiltonian Loss
    ax = axes[0, 1]
    add_panel_label(ax, 'b')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'H')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:5]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hamiltonian Loss')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Gradient Norm
    ax = axes[0, 2]
    add_panel_label(ax, 'c')

    for cond_name, seed_data_list in all_condition_data.items():
        iters, mean_vals, std_vals = aggregate_curves(seed_data_list, 'grad_norm')
        if iters is not None:
            ax.plot(iters, mean_vals, label=SHORT_LABELS[cond_name],
                   color=COLORS[cond_name]['line'], linewidth=1.5, alpha=0.9)
            if np.any(std_vals > 0):
                ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                               color=COLORS[cond_name]['fill'], alpha=0.4, linewidth=0)

    if task_boundaries:
        for boundary in task_boundaries[:5]:
            ax.axvline(x=boundary, color='#888888', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (d): Average MSE Bar Chart
    ax = axes[1, 0]
    add_panel_label(ax, 'd')

    cond_names = [c for c in conditions.keys() if c in agg_metrics]
    x_pos = np.arange(len(cond_names))
    means = [agg_metrics[c]['Avg_Metric_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Avg_Metric_std'] for c in cond_names]
    colors_list = [COLORS[c]['line'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Avg MSE')
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    # Panel (e): Backward Transfer Bar Chart
    ax = axes[1, 1]
    add_panel_label(ax, 'e')

    means = [agg_metrics[c]['BWT_mean'] for c in cond_names]
    stds = [agg_metrics[c]['BWT_std'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        offset = 0.0005 if mean >= 0 else -0.001
        va = 'bottom' if mean >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
               f'{mean:.4f}', ha='center', va=va, fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('BWT (↑ better)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    # Panel (f): Forgetting Bar Chart
    ax = axes[1, 2]
    add_panel_label(ax, 'f')

    means = [agg_metrics[c]['Forgetting_mean'] for c in cond_names]
    stds = [agg_metrics[c]['Forgetting_std'] for c in cond_names]

    bars = ax.bar(x_pos, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.8, capsize=4,
                  error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_LABELS[c] for c in cond_names], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Forgetting (↓ better)')
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

    plt.tight_layout()

    # Save
    output_dir = data_dir.parent / "results_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f'{output_name}.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / f'{output_name}.pdf', facecolor='white')
    print(f"  ✓ Saved: {output_dir / output_name}.png/pdf")

    if PAPER_FIG_DIR.exists():
        fig.savefig(PAPER_FIG_DIR / f'{output_name}.png', dpi=300, facecolor='white')
        fig.savefig(PAPER_FIG_DIR / f'{output_name}.pdf', facecolor='white')
        print(f"  ✓ Also saved to: {PAPER_FIG_DIR / output_name}.pdf")

    plt.close()


def main():
    """Generate all publication figures."""
    print("="*70)
    print("GENERATING UPDATED PUBLICATION FIGURES")
    print("="*70)

    base_dir = Path("runs__/analysis/data_analysis")

    # 1. Sine 2-task (3 panels, no forgetting)
    sine_2task_conditions = {
        'C1: Baseline': 'sine_condition1_run0/regression_sine_fcnn_run0_records.pkl',
        'C2: Heuristics': 'sine_condition2_run0/regression_sine_fcnn_run0_records.pkl',
        'C3: Arch Search': 'sine_condition3_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
        'C4: AWB Full': 'sine_condition4_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
    }
    generate_2task_figure(
        dataset_name="Sine Regression",
        data_dir=base_dir / "Sine_2Tasks",
        conditions=sine_2task_conditions,
        output_name="sine_2task_results",
        is_classification=False,
        metric_label="MSE"
    )

    # 2. MNIST 2-task (3 panels, no forgetting)
    mnist_conditions = {
        'C1: Baseline': 'mnist_condition1_run0/classification_mnist_cnn_run0_records.pkl',
        'C2: Heuristics': 'mnist_condition2_run0/classification_mnist_cnn_run0_records.pkl',
        'C3: Arch Search': 'mnist_condition3_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
        'C4: AWB Full': 'mnist_condition4_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
    }
    generate_2task_figure(
        dataset_name="MNIST Classification",
        data_dir=base_dir / "mnist",
        conditions=mnist_conditions,
        output_name="mnist_2task_results",
        is_classification=True,
        metric_label="Accuracy"
    )

    # 3. Sine 10-task (6 panels with BWT and gradient norm)
    sine_10task_conditions = {
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
    generate_10task_figure(
        dataset_name="Sine Regression",
        data_dir=base_dir / "sine_3seed" / "results",
        conditions=sine_10task_conditions,
        output_name="sine_10task_3seed_results"
    )

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)


if __name__ == '__main__':
    main()
