#!/usr/bin/env python3
"""
Generate comprehensive comparison plots matching JMLR paper format.
Loads pkl files, computes CL metrics, and generates publication-quality plots.
Ignores warmup phases - only uses main_training data.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# Data directory
DATA_DIR = Path("runs__/analysis/data_analysis")
OUTPUT_DIR = Path("runs__/analysis")

# Color scheme matching reference plots
COLORS = {
    'baseline': '#1f77b4',      # blue
    'heuristics': '#ff7f0e',    # orange
    'arch_search': '#2ca02c',   # green
    'awb_full': '#d62728',      # red
}

CONDITION_LABELS = {
    'baseline': 'baseline',
    'heuristics': 'heuristics',
    'arch_search': 'arch_search',
    'awb_full': 'awb_full',
}


def load_pkl_data(dataset='sine'):
    """Load pkl data for all 4 conditions."""
    if dataset == 'sine':
        conditions = {
            'baseline': 'sine_condition1_run0/regression_sine_fcnn_run0_records.pkl',
            'heuristics': 'sine_condition2_run0/regression_sine_fcnn_run0_records.pkl',
            'arch_search': 'sine_condition3_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
            'awb_full': 'sine_condition4_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
        }
    else:  # mnist
        conditions = {
            'baseline': 'mnist_condition1_run0/classification_mnist_cnn_run0_records.pkl',
            'heuristics': 'mnist_condition2_run0/classification_mnist_cnn_run0_records.pkl',
            'arch_search': 'mnist_condition3_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
            'awb_full': 'mnist_condition4_awb_run0/classification_mnist_cnn_awb_run0_records.pkl',
        }

    all_data = {}
    for cond_name, pkl_file in conditions.items():
        pkl_path = DATA_DIR / pkl_file
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                all_data[cond_name] = pickle.load(f)
            print(f"Loaded: {cond_name} ({pkl_path.name})")
        else:
            print(f"Missing: {cond_name} ({pkl_path})")

    return all_data


def extract_training_metrics(data):
    """Extract main_training metrics across all tasks (ignoring warmup)."""
    metrics = defaultdict(list)
    task_boundaries = []  # (start_iter, end_iter) for each task
    global_iter = 0

    n_tasks = data['metadata']['n_tasks']

    for task_id in range(n_tasks):
        if task_id not in data['tasks']:
            continue

        task_data = data['tasks'][task_id]

        if 'main_training' not in task_data:
            continue

        training = task_data['main_training']
        task_start = global_iter

        # Extract all metrics
        metric_keys = ['H', 'V', 'dV', 'dV_dx', 'dV_dtheta', 'grad_norm',
                       'train_metric', 'test_current', 'test_experience']

        for key in metric_keys:
            if key in training:
                metrics[key].extend(training[key])

        # Track iterations
        if 'iterations' in training:
            n_iters = len(training['iterations'])
            global_iter += n_iters

        task_boundaries.append((task_start, global_iter))

    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    return dict(metrics), task_boundaries, n_tasks


def compute_cl_metrics(data):
    """Compute ACC, BWT, FWT, Forgetting from task_performance_matrix."""
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

    # ACC: Average of final row (performance after all training)
    acc = np.mean(matrix[-1, :])

    # BWT: Average backward transfer (excluding diagonal)
    # BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    bwt_values = []
    for i in range(n_tasks - 1):
        bwt_values.append(matrix[-1, i] - matrix[i, i])
    bwt = np.mean(bwt_values) if bwt_values else 0.0

    # FWT: Forward transfer
    # FWT = (1/(T-1)) * sum_{i=2}^{T} (R_{i-1,i} - b_i) where b_i is random baseline
    # Simplified: compare to task 0 performance
    fwt_values = []
    for i in range(1, n_tasks):
        if i > 0:
            fwt_values.append(matrix[i-1, i] - matrix[0, 0])
    fwt = np.mean(fwt_values) if fwt_values else 0.0

    # Forgetting: Maximum forgetting per task
    # F_i = max_{j in {1,...,T-1}} (R_{j,i} - R_{T,i})
    forgetting_values = []
    for i in range(n_tasks - 1):
        max_perf = np.max(matrix[:, i])
        final_perf = matrix[-1, i]
        forgetting_values.append(max(0, max_perf - final_perf))
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {
        'ACC': acc,
        'BWT': bwt,
        'FWT': fwt,
        'Forgetting': forgetting,
        'matrix': matrix,
        'n_tasks': n_tasks
    }


def add_task_shading(ax, task_boundaries, alpha=0.1):
    """Add alternating task boundary shading."""
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#ffccff',
              '#ccffff', '#ffddcc', '#ddccff', '#ccffdd', '#ffccdd']

    for i, (start, end) in enumerate(task_boundaries):
        ax.axvspan(start, end, alpha=alpha, color=colors[i % len(colors)], zorder=0)


def plot_comprehensive_overview(all_data, all_metrics, all_boundaries, dataset, output_dir):
    """Generate comprehensive overview plot matching reference format."""
    fig = plt.figure(figsize=(16, 14))

    # Create grid: 4 rows, 3 cols (last row spans all)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2], hspace=0.3, wspace=0.25)

    fig.suptitle(f'Comprehensive Training Overview with Transfer Metrics - {dataset.upper()} Dataset',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: H, V, Gradient Norm
    row1_metrics = [('H', 'Total Loss (H)'), ('V', 'Experience Replay Loss (V)'), ('grad_norm', 'Gradient Norm')]
    # Row 2: dV, dV_dx, dV_dtheta
    row2_metrics = [('dV', 'Total Regularization (dV)'), ('dV_dx', 'Regularization w.r.t. Input'),
                    ('dV_dtheta', 'Regularization w.r.t. Parameters')]
    # Row 3: Train, Test Current, Test Experience
    row3_metrics = [('train_metric', 'Train Metric'), ('test_current', 'Test Current Task'),
                    ('test_experience', 'Test Experience Replay')]

    all_rows = [row1_metrics, row2_metrics, row3_metrics]

    # Get reference boundaries for shading
    ref_cond = list(all_boundaries.keys())[0]
    ref_bounds = all_boundaries[ref_cond]

    for row_idx, row_metrics in enumerate(all_rows):
        for col_idx, (metric_key, title) in enumerate(row_metrics):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Add task shading
            add_task_shading(ax, ref_bounds)

            for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
                if cond_name not in all_metrics:
                    continue
                metrics = all_metrics[cond_name]
                if metric_key in metrics and len(metrics[metric_key]) > 0:
                    values = metrics[metric_key]
                    x = np.arange(len(values))
                    ax.plot(x, values, label=cond_name, color=COLORS[cond_name],
                           alpha=0.8, linewidth=1.5)

            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_key)
            ax.set_title(title)
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)

    # Row 4: BWT and Forgetting over time (spans all columns)
    ax_bwt = fig.add_subplot(gs[3, :])
    ax_bwt.set_title('Backward Transfer (BWT) and Forgetting Over Time', fontsize=12, fontweight='bold')

    # Add task shading
    add_task_shading(ax_bwt, ref_bounds)

    # Compute running BWT and Forgetting for each condition
    for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
        if cond_name not in all_data:
            continue

        cl_metrics = compute_cl_metrics(all_data[cond_name])
        if cl_metrics is None:
            continue

        matrix = cl_metrics['matrix']
        n_tasks = cl_metrics['n_tasks']

        # Compute BWT after each task
        bwt_over_time = []
        forgetting_over_time = []

        for t in range(1, n_tasks):
            # BWT up to task t
            bwt_vals = [matrix[t, i] - matrix[i, i] for i in range(t)]
            bwt_over_time.append(np.mean(bwt_vals) if bwt_vals else 0)

            # Forgetting up to task t
            forg_vals = []
            for i in range(t):
                max_perf = np.max(matrix[:t+1, i])
                forg_vals.append(max(0, max_perf - matrix[t, i]))
            forgetting_over_time.append(np.mean(forg_vals) if forg_vals else 0)

        # Map to iteration space
        task_iters = [b[1] for b in all_boundaries[cond_name]][:-1]  # End of each task
        if len(task_iters) >= len(bwt_over_time):
            task_iters = task_iters[:len(bwt_over_time)]

        if task_iters:
            ax_bwt.plot(task_iters, bwt_over_time, '-', label=f'{cond_name} BWT',
                       color=COLORS[cond_name], linewidth=2)
            ax_bwt.plot(task_iters, forgetting_over_time, '--', label=f'{cond_name} Forgetting',
                       color=COLORS[cond_name], linewidth=2, alpha=0.7)

    ax_bwt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_bwt.set_xlabel('Iteration')
    ax_bwt.set_ylabel('Transfer Metric Value')
    ax_bwt.legend(loc='upper left', fontsize=7, ncol=4)
    ax_bwt.grid(True, alpha=0.3)

    # Add explanation text
    ax_bwt.text(0.02, 0.98, 'Solid lines: BWT (negative = forgetting old tasks)\n'
                           'Dashed lines: Forgetting measure (higher = more forgetting)',
               transform=ax_bwt.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_comprehensive_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{dataset}_comprehensive_overview.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_comprehensive_overview.png'}")
    plt.close()


def plot_h_v_comparison(all_metrics, all_boundaries, dataset, output_dir):
    """Generate H and V comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Loss Components: H and V - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')

    ref_cond = list(all_boundaries.keys())[0]
    ref_bounds = all_boundaries[ref_cond]

    for idx, (metric_key, title) in enumerate([('H', 'Total Loss (H)'), ('V', 'Experience Replay Loss (V)')]):
        ax = axes[idx]
        add_task_shading(ax, ref_bounds)

        for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
            if cond_name not in all_metrics:
                continue
            metrics = all_metrics[cond_name]
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                values = metrics[metric_key]
                ax.plot(values, label=cond_name, color=COLORS[cond_name], linewidth=1.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'{metric_key} Loss')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_H_V_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_H_V_comparison.png'}")
    plt.close()


def plot_learning_curves(all_metrics, all_boundaries, dataset, output_dir):
    """Generate learning curves plot (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Learning Curves - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')

    ref_cond = list(all_boundaries.keys())[0]
    ref_bounds = all_boundaries[ref_cond]

    curve_metrics = [
        ('H', 'Total Loss (H)'),
        ('train_metric', 'Train Metric'),
        ('test_current', 'Test Current Task'),
        ('test_experience', 'Test Experience Replay'),
    ]

    for idx, (metric_key, title) in enumerate(curve_metrics):
        ax = axes[idx // 2, idx % 2]
        add_task_shading(ax, ref_bounds)

        for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
            if cond_name not in all_metrics:
                continue
            metrics = all_metrics[cond_name]
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                values = metrics[metric_key]
                ax.plot(values, label=cond_name, color=COLORS[cond_name], linewidth=1.5)

        ax.set_xlabel('Global Iteration')
        ax.set_ylabel(metric_key.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_learning_curves.png'}")
    plt.close()


def plot_performance_matrix(data, cond_name, dataset, output_dir):
    """Generate task performance matrix heatmap."""
    cl_metrics = compute_cl_metrics(data)
    if cl_metrics is None:
        print(f"  No performance matrix for {cond_name}")
        return

    matrix = cl_metrics['matrix']
    n_tasks = cl_metrics['n_tasks']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance (MSE/Accuracy)', fontsize=10)

    # Add text annotations
    for i in range(n_tasks):
        for j in range(n_tasks):
            val = matrix[i, j]
            text_color = 'white' if val > (matrix.max() + matrix.min()) / 2 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=text_color, fontsize=7)

    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xlabel('Task i (being evaluated)', fontsize=11)
    ax.set_ylabel('After Training Task j', fontsize=11)
    ax.set_title(f'Task Performance Matrix - {dataset.title()} {cond_name.replace("_", " ").title()}',
                fontsize=12, fontweight='bold')

    # Add explanation
    fig.text(0.5, 0.02, 'Diagonal: Performance on current task | Off-diagonal: Backward transfer',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_{cond_name}_performance_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_{cond_name}_performance_matrix.png'}")
    plt.close()


def plot_dv_terms(all_metrics, all_boundaries, dataset, output_dir):
    """Generate dV terms comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Regularization Terms (dV) - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')

    ref_cond = list(all_boundaries.keys())[0]
    ref_bounds = all_boundaries[ref_cond]

    dv_metrics = [
        ('dV', 'Total Regularization (dV)'),
        ('dV_dx', 'Regularization w.r.t. Input'),
        ('dV_dtheta', 'Regularization w.r.t. Parameters'),
    ]

    for idx, (metric_key, title) in enumerate(dv_metrics):
        ax = axes[idx]
        add_task_shading(ax, ref_bounds)

        for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
            if cond_name not in all_metrics:
                continue
            metrics = all_metrics[cond_name]
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                values = metrics[metric_key]
                ax.plot(values, label=cond_name, color=COLORS[cond_name], linewidth=1.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_dV_terms_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_dV_terms_comparison.png'}")
    plt.close()


def plot_test_metrics(all_metrics, all_boundaries, dataset, output_dir):
    """Generate test metrics comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Test Metrics Comparison - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')

    ref_cond = list(all_boundaries.keys())[0]
    ref_bounds = all_boundaries[ref_cond]

    test_metrics = [
        ('test_current', 'Test Current Task'),
        ('test_experience', 'Test Experience Replay'),
    ]

    for idx, (metric_key, title) in enumerate(test_metrics):
        ax = axes[idx]
        add_task_shading(ax, ref_bounds)

        for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
            if cond_name not in all_metrics:
                continue
            metrics = all_metrics[cond_name]
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                values = metrics[metric_key]
                ax.plot(values, label=cond_name, color=COLORS[cond_name], linewidth=1.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Test Performance')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_test_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_test_metrics_comparison.png'}")
    plt.close()


def plot_metrics_comparison(all_metrics, all_boundaries, dataset, output_dir):
    """Generate metrics bar comparison."""
    # Compute final metrics for each condition
    conditions = ['baseline', 'heuristics', 'arch_search', 'awb_full']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Final Metrics Comparison - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')

    metric_names = ['train_metric', 'test_current', 'test_experience']
    titles = ['Final Train Metric', 'Final Test (Current)', 'Final Test (Experience)']

    for idx, (metric_key, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx]

        final_values = []
        valid_conditions = []
        colors_list = []

        for cond in conditions:
            if cond in all_metrics and metric_key in all_metrics[cond]:
                values = all_metrics[cond][metric_key]
                if len(values) > 0:
                    final_values.append(values[-1])
                    valid_conditions.append(cond)
                    colors_list.append(COLORS[cond])

        if final_values:
            bars = ax.bar(valid_conditions, final_values, color=colors_list, alpha=0.8)
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, val in zip(bars, final_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset}_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{dataset}_metrics_comparison.png'}")
    plt.close()


def generate_metrics_summary(all_data, dataset):
    """Generate metrics summary for a dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset.upper()} Dataset - CL Metrics Summary")
    print(f"{'='*70}")

    summary = {}
    for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
        if cond_name not in all_data:
            continue

        cl_metrics = compute_cl_metrics(all_data[cond_name])
        if cl_metrics is None:
            continue

        summary[cond_name] = cl_metrics

        print(f"\n{cond_name}:")
        print(f"  ACC (Avg Performance): {cl_metrics['ACC']:.6f}")
        print(f"  BWT (Backward Transfer): {cl_metrics['BWT']:.6f}")
        print(f"  FWT (Forward Transfer): {cl_metrics['FWT']:.6f}")
        print(f"  Forgetting: {cl_metrics['Forgetting']:.6f}")

    return summary


def main():
    """Main execution."""
    print("="*70)
    print("Comprehensive Plot Generation (JMLR Paper Format)")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for dataset in ['sine', 'mnist']:
        print(f"\n{'='*70}")
        print(f"Processing {dataset.upper()} Dataset")
        print(f"{'='*70}")

        # Load data
        all_data = load_pkl_data(dataset)

        if not all_data:
            print(f"No data found for {dataset}")
            continue

        # Extract metrics
        all_metrics = {}
        all_boundaries = {}

        for cond_name, data in all_data.items():
            metrics, boundaries, n_tasks = extract_training_metrics(data)
            all_metrics[cond_name] = metrics
            all_boundaries[cond_name] = boundaries
            print(f"  {cond_name}: {len(metrics.get('H', []))} iterations, {n_tasks} tasks")

        # Create output directory for this dataset
        dataset_output = OUTPUT_DIR / f"{dataset}_comparison_plots"
        dataset_output.mkdir(parents=True, exist_ok=True)

        # Generate all plots
        print(f"\nGenerating plots for {dataset}...")

        plot_comprehensive_overview(all_data, all_metrics, all_boundaries, dataset, dataset_output)
        plot_h_v_comparison(all_metrics, all_boundaries, dataset, dataset_output)
        plot_learning_curves(all_metrics, all_boundaries, dataset, dataset_output)
        plot_dv_terms(all_metrics, all_boundaries, dataset, dataset_output)
        plot_test_metrics(all_metrics, all_boundaries, dataset, dataset_output)
        plot_metrics_comparison(all_metrics, all_boundaries, dataset, dataset_output)

        # Generate performance matrices for each condition
        for cond_name, data in all_data.items():
            plot_performance_matrix(data, cond_name, dataset, dataset_output)

        # Generate metrics summary
        all_summaries[dataset] = generate_metrics_summary(all_data, dataset)

    # Write summary to file
    print("\n" + "="*70)
    print("Writing metrics summary...")
    print("="*70)

    with open(OUTPUT_DIR / 'metrics_summary.md', 'w') as f:
        f.write("# Continual Learning Metrics Summary\n\n")

        for dataset in ['sine', 'mnist']:
            if dataset not in all_summaries:
                continue

            f.write(f"## {dataset.upper()} Dataset\n\n")
            f.write("| Condition | ACC | BWT | FWT | Forgetting |\n")
            f.write("|-----------|-----|-----|-----|------------|\n")

            for cond_name in ['baseline', 'heuristics', 'arch_search', 'awb_full']:
                if cond_name not in all_summaries[dataset]:
                    continue
                m = all_summaries[dataset][cond_name]
                f.write(f"| {cond_name} | {m['ACC']:.6f} | {m['BWT']:.6f} | {m['FWT']:.6f} | {m['Forgetting']:.6f} |\n")

            f.write("\n")

        f.write("\n## Metric Interpretations\n\n")
        f.write("- **ACC (Average Accuracy/MSE)**: Mean performance across all tasks after training.\n")
        f.write("- **BWT (Backward Transfer)**: Negative values indicate catastrophic forgetting.\n")
        f.write("- **FWT (Forward Transfer)**: Positive values indicate beneficial knowledge transfer.\n")
        f.write("- **Forgetting**: Average maximum performance drop per task. Lower is better.\n")

    print(f"Saved: {OUTPUT_DIR / 'metrics_summary.md'}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
