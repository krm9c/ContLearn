"""
Comprehensive plotting script for ContLearn training results.

Generates visualizations for:
- Training metrics over iterations
- Test metrics (current task vs experience replay) over iterations
- Loss components over iterations
- Eigenvalue distributions per layer over iterations
- Cross-run comparisons and statistics

Usage:
    python plot_results.py <path_to_records.pkl> [--output-dir figures]
    python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl
    python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_records(filepath: str) -> Tuple[Dict[str, Any], bool]:
    """Load records from pickle file.

    Returns:
        (records_dict, is_multi_run): Loaded records and flag indicating if multiple runs
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Check if this is all runs or single run
    is_multi_run = 'runs' in data and 'metadata' in data and 'total_runs' in data['metadata']

    return data, is_multi_run


def extract_time_series(run_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract time series data from a single run."""
    iterations = sorted(run_data['iterations'].keys())

    series = {
        'iterations': np.array(iterations),
        'task_ids': np.array([run_data['iterations'][i]['task_id'] for i in iterations]),
        'steps': np.array([run_data['iterations'][i]['step'] for i in iterations]),
    }

    # Extract losses
    for loss_key in ['H', 'V', 'dV', 'dV_dx', 'dV_dtheta']:
        series[f'loss_{loss_key}'] = np.array([
            run_data['iterations'][i]['losses'][loss_key] for i in iterations
        ])

    # Handle optional dV_dadj for graph problems
    if 'dV_dadj' in run_data['iterations'][iterations[0]]['losses']:
        series['loss_dV_dadj'] = np.array([
            run_data['iterations'][i]['losses']['dV_dadj'] for i in iterations
        ])

    # Extract gradients
    series['grad_norm'] = np.array([
        run_data['iterations'][i]['gradients']['grad_norm'] for i in iterations
    ])

    # Extract metrics
    series['metric_train'] = np.array([
        run_data['iterations'][i]['metrics']['train'] for i in iterations
    ])
    series['metric_test_current'] = np.array([
        run_data['iterations'][i]['metrics']['test_current'] for i in iterations
    ])
    series['metric_test_experience'] = np.array([
        run_data['iterations'][i]['metrics']['test_experience'] for i in iterations
    ])

    return series


def add_task_shading(ax, series: Dict[str, np.ndarray], metadata: Dict[str, Any], alpha: float = 0.1):
    """Add shaded regions to indicate different tasks.

    Args:
        ax: Matplotlib axis object
        series: Time series data with 'iterations' and 'task_ids'
        metadata: Metadata containing task information
        alpha: Transparency for shaded regions
    """
    task_ids = series['task_ids']
    iterations = series['iterations']

    # Get unique tasks
    unique_tasks = np.unique(task_ids)

    # Define colors for different tasks (cycle through if more tasks than colors)
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Add shaded region for each task
    for i, task_id in enumerate(unique_tasks):
        # Find iterations for this task
        task_mask = task_ids == task_id
        task_iters = iterations[task_mask]

        if len(task_iters) > 0:
            # Get min and max iteration for this task
            min_iter = task_iters.min()
            max_iter = task_iters.max()

            # Add shaded region
            color = colors[int(task_id) % len(colors)]
            ax.axvspan(min_iter, max_iter, alpha=alpha, color=color, zorder=0)


def plot_losses(run_data: Dict[str, Any], output_dir: str, run_id: str = ''):
    """Plot all loss components over iterations."""
    metadata = run_data['metadata']
    series = extract_time_series(run_data)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Loss Components - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=14, fontweight='bold')

    # Plot Hamiltonian (H)
    axes[0, 0].plot(series['iterations'], series['loss_H'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Hamiltonian (H)')
    axes[0, 0].set_title('Total Loss (H)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot V (primary loss: MSE or Cross Entropy)
    axes[0, 1].plot(series['iterations'], series['loss_V'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(f'V ({metadata["loss_function"]})')
    axes[0, 1].set_title(f'Primary Loss (V - {metadata["loss_function"]})')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot dV
    axes[0, 2].plot(series['iterations'], series['loss_dV'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('dV')
    axes[0, 2].set_title('dV Term')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot dV/dx
    axes[1, 0].plot(series['iterations'], series['loss_dV_dx'], 'c-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('dV/dx')
    axes[1, 0].set_title('Gradient w.r.t. Input')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot dV/dtheta
    axes[1, 1].plot(series['iterations'], series['loss_dV_dtheta'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('dV/dθ')
    axes[1, 1].set_title('Gradient w.r.t. Parameters')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot gradient norm or dV/dadj for graphs
    if 'loss_dV_dadj' in series:
        axes[1, 2].plot(series['iterations'], series['loss_dV_dadj'], 'orange', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('dV/dAdj')
        axes[1, 2].set_title('Gradient w.r.t. Adjacency')
    else:
        axes[1, 2].plot(series['iterations'], series['grad_norm'], 'orange', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('||dH/dθ||')
        axes[1, 2].set_title('Gradient Norm')
    axes[1, 2].grid(True, alpha=0.3)

    # Add task shading to all subplots
    for ax in axes.flat:
        add_task_shading(ax, series, metadata, alpha=0.08)

    # Add vertical lines for task boundaries
    task_changes = np.where(np.diff(series['task_ids']) != 0)[0] + 1
    for ax in axes.flat:
        for tc in task_changes:
            ax.axvline(series['iterations'][tc], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_losses.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_metrics(run_data: Dict[str, Any], output_dir: str, run_id: str = ''):
    """Plot training and test metrics over iterations."""
    metadata = run_data['metadata']
    series = extract_time_series(run_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Metrics - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=14, fontweight='bold')

    # Plot training metric
    axes[0].plot(series['iterations'], series['metric_train'], 'b-', linewidth=2, label='Train')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel(f'{metadata["metric_function"]}', fontsize=12)
    axes[0].set_title('Training Metric', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Plot test metrics (current task vs experience replay)
    axes[1].plot(series['iterations'], series['metric_test_current'], 'g-', linewidth=2,
                label='Test (Current Task)', marker='o', markersize=4, markevery=max(1, len(series['iterations'])//20))
    axes[1].plot(series['iterations'], series['metric_test_experience'], 'r-', linewidth=2,
                label='Test (Experience Replay)', marker='s', markersize=4, markevery=max(1, len(series['iterations'])//20))
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel(f'{metadata["metric_function"]}', fontsize=12)
    axes[1].set_title('Test Metrics: Current vs Experience', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    # Add task shading to all subplots
    for ax in axes:
        add_task_shading(ax, series, metadata, alpha=0.08)

    # Add vertical lines for task boundaries
    task_changes = np.where(np.diff(series['task_ids']) != 0)[0] + 1
    for ax in axes:
        for tc in task_changes:
            ax.axvline(series['iterations'][tc], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_metrics.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_eigenvalues_violin(run_data: Dict[str, Any], output_dir: str, run_id: str = ''):
    """Plot eigenvalue evolution as violin plots."""
    metadata = run_data['metadata']
    iterations = sorted(run_data['iterations'].keys())

    # Get layer names from first iteration
    first_iter = run_data['iterations'][iterations[0]]
    layer_names_A = sorted(first_iter['eigenvalues']['A'].keys())
    layer_names_B = sorted(first_iter['eigenvalues']['B'].keys())

    # Create subplot grid
    n_layers = max(len(layer_names_A), len(layer_names_B))
    fig = plt.figure(figsize=(20, 4 * n_layers))
    gs = gridspec.GridSpec(n_layers, 2, figure=fig, hspace=0.3)
    fig.suptitle(f'Eigenvalue Evolution (Violin Plot) - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=14, fontweight='bold')

    # Plot A matrices
    for idx, layer_name in enumerate(layer_names_A):
        ax = fig.add_subplot(gs[idx, 0])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['A'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.real(eigs))  # Take real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create violin plot
            positions = iter_indices
            width = max(1, (max(positions) - min(positions)) / len(positions) * 0.7)

            parts = ax.violinplot(eigenvals_per_iter, positions=positions, widths=width,
                                 showmeans=False, showmedians=True, showextrema=True)

            # Style the violin plots
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_edgecolor('darkblue')
                pc.set_alpha(0.7)
                pc.set_linewidth(1.5)

            # Style the other elements
            parts['cmedians'].set_edgecolor('crimson')
            parts['cmedians'].set_linewidth(2.5)
            parts['cbars'].set_edgecolor('darkblue')
            parts['cbars'].set_linewidth(1.5)
            parts['cmaxes'].set_edgecolor('darkblue')
            parts['cmaxes'].set_linewidth(1.5)
            parts['cmins'].set_edgecolor('darkblue')
            parts['cmins'].set_linewidth(1.5)

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Magnitude', fontsize=11)
            ax.set_title(f'A Matrix - {layer_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')

            # Add task shading
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            subset_series = {'iterations': np.array(iter_indices), 'task_ids': task_ids_arr}
            add_task_shading(ax, subset_series, metadata, alpha=0.08)

            # Add task boundaries
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    # Plot B matrices
    for idx, layer_name in enumerate(layer_names_B):
        ax = fig.add_subplot(gs[idx, 1])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['B'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.real(eigs))  # Take real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create violin plot
            positions = iter_indices
            width = max(1, (max(positions) - min(positions)) / len(positions) * 0.7)

            parts = ax.violinplot(eigenvals_per_iter, positions=positions, widths=width,
                                 showmeans=False, showmedians=True, showextrema=True)

            # Style the violin plots
            for pc in parts['bodies']:
                pc.set_facecolor('lightgreen')
                pc.set_edgecolor('darkgreen')
                pc.set_alpha(0.7)
                pc.set_linewidth(1.5)

            # Style the other elements
            parts['cmedians'].set_edgecolor('crimson')
            parts['cmedians'].set_linewidth(2.5)
            parts['cbars'].set_edgecolor('darkgreen')
            parts['cbars'].set_linewidth(1.5)
            parts['cmaxes'].set_edgecolor('darkgreen')
            parts['cmaxes'].set_linewidth(1.5)
            parts['cmins'].set_edgecolor('darkgreen')
            parts['cmins'].set_linewidth(1.5)

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Magnitude', fontsize=11)
            ax.set_title(f'B Matrix - {layer_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')

            # Add task shading
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            subset_series = {'iterations': np.array(iter_indices), 'task_ids': task_ids_arr}
            add_task_shading(ax, subset_series, metadata, alpha=0.08)

            # Add task boundaries
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_eigenvalues.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_eigenvalues_heatmap(run_data: Dict[str, Any], output_dir: str, run_id: str = ''):
    """Plot eigenvalue evolution as heatmaps."""
    metadata = run_data['metadata']
    iterations = sorted(run_data['iterations'].keys())

    # Get layer names from first iteration
    first_iter = run_data['iterations'][iterations[0]]
    layer_names_A = sorted(first_iter['eigenvalues']['A'].keys())
    layer_names_B = sorted(first_iter['eigenvalues']['B'].keys())

    # Create subplot grid
    n_layers = max(len(layer_names_A), len(layer_names_B))
    fig = plt.figure(figsize=(20, 5 * n_layers))
    gs = gridspec.GridSpec(n_layers, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f'Eigenvalue Evolution (Heatmap) - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=14, fontweight='bold')

    # Plot A matrices
    for idx, layer_name in enumerate(layer_names_A):
        ax = fig.add_subplot(gs[idx, 0])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['A'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.abs(np.real(eigs)))  # Take absolute value of real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create heatmap matrix: rows = eigenvalue rank, columns = iterations
            max_len = max(len(eigs) for eigs in eigenvals_per_iter)
            matrix = np.full((max_len, len(iter_indices)), np.nan)

            for i, eigs in enumerate(eigenvals_per_iter):
                sorted_eigs = np.sort(eigs)[::-1]  # Sort by magnitude, descending
                matrix[:len(sorted_eigs), i] = sorted_eigs

            # Plot heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                          extent=[min(iter_indices)-0.5, max(iter_indices)+0.5, max_len-0.5, -0.5],
                          vmin=0)

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Rank', fontsize=11)
            ax.set_title(f'A Matrix - {layer_name}', fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Magnitude', fontsize=10)

            # Add task boundaries as vertical lines
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='black', linestyle='--', alpha=0.8, linewidth=2)

    # Plot B matrices
    for idx, layer_name in enumerate(layer_names_B):
        ax = fig.add_subplot(gs[idx, 1])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['B'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.abs(np.real(eigs)))  # Take absolute value of real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create heatmap matrix: rows = eigenvalue rank, columns = iterations
            max_len = max(len(eigs) for eigs in eigenvals_per_iter)
            matrix = np.full((max_len, len(iter_indices)), np.nan)

            for i, eigs in enumerate(eigenvals_per_iter):
                sorted_eigs = np.sort(eigs)[::-1]  # Sort by magnitude, descending
                matrix[:len(sorted_eigs), i] = sorted_eigs

            # Plot heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='YlGn', interpolation='nearest',
                          extent=[min(iter_indices)-0.5, max(iter_indices)+0.5, max_len-0.5, -0.5],
                          vmin=0)

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Rank', fontsize=11)
            ax.set_title(f'B Matrix - {layer_name}', fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Magnitude', fontsize=10)

            # Add task boundaries as vertical lines
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='black', linestyle='--', alpha=0.8, linewidth=2)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_eigenvalues.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_eigenvalues(run_data: Dict[str, Any], output_dir: str, run_id: str = '', style: str = 'violin'):
    """Plot eigenvalue distributions per layer over iterations.

    Args:
        run_data: Single run data dictionary
        output_dir: Output directory for saving plots
        run_id: Run identifier
        style: Plot style - 'box' for box plots, 'heatmap' for heatmaps
    """
    metadata = run_data['metadata']
    iterations = sorted(run_data['iterations'].keys())

    # Get layer names from first iteration
    first_iter = run_data['iterations'][iterations[0]]
    layer_names_A = sorted(first_iter['eigenvalues']['A'].keys())
    layer_names_B = sorted(first_iter['eigenvalues']['B'].keys())

    if not layer_names_A and not layer_names_B:
        print(f'No eigenvalues found for run {run_id}, skipping eigenvalue plot')
        return

    if style == 'heatmap':
        return plot_eigenvalues_heatmap(run_data, output_dir, run_id)
    elif style == 'violin':
        return plot_eigenvalues_violin(run_data, output_dir, run_id)

    # Create subplot grid for box plots
    n_layers = max(len(layer_names_A), len(layer_names_B))
    fig = plt.figure(figsize=(20, 4 * n_layers))
    gs = gridspec.GridSpec(n_layers, 2, figure=fig)
    fig.suptitle(f'Eigenvalue Evolution - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=14, fontweight='bold')

    # Plot A matrices
    for idx, layer_name in enumerate(layer_names_A):
        ax = fig.add_subplot(gs[idx, 0])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['A'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.real(eigs))  # Take real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create box plot with improved styling
            positions = iter_indices
            width = max(1, (max(positions) - min(positions)) / len(positions) * 0.6)

            bp = ax.boxplot(eigenvals_per_iter, positions=positions, widths=width,
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', edgecolor='darkblue',
                                       linewidth=1.5, alpha=0.8),
                           medianprops=dict(color='crimson', linewidth=2.5),
                           whiskerprops=dict(color='darkblue', linewidth=1.5, linestyle='-'),
                           capprops=dict(color='darkblue', linewidth=1.5))

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Magnitude', fontsize=11)
            ax.set_title(f'A Matrix - {layer_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')

            # Add task shading
            # Build series dict for this subset
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            subset_series = {'iterations': np.array(iter_indices), 'task_ids': task_ids_arr}
            add_task_shading(ax, subset_series, metadata, alpha=0.08)

            # Add task boundaries
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    # Plot B matrices
    for idx, layer_name in enumerate(layer_names_B):
        ax = fig.add_subplot(gs[idx, 1])

        # Collect eigenvalues for this layer across iterations
        eigenvals_per_iter = []
        iter_indices = []

        for it in iterations:
            eigs = run_data['iterations'][it]['eigenvalues']['B'].get(layer_name, np.array([]))
            if len(eigs) > 0:
                eigenvals_per_iter.append(np.real(eigs))  # Take real part
                iter_indices.append(it)

        if eigenvals_per_iter:
            # Create box plot with improved styling
            positions = iter_indices
            width = max(1, (max(positions) - min(positions)) / len(positions) * 0.6)

            bp = ax.boxplot(eigenvals_per_iter, positions=positions, widths=width,
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor='lightgreen', edgecolor='darkgreen',
                                       linewidth=1.5, alpha=0.8),
                           medianprops=dict(color='crimson', linewidth=2.5),
                           whiskerprops=dict(color='darkgreen', linewidth=1.5, linestyle='-'),
                           capprops=dict(color='darkgreen', linewidth=1.5))

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Eigenvalue Magnitude', fontsize=11)
            ax.set_title(f'B Matrix - {layer_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')

            # Add task shading
            # Build series dict for this subset
            task_ids_arr = np.array([run_data['iterations'][i]['task_id'] for i in iter_indices])
            subset_series = {'iterations': np.array(iter_indices), 'task_ids': task_ids_arr}
            add_task_shading(ax, subset_series, metadata, alpha=0.08)

            # Add task boundaries
            task_changes_idx = np.where(np.diff(task_ids_arr) != 0)[0]
            for tc_idx in task_changes_idx:
                ax.axvline(iter_indices[tc_idx + 1], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_eigenvalues.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_combined_metrics(run_data: Dict[str, Any], output_dir: str, run_id: str = ''):
    """Combined plot: losses, metrics, and gradient norm in one figure."""
    metadata = run_data['metadata']
    series = extract_time_series(run_data)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Training Overview - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]} (Run {run_id})',
                 fontsize=16, fontweight='bold')

    # Row 1: Main losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(series['iterations'], series['loss_H'], 'b-', linewidth=2)
    ax1.set_ylabel('Hamiltonian (H)', fontsize=11)
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(series['iterations'], series['loss_V'], 'r-', linewidth=2)
    ax2.set_ylabel(f'V ({metadata["loss_function"]})', fontsize=11)
    ax2.set_title('Primary Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(series['iterations'], series['grad_norm'], 'orange', linewidth=2)
    ax3.set_ylabel('||dH/dθ||', fontsize=11)
    ax3.set_title('Gradient Norm', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Row 2: Gradient components
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(series['iterations'], series['loss_dV_dx'], 'c-', linewidth=2)
    ax4.set_ylabel('dV/dx', fontsize=11)
    ax4.set_title('Gradient w.r.t. Input', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(series['iterations'], series['loss_dV_dtheta'], 'm-', linewidth=2)
    ax5.set_ylabel('dV/dθ', fontsize=11)
    ax5.set_title('Gradient w.r.t. Parameters', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(series['iterations'], series['loss_dV'], 'g-', linewidth=2)
    ax6.set_ylabel('dV', fontsize=11)
    ax6.set_title('dV Term', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Row 3: Metrics (spanning all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(series['iterations'], series['metric_train'], 'b-', linewidth=2.5,
            label='Train', marker='o', markersize=5, markevery=max(1, len(series['iterations'])//20))
    ax7.plot(series['iterations'], series['metric_test_current'], 'g-', linewidth=2.5,
            label='Test (Current Task)', marker='s', markersize=5, markevery=max(1, len(series['iterations'])//20))
    ax7.plot(series['iterations'], series['metric_test_experience'], 'r-', linewidth=2.5,
            label='Test (Experience)', marker='^', markersize=5, markevery=max(1, len(series['iterations'])//20))
    ax7.set_xlabel('Iteration', fontsize=12)
    ax7.set_ylabel(f'{metadata["metric_function"]}', fontsize=12)
    ax7.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=11, loc='best')

    # Add task shading to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        add_task_shading(ax, series, metadata, alpha=0.08)

    # Add task boundaries to all subplots
    task_changes = np.where(np.diff(series['task_ids']) != 0)[0] + 1
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        for tc in task_changes:
            ax.axvline(series['iterations'][tc], color='k', linestyle='--', alpha=0.5, linewidth=1.5)

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_run{run_id}_overview.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_multi_run_comparison(all_runs_data: Dict[str, Any], output_dir: str):
    """Plot comparison across multiple runs."""
    metadata = all_runs_data['metadata']

    # Extract data from all runs
    all_series = {}
    for run_id, run_data in all_runs_data['runs'].items():
        all_series[run_id] = extract_time_series(run_data)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Run Comparison - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Hamiltonian across runs
    for run_id, series in all_series.items():
        axes[0, 0].plot(series['iterations'], series['loss_H'], linewidth=2, label=f'Run {run_id}', alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Hamiltonian (H)')
    axes[0, 0].set_title('Total Loss Across Runs')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Primary loss across runs
    for run_id, series in all_series.items():
        axes[0, 1].plot(series['iterations'], series['loss_V'], linewidth=2, label=f'Run {run_id}', alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Primary Loss (V)')
    axes[0, 1].set_title('Primary Loss Across Runs')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Test (current) metric across runs
    for run_id, series in all_series.items():
        axes[1, 0].plot(series['iterations'], series['metric_test_current'], linewidth=2,
                       label=f'Run {run_id}', alpha=0.7)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Test Metric (Current)')
    axes[1, 0].set_title('Test Performance (Current Task) Across Runs')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Test (experience) metric across runs
    for run_id, series in all_series.items():
        axes[1, 1].plot(series['iterations'], series['metric_test_experience'], linewidth=2,
                       label=f'Run {run_id}', alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Test Metric (Experience)')
    axes[1, 1].set_title('Test Performance (Experience Replay) Across Runs')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Add task shading to all subplots (use first run's task structure)
    first_run_id = list(all_series.keys())[0]
    first_series = all_series[first_run_id]
    first_run_metadata = all_runs_data['runs'][first_run_id]['metadata']
    for ax in axes.flat:
        add_task_shading(ax, first_series, first_run_metadata, alpha=0.08)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_allruns_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def plot_multi_run_statistics(all_runs_data: Dict[str, Any], output_dir: str):
    """Plot statistics (mean ± std) across multiple runs."""
    metadata = all_runs_data['metadata']

    # Extract data from all runs
    all_series = {}
    for run_id, run_data in all_runs_data['runs'].items():
        all_series[run_id] = extract_time_series(run_data)

    # Find common iterations across all runs
    common_iters = set(all_series[list(all_series.keys())[0]]['iterations'])
    for series in all_series.values():
        common_iters = common_iters.intersection(set(series['iterations']))
    common_iters = sorted(list(common_iters))

    if not common_iters:
        print('No common iterations across runs, skipping statistics plot')
        return

    # Compute statistics for each metric
    def compute_stats(metric_name):
        values = []
        for run_id, series in all_series.items():
            # Find indices of common iterations
            idx = [np.where(series['iterations'] == it)[0][0] for it in common_iters]
            values.append(series[metric_name][idx])
        values = np.array(values)
        return np.mean(values, axis=0), np.std(values, axis=0)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Run Statistics (Mean ± Std) - {metadata["prob"]} {metadata["dataset"]} {metadata["network"]}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Hamiltonian
    mean_H, std_H = compute_stats('loss_H')
    axes[0, 0].plot(common_iters, mean_H, 'b-', linewidth=2.5, label='Mean')
    axes[0, 0].fill_between(common_iters, mean_H - std_H, mean_H + std_H, alpha=0.3, label='±1 Std')
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Hamiltonian (H)', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # Plot 2: Primary loss
    mean_V, std_V = compute_stats('loss_V')
    axes[0, 1].plot(common_iters, mean_V, 'r-', linewidth=2.5, label='Mean')
    axes[0, 1].fill_between(common_iters, mean_V - std_V, mean_V + std_V, alpha=0.3, label='±1 Std')
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel('Primary Loss (V)', fontsize=11)
    axes[0, 1].set_title('Primary Loss', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # Plot 3: Test (current) metric
    mean_test_curr, std_test_curr = compute_stats('metric_test_current')
    axes[1, 0].plot(common_iters, mean_test_curr, 'g-', linewidth=2.5, label='Mean')
    axes[1, 0].fill_between(common_iters, mean_test_curr - std_test_curr,
                            mean_test_curr + std_test_curr, alpha=0.3, label='±1 Std')
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel('Test Metric (Current)', fontsize=11)
    axes[1, 0].set_title('Test Performance (Current Task)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # Plot 4: Test (experience) metric
    mean_test_exp, std_test_exp = compute_stats('metric_test_experience')
    axes[1, 1].plot(common_iters, mean_test_exp, 'm-', linewidth=2.5, label='Mean')
    axes[1, 1].fill_between(common_iters, mean_test_exp - std_test_exp,
                            mean_test_exp + std_test_exp, alpha=0.3, label='±1 Std')
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel('Test Metric (Experience)', fontsize=11)
    axes[1, 1].set_title('Test Performance (Experience Replay)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)

    # Add task shading to all subplots (use first run's task structure)
    first_run_id = list(all_series.keys())[0]
    first_series = all_series[first_run_id]
    first_run_metadata = all_runs_data['runs'][first_run_id]['metadata']
    # Build subset series for common iterations
    subset_series = {
        'iterations': np.array(common_iters),
        'task_ids': first_series['task_ids'][[np.where(first_series['iterations'] == it)[0][0] for it in common_iters]]
    }
    for ax in axes.flat:
        add_task_shading(ax, subset_series, first_run_metadata, alpha=0.08)

    plt.tight_layout()

    # Save figure
    filename = f'{metadata["prob"]}_{metadata["dataset"]}_{metadata["network"]}_allruns_statistics.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive plots for ContLearn training results'
    )
    parser.add_argument('record_file', type=str,
                       help='Path to pickle file (single run or all runs)')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures (default: figures)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Load records
    print(f'Loading records from: {args.record_file}')
    data, is_multi_run = load_records(args.record_file)

    if is_multi_run:
        print(f'Detected multi-run file with {len(data["runs"])} runs')

        # Generate individual plots for each run
        for run_id, run_data in data['runs'].items():
            print(f'\nGenerating plots for run {run_id}...')
            plot_losses(run_data, args.output_dir, run_id=run_id)
            plot_metrics(run_data, args.output_dir, run_id=run_id)
            plot_eigenvalues(run_data, args.output_dir, run_id=run_id)
            plot_combined_metrics(run_data, args.output_dir, run_id=run_id)

        # Generate multi-run comparison plots
        if len(data['runs']) > 1:
            print('\nGenerating multi-run comparison plots...')
            plot_multi_run_comparison(data, args.output_dir)
            plot_multi_run_statistics(data, args.output_dir)

    else:
        print('Detected single-run file')
        run_id = data['metadata'].get('run_id', '0')
        print(f'Generating plots for run {run_id}...')

        plot_losses(data, args.output_dir, run_id=str(run_id))
        plot_metrics(data, args.output_dir, run_id=str(run_id))
        plot_eigenvalues(data, args.output_dir, run_id=str(run_id))
        plot_combined_metrics(data, args.output_dir, run_id=str(run_id))

    print(f'\n✓ All plots generated successfully in {args.output_dir}/')


if __name__ == '__main__':
    main()
