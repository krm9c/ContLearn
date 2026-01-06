#!/usr/bin/env python
"""
Plot Hamiltonian loss for all 4 conditions with architecture changes marked at task boundaries.
Creates a 4-panel figure (2x2) showing architecture evolution for each condition.
Architecture labels are placed at task boundaries.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Nature Communications style settings (matching other paper figures)
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

# Pastel color scheme matching paper figures
COLORS = {
    'C1': {'line': '#E57373', 'fill': '#FFCDD2'},  # Pastel red
    'C2': {'line': '#FFB74D', 'fill': '#FFE0B2'},  # Pastel orange
    'C3': {'line': '#64B5F6', 'fill': '#BBDEFB'},  # Pastel blue
    'C4': {'line': '#BA68C8', 'fill': '#E1BEE7'},  # Pastel purple
}

CONDITION_LABELS = {
    'C3': 'C3: Arch Search',
    'C4': 'C4: AWB Full',
}

def load_pickle(path):
    """Load pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_hamiltonian_and_arch(data):
    """Extract Hamiltonian loss values and architecture info across all tasks."""
    H_values = []
    epochs = []
    arch_changes = []

    current_epoch = 0

    for task_id in sorted(data['tasks'].keys()):
        task_data = data['tasks'][task_id]
        main_training = task_data['main_training']

        # Get H values for this task
        H = main_training['H']
        task_epochs = len(H)

        H_values.extend(H)
        epochs.extend(range(current_epoch, current_epoch + task_epochs))

        # Get architecture info
        arch_info = data['architecture_history'].get(task_id, {})
        arch = arch_info.get('final_arch', {})
        arch_changed = arch_info.get('arch_changed', False)
        sizes = arch.get('sizes', [])

        arch_changes.append({
            'epoch': current_epoch,
            'task_id': task_id,
            'sizes': sizes,
            'changed': arch_changed,
            'end_epoch': current_epoch + task_epochs
        })

        current_epoch += task_epochs

    return np.array(epochs), np.array(H_values), arch_changes

def plot_condition(ax, data, condition_key):
    """Plot Hamiltonian loss for a single condition with architecture annotations."""
    epochs, H_values, arch_changes = extract_hamiltonian_and_arch(data)

    color = COLORS[condition_key]
    label = CONDITION_LABELS[condition_key]

    # Plot H loss
    ax.plot(epochs, H_values, color=color['line'], linewidth=1.5, label=label)

    # Get y limits for positioning
    y_min, y_max = H_values.min(), H_values.max()
    y_range = y_max - y_min

    # All conditions show architecture labels at task boundaries
    for i, arch_info in enumerate(arch_changes):
        epoch = arch_info['epoch']
        task_id = arch_info['task_id']
        sizes = arch_info['sizes']

        # Vertical line at task boundary
        ax.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # Create architecture label
        if len(sizes) >= 4:
            hidden = sizes[1:-1]
            arch_str = f"T{task_id}:[{hidden[0]},{hidden[1]}]"
        elif len(sizes) >= 3:
            arch_str = f"T{task_id}:[{sizes[1]}]"
        else:
            arch_str = f"T{task_id}"

        # Alternate label positions (top/bottom) to avoid overlap
        if i % 2 == 0:
            y_pos = y_max - y_range * 0.05
            va = 'top'
        else:
            y_pos = y_max - y_range * 0.30
            va = 'top'

        # Place label at the task boundary
        ax.text(epoch, y_pos, arch_str,
                fontsize=9, fontweight='bold',
                color=color['line'],
                ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color['fill'],
                         edgecolor=color['line'], alpha=0.9, linewidth=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hamiltonian Loss (H)')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)

def main():
    # Data paths
    base_path = Path('runs__/analysis/data_analysis/sine_3seed/results')

    # Only C3 and C4 conditions
    conditions = {
        'C3': 'sine_condition3_arch_no_transfer_run0/sine_condition3_arch_no_transfer_run0_awb_run0/regression_sine_fcnn_awb_run0_records.pkl',
        'C4': 'sine_condition4_awb_full_run0/sine_condition4_awb_full_run0_run0/regression_sine_fcnn_awb_run0_records.pkl',
    }

    # Create vertical 2x1 figure (long plot with C3 on top, C4 below)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes = axes.flatten()

    # Load and plot each condition
    for idx, (cond_key, pkl_file) in enumerate(conditions.items()):
        pkl_path = base_path / pkl_file
        if pkl_path.exists():
            data = load_pickle(pkl_path)
            plot_condition(axes[idx], data, cond_key)
        else:
            print(f"Warning: {pkl_path} not found")
            axes[idx].text(0.5, 0.5, f'{cond_key} data not found',
                          transform=axes[idx].transAxes, ha='center', va='center')

    plt.tight_layout()

    # Save to paper figures directory and analysis figures
    output_paths = [
        Path('runs__/analysis/figures/hamiltonian_arch_evolution_c3c4.pdf'),
        Path('runs__/analysis/figures/hamiltonian_arch_evolution_c3c4.png'),
        Path('/Users/kraghavan/Desktop/JMLR_paper/Allyson-nonsmooth-dynamics/paperFigures/hamiltonian_arch_evolution_c3c4.pdf'),
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()

if __name__ == '__main__':
    main()
