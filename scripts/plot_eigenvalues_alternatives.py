"""
Alternative visualization options for eigenvalue evolution.

This script demonstrates different ways to visualize eigenvalue distributions:
1. Enhanced box plots (default)
2. Violin plots (smooth distribution)
3. Line plots with confidence bands
4. Heatmaps
5. Ridge plots

Usage:
    python plot_eigenvalues_alternatives.py <record_file.pkl> --style [box|violin|line|heatmap|ridge]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import argparse


def load_data(filepath):
    """Load eigenvalue data from record file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_box_enhanced(eigenvals_per_iter, iter_indices, ax, color_fill, color_edge, title):
    """Enhanced box plot with improved styling."""
    width = max(1, (max(iter_indices) - min(iter_indices)) / len(iter_indices) * 0.6)

    bp = ax.boxplot(eigenvals_per_iter, positions=iter_indices, widths=width,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=color_fill, edgecolor=color_edge,
                                linewidth=1.5, alpha=0.8),
                    medianprops=dict(color='crimson', linewidth=2.5),
                    whiskerprops=dict(color=color_edge, linewidth=1.5, linestyle='-'),
                    capprops=dict(color=color_edge, linewidth=1.5))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')


def plot_violin(eigenvals_per_iter, iter_indices, ax, color, title):
    """Violin plot showing full distribution."""
    parts = ax.violinplot(eigenvals_per_iter, positions=iter_indices,
                          widths=max(1, (max(iter_indices) - min(iter_indices)) / len(iter_indices) * 0.8),
                          showmeans=True, showmedians=True, showextrema=True)

    # Style the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # Style the other elements
    parts['cmeans'].set_edgecolor('darkred')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_edgecolor('crimson')
    parts['cmedians'].set_linewidth=2.5
    parts['cbars'].set_edgecolor('black')
    parts['cmaxes'].set_edgecolor('black')
    parts['cmins'].set_edgecolor('black')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')


def plot_line_with_bands(eigenvals_per_iter, iter_indices, ax, color, title):
    """Line plot with confidence bands."""
    # Compute statistics
    medians = [np.median(eigs) for eigs in eigenvals_per_iter]
    q25 = [np.percentile(eigs, 25) for eigs in eigenvals_per_iter]
    q75 = [np.percentile(eigs, 75) for eigs in eigenvals_per_iter]
    mins = [np.min(eigs) for eigs in eigenvals_per_iter]
    maxs = [np.max(eigs) for eigs in eigenvals_per_iter]

    # Plot median line
    ax.plot(iter_indices, medians, color=color, linewidth=2.5, label='Median', marker='o', markersize=4)

    # Fill between quartiles (IQR)
    ax.fill_between(iter_indices, q25, q75, color=color, alpha=0.3, label='IQR (25-75%)')

    # Fill between min-max
    ax.fill_between(iter_indices, mins, maxs, color=color, alpha=0.1, label='Range')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    ax.legend(fontsize=8, loc='best')


def plot_heatmap(eigenvals_per_iter, iter_indices, ax, title):
    """Heatmap showing eigenvalue evolution."""
    # Normalize each iteration's eigenvalues to same length (pad with NaN)
    max_len = max(len(eigs) for eigs in eigenvals_per_iter)
    matrix = np.full((max_len, len(iter_indices)), np.nan)

    for i, eigs in enumerate(eigenvals_per_iter):
        sorted_eigs = np.sort(np.abs(eigs))[::-1]  # Sort by magnitude, descending
        matrix[:len(sorted_eigs), i] = sorted_eigs

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                   extent=[min(iter_indices), max(iter_indices), max_len, 0])

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eigenvalue Rank')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Eigenvalue Magnitude')


def plot_ridge(eigenvals_per_iter, iter_indices, ax, color, title):
    """Ridge plot (overlapping density plots)."""
    from scipy.stats import gaussian_kde

    # Normalize y-offsets
    n_plots = len(iter_indices)
    y_offset = 0

    for i, (iteration, eigs) in enumerate(zip(iter_indices, eigenvals_per_iter)):
        # Compute KDE
        if len(eigs) > 1:
            kde = gaussian_kde(eigs)
            x_range = np.linspace(np.min(eigs), np.max(eigs), 100)
            density = kde(x_range)

            # Normalize density
            density = density / density.max() * 0.8  # Scale for overlap

            # Plot
            ax.fill_between(x_range, y_offset, y_offset + density,
                           color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.plot(x_range, y_offset + density, color='black', linewidth=1)

        y_offset += 1

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Eigenvalue Magnitude')
    ax.set_ylabel('Iteration')
    ax.set_yticks(range(n_plots))
    ax.set_yticklabels([str(it) for it in iter_indices])
    ax.grid(True, alpha=0.3, axis='x')


def main():
    parser = argparse.ArgumentParser(description='Alternative eigenvalue visualizations')
    parser.add_argument('record_file', help='Path to record pickle file')
    parser.add_argument('--style', choices=['box', 'violin', 'line', 'heatmap', 'ridge'],
                       default='box', help='Visualization style')
    parser.add_argument('--layer', default='layer_0', help='Layer to visualize')
    parser.add_argument('--output', default='eigenvalue_viz.png', help='Output filename')

    args = parser.parse_args()

    # Load data
    data = load_data(args.record_file)
    iterations = sorted(data['iterations'].keys())

    # Extract eigenvalues for specified layer
    eigenvals_A = []
    eigenvals_B = []
    iter_indices = []

    for it in iterations:
        eigs_A = data['iterations'][it]['eigenvalues']['A'].get(args.layer, np.array([]))
        eigs_B = data['iterations'][it]['eigenvalues']['B'].get(args.layer, np.array([]))

        if len(eigs_A) > 0 and len(eigs_B) > 0:
            eigenvals_A.append(np.real(eigs_A))
            eigenvals_B.append(np.real(eigs_B))
            iter_indices.append(it)

    if not eigenvals_A:
        print(f"No eigenvalues found for layer {args.layer}")
        return

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Eigenvalue Evolution - {args.layer} ({args.style} style)',
                 fontsize=14, fontweight='bold')

    if args.style == 'box':
        plot_box_enhanced(eigenvals_A, iter_indices, ax1, 'lightblue', 'darkblue', 'A Matrix')
        plot_box_enhanced(eigenvals_B, iter_indices, ax2, 'lightgreen', 'darkgreen', 'B Matrix')

    elif args.style == 'violin':
        plot_violin(eigenvals_A, iter_indices, ax1, 'lightblue', 'A Matrix')
        plot_violin(eigenvals_B, iter_indices, ax2, 'lightgreen', 'B Matrix')

    elif args.style == 'line':
        plot_line_with_bands(eigenvals_A, iter_indices, ax1, 'blue', 'A Matrix')
        plot_line_with_bands(eigenvals_B, iter_indices, ax2, 'green', 'B Matrix')

    elif args.style == 'heatmap':
        plot_heatmap(eigenvals_A, iter_indices, ax1, 'A Matrix')
        plot_heatmap(eigenvals_B, iter_indices, ax2, 'B Matrix')

    elif args.style == 'ridge':
        plot_ridge(eigenvals_A, iter_indices, ax1, 'lightblue', 'A Matrix')
        plot_ridge(eigenvals_B, iter_indices, ax2, 'lightgreen', 'B Matrix')

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Eigenvalue Magnitude', fontsize=11)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved: {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
