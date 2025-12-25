#!/usr/bin/env python
"""
Compute continual learning metrics from validation experiments.

Implements standard CL metrics from literature:
- Lopez-Paz & Ranzato (NeurIPS 2017)
- Chaudhry et al. (ECCV 2018)

Usage:
    python compute_metrics.py --results-dir ../results --output metrics_summary.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json


def load_records(pkl_path: Path) -> Dict:
    """Load a single experiment's records."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_performance_matrix(record_dict: Dict) -> np.ndarray:
    """Extract performance matrix A[j][i] from record_dict.

    A[j][i] = performance on task i after training task j

    Args:
        record_dict: Loaded records dictionary

    Returns:
        matrix: numpy array of shape (n_tasks, n_tasks)
    """
    task_perf = record_dict.get('task_performance_matrix', {})

    if not task_perf:
        # Fallback: try to extract from final metrics
        print("Warning: No task_performance_matrix found. Using fallback extraction.")
        return None

    n_tasks = len(task_perf)
    matrix = np.zeros((n_tasks, n_tasks))

    for j_str, perf_dict in task_perf.items():
        j = int(j_str)
        for i_str, perf_value in perf_dict.items():
            i = int(i_str)
            matrix[j, i] = perf_value

    return matrix


def compute_acc(matrix: np.ndarray) -> float:
    """Compute Average Accuracy.

    ACC = (1/T) Σ_{i=0}^{T-1} A[T-1][i]

    Final performance on all tasks.

    Args:
        matrix: Performance matrix A[j][i]

    Returns:
        float: Average accuracy
    """
    T = matrix.shape[0]
    return np.mean(matrix[T-1, :])


def compute_bwt(matrix: np.ndarray) -> float:
    """Compute Backward Transfer (Catastrophic Forgetting).

    BWT = (1/(T-1)) Σ_{i=0}^{T-2} (A[T-1][i] - A[i][i])

    Negative values indicate forgetting.
    From Lopez-Paz & Ranzato (NeurIPS 2017).

    Args:
        matrix: Performance matrix A[j][i]

    Returns:
        float: Backward transfer (negative = forgetting)
    """
    T = matrix.shape[0]
    if T == 1:
        return 0.0

    bwt_sum = 0.0
    for i in range(T-1):
        bwt_sum += (matrix[T-1, i] - matrix[i, i])

    return bwt_sum / (T - 1)


def compute_forgetting(matrix: np.ndarray) -> float:
    """Compute Average Forgetting.

    F = (1/(T-1)) Σ_{i=0}^{T-2} max_{j>=i} (A[i][i] - A[j][i])

    Maximum performance drop on task i across all subsequent tasks.
    From Chaudhry et al. (ECCV 2018).

    Args:
        matrix: Performance matrix A[j][i]

    Returns:
        float: Average forgetting
    """
    T = matrix.shape[0]
    if T == 1:
        return 0.0

    forgetting_sum = 0.0
    for i in range(T-1):
        # Find maximum forgetting for task i
        max_forget = 0.0
        for j in range(i, T):
            forget = matrix[i, i] - matrix[j, i]
            max_forget = max(max_forget, forget)
        forgetting_sum += max_forget

    return forgetting_sum / (T - 1)


def compute_fwt(matrix: np.ndarray) -> float:
    """Compute Forward Transfer.

    FWT = (1/(T-1)) Σ_{i=1}^{T-1} (A[i-1][i] - A_random)

    Performance on new task before training on it.
    Assumes A_random = 0 for simplicity (can be updated).

    Args:
        matrix: Performance matrix A[j][i]

    Returns:
        float: Forward transfer
    """
    T = matrix.shape[0]
    if T == 1:
        return 0.0

    # Random baseline (assumed 0 for classification, can be parametrized)
    A_random = 0.0

    fwt_sum = 0.0
    for i in range(1, T):
        # Performance on task i before training it (after training i-1)
        fwt_sum += (matrix[i-1, i] - A_random)

    return fwt_sum / (T - 1)


def compute_all_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """Compute all CL metrics.

    Args:
        matrix: Performance matrix A[j][i]

    Returns:
        dict: All metrics
    """
    return {
        'ACC': compute_acc(matrix),
        'BWT': compute_bwt(matrix),
        'Forgetting': compute_forgetting(matrix),
        'FWT': compute_fwt(matrix),
    }


def process_experiment(exp_path: Path) -> Dict:
    """Process a single experiment directory.

    Args:
        exp_path: Path to experiment results

    Returns:
        dict: Metrics for this experiment
    """
    # Find records.pkl file
    pkl_files = list(exp_path.glob('*_records.pkl'))

    if not pkl_files:
        print(f"Warning: No records.pkl found in {exp_path}")
        return None

    pkl_path = pkl_files[0]
    records = load_records(pkl_path)

    # Extract performance matrix
    matrix = extract_performance_matrix(records)

    if matrix is None:
        print(f"Warning: Could not extract performance matrix from {exp_path}")
        return None

    # Compute metrics
    metrics = compute_all_metrics(matrix)

    # Add metadata
    metadata = records.get('metadata', {})
    metrics['dataset'] = metadata.get('dataset', 'unknown')
    metrics['condition'] = exp_path.parent.name
    metrics['run_id'] = metadata.get('run_id', int(exp_path.name.replace('run_', '')))
    metrics['n_tasks'] = matrix.shape[0]

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compute CL metrics from validation results')
    parser.add_argument('--results-dir', required=True, help='Path to results directory')
    parser.add_argument('--output', default='metrics_summary.csv', help='Output CSV file')
    parser.add_argument('--phase', choices=['quick', 'full'], default='full', help='Analysis phase')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Collect all experiments
    experiments = []

    # Phase selection
    if args.phase == 'quick':
        datasets = ['sine', 'mnist']
    else:
        datasets = ['sine', 'mnist', 'permuted_mnist', 'cifar10', 'cifar100', 'synthetic_graph']

    conditions = [
        'condition1_baseline',
        'condition2_heuristics',
        'condition3_arch_no_transfer',
        'condition4_awb_full'
    ]

    # Process all experiments
    all_metrics = []

    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue

        for condition in conditions:
            condition_dir = dataset_dir / condition
            if not condition_dir.exists():
                print(f"Warning: Condition directory not found: {condition_dir}")
                continue

            # Process all runs
            for run_dir in sorted(condition_dir.glob('run_*')):
                if not run_dir.is_dir():
                    continue

                print(f"Processing: {dataset}/{condition}/{run_dir.name}")
                metrics = process_experiment(run_dir)

                if metrics:
                    all_metrics.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Compute summary statistics
    summary = df.groupby(['dataset', 'condition']).agg({
        'ACC': ['mean', 'std'],
        'BWT': ['mean', 'std'],
        'Forgetting': ['mean', 'std'],
        'FWT': ['mean', 'std'],
        'n_tasks': 'first'
    }).reset_index()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    summary.to_csv(output_path.replace('.csv', '_summary.csv'), index=False)

    print(f"\n✓ Metrics computed for {len(all_metrics)} experiments")
    print(f"  Detailed results: {output_path}")
    print(f"  Summary: {output_path.replace('.csv', '_summary.csv')}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
