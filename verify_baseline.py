#!/usr/bin/env python
"""Verify baseline graph CL experiment has no significant train/test gap."""

import pickle
import sys
import time


def verify_baseline(records_path, gap_threshold=0.10):
    """Check train/test accuracy gap from saved records.

    Args:
        records_path: Path to the pickle file with training records
        gap_threshold: Maximum acceptable train/test gap (default: 10%)
    """
    with open(records_path, 'rb') as f:
        records = pickle.load(f)

    print("=" * 60)
    print("Baseline Verification Results")
    print("=" * 60)

    all_ok = True
    task_metrics = []

    for task_id in sorted(records.get('tasks', {}).keys()):
        task_data = records['tasks'][task_id].get('main_training', {})
        if not task_data:
            continue

        train_acc = task_data.get('train_metric', [0])[-1]
        test_cur = task_data.get('test_current', [0])[-1]
        test_exp = task_data.get('test_experience', [0])[-1] if 'test_experience' in task_data else 0
        gap = train_acc - test_cur

        status = "OK" if abs(gap) < gap_threshold else "GAP"
        if abs(gap) >= gap_threshold:
            all_ok = False

        # Check for data leakage (Te/Exp should NOT be 100%)
        leakage = test_exp > 0.99
        if leakage:
            status = "LEAK"
            all_ok = False

        task_metrics.append({
            'task_id': task_id,
            'train': train_acc,
            'test_cur': test_cur,
            'test_exp': test_exp,
            'gap': gap
        })

        print(f"\nTask {task_id}:")
        print(f"  Train Acc:    {train_acc:.4f}")
        print(f"  Test/Current: {test_cur:.4f}")
        print(f"  Test/Exp:     {test_exp:.4f}")
        print(f"  Gap:          {gap:.4f} [{status}]")

    # Summary statistics
    if task_metrics:
        avg_train = sum(m['train'] for m in task_metrics) / len(task_metrics)
        avg_test = sum(m['test_cur'] for m in task_metrics) / len(task_metrics)
        avg_gap = sum(m['gap'] for m in task_metrics) / len(task_metrics)

        print("\n" + "-" * 60)
        print("Summary:")
        print(f"  Average Train Acc: {avg_train:.4f}")
        print(f"  Average Test Acc:  {avg_test:.4f}")
        print(f"  Average Gap:       {avg_gap:.4f}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        print(f"  (Gap threshold: {gap_threshold*100:.0f}%)")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    # Updated default path to new minimal baseline config output
    default_path = "outputs/synthetic_graph_minimal_baseline/classification_synthetic_taskshift_gcn_run0_records.pkl"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    try:
        success = verify_baseline(path)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"Error: Records file not found at {path}")
        print("\nRun the experiment first with:")
        print("  python run.py runs__/configs/synthetic_graph_minimal_baseline.json")
        print("\nOr specify a different records file:")
        print("  python verify_baseline.py <path_to_records.pkl>")
        sys.exit(1)
