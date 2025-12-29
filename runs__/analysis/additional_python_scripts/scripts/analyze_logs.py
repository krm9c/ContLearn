#!/usr/bin/env python3
"""Analyze training logs for performance issues and bottlenecks."""

import re
from pathlib import Path
from collections import defaultdict

def get_log_status(log_path):
    """Determine if log completed successfully."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    last_50 = lines[-50:] if len(lines) >= 50 else lines
    last_text = ''.join(last_50)

    completed = 'Training complete!' in last_text
    has_final_metrics = 'Task' in last_text and ':' in last_text
    has_plots = 'Generated' in last_text and 'plots' in last_text

    return {
        'total_lines': total_lines,
        'completed': completed,
        'has_final_metrics': has_final_metrics,
        'has_plots': has_plots,
        'status': 'COMPLETE' if completed else ('PARTIAL' if total_lines > 100 else 'FAILED_EARLY')
    }

def extract_iteration_speeds(log_path, sample_size=50):
    """Extract iteration speeds to identify slowdowns."""
    speeds = []
    with open(log_path, 'r') as f:
        for line in f:
            # Match patterns like "10.5it/s" or "1.5s/it"
            its_match = re.search(r'(\d+\.?\d*)it/s', line)
            sit_match = re.search(r'(\d+\.?\d*)s/it', line)

            if its_match:
                speeds.append(('it/s', float(its_match.group(1))))
            elif sit_match:
                speeds.append(('s/it', float(sit_match.group(1))))

    if not speeds:
        return None

    # Sample evenly
    step = max(1, len(speeds) // sample_size)
    sampled = speeds[::step]

    return sampled

def extract_final_metrics(log_path):
    """Extract final task metrics."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    metrics = {}
    for line in reversed(lines[-30:]):
        # Look for "Task X: VALUE" pattern
        match = re.search(r'Task (\d+):\s+([0-9.e+-]+)', line)
        if match:
            task_id = int(match.group(1))
            value = float(match.group(2))
            metrics[task_id] = value

    return metrics

def analyze_dataset(dataset_name, logs_dir):
    """Analyze all conditions for a dataset."""
    logs_dir = Path(logs_dir)
    conditions = ['baseline', 'heuristics', 'arch_no_transfer', 'awb_full']

    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()}")
    print('='*70)

    results = {}

    for condition in conditions:
        # Find log files for this condition
        pattern = f"{dataset_name}_*{condition}*.log"
        log_files = list(logs_dir.glob(pattern))

        if not log_files:
            print(f"\n{condition.upper()}: NO LOGS FOUND")
            continue

        # Use the most recent log
        log_file = sorted(log_files)[-1]

        print(f"\n{condition.upper()}:")
        print(f"  File: {log_file.name}")

        # Check status
        status = get_log_status(log_file)
        print(f"  Status: {status['status']} ({status['total_lines']} lines)")

        if status['status'] == 'FAILED_EARLY':
            print(f"  ⚠️  CRITICAL: Training crashed/terminated early!")
            # Check last few lines for clues
            with open(log_file, 'r') as f:
                last_lines = f.readlines()[-5:]
            print(f"  Last output:")
            for line in last_lines[-3:]:
                print(f"    {line.rstrip()}")

        elif status['completed']:
            # Get final metrics
            metrics = extract_final_metrics(log_file)
            if metrics:
                values = list(metrics.values())
                print(f"  Tasks completed: {len(metrics)}")
                print(f"  Avg metric: {sum(values)/len(values):.6f}")
                print(f"  Final task metric: {values[-1]:.6f}")

            # Analyze iteration speeds
            speeds = extract_iteration_speeds(log_file)
            if speeds:
                # Convert to it/s for comparison
                it_s_values = []
                for unit, val in speeds:
                    if unit == 'it/s':
                        it_s_values.append(val)
                    else:  # s/it
                        it_s_values.append(1.0 / val if val > 0 else 0)

                if it_s_values:
                    avg_speed = sum(it_s_values) / len(it_s_values)
                    min_speed = min(it_s_values)
                    max_speed = max(it_s_values)

                    print(f"  Iteration speed (it/s):")
                    print(f"    Avg: {avg_speed:.2f}, Min: {min_speed:.2f}, Max: {max_speed:.2f}")

                    if max_speed / min_speed > 5:
                        print(f"    ⚠️  BOTTLENECK: {max_speed/min_speed:.1f}x speed variation!")

        results[condition] = {
            'status': status,
            'metrics': extract_final_metrics(log_file) if status['completed'] else None,
            'speeds': extract_iteration_speeds(log_file)
        }

    return results

def main():
    logs_dir = Path('/Users/kraghavan/Desktop/JMLR_paper/ContLearn/kkt_run/kkt/logs')

    for dataset in ['sine', 'mnist', 'cifar10', 'synthetic_graph']:
        analyze_dataset(dataset, logs_dir)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print('='*70)

if __name__ == '__main__':
    main()
