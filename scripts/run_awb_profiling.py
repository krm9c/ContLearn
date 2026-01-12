#!/usr/bin/env python3
"""
Detailed AWB (Condition 4) Training Profiling

This script profiles the full AWB pipeline to identify bottlenecks in:
- Preliminary training
- Architecture decision
- A/B matrix training
- Weight transfer
- Main training with transferred weights

Usage:
    python scripts/run_awb_profiling.py [config_path] [--output FILE]

Added by Claude: January 2025
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


def get_system_info():
    """Collect system information."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'jax_version': jax.__version__,
        'jax_backend': jax.default_backend(),
        'jax_devices': [str(d) for d in jax.devices()],
    }

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 4:
                info['gpu'] = {
                    'name': parts[0],
                    'memory_total': parts[1],
                    'memory_used': parts[2],
                    'utilization': parts[3],
                }
    except:
        pass

    return info


class AWBProfiler:
    """Collects detailed timing for AWB pipeline phases."""

    def __init__(self):
        self.phase_timings = defaultdict(list)
        self.batch_timings = defaultdict(list)
        self.gpu_samples = []
        self.current_task = 0
        self.current_phase = "init"

    def start_phase(self, phase_name: str, task_id: int = None):
        """Mark start of a phase."""
        if task_id is not None:
            self.current_task = task_id
        self.current_phase = phase_name
        self._phase_start = time.perf_counter()

    def end_phase(self, phase_name: str = None):
        """Mark end of a phase and record timing."""
        phase = phase_name or self.current_phase
        duration = (time.perf_counter() - self._phase_start) * 1000
        self.phase_timings[phase].append({
            'task': self.current_task,
            'duration_ms': duration,
        })
        return duration

    def record_batch(self, phase: str, duration_ms: float):
        """Record a batch timing within a phase."""
        self.batch_timings[phase].append({
            'task': self.current_task,
            'duration_ms': duration_ms,
        })

    def sample_gpu(self):
        """Sample current GPU utilization."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                self.gpu_samples.append({
                    'task': self.current_task,
                    'phase': self.current_phase,
                    'utilization': float(parts[0].strip()),
                    'memory_mb': float(parts[1].strip()),
                })
        except:
            pass

    def get_summary(self):
        """Get summary statistics."""
        summary = {'phases': {}, 'batches': {}}

        for phase, measurements in self.phase_timings.items():
            durations = [m['duration_ms'] for m in measurements]
            if durations:
                summary['phases'][phase] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'mean_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                }

        for phase, measurements in self.batch_timings.items():
            durations = [m['duration_ms'] for m in measurements]
            if durations:
                summary['batches'][phase] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'mean_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                }

        if self.gpu_samples:
            utils = [s['utilization'] for s in self.gpu_samples]
            summary['gpu'] = {
                'mean_percent': sum(utils) / len(utils),
                'min_percent': min(utils),
                'max_percent': max(utils),
                'samples': len(utils),
            }

        return summary


def run_awb_training_with_profiling(config_path: str, output_path: str):
    """Run AWB training with detailed phase-level profiling."""

    from cl.config import load_config
    from cl.runners.generic_runner import train_model

    # Load config
    config = load_config(config_path)

    print("="*70)
    print("AWB (CONDITION 4) DETAILED PROFILING")
    print("="*70)
    print(f"\nConfig: {config_path}")
    print(f"\nSettings:")
    print(f"  n_task: {config.get('n_task')}")
    print(f"  epochs_per_task: {config.get('epochs_per_task')}")
    print(f"  batch_size: {config.get('batch_size')}")
    print(f"  awb_enabled: {config.get('awb_enabled')}")
    print(f"  awb_preliminary_epochs: {config.get('awb_preliminary_epochs')}")
    print(f"  awb_ab_training_epochs: {config.get('awb_ab_training_epochs')}")
    print(f"  task_warmup_epochs: {config.get('task_warmup_epochs')}")

    # Enable profiling
    config['profiling_enabled'] = True
    config['detailed_profiling'] = True

    # Collect system info
    system_info = get_system_info()
    print(f"\nSystem:")
    print(f"  JAX backend: {system_info['jax_backend']}")
    print(f"  Devices: {system_info['jax_devices']}")
    if system_info.get('gpu'):
        print(f"  GPU: {system_info['gpu']['name']}")

    # Run training with timing
    print("\n" + "="*70)
    print("STARTING AWB TRAINING")
    print("="*70)

    total_start = time.perf_counter()

    # Sample GPU before
    gpu_before = None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_before = result.stdout.strip()
    except:
        pass

    # Run training
    try:
        result = train_model(config)
        success = True
    except Exception as e:
        import traceback
        print(f"\nTraining failed: {e}")
        traceback.print_exc()
        success = False
        result = {'error': str(e), 'traceback': traceback.format_exc()}

    total_time = time.perf_counter() - total_start

    # Sample GPU after
    gpu_after = None
    try:
        result_gpu = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result_gpu.returncode == 0:
            gpu_after = result_gpu.stdout.strip()
    except:
        pass

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Success: {success}")

    # Try to get profiling data from the collector
    profiling_data = {}
    try:
        from cl.core.profiling import get_collector
        collector = get_collector()
        if collector:
            profiling_data = collector.get_stats()
            print("\n--- Profiling Summary from Collector ---")
            collector.print_summary()
    except Exception as e:
        print(f"\nCould not get profiling data: {e}")

    # Build report
    report = {
        'generated_at': datetime.now().isoformat(),
        'config_path': config_path,
        'config': {
            'n_task': config.get('n_task'),
            'epochs_per_task': config.get('epochs_per_task'),
            'batch_size': config.get('batch_size'),
            'awb_enabled': config.get('awb_enabled'),
            'awb_preliminary_epochs': config.get('awb_preliminary_epochs'),
            'awb_ab_training_epochs': config.get('awb_ab_training_epochs'),
            'task_warmup_enabled': config.get('task_warmup_enabled'),
            'task_warmup_epochs': config.get('task_warmup_epochs'),
            'debug_mode': config.get('debug_mode'),
            'debug_limit': config.get('debug_limit'),
        },
        'system_info': system_info,
        'totals': {
            'total_time_sec': total_time,
            'success': success,
        },
        'gpu_before': gpu_before,
        'gpu_after': gpu_after,
        'profiling_data': profiling_data,
    }

    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_file.absolute()}")

    # Print analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if profiling_data and 'timings' in profiling_data:
        timings = profiling_data['timings']
        print("\nTime by Component:")
        print("-"*50)

        # Sort by total time
        sorted_timings = sorted(
            timings.items(),
            key=lambda x: x[1].get('total_ms', 0),
            reverse=True
        )

        total_tracked = sum(t.get('total_ms', 0) for _, t in sorted_timings)

        for name, stats in sorted_timings[:15]:  # Top 15
            total_ms = stats.get('total_ms', 0)
            mean_ms = stats.get('mean_ms', 0)
            count = stats.get('count', 0)
            pct = (total_ms / total_tracked * 100) if total_tracked > 0 else 0
            print(f"  {name:<30} {total_ms:>10.1f}ms ({pct:>5.1f}%) n={count}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Profile AWB (Condition 4) training')
    parser.add_argument('config', nargs='?',
                        default='runs__/configs/mnist_condition4_profiling.json',
                        help='Config file path')
    parser.add_argument('--output', '-o', default='awb_profiling_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    run_awb_training_with_profiling(args.config, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
