#!/usr/bin/env python3
"""
Detailed Training Loop Profiling

This script runs training with detailed timing for each component to identify
exactly where time is being spent in the training loop.

Usage:
    python scripts/run_detailed_profiling.py [config_path] [--output FILE]

Output:
    JSON file with detailed timing breakdown for each component

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


class DetailedProfiler:
    """Collects detailed timing for each training component."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_epoch = 0
        self.current_batch = 0
        self.gpu_samples = []

    def record(self, name: str, duration_ms: float):
        """Record a timing measurement."""
        self.timings[name].append({
            'epoch': self.current_epoch,
            'batch': self.current_batch,
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
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'utilization': float(parts[0].strip()),
                    'memory_mb': float(parts[1].strip()),
                })
        except:
            pass

    def get_summary(self):
        """Get summary statistics for all components."""
        summary = {}

        for name, measurements in self.timings.items():
            durations = [m['duration_ms'] for m in measurements]
            if durations:
                summary[name] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'mean_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'percent_of_total': 0,  # Calculated later
                }

        # Calculate percentages
        total_time = sum(s['total_ms'] for s in summary.values())
        if total_time > 0:
            for name in summary:
                summary[name]['percent_of_total'] = (summary[name]['total_ms'] / total_time) * 100

        # GPU stats
        if self.gpu_samples:
            utils = [s['utilization'] for s in self.gpu_samples]
            summary['gpu_utilization'] = {
                'mean_percent': sum(utils) / len(utils),
                'min_percent': min(utils),
                'max_percent': max(utils),
                'samples': len(utils),
            }

        return summary

    def get_full_report(self):
        """Get full report with all data."""
        return {
            'summary': self.get_summary(),
            'raw_timings': dict(self.timings),
            'gpu_samples': self.gpu_samples,
        }


def run_detailed_training(config_path: str, output_path: str):
    """Run training with detailed profiling."""

    from cl.config import load_config
    from cl.datasets.mnist import MNISTDataset
    from cl.datasets.jax_dataloader import PrefetchDataLoader
    from cl.models.cnn import CNN
    from cl.core.hamiltonian import _hamiltonian_core_class_standard
    import equinox as eqx
    import optax

    # Load config
    config = load_config(config_path)
    print(f"\nConfig loaded:")
    print(f"  batch_size: {config.get('batch_size')}")
    print(f"  epochs_per_task: {config.get('epochs_per_task')}")
    print(f"  eval_interval: {config.get('eval_interval')}")
    print(f"  log_interval: {config.get('log_interval')}")

    profiler = DetailedProfiler()

    # Create dataset
    print("\nCreating dataset...")
    dataset_config = {
        'n_task': config.get('n_task', 2),
        'batch_size': config.get('batch_size', 1024),
        'debug_mode': config.get('debug_mode', True),
        'debug_limit': config.get('debug_limit', 10000),
        'problem': 'classification',
        'network': 'cnn',
    }
    dataset = MNISTDataset(dataset_config)

    # Create model
    print("Creating model...")
    batch_size = config.get('batch_size', 1024)
    filter_size = config.get('filter_size', 4)
    channel_out = config.get('channel_out', 3)
    input_size = 28

    conv_output = (input_size - filter_size) + 1
    pool_output = (conv_output - 2) // 2 + 1
    flatten_size = channel_out * pool_output * pool_output
    feed_sizes = [flatten_size, 512, 64, 10]

    key = jax.random.PRNGKey(42)
    model = CNN(key=key, filter_size=filter_size, feed_sizes=feed_sizes,
                channel_in=1, input_size=input_size, channel_out=channel_out)
    params, static = eqx.partition(model, eqx.is_array)

    # Create optimizer
    optim = optax.adam(config.get('lr', 0.0001))
    opt_state = optim.init(params)

    # JIT compile functions
    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @jax.jit
    def compute_accuracy(params, static, x, y):
        model = eqx.combine(params, static)
        logits = jax.vmap(model)(x)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == y)

    def to_jax(tensor):
        """Convert to JAX array if needed (handles both PyTorch and JAX inputs)."""
        if hasattr(tensor, 'numpy'):
            # PyTorch tensor
            return jnp.array(tensor.numpy())
        else:
            # Already a JAX array
            return tensor

    # Training loop
    n_tasks = config.get('n_task', 1)
    epochs_per_task = config.get('epochs_per_task', 5)
    eval_interval = config.get('eval_interval', 50)
    log_interval = config.get('log_interval', 10)

    print(f"\nStarting training: {n_tasks} tasks, {epochs_per_task} epochs each")
    print("="*70)

    total_start = time.perf_counter()
    total_batches = 0
    rng_key = jax.random.PRNGKey(0)

    for task_id in range(n_tasks):
        print(f"\n--- Task {task_id} ---")

        # Generate data for this task
        t0 = time.perf_counter()
        trainloader, exploader = dataset.generate_dataset(
            task_id=task_id, batch_size=batch_size, phase='training'
        )

        # Wrap with prefetch if enabled
        if config.get('use_jax_prefetch', True):
            trainloader = PrefetchDataLoader(trainloader, prefetch_size=3, loss_type='classification')
            exploader = PrefetchDataLoader(exploader, prefetch_size=3, loss_type='classification')

        profiler.record('task_setup', (time.perf_counter() - t0) * 1000)

        # Convert to JAX arrays
        t0 = time.perf_counter()
        train_batches = [(to_jax(x), to_jax(y)) for x, y in trainloader]
        exp_batches = [(to_jax(x), to_jax(y)) for x, y in exploader]
        profiler.record('data_to_jax', (time.perf_counter() - t0) * 1000)

        print(f"  {len(train_batches)} training batches, {len(exp_batches)} experience batches")

        for epoch in range(epochs_per_task):
            profiler.current_epoch = task_id * epochs_per_task + epoch
            epoch_start = time.perf_counter()

            for batch_idx, (batch, batch_exp) in enumerate(zip(train_batches, exp_batches)):
                profiler.current_batch = batch_idx
                total_batches += 1

                # Sample GPU periodically
                if batch_idx % 10 == 0:
                    profiler.sample_gpu()

                # Data prep
                t0 = time.perf_counter()
                x, y = batch
                exp_x, exp_y = batch_exp
                min_batch = min(x.shape[0], exp_x.shape[0])
                x, y = x[:min_batch], y[:min_batch]
                exp_x, exp_y = exp_x[:min_batch], exp_y[:min_batch]
                rng_key, subkey = jax.random.split(rng_key)
                delta_x = jax.random.normal(subkey, exp_x.shape) * 0.01
                profiler.record('data_prep', (time.perf_counter() - t0) * 1000)

                # Hamiltonian gradient
                t0 = time.perf_counter()
                grad, losses = _hamiltonian_core_class_standard(
                    params, static, x, y, exp_x, exp_y, delta_x,
                    jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                    jnp.array(1000.0), jnp.array(1.0)
                )
                jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
                profiler.record('hamiltonian', (time.perf_counter() - t0) * 1000)

                # Optimizer step
                t0 = time.perf_counter()
                params, opt_state = optimizer_step(grad, opt_state, params)
                jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
                profiler.record('optimizer_step', (time.perf_counter() - t0) * 1000)

                # Evaluation (if interval)
                if total_batches % eval_interval == 0:
                    t0 = time.perf_counter()
                    # Quick eval on first batch
                    acc = compute_accuracy(params, static, x, y)
                    acc.block_until_ready()
                    profiler.record('evaluation', (time.perf_counter() - t0) * 1000)

                # Logging overhead
                if total_batches % log_interval == 0:
                    t0 = time.perf_counter()
                    # Simulate logging (extract loss values)
                    _ = float(losses[0])
                    profiler.record('logging', (time.perf_counter() - t0) * 1000)

            epoch_time = time.perf_counter() - epoch_start
            print(f"  Epoch {epoch}: {epoch_time:.2f}s ({len(train_batches)/epoch_time:.1f} batches/sec)")

    total_time = time.perf_counter() - total_start
    total_samples = total_batches * batch_size

    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Throughput: {total_samples/total_time:.0f} samples/sec")

    # Get summary
    summary = profiler.get_summary()

    print("\n" + "-"*70)
    print("TIME BREAKDOWN BY COMPONENT")
    print("-"*70)
    print(f"{'Component':<20} {'Total (ms)':<12} {'Mean (ms)':<12} {'Count':<8} {'%':<8}")
    print("-"*70)

    # Sort by total time
    sorted_components = sorted(
        [(k, v) for k, v in summary.items() if k != 'gpu_utilization'],
        key=lambda x: x[1]['total_ms'],
        reverse=True
    )

    for name, stats in sorted_components:
        print(f"{name:<20} {stats['total_ms']:<12.1f} {stats['mean_ms']:<12.3f} {stats['count']:<8} {stats['percent_of_total']:<8.1f}")

    if 'gpu_utilization' in summary:
        gpu = summary['gpu_utilization']
        print(f"\nGPU Utilization: {gpu['mean_percent']:.1f}% (min: {gpu['min_percent']:.0f}%, max: {gpu['max_percent']:.0f}%)")

    # Save full report
    report = {
        'generated_at': datetime.now().isoformat(),
        'config': {
            'batch_size': batch_size,
            'epochs_per_task': epochs_per_task,
            'n_task': n_tasks,
            'eval_interval': eval_interval,
            'log_interval': log_interval,
        },
        'system_info': get_system_info(),
        'totals': {
            'total_time_sec': total_time,
            'total_batches': total_batches,
            'total_samples': total_samples,
            'throughput_samples_per_sec': total_samples / total_time,
        },
        'component_summary': summary,
        'raw_data': profiler.get_full_report(),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {output_file.absolute()}")
    print("\nShare this file with Claude for analysis.")

    return report


def main():
    parser = argparse.ArgumentParser(description='Run detailed training loop profiling')
    parser.add_argument('config', nargs='?',
                        default='runs__/configs/mnist_condition1_profiling_optimized.json',
                        help='Config file path')
    parser.add_argument('--output', '-o', default='detailed_profiling_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("="*70)
    print("DETAILED TRAINING LOOP PROFILING")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    run_detailed_training(args.config, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
