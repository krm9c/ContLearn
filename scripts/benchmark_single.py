#!/usr/bin/env python3
"""
Single Configuration Benchmark Runner

Runs a single benchmark configuration and saves results.
Designed to be called from run_optimization_benchmark.sh for proper XLA flag isolation.

Usage:
    python scripts/benchmark_single.py --name NAME --xla BOOL --fused BOOL --awb BOOL --output FILE

Added by Claude: January 2026
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def str_to_bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_gpu_stats():
    """Get GPU stats from nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 4:
                return {
                    'name': parts[0],
                    'memory_total': parts[1],
                    'memory_used': parts[2],
                    'utilization': parts[3],
                }
    except Exception:
        pass
    return None


def sample_gpu_utilization():
    """Sample current GPU utilization."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def run_benchmark(name: str, xla_enabled: bool, fused_enabled: bool, awb_enabled: bool,
                  output_file: str, quick: bool = False):
    """Run a single benchmark configuration."""

    # Apply XLA flags BEFORE importing JAX
    if xla_enabled:
        from cl.core.profiling import set_xla_flags, configure_jax_for_gpu
        set_xla_flags(enable=True, verbose=True)

    # Now import JAX and other dependencies
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    # Configure JAX after import
    if xla_enabled:
        from cl.core.profiling import configure_jax_for_gpu, get_optimization_status
        configure_jax_for_gpu(verbose=True)
    else:
        from cl.core.profiling import get_optimization_status

    from cl.datasets.mnist import MNISTDataset
    from cl.datasets.jax_dataloader import PrefetchDataLoader
    from cl.models.cnn import CNN
    from cl.core.hamiltonian import (
        _hamiltonian_core_class_standard,
        _hamiltonian_core_class_awb,
        _fused_train_step_class_standard,
        _fused_train_step_class_awb,
    )

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*70}")
    print(f"  XLA enabled: {xla_enabled}")
    print(f"  Fused enabled: {fused_enabled}")
    print(f"  AWB enabled: {awb_enabled}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  JAX devices: {jax.devices()}")

    # Configuration
    seed = 42
    batch_size = 1024
    epochs = 2 if quick else 5
    n_tasks = 1 if quick else 2
    debug_limit = 5000 if quick else 10000

    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Tasks: {n_tasks}")
    print(f"  Debug limit: {debug_limit}")

    # Timing collector
    timings = defaultdict(list)
    gpu_samples = []

    # Set reproducibility
    key = jax.random.PRNGKey(seed)

    # Create dataset
    dataset_config = {
        'n_task': n_tasks,
        'batch_size': batch_size,
        'debug_mode': True,
        'debug_limit': debug_limit,
        'problem': 'classification',
        'network': 'cnn',
        'seed': seed,
    }
    dataset = MNISTDataset(dataset_config)

    # Create model
    filter_size = 4
    channel_out = 3
    input_size = 28
    conv_output = (input_size - filter_size) + 1
    pool_output = (conv_output - 2) // 2 + 1
    flatten_size = channel_out * pool_output * pool_output
    feed_sizes = [flatten_size, 512, 64, 10]

    key, subkey = jax.random.split(key)
    model = CNN(key=subkey, filter_size=filter_size, feed_sizes=feed_sizes,
                channel_in=1, input_size=input_size, channel_out=channel_out)

    # Partition model
    if awb_enabled:
        params, static = model.partition_for_AB_training()
    else:
        params, static = eqx.partition(model, eqx.is_array)

    # Create optimizer
    optim = optax.adam(0.0001)
    opt_state = optim.init(params)

    # Select hamiltonian function
    if fused_enabled:
        if awb_enabled:
            hamiltonian_fn = _fused_train_step_class_awb
        else:
            hamiltonian_fn = _fused_train_step_class_standard
        print(f"  Using FUSED train step")
    else:
        if awb_enabled:
            hamiltonian_fn = _hamiltonian_core_class_awb
        else:
            hamiltonian_fn = _hamiltonian_core_class_standard
        print(f"  Using STANDARD train step")

    # JIT compile optimizer step
    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Helper
    def to_jax(tensor):
        if hasattr(tensor, 'numpy'):
            return jnp.array(tensor.numpy())
        return tensor

    # Pre-compute sqrt(param_count)
    leaves = jax.tree_util.tree_leaves(params)
    param_count = sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    sqrt_param_count = jnp.sqrt(jnp.array(param_count, dtype=jnp.float64))
    print(f"  Parameter count: {param_count}")

    # Training loop
    total_batches = 0
    first_batch_time = None
    total_start = time.perf_counter()

    for task_id in range(n_tasks):
        print(f"\n  Task {task_id}...")

        # Generate data
        trainloader, exploader = dataset.generate_dataset(
            task_id=task_id, batch_size=batch_size, phase='training'
        )

        # Wrap with prefetch
        trainloader = PrefetchDataLoader(trainloader, prefetch_size=3, loss_type='classification')
        exploader = PrefetchDataLoader(exploader, prefetch_size=3, loss_type='classification')

        # Convert to JAX
        train_batches = [(to_jax(x), to_jax(y)) for x, y in trainloader]
        exp_batches = [(to_jax(x), to_jax(y)) for x, y in exploader]
        print(f"    {len(train_batches)} batches")

        for epoch in range(epochs):
            epoch_start = time.perf_counter()

            for batch_idx, (batch, batch_exp) in enumerate(zip(train_batches, exp_batches)):
                batch_start = time.perf_counter()
                total_batches += 1

                # Sample GPU every 10 batches
                if batch_idx % 10 == 0:
                    util = sample_gpu_utilization()
                    if util is not None:
                        gpu_samples.append(util)

                # Prepare data
                t0 = time.perf_counter()
                x, y = batch
                exp_x, exp_y = batch_exp
                min_batch = min(x.shape[0], exp_x.shape[0])
                x, y = x[:min_batch], y[:min_batch]
                exp_x, exp_y = exp_x[:min_batch], exp_y[:min_batch]
                y = y.astype(jnp.int64)
                exp_y = exp_y.astype(jnp.int64)
                key, subkey = jax.random.split(key)
                delta_x = jax.random.normal(subkey, exp_x.shape) * 0.01
                timings['data_prep'].append((time.perf_counter() - t0) * 1000)

                # Hamiltonian gradient
                t0 = time.perf_counter()
                if fused_enabled:
                    grad, losses = hamiltonian_fn(
                        params, static, opt_state, x, y, exp_x, exp_y, delta_x,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        sqrt_param_count, jnp.array(1.0), jnp.array(0.0)
                    )
                else:
                    grad, losses = hamiltonian_fn(
                        params, static, x, y, exp_x, exp_y, delta_x,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        sqrt_param_count, jnp.array(1.0)
                    )
                # Block until ready
                jax.tree_util.tree_map(
                    lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a,
                    grad
                )
                timings['hamiltonian'].append((time.perf_counter() - t0) * 1000)

                # Optimizer step
                t0 = time.perf_counter()
                params, opt_state = optimizer_step(grad, opt_state, params)
                jax.tree_util.tree_map(
                    lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a,
                    params
                )
                timings['optimizer'].append((time.perf_counter() - t0) * 1000)

                # Record first batch time
                if first_batch_time is None:
                    first_batch_time = (time.perf_counter() - batch_start) * 1000
                    print(f"    First batch: {first_batch_time:.0f}ms (includes JIT)")

            epoch_time = time.perf_counter() - epoch_start
            batches_per_sec = len(train_batches) / epoch_time
            hamil_mean = sum(timings['hamiltonian'][-len(train_batches):]) / len(train_batches)
            print(f"    Epoch {epoch}: {epoch_time:.2f}s ({batches_per_sec:.1f} batch/s, hamil={hamil_mean:.1f}ms)")

    total_time = time.perf_counter() - total_start
    total_samples = total_batches * batch_size

    # Compute statistics
    def stats(values):
        if not values:
            return {'mean': 0, 'min': 0, 'max': 0, 'total': 0, 'count': 0}
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values),
            'count': len(values),
            'first': values[0] if values else 0,
            'rest_mean': sum(values[1:]) / len(values[1:]) if len(values) > 1 else 0,
        }

    hamil_stats = stats(timings['hamiltonian'])
    opt_stats = stats(timings['optimizer'])
    data_stats = stats(timings['data_prep'])
    gpu_stats = stats(gpu_samples) if gpu_samples else {'mean': 0, 'min': 0, 'max': 0}

    # Component breakdown
    total_profiled = sum(sum(v) for v in timings.values())
    breakdown = {}
    for comp_name, values in timings.items():
        breakdown[comp_name] = {
            'percent': (sum(values) / total_profiled * 100) if total_profiled > 0 else 0,
            'total_ms': sum(values),
            'mean_ms': sum(values) / len(values) if values else 0,
        }

    # Build result
    result = {
        'config_name': name,
        'generated_at': datetime.now().isoformat(),
        'config': {
            'xla_enabled': xla_enabled,
            'fused_enabled': fused_enabled,
            'awb_enabled': awb_enabled,
            'batch_size': batch_size,
            'epochs': epochs,
            'n_tasks': n_tasks,
            'debug_limit': debug_limit,
            'seed': seed,
        },
        'totals': {
            'total_time_sec': total_time,
            'total_batches': total_batches,
            'total_samples': total_samples,
            'throughput_samples_per_sec': total_samples / total_time,
            'throughput_batches_per_sec': total_batches / total_time,
        },
        'gpu': {
            'utilization_mean': gpu_stats['mean'],
            'utilization_min': gpu_stats.get('min', 0),
            'utilization_max': gpu_stats.get('max', 0),
            'samples': len(gpu_samples),
            'info': get_gpu_stats(),
        },
        'hamiltonian': hamil_stats,
        'optimizer': opt_stats,
        'data_prep': data_stats,
        'first_batch_time_ms': first_batch_time or 0,
        'component_breakdown': breakdown,
        'xla_status': get_optimization_status(),
    }

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_samples/total_time:.0f} samples/sec")
    print(f"  GPU utilization: {gpu_stats['mean']:.1f}% (min={gpu_stats.get('min', 0):.0f}%, max={gpu_stats.get('max', 0):.0f}%)")
    print(f"  Hamiltonian: {hamil_stats['mean']:.1f}ms mean (first={hamil_stats['first']:.1f}ms, rest={hamil_stats['rest_mean']:.1f}ms)")
    print(f"  Optimizer: {opt_stats['mean']:.1f}ms mean")

    print(f"\nComponent Breakdown:")
    for comp_name, comp_stats in sorted(breakdown.items(), key=lambda x: x[1]['percent'], reverse=True):
        print(f"  {comp_name}: {comp_stats['percent']:.1f}%")

    # Save result
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResult saved to: {output_path.absolute()}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Run single benchmark configuration')
    parser.add_argument('--name', required=True, help='Configuration name')
    parser.add_argument('--xla', type=str_to_bool, required=True, help='Enable XLA flags')
    parser.add_argument('--fused', type=str_to_bool, required=True, help='Enable fused train step')
    parser.add_argument('--awb', type=str_to_bool, required=True, help='Enable AWB mode')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode')

    args = parser.parse_args()

    run_benchmark(
        name=args.name,
        xla_enabled=args.xla,
        fused_enabled=args.fused,
        awb_enabled=args.awb,
        output_file=args.output,
        quick=args.quick,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
