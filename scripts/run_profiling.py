#!/usr/bin/env python3
"""
Run All Profiling - Comprehensive GPU Profiling for Condition 1

This script runs all profiling benchmarks and saves results to a single JSON file
that can be shared for analysis.

Usage:
    python scripts/run_profiling.py [--output FILE] [--quick]

Output:
    A JSON file with all profiling results (default: profiling_results.json)

Added by Claude: January 2025
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


def get_system_info():
    """Collect comprehensive system information."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'jax_version': jax.__version__,
        'jax_backend': jax.default_backend(),
        'jax_devices': [str(d) for d in jax.devices()],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
    }

    # Get JAX device details
    try:
        devices = jax.devices()
        info['device_details'] = []
        for d in devices:
            detail = {'id': str(d), 'platform': d.platform}
            if hasattr(d, 'device_kind'):
                detail['kind'] = d.device_kind
            info['device_details'].append(detail)
    except Exception as e:
        info['device_error'] = str(e)

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            info['gpus'] = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    info['gpus'].append({
                        'name': parts[0],
                        'memory_total': parts[1],
                        'memory_free': parts[2],
                        'memory_used': parts[3],
                        'utilization': parts[4],
                        'temperature': parts[5],
                        'power': parts[6] if len(parts) > 6 else 'N/A',
                    })
            info['nvidia_smi_available'] = True
    except Exception as e:
        info['nvidia_smi_available'] = False
        info['nvidia_smi_error'] = str(e)

    return info


def run_jax_warmup():
    """Run JAX warmup and measure JIT compilation overhead."""
    print("\n[1/5] JAX Warmup Benchmark...")

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))

    # First call (includes compilation)
    start = time.perf_counter()
    y = x @ x
    y.block_until_ready()
    first_call = time.perf_counter() - start

    # Subsequent calls
    times = []
    for _ in range(10):
        start = time.perf_counter()
        y = x @ x
        y.block_until_ready()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)

    results = {
        'first_call_ms': first_call * 1000,
        'subsequent_avg_ms': avg_time * 1000,
        'jit_overhead_ratio': first_call / avg_time if avg_time > 0 else 0,
    }

    print(f"    First call: {results['first_call_ms']:.2f}ms")
    print(f"    Subsequent: {results['subsequent_avg_ms']:.2f}ms")
    print(f"    JIT overhead: {results['jit_overhead_ratio']:.1f}x")

    return results


def run_data_loading_benchmark(config):
    """Benchmark data loading with and without prefetch."""
    print("\n[2/5] Data Loading Benchmark...")

    from cl.datasets.mnist import MNISTDataset
    from cl.datasets.jax_dataloader import PrefetchDataLoader, benchmark_dataloader

    # Create dataset
    dataset_config = {
        'n_task': config.get('n_task', 2),
        'batch_size': config.get('batch_size', 512),
        'debug_mode': True,
        'debug_limit': config.get('debug_limit', 5000),
        'problem': 'classification',
        'network': 'cnn',
    }

    dataset = MNISTDataset(dataset_config)
    batch_size = config.get('batch_size', 512)

    results = {'batch_size': batch_size}

    # Without prefetch
    print("    Testing without prefetch...")
    trainloader, _ = dataset.generate_dataset(task_id=0, batch_size=batch_size, phase='training')
    no_prefetch = benchmark_dataloader(trainloader, num_batches=50, warmup=5)
    results['without_prefetch'] = no_prefetch
    print(f"      {no_prefetch['batches_per_sec']:.1f} batches/sec")

    # With prefetch
    print("    Testing with prefetch...")
    trainloader, _ = dataset.generate_dataset(task_id=0, batch_size=batch_size, phase='training')
    prefetch_loader = PrefetchDataLoader(trainloader, prefetch_size=3, loss_type='classification')
    with_prefetch = benchmark_dataloader(prefetch_loader, num_batches=50, warmup=5)
    results['with_prefetch'] = with_prefetch
    print(f"      {with_prefetch['batches_per_sec']:.1f} batches/sec")

    # Speedup
    speedup = with_prefetch['batches_per_sec'] / no_prefetch['batches_per_sec']
    results['prefetch_speedup'] = speedup
    print(f"    Prefetch speedup: {speedup:.2f}x")

    return results


def run_hamiltonian_benchmark(config):
    """Benchmark Hamiltonian gradient computation."""
    print("\n[3/5] Hamiltonian Gradient Benchmark...")

    from cl.models.cnn import CNN
    from cl.core.hamiltonian import _hamiltonian_core_class_standard
    import equinox as eqx

    batch_size = config.get('batch_size', 512)
    filter_size = 4
    channel_out = 3
    input_size = 28

    # Calculate flatten size
    conv_output = (input_size - filter_size) + 1
    pool_output = (conv_output - 2) // 2 + 1
    flatten_size = channel_out * pool_output * pool_output

    feed_sizes = [flatten_size, 512, 64, 10]

    key = jax.random.PRNGKey(42)
    model = CNN(key=key, filter_size=filter_size, feed_sizes=feed_sizes,
                channel_in=1, input_size=input_size, channel_out=channel_out)
    params, static = eqx.partition(model, eqx.is_array)

    # Synthetic data
    x = jax.random.normal(key, (batch_size, 1, 28, 28))
    y = jax.random.randint(key, (batch_size,), 0, 10)
    exp_x = jax.random.normal(key, (batch_size, 1, 28, 28))
    exp_y = jax.random.randint(key, (batch_size,), 0, 10)
    deltax = jax.random.normal(key, (batch_size, 1, 28, 28)) * 0.01

    # First call (JIT compilation)
    print("    Measuring JIT compilation...")
    start = time.perf_counter()
    grad, losses = _hamiltonian_core_class_standard(
        params, static, x, y, exp_x, exp_y, deltax,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(1000.0), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    compile_time = time.perf_counter() - start

    # Subsequent calls
    print("    Measuring execution (100 iterations)...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        grad, losses = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        times.append(time.perf_counter() - start)

    results = {
        'batch_size': batch_size,
        'compile_time_ms': compile_time * 1000,
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        'jit_overhead_ratio': compile_time / (sum(times) / len(times)),
        'throughput_samples_per_sec': batch_size / (sum(times) / len(times)),
    }

    print(f"    Compile time: {results['compile_time_ms']:.1f}ms")
    print(f"    Mean execution: {results['mean_ms']:.2f}ms (+/- {results['std_ms']:.2f}ms)")
    print(f"    Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")

    return results


def run_batch_size_sweep(config):
    """Test different batch sizes."""
    print("\n[4/5] Batch Size Sweep...")

    from cl.models.cnn import CNN
    from cl.core.hamiltonian import _hamiltonian_core_class_standard
    import equinox as eqx

    batch_sizes = [128, 256, 512, 1024]
    results = {}

    for bs in batch_sizes:
        print(f"    Testing batch_size={bs}...")

        filter_size = 4
        channel_out = 3
        input_size = 28
        conv_output = (input_size - filter_size) + 1
        pool_output = (conv_output - 2) // 2 + 1
        flatten_size = channel_out * pool_output * pool_output
        feed_sizes = [flatten_size, 512, 64, 10]

        key = jax.random.PRNGKey(42)
        model = CNN(key=key, filter_size=filter_size, feed_sizes=feed_sizes,
                    channel_in=1, input_size=input_size, channel_out=channel_out)
        params, static = eqx.partition(model, eqx.is_array)

        x = jax.random.normal(key, (bs, 1, 28, 28))
        y = jax.random.randint(key, (bs,), 0, 10)
        exp_x = jax.random.normal(key, (bs, 1, 28, 28))
        exp_y = jax.random.randint(key, (bs,), 0, 10)
        deltax = jax.random.normal(key, (bs, 1, 28, 28)) * 0.01

        # Warmup
        grad, _ = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

        # Measure
        times = []
        for _ in range(20):
            start = time.perf_counter()
            grad, _ = _hamiltonian_core_class_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)
        throughput = bs / mean_time

        results[f'batch_{bs}'] = {
            'batch_size': bs,
            'mean_ms': mean_time * 1000,
            'throughput_samples_per_sec': throughput,
        }
        print(f"      {throughput:.0f} samples/sec ({mean_time*1000:.2f}ms/batch)")

    # Find optimal
    throughputs = {k: v['throughput_samples_per_sec'] for k, v in results.items()}
    optimal = max(throughputs, key=throughputs.get)
    results['optimal'] = optimal
    print(f"    Optimal: {optimal}")

    return results


def run_full_training_sample(config):
    """Run a short training sample to measure real-world performance."""
    print("\n[5/5] Full Training Sample (2 epochs)...")

    from cl.runners.generic_runner import train_model
    from cl.config.params import apply_defaults

    # Quick training config
    train_config = apply_defaults({
        'data': 'mnist',
        'n_task': 1,
        'epochs_per_task': 2,
        'batch_size': config.get('batch_size', 512),
        'lr': 0.0001,
        'use_jax_prefetch': True,
        'prefetch_size': 3,
        'debug_mode': True,
        'debug_limit': 2000,
        'log_interval': 100,
        'eval_interval': 100,
        'model_path': 'runs__/profiling/temp_model',
    })

    start_time = time.perf_counter()
    try:
        result = train_model(train_config)
        elapsed = time.perf_counter() - start_time
        success = True
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        success = False
        result = {'error': str(e)}

    results = {
        'success': success,
        'total_time_sec': elapsed,
        'samples_trained': 2000 * 2 if success else 0,  # debug_limit * epochs
        'samples_per_sec': (2000 * 2) / elapsed if success and elapsed > 0 else 0,
    }

    if success:
        print(f"    Completed in {elapsed:.1f}s")
        print(f"    Throughput: {results['samples_per_sec']:.0f} samples/sec")
    else:
        print(f"    Failed: {result.get('error', 'Unknown error')}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run all GPU profiling benchmarks')
    parser.add_argument('--output', '-o', default='profiling_results.json',
                        help='Output JSON file (default: profiling_results.json)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: skip full training sample')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Base batch size (default: 512)')

    args = parser.parse_args()

    print("="*70)
    print("COMPREHENSIVE GPU PROFILING FOR CONDITION 1")
    print("="*70)
    print(f"\nOutput file: {args.output}")

    config = {
        'batch_size': args.batch_size,
        'debug_limit': 5000,
        'n_task': 1,
    }

    results = {
        'generated_at': datetime.now().isoformat(),
        'config': config,
    }

    # Collect system info
    print("\nCollecting system information...")
    results['system_info'] = get_system_info()
    print(f"  Backend: {results['system_info']['jax_backend']}")
    print(f"  Devices: {results['system_info']['jax_devices']}")
    if results['system_info'].get('gpus'):
        for gpu in results['system_info']['gpus']:
            print(f"  GPU: {gpu['name']} ({gpu['utilization']} util, {gpu['memory_used']}/{gpu['memory_total']})")

    # Run benchmarks
    try:
        results['jax_warmup'] = run_jax_warmup()
    except Exception as e:
        print(f"    FAILED: {e}")
        results['jax_warmup'] = {'error': str(e), 'traceback': traceback.format_exc()}

    try:
        results['data_loading'] = run_data_loading_benchmark(config)
    except Exception as e:
        print(f"    FAILED: {e}")
        results['data_loading'] = {'error': str(e), 'traceback': traceback.format_exc()}

    try:
        results['hamiltonian'] = run_hamiltonian_benchmark(config)
    except Exception as e:
        print(f"    FAILED: {e}")
        results['hamiltonian'] = {'error': str(e), 'traceback': traceback.format_exc()}

    try:
        results['batch_size_sweep'] = run_batch_size_sweep(config)
    except Exception as e:
        print(f"    FAILED: {e}")
        results['batch_size_sweep'] = {'error': str(e), 'traceback': traceback.format_exc()}

    if not args.quick:
        try:
            results['full_training'] = run_full_training_sample(config)
        except Exception as e:
            print(f"    FAILED: {e}")
            results['full_training'] = {'error': str(e), 'traceback': traceback.format_exc()}

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nShare this file with Claude for analysis:")
    print(f"  cat {output_path}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
