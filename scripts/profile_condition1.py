#!/usr/bin/env python3
"""
Comprehensive GPU Profiling Script for Condition 1 (Baseline - No AWB)

This script profiles the MNIST condition 1 training to identify GPU utilization
bottlenecks. It collects detailed timing data for each component of the training
pipeline and generates a comprehensive report.

Usage:
    python scripts/profile_condition1.py [config_path] [--output-dir DIR]

Arguments:
    config_path: Path to profiling config (default: runs__/configs/mnist_condition1_profiling.json)
    --output-dir: Directory for profiling reports (default: runs__/profiling/reports)

Output:
    - JSON report with detailed timing breakdown
    - Console summary with bottleneck analysis
    - GPU utilization statistics

Requirements:
    - CUDA GPU with nvidia-smi available
    - JAX with GPU support

Added by Claude: January 2025 - For identifying GPU utilization bottlenecks
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


def get_system_info():
    """Collect system and environment information."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'jax_version': jax.__version__,
        'jax_backend': jax.default_backend(),
        'jax_devices': [str(d) for d in jax.devices()],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
    }

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            info['nvidia_gpus'] = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    info['nvidia_gpus'].append({
                        'name': parts[0],
                        'memory_total': parts[1],
                        'driver_version': parts[2],
                    })
            info['nvidia_smi_available'] = True
        else:
            info['nvidia_smi_available'] = False
    except Exception as e:
        info['nvidia_smi_available'] = False
        info['nvidia_smi_error'] = str(e)

    return info


def run_warmup_benchmark():
    """Run JAX warmup and measure JIT compilation overhead."""
    print("\n" + "="*70)
    print("PHASE 1: JAX Warmup and JIT Benchmark")
    print("="*70)

    # Simple matrix multiplication to warm up JAX/XLA
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
        'jit_speedup': first_call / avg_time if avg_time > 0 else 0,
    }

    print(f"\nJIT Warmup Results:")
    print(f"  First call (w/ compilation): {results['first_call_ms']:.2f}ms")
    print(f"  Subsequent calls (avg):      {results['subsequent_avg_ms']:.2f}ms")
    print(f"  JIT Speedup:                 {results['jit_speedup']:.1f}x")

    return results


def create_dataset(config):
    """Create dataset instance based on config."""
    from cl.datasets.sine import SineDataset
    from cl.datasets.mnist import MNISTDataset, PermutedMNISTDataset
    from cl.datasets.cifar import CIFAR10Dataset, CIFAR100Dataset

    data_type = config.get('data', 'mnist')

    # Build dataset config
    dataset_config = {
        'n_task': config.get('n_task', 10),
        'batch_size': config.get('batch_size', 512),
        'seed': config.get('seed', 42),
        'debug_mode': config.get('debug_mode', False),
        'debug_limit': config.get('debug_limit', None),
    }

    if data_type == 'sine':
        dataset_config.update({
            'n_layers': config.get('n_layers', 4),
            'hln': config.get('hln', 64),
        })
        return SineDataset(dataset_config)
    elif data_type == 'mnist':
        return MNISTDataset(dataset_config)
    elif data_type == 'permuted_mnist':
        return PermutedMNISTDataset(dataset_config)
    elif data_type == 'cifar10':
        return CIFAR10Dataset(dataset_config)
    elif data_type == 'cifar100':
        return CIFAR100Dataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {data_type}")


def run_data_loading_benchmark(config):
    """Benchmark data loading pipeline with and without prefetch."""
    print("\n" + "="*70)
    print("PHASE 2: Data Loading Benchmark")
    print("="*70)

    from cl.datasets.jax_dataloader import PrefetchDataLoader, benchmark_dataloader

    # Create dataset
    dataset_name = config.get('data', 'mnist')
    batch_size = config.get('batch_size', 512)

    print(f"\nDataset: {dataset_name}, Batch size: {batch_size}")

    # Get dataset class
    dataset = create_dataset(config)

    # Generate data for task 0
    trainloader, exploader = dataset.generate_dataset(task_id=0, batch_size=batch_size, phase='train')

    results = {}

    # Benchmark without prefetch
    print("\nBenchmarking WITHOUT prefetch...")
    try:
        no_prefetch_stats = benchmark_dataloader(trainloader, num_batches=50, warmup=5)
        results['without_prefetch'] = no_prefetch_stats
        print(f"  Batches/sec: {no_prefetch_stats['batches_per_sec']:.2f}")
        print(f"  Avg batch time: {no_prefetch_stats['avg_batch_time_ms']:.2f}ms")
    except Exception as e:
        print(f"  Error: {e}")
        results['without_prefetch'] = {'error': str(e)}

    # Benchmark with prefetch
    print("\nBenchmarking WITH prefetch...")
    try:
        # Re-create loaders
        trainloader, _ = dataset.generate_dataset(task_id=0, batch_size=batch_size, phase='train')
        prefetch_loader = PrefetchDataLoader(trainloader, prefetch_size=3, loss_type='classification')
        prefetch_stats = benchmark_dataloader(prefetch_loader, num_batches=50, warmup=5)
        results['with_prefetch'] = prefetch_stats
        print(f"  Batches/sec: {prefetch_stats['batches_per_sec']:.2f}")
        print(f"  Avg batch time: {prefetch_stats['avg_batch_time_ms']:.2f}ms")

        # Compute speedup
        if 'without_prefetch' in results and 'batches_per_sec' in results['without_prefetch']:
            speedup = prefetch_stats['batches_per_sec'] / results['without_prefetch']['batches_per_sec']
            results['prefetch_speedup'] = speedup
            print(f"\n  Prefetch Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"  Error: {e}")
        results['with_prefetch'] = {'error': str(e)}

    return results


def run_hamiltonian_benchmark(config):
    """Benchmark Hamiltonian gradient computation."""
    print("\n" + "="*70)
    print("PHASE 3: Hamiltonian Computation Benchmark")
    print("="*70)

    from cl.models.mlp import MLP
    from cl.core.hamiltonian import _hamiltonian_core_class_standard
    import equinox as eqx

    # Create model
    network_type = config.get('network', 'fcnn')
    batch_size = config.get('batch_size', 512)

    print(f"\nNetwork: {network_type}, Batch size: {batch_size}")

    # Create MLP for MNIST (784 input, 10 output)
    key = jax.random.PRNGKey(42)
    # Default MLP configuration for MNIST
    input_size = 784
    output_size = 10
    n_layers = config.get('n_layers', 4)
    hln = config.get('hln', 256)
    feed_sizes = [input_size] + [hln] * (n_layers - 1) + [output_size]

    model = MLP(jax.random.PRNGKey(0), feed_sizes=feed_sizes, awb_arch=None)
    params, static = eqx.partition(model, eqx.is_array)

    # Generate synthetic data
    x = jax.random.normal(key, (batch_size, 784))
    y = jax.random.randint(key, (batch_size,), 0, 10)
    exp_x = jax.random.normal(key, (batch_size, 784))
    exp_y = jax.random.randint(key, (batch_size,), 0, 10)
    deltax = jax.random.normal(key, (batch_size, 784)) * 0.01

    results = {}

    # First call (includes compilation)
    print("\nMeasuring JIT compilation time...")
    start = time.perf_counter()
    grad, losses = _hamiltonian_core_class_standard(
        params, static, x, y, exp_x, exp_y, deltax,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(1000.0), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    compile_time = time.perf_counter() - start
    results['first_call_ms'] = compile_time * 1000
    print(f"  First call (w/ compilation): {results['first_call_ms']:.2f}ms")

    # Subsequent calls
    print("\nMeasuring execution time (100 iterations)...")
    times = []
    for i in range(100):
        start = time.perf_counter()
        grad, losses = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        times.append(time.perf_counter() - start)

    results['execution_times_ms'] = [t * 1000 for t in times]
    results['mean_ms'] = sum(results['execution_times_ms']) / len(results['execution_times_ms'])
    results['min_ms'] = min(results['execution_times_ms'])
    results['max_ms'] = max(results['execution_times_ms'])
    results['jit_speedup'] = results['first_call_ms'] / results['mean_ms'] if results['mean_ms'] > 0 else 0

    print(f"  Mean execution time: {results['mean_ms']:.2f}ms")
    print(f"  Min: {results['min_ms']:.2f}ms, Max: {results['max_ms']:.2f}ms")
    print(f"  JIT Speedup: {results['jit_speedup']:.1f}x")

    # Estimate throughput
    samples_per_sec = batch_size / (results['mean_ms'] / 1000)
    results['samples_per_sec'] = samples_per_sec
    print(f"  Throughput: {samples_per_sec:.0f} samples/sec")

    return results


def run_full_training_profile(config, output_dir):
    """Run full training with detailed profiling."""
    print("\n" + "="*70)
    print("PHASE 4: Full Training Profile")
    print("="*70)

    from cl.runners.generic_runner import train_model
    from cl.core.profiling import (enable_profiling, set_detailed_profiling,
                                    init_collector, get_collector)

    # Enable profiling
    config['profiling_enabled'] = True
    config['detailed_profiling'] = True

    print(f"\nRunning training with config:")
    print(f"  Dataset: {config.get('data', 'mnist')}")
    print(f"  Tasks: {config.get('n_task', 2)}")
    print(f"  Epochs/task: {config.get('epochs_per_task', 20)}")
    print(f"  Batch size: {config.get('batch_size', 512)}")
    print(f"  Prefetch: {config.get('use_jax_prefetch', True)}")

    # Initialize profiling
    enable_profiling(True)
    set_detailed_profiling(True)
    collector = init_collector(config)

    # Run training
    start_time = time.perf_counter()
    try:
        result = train_model(config)
        success = True
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
        result = None

    elapsed = time.perf_counter() - start_time

    # Get profiling results
    collector = get_collector()
    if collector:
        # Generate report
        report_path = Path(output_dir) / f"profile_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = collector.generate_report(str(report_path))

        # Print summary
        collector.print_summary()

        return {
            'success': success,
            'total_time_sec': elapsed,
            'report_path': str(report_path),
            'stats': collector.get_stats(),
        }
    else:
        return {
            'success': success,
            'total_time_sec': elapsed,
            'error': 'Collector not available',
        }


def generate_final_report(all_results, output_dir):
    """Generate comprehensive final report."""
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)

    report = {
        'generated_at': datetime.now().isoformat(),
        'system_info': all_results.get('system_info', {}),
        'benchmarks': {
            'jax_warmup': all_results.get('jax_warmup', {}),
            'data_loading': all_results.get('data_loading', {}),
            'hamiltonian': all_results.get('hamiltonian', {}),
        },
        'training_profile': all_results.get('training_profile', {}),
    }

    # Analyze bottlenecks
    bottlenecks = []

    # Check GPU utilization
    training = all_results.get('training_profile', {})
    stats = training.get('stats', {})
    if 'gpu' in stats:
        gpu_util = stats['gpu'].get('utilization_mean', 0)
        if gpu_util < 30:
            bottlenecks.append({
                'type': 'GPU Underutilization (Critical)',
                'value': f"{gpu_util:.1f}%",
                'recommendation': 'GPU is severely underutilized. Main bottleneck is likely '
                                 'data loading or CPU-GPU synchronization.'
            })
        elif gpu_util < 60:
            bottlenecks.append({
                'type': 'GPU Underutilization (Moderate)',
                'value': f"{gpu_util:.1f}%",
                'recommendation': 'GPU could be better utilized. Consider increasing batch size '
                                 'or optimizing data pipeline.'
            })

    # Check data loading
    data_loading = all_results.get('data_loading', {})
    if 'prefetch_speedup' in data_loading:
        speedup = data_loading['prefetch_speedup']
        if speedup > 2:
            bottlenecks.append({
                'type': 'Data Loading Bottleneck',
                'value': f"{speedup:.1f}x speedup with prefetch",
                'recommendation': 'Data loading is a significant bottleneck. Ensure prefetch is enabled.'
            })

    # Check Hamiltonian computation
    hamiltonian = all_results.get('hamiltonian', {})
    if 'jit_speedup' in hamiltonian:
        jit_speedup = hamiltonian['jit_speedup']
        if jit_speedup > 10:
            bottlenecks.append({
                'type': 'JIT Compilation Overhead',
                'value': f"{jit_speedup:.1f}x",
                'recommendation': 'First iteration is very slow due to JIT compilation. '
                                 'This amortizes over training but consider pre-warming.'
            })

    report['bottleneck_analysis'] = bottlenecks

    # Save final report
    report_path = Path(output_dir) / f"final_profile_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nFinal report saved to: {report_path}")

    # Print summary
    print("\n" + "-"*50)
    print("BOTTLENECK SUMMARY")
    print("-"*50)

    if bottlenecks:
        for i, b in enumerate(bottlenecks, 1):
            print(f"\n{i}. {b['type']}")
            print(f"   Value: {b['value']}")
            print(f"   Recommendation: {b['recommendation']}")
    else:
        print("\nNo major bottlenecks detected.")

    print("\n" + "="*70)

    return report


def main():
    parser = argparse.ArgumentParser(description='Profile MNIST Condition 1 training')
    parser.add_argument('config', nargs='?',
                       default='runs__/configs/mnist_condition1_profiling.json',
                       help='Path to profiling config')
    parser.add_argument('--output-dir', default='runs__/profiling/reports',
                       help='Directory for profiling reports')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip full training profile (only run benchmarks)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer iterations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MNIST CONDITION 1 GPU PROFILING")
    print("="*70)
    print(f"\nConfig: {args.config}")
    print(f"Output: {output_dir}")

    # Load config using proper loader that applies dataset-specific defaults
    from cl.config import load_config

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"\nLoaded config from {config_path}")
        print(f"  Dataset: {config.get('data')}")
        print(f"  Network: {config.get('network')}")
        print(f"  Problem type: {config.get('prob')}")
    else:
        print(f"\nConfig not found, using defaults")
        # Use load_config's apply_defaults by creating a temp config
        from cl.config.params import apply_defaults
        config = apply_defaults({
            'data': 'mnist',
            'n_task': 2,
            'epochs_per_task': 10,
            'batch_size': 512,
            'profiling_enabled': True,
            'detailed_profiling': True,
            'use_jax_prefetch': True,
            'prefetch_size': 3,
            'debug_mode': True,
            'debug_limit': 2000,
        })

    if args.quick:
        config['n_task'] = 1
        config['epochs_per_task'] = 5
        config['debug_limit'] = 1000

    all_results = {}

    # Phase 1: System info and JAX warmup
    all_results['system_info'] = get_system_info()
    print(f"\nSystem: {all_results['system_info']['jax_backend']} backend")
    print(f"Devices: {all_results['system_info']['jax_devices']}")

    all_results['jax_warmup'] = run_warmup_benchmark()

    # Phase 2: Data loading benchmark
    try:
        all_results['data_loading'] = run_data_loading_benchmark(config)
    except Exception as e:
        print(f"\nData loading benchmark failed: {e}")
        all_results['data_loading'] = {'error': str(e)}

    # Phase 3: Hamiltonian benchmark
    try:
        all_results['hamiltonian'] = run_hamiltonian_benchmark(config)
    except Exception as e:
        print(f"\nHamiltonian benchmark failed: {e}")
        all_results['hamiltonian'] = {'error': str(e)}

    # Phase 4: Full training profile
    if not args.skip_training:
        try:
            all_results['training_profile'] = run_full_training_profile(config, output_dir)
        except Exception as e:
            print(f"\nFull training profile failed: {e}")
            import traceback
            traceback.print_exc()
            all_results['training_profile'] = {'error': str(e)}
    else:
        print("\n[Skipping full training profile]")

    # Generate final report
    final_report = generate_final_report(all_results, output_dir)

    print("\nProfiling complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
