#!/usr/bin/env python3
"""
Comprehensive Optimization Benchmark Script

Compares GPU utilization and throughput with different optimization settings:
1. XLA optimization flags (Option 4)
2. Fused train step (Option 2)
3. Condition 1 (baseline) vs Condition 4 (AWB)

Ensures reproducibility with fixed seeds and controlled environment.

Usage:
    python scripts/benchmark_optimizations.py [--output DIR] [--quick]

Output:
    JSON file with benchmark results in specified directory

Added by Claude: January 2026
"""

# IMPORTANT: Set XLA flags BEFORE importing JAX
import os
import sys
from pathlib import Path

# Add project root to path BEFORE any other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_benchmark_with_xla(enable_xla: bool, verbose: bool = True):
    """Run benchmark with or without XLA optimization flags.

    This function must be called at the very start before JAX is imported.
    """
    from cl.core.profiling import set_xla_flags, configure_jax_for_gpu, get_optimization_status

    if enable_xla:
        set_xla_flags(enable=True, verbose=verbose)
    else:
        # Clear any existing XLA flags for fair comparison
        os.environ.pop('XLA_FLAGS', None)
        os.environ.pop('TF_GPU_THREAD_MODE', None)
        os.environ.pop('TF_GPU_THREAD_COUNT', None)

    # Now import JAX
    import jax

    # Configure JAX after import
    if enable_xla:
        configure_jax_for_gpu(verbose=verbose)

    return get_optimization_status()


# Now we can import the rest
import json
import time
import argparse
import subprocess
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    xla_enabled: bool
    fused_enabled: bool
    awb_enabled: bool  # True = Condition 4, False = Condition 1
    batch_size: int = 1024
    epochs: int = 5
    n_tasks: int = 2
    seed: int = 42
    debug_limit: int = 10000


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    total_time_sec: float
    total_batches: int
    total_samples: int
    throughput_samples_per_sec: float
    gpu_utilization_mean: float
    gpu_utilization_min: float
    gpu_utilization_max: float
    hamiltonian_mean_ms: float
    hamiltonian_total_ms: float
    optimizer_mean_ms: float
    optimizer_total_ms: float
    first_batch_time_ms: float
    xla_status: Dict[str, Any]
    component_breakdown: Dict[str, float]


def get_gpu_stats() -> Optional[Dict[str, Any]]:
    """Get current GPU stats from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu',
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
                    'temperature': parts[4] if len(parts) > 4 else 'N/A',
                }
    except Exception:
        pass
    return None


def sample_gpu_utilization() -> Optional[float]:
    """Sample current GPU utilization percentage."""
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


def run_single_benchmark(config: BenchmarkConfig, verbose: bool = True) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    from cl.datasets.mnist import MNISTDataset
    from cl.datasets.jax_dataloader import PrefetchDataLoader
    from cl.models.cnn import CNN
    from cl.core.hamiltonian import (
        _hamiltonian_core_class_standard,
        _hamiltonian_core_class_awb,
        _fused_train_step_class_standard,
        _fused_train_step_class_awb,
    )
    from cl.core.profiling import get_optimization_status

    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {config.name}")
        print(f"{'='*70}")
        print(f"  XLA enabled: {config.xla_enabled}")
        print(f"  Fused enabled: {config.fused_enabled}")
        print(f"  AWB enabled: {config.awb_enabled}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Tasks: {config.n_tasks}")

    # Timing collector
    timings = defaultdict(list)
    gpu_samples = []

    # Set reproducibility
    key = jax.random.PRNGKey(config.seed)

    # Create dataset
    dataset_config = {
        'n_task': config.n_tasks,
        'batch_size': config.batch_size,
        'debug_mode': True,
        'debug_limit': config.debug_limit,
        'problem': 'classification',
        'network': 'cnn',
        'seed': config.seed,
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
    if config.awb_enabled:
        # AWB mode: partition for A/B training
        params, static = model.partition_for_AB_training()
    else:
        # Standard mode
        params, static = eqx.partition(model, eqx.is_array)

    # Create optimizer
    optim = optax.adam(0.0001)
    opt_state = optim.init(params)

    # Select hamiltonian function based on config
    if config.fused_enabled:
        if config.awb_enabled:
            hamiltonian_fn = _fused_train_step_class_awb
        else:
            hamiltonian_fn = _fused_train_step_class_standard
    else:
        if config.awb_enabled:
            hamiltonian_fn = _hamiltonian_core_class_awb
        else:
            hamiltonian_fn = _hamiltonian_core_class_standard

    # JIT compile optimizer step
    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Helper to convert tensors
    def to_jax(tensor):
        if hasattr(tensor, 'numpy'):
            return jnp.array(tensor.numpy())
        return tensor

    # Pre-compute sqrt(param_count)
    leaves = jax.tree_util.tree_leaves(params)
    param_count = sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    sqrt_param_count = jnp.sqrt(jnp.array(param_count, dtype=jnp.float64))

    # Training loop
    total_batches = 0
    first_batch_time = None
    total_start = time.perf_counter()

    for task_id in range(config.n_tasks):
        if verbose:
            print(f"\n  Task {task_id}...")

        # Generate data
        trainloader, exploader = dataset.generate_dataset(
            task_id=task_id, batch_size=config.batch_size, phase='training'
        )

        # Wrap with prefetch
        trainloader = PrefetchDataLoader(trainloader, prefetch_size=3, loss_type='classification')
        exploader = PrefetchDataLoader(exploader, prefetch_size=3, loss_type='classification')

        # Convert to JAX
        train_batches = [(to_jax(x), to_jax(y)) for x, y in trainloader]
        exp_batches = [(to_jax(x), to_jax(y)) for x, y in exploader]

        for epoch in range(config.epochs):
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
                if config.fused_enabled:
                    grad, losses = hamiltonian_fn(
                        params, static, opt_state, x, y, exp_x, exp_y, delta_x,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        sqrt_param_count, jnp.array(1.0), jnp.array(0.0)  # max_grad_norm=0 (disabled)
                    )
                else:
                    grad, losses = hamiltonian_fn(
                        params, static, x, y, exp_x, exp_y, delta_x,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        sqrt_param_count, jnp.array(1.0)
                    )
                jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, grad)
                timings['hamiltonian'].append((time.perf_counter() - t0) * 1000)

                # Optimizer step
                t0 = time.perf_counter()
                params, opt_state = optimizer_step(grad, opt_state, params)
                jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, params)
                timings['optimizer'].append((time.perf_counter() - t0) * 1000)

                # Record first batch time (includes JIT compilation)
                if first_batch_time is None:
                    first_batch_time = (time.perf_counter() - batch_start) * 1000

            if verbose and epoch == 0:
                epoch_batches = len(train_batches)
                hamil_mean = sum(timings['hamiltonian'][-epoch_batches:]) / epoch_batches
                print(f"    Epoch {epoch}: hamiltonian={hamil_mean:.1f}ms/batch")

    total_time = time.perf_counter() - total_start
    total_samples = total_batches * config.batch_size

    # Compute statistics
    def stats(values):
        if not values:
            return {'mean': 0, 'min': 0, 'max': 0, 'total': 0}
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values),
        }

    hamil_stats = stats(timings['hamiltonian'])
    opt_stats = stats(timings['optimizer'])
    gpu_stats = stats(gpu_samples) if gpu_samples else {'mean': 0, 'min': 0, 'max': 0}

    # Component breakdown (percentage of total profiled time)
    total_profiled = sum(sum(v) for v in timings.values())
    breakdown = {}
    for name, values in timings.items():
        breakdown[name] = (sum(values) / total_profiled * 100) if total_profiled > 0 else 0

    result = BenchmarkResult(
        config_name=config.name,
        total_time_sec=total_time,
        total_batches=total_batches,
        total_samples=total_samples,
        throughput_samples_per_sec=total_samples / total_time,
        gpu_utilization_mean=gpu_stats['mean'],
        gpu_utilization_min=gpu_stats.get('min', 0),
        gpu_utilization_max=gpu_stats.get('max', 0),
        hamiltonian_mean_ms=hamil_stats['mean'],
        hamiltonian_total_ms=hamil_stats['total'],
        optimizer_mean_ms=opt_stats['mean'],
        optimizer_total_ms=opt_stats['total'],
        first_batch_time_ms=first_batch_time or 0,
        xla_status=get_optimization_status(),
        component_breakdown=breakdown,
    )

    if verbose:
        print(f"\n  Results:")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Throughput: {total_samples/total_time:.0f} samples/sec")
        print(f"    GPU utilization: {gpu_stats['mean']:.1f}%")
        print(f"    Hamiltonian: {hamil_stats['mean']:.1f}ms (first: {timings['hamiltonian'][0]:.1f}ms)")

    return result


def run_all_benchmarks(output_dir: str, quick: bool = False, verbose: bool = True):
    """Run all benchmark configurations and save results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define benchmark configurations
    if quick:
        epochs = 2
        n_tasks = 1
        debug_limit = 5000
    else:
        epochs = 5
        n_tasks = 2
        debug_limit = 10000

    configs = [
        # Baseline comparisons (Condition 1)
        BenchmarkConfig(
            name="baseline_no_xla",
            xla_enabled=False,
            fused_enabled=False,
            awb_enabled=False,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="baseline_xla_only",
            xla_enabled=True,
            fused_enabled=False,
            awb_enabled=False,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="baseline_fused_only",
            xla_enabled=False,
            fused_enabled=True,
            awb_enabled=False,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="baseline_xla_and_fused",
            xla_enabled=True,
            fused_enabled=True,
            awb_enabled=False,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        # AWB comparisons (Condition 4)
        BenchmarkConfig(
            name="awb_no_xla",
            xla_enabled=False,
            fused_enabled=False,
            awb_enabled=True,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="awb_xla_only",
            xla_enabled=True,
            fused_enabled=False,
            awb_enabled=True,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="awb_fused_only",
            xla_enabled=False,
            fused_enabled=True,
            awb_enabled=True,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
        BenchmarkConfig(
            name="awb_xla_and_fused",
            xla_enabled=True,
            fused_enabled=True,
            awb_enabled=True,
            epochs=epochs,
            n_tasks=n_tasks,
            debug_limit=debug_limit,
        ),
    ]

    results = []

    print("\n" + "="*70)
    print("COMPREHENSIVE OPTIMIZATION BENCHMARK")
    print("="*70)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Quick mode: {quick}")
    print(f"Configurations to test: {len(configs)}")
    print(f"GPU: {get_gpu_stats()}")

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running {config.name}...")

        try:
            # Note: XLA flags should ideally be set before JAX is imported.
            # Since we're in the same process, subsequent changes may have limited effect.
            # For rigorous testing, each config should be run in a separate subprocess.
            result = run_single_benchmark(config, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison report
    report = {
        'generated_at': datetime.now().isoformat(),
        'quick_mode': quick,
        'system_info': {
            'gpu': get_gpu_stats(),
        },
        'results': [asdict(r) for r in results],
        'comparison': generate_comparison(results),
    }

    # Save report
    report_file = output_path / 'benchmark_results.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {report_file.absolute()}")

    # Print comparison summary
    print_comparison_summary(results)

    return report


def generate_comparison(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Generate comparison metrics between results."""
    if not results:
        return {}

    # Find baselines
    baseline_no_opt = next((r for r in results if r.config_name == "baseline_no_xla"), None)
    awb_no_opt = next((r for r in results if r.config_name == "awb_no_xla"), None)

    comparison = {}

    for r in results:
        baseline = awb_no_opt if 'awb' in r.config_name else baseline_no_opt
        if baseline:
            speedup = r.throughput_samples_per_sec / baseline.throughput_samples_per_sec
            gpu_improvement = r.gpu_utilization_mean - baseline.gpu_utilization_mean
            comparison[r.config_name] = {
                'speedup_vs_baseline': speedup,
                'gpu_util_improvement_pct': gpu_improvement,
                'hamiltonian_speedup': baseline.hamiltonian_mean_ms / r.hamiltonian_mean_ms if r.hamiltonian_mean_ms > 0 else 0,
            }

    return comparison


def print_comparison_summary(results: List[BenchmarkResult]):
    """Print a summary comparison table."""
    if not results:
        return

    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<25} {'Throughput':<15} {'GPU Util':<12} {'Hamiltonian':<15} {'First Batch':<12}")
    print(f"{'':25} {'(samples/s)':<15} {'(%)':<12} {'(ms/batch)':<15} {'(ms)':<12}")
    print(f"{'-'*90}")

    for r in results:
        print(f"{r.config_name:<25} {r.throughput_samples_per_sec:<15.0f} {r.gpu_utilization_mean:<12.1f} {r.hamiltonian_mean_ms:<15.1f} {r.first_batch_time_ms:<12.0f}")

    # Calculate improvements
    baseline = next((r for r in results if r.config_name == "baseline_no_xla"), None)
    best_baseline = next((r for r in results if r.config_name == "baseline_xla_and_fused"), None)
    awb_baseline = next((r for r in results if r.config_name == "awb_no_xla"), None)
    best_awb = next((r for r in results if r.config_name == "awb_xla_and_fused"), None)

    print(f"\n{'-'*90}")
    print("IMPROVEMENTS:")

    if baseline and best_baseline:
        speedup = best_baseline.throughput_samples_per_sec / baseline.throughput_samples_per_sec
        print(f"  Baseline (XLA+Fused vs None): {speedup:.2f}x throughput")

    if awb_baseline and best_awb:
        speedup = best_awb.throughput_samples_per_sec / awb_baseline.throughput_samples_per_sec
        gpu_improve = best_awb.gpu_utilization_mean - awb_baseline.gpu_utilization_mean
        print(f"  AWB (XLA+Fused vs None): {speedup:.2f}x throughput, +{gpu_improve:.1f}% GPU util")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive optimization benchmarks')
    parser.add_argument('--output', '-o', default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode (fewer epochs/tasks for faster testing)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')

    args = parser.parse_args()

    # Run benchmarks
    run_all_benchmarks(args.output, quick=args.quick, verbose=args.verbose)

    return 0


if __name__ == '__main__':
    sys.exit(main())
