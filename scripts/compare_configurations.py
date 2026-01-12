#!/usr/bin/env python3
"""
Configuration Comparison Script for GPU Profiling

This script compares different training configurations to identify
the optimal settings for GPU utilization. It runs multiple configurations
and generates a comparison report.

Usage:
    python scripts/compare_configurations.py [--output-dir DIR]

Configurations compared:
    1. With vs Without JAX Prefetch
    2. Different batch sizes (128, 256, 512, 1024)
    3. Different prefetch sizes (1, 2, 3, 5)

Added by Claude: January 2025
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


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


def run_single_config(config: Dict, num_epochs: int = 5, task_id: int = 0) -> Dict:
    """Run a single configuration and return timing results."""
    from cl.datasets.jax_dataloader import PrefetchDataLoader
    from cl.models.mlp import MLP
    from cl.models.cnn import CNN
    from cl.core.hamiltonian import _hamiltonian_core_class_standard
    import equinox as eqx
    import optax

    results = {
        'config': {
            'batch_size': config.get('batch_size', 512),
            'use_jax_prefetch': config.get('use_jax_prefetch', True),
            'prefetch_size': config.get('prefetch_size', 3),
            'network': config.get('network', 'cnn'),
        }
    }

    # Create dataset
    dataset = create_dataset(config)
    trainloader, exploader = dataset.generate_dataset(
        task_id=task_id,
        batch_size=config.get('batch_size', 512),
        phase='train'
    )

    # Determine loss type from config
    loss_type = 'regression' if config.get('prob') == 'regression' else 'classification'

    # Optionally wrap with prefetch
    if config.get('use_jax_prefetch', True):
        trainloader = PrefetchDataLoader(
            trainloader,
            prefetch_size=config.get('prefetch_size', 3),
            loss_type=loss_type
        )
        exploader = PrefetchDataLoader(
            exploader,
            prefetch_size=config.get('prefetch_size', 3),
            loss_type=loss_type
        )

    # Create model based on network type
    key = jax.random.PRNGKey(42)
    network = config.get('network', 'cnn')

    if network == 'fcnn':
        # MLP for regression or flattened input
        input_size = dataset.input_size
        output_size = dataset.output_size
        n_layers = config.get('n_layers', 4)
        hln = config.get('hln', 256)
        feed_sizes = [input_size] + [hln] * (n_layers - 1) + [output_size]
        model = MLP(jax.random.PRNGKey(0), feed_sizes=feed_sizes, awb_arch=None)
    elif network == 'cnn':
        # CNN for image classification (MNIST)
        channel_out = config.get('channel_out', 3)
        filter_size = config.get('filter_size', 4)
        input_size = 28  # MNIST
        num_classes = dataset.output_size

        # Calculate flatten_size
        conv_output = (input_size - filter_size + 1)
        pool_output = conv_output // 2
        flatten_size = channel_out * pool_output * pool_output

        feed_sizes = [flatten_size, 512, 64, num_classes]

        model = CNN(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_out=channel_out,
            num_classes=num_classes,
            awb_arch=None
        )
    else:
        raise ValueError(f"Unsupported network type: {network}")

    params, static = eqx.partition(model, eqx.is_array)

    # Create optimizer
    optim = optax.adam(config.get('lr', 0.0001))
    opt_state = optim.init(params)

    # Pre-convert data to JAX
    train_batches = list(trainloader)
    exp_batches = list(exploader)

    results['num_batches'] = len(train_batches)
    results['samples_per_batch'] = config.get('batch_size', 512)

    # JIT compile optimizer step
    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Warm up JIT
    if train_batches and exp_batches:
        x, y = train_batches[0]
        exp_x, exp_y = exp_batches[0]
        min_batch = min(x.shape[0], exp_x.shape[0])
        x, y = x[:min_batch], y[:min_batch]
        exp_x, exp_y = exp_x[:min_batch], exp_y[:min_batch]

        rng_key = jax.random.PRNGKey(0)
        delta_x = jax.random.normal(rng_key, exp_x.shape) * 0.01

        # Warmup call
        grad, losses = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, delta_x,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        params, opt_state = optimizer_step(grad, opt_state, params)

    # Measure epoch times
    epoch_times = []
    batch_times = []
    rng_key = jax.random.PRNGKey(1)

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        batch_count = 0

        for batch, batch_ex in zip(train_batches, exp_batches):
            batch_start = time.perf_counter()

            x, y = batch
            exp_x, exp_y = batch_ex
            min_batch = min(x.shape[0], exp_x.shape[0])
            x, y = x[:min_batch], y[:min_batch]
            exp_x, exp_y = exp_x[:min_batch], exp_y[:min_batch]

            rng_key, subkey = jax.random.split(rng_key)
            delta_x = jax.random.normal(subkey, exp_x.shape) * 0.01

            # Hamiltonian gradient
            grad, losses = _hamiltonian_core_class_standard(
                params, static, x, y, exp_x, exp_y, delta_x,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

            # Optimizer step
            params, opt_state = optimizer_step(grad, opt_state, params)

            batch_times.append(time.perf_counter() - batch_start)
            batch_count += 1

        epoch_times.append(time.perf_counter() - epoch_start)

    # Compute statistics
    results['epoch_times_sec'] = epoch_times
    results['mean_epoch_sec'] = sum(epoch_times) / len(epoch_times)
    results['batch_times_ms'] = [t * 1000 for t in batch_times]
    results['mean_batch_ms'] = sum(results['batch_times_ms']) / len(results['batch_times_ms'])
    results['min_batch_ms'] = min(results['batch_times_ms'])
    results['max_batch_ms'] = max(results['batch_times_ms'])
    results['throughput_samples_per_sec'] = (
        results['samples_per_batch'] / (results['mean_batch_ms'] / 1000)
    )

    return results


def compare_prefetch(base_config: Dict, output_dir: Path) -> Dict:
    """Compare with and without JAX prefetch."""
    print("\n" + "="*70)
    print("COMPARISON 1: With vs Without JAX Prefetch")
    print("="*70)

    results = {}

    # Without prefetch
    print("\nRunning WITHOUT prefetch...")
    config_no_prefetch = base_config.copy()
    config_no_prefetch['use_jax_prefetch'] = False
    results['without_prefetch'] = run_single_config(config_no_prefetch)
    print(f"  Mean batch time: {results['without_prefetch']['mean_batch_ms']:.2f}ms")
    print(f"  Throughput: {results['without_prefetch']['throughput_samples_per_sec']:.0f} samples/sec")

    # With prefetch
    print("\nRunning WITH prefetch...")
    config_with_prefetch = base_config.copy()
    config_with_prefetch['use_jax_prefetch'] = True
    config_with_prefetch['prefetch_size'] = 3
    results['with_prefetch'] = run_single_config(config_with_prefetch)
    print(f"  Mean batch time: {results['with_prefetch']['mean_batch_ms']:.2f}ms")
    print(f"  Throughput: {results['with_prefetch']['throughput_samples_per_sec']:.0f} samples/sec")

    # Compute speedup
    speedup = (results['with_prefetch']['throughput_samples_per_sec'] /
               results['without_prefetch']['throughput_samples_per_sec'])
    results['prefetch_speedup'] = speedup
    print(f"\n  Speedup with prefetch: {speedup:.2f}x")

    return results


def compare_batch_sizes(base_config: Dict, output_dir: Path) -> Dict:
    """Compare different batch sizes."""
    print("\n" + "="*70)
    print("COMPARISON 2: Batch Size Sweep")
    print("="*70)

    batch_sizes = [128, 256, 512, 1024]
    results = {}

    for bs in batch_sizes:
        print(f"\nRunning with batch_size={bs}...")
        config = base_config.copy()
        config['batch_size'] = bs
        config['use_jax_prefetch'] = True

        try:
            results[f'batch_{bs}'] = run_single_config(config)
            print(f"  Mean batch time: {results[f'batch_{bs}']['mean_batch_ms']:.2f}ms")
            print(f"  Throughput: {results[f'batch_{bs}']['throughput_samples_per_sec']:.0f} samples/sec")
        except Exception as e:
            print(f"  Failed: {e}")
            results[f'batch_{bs}'] = {'error': str(e)}

    # Find optimal
    throughputs = {k: v.get('throughput_samples_per_sec', 0)
                   for k, v in results.items() if 'error' not in v}
    if throughputs:
        optimal = max(throughputs, key=throughputs.get)
        results['optimal_batch_size'] = int(optimal.split('_')[1])
        print(f"\n  Optimal batch size: {results['optimal_batch_size']}")

    return results


def compare_prefetch_sizes(base_config: Dict, output_dir: Path) -> Dict:
    """Compare different prefetch queue sizes."""
    print("\n" + "="*70)
    print("COMPARISON 3: Prefetch Size Sweep")
    print("="*70)

    prefetch_sizes = [1, 2, 3, 5, 8]
    results = {}

    for ps in prefetch_sizes:
        print(f"\nRunning with prefetch_size={ps}...")
        config = base_config.copy()
        config['use_jax_prefetch'] = True
        config['prefetch_size'] = ps

        try:
            results[f'prefetch_{ps}'] = run_single_config(config)
            print(f"  Mean batch time: {results[f'prefetch_{ps}']['mean_batch_ms']:.2f}ms")
            print(f"  Throughput: {results[f'prefetch_{ps}']['throughput_samples_per_sec']:.0f} samples/sec")
        except Exception as e:
            print(f"  Failed: {e}")
            results[f'prefetch_{ps}'] = {'error': str(e)}

    # Find optimal
    throughputs = {k: v.get('throughput_samples_per_sec', 0)
                   for k, v in results.items() if 'error' not in v}
    if throughputs:
        optimal = max(throughputs, key=throughputs.get)
        results['optimal_prefetch_size'] = int(optimal.split('_')[1])
        print(f"\n  Optimal prefetch size: {results['optimal_prefetch_size']}")

    return results


def generate_comparison_report(all_results: Dict, output_dir: Path) -> Dict:
    """Generate comprehensive comparison report."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    report = {
        'generated_at': datetime.now().isoformat(),
        'comparisons': all_results,
    }

    # Summary table
    print("\n{:<25} {:>15} {:>15}".format("Configuration", "Throughput", "Relative"))
    print("-"*55)

    # Get baseline (512 batch, with prefetch)
    baseline = None
    if 'prefetch_comparison' in all_results:
        baseline = all_results['prefetch_comparison'].get('with_prefetch', {}).get(
            'throughput_samples_per_sec', 1)

    # Print all configurations
    for comparison_name, comparison_data in all_results.items():
        for config_name, config_data in comparison_data.items():
            if isinstance(config_data, dict) and 'throughput_samples_per_sec' in config_data:
                throughput = config_data['throughput_samples_per_sec']
                relative = throughput / baseline if baseline else 1.0
                print(f"{config_name:<25} {throughput:>12.0f}/s {relative:>13.2f}x")

    # Recommendations
    print("\n" + "-"*55)
    print("RECOMMENDATIONS")
    print("-"*55)

    recommendations = []

    # Check prefetch benefit
    prefetch_cmp = all_results.get('prefetch_comparison', {})
    if 'prefetch_speedup' in prefetch_cmp:
        speedup = prefetch_cmp['prefetch_speedup']
        if speedup > 1.5:
            recommendations.append(f"Enable JAX prefetch (provides {speedup:.1f}x speedup)")
        elif speedup < 1.1:
            recommendations.append("JAX prefetch provides minimal benefit - data loading is not the bottleneck")

    # Check batch size
    batch_cmp = all_results.get('batch_size_comparison', {})
    if 'optimal_batch_size' in batch_cmp:
        recommendations.append(f"Use batch size {batch_cmp['optimal_batch_size']} for optimal throughput")

    # Check prefetch size
    prefetch_size_cmp = all_results.get('prefetch_size_comparison', {})
    if 'optimal_prefetch_size' in prefetch_size_cmp:
        recommendations.append(f"Use prefetch_size={prefetch_size_cmp['optimal_prefetch_size']} for optimal throughput")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    report['recommendations'] = recommendations

    # Save report
    report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")
    print("="*70)

    return report


def main():
    parser = argparse.ArgumentParser(description='Compare different training configurations')
    parser.add_argument('--output-dir', default='runs__/profiling/reports',
                       help='Directory for reports')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Epochs per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer configurations')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CONFIGURATION COMPARISON PROFILING")
    print("="*70)
    print(f"\nOutput: {output_dir}")
    print(f"Epochs per config: {args.epochs}")

    # Base configuration with dataset-specific defaults applied
    from cl.config.params import apply_defaults
    base_config = apply_defaults({
        'data': 'mnist',
        'batch_size': 512,
        'lr': 0.0001,
        'use_jax_prefetch': True,
        'prefetch_size': 3,
        'debug_mode': True,
        'debug_limit': 5000,
    })
    print(f"\nBase config:")
    print(f"  Dataset: {base_config.get('data')}")
    print(f"  Network: {base_config.get('network')}")
    print(f"  Problem type: {base_config.get('prob')}")

    all_results = {}

    # Run comparisons
    try:
        all_results['prefetch_comparison'] = compare_prefetch(base_config, output_dir)
    except Exception as e:
        print(f"\nPrefetch comparison failed: {e}")
        all_results['prefetch_comparison'] = {'error': str(e)}

    if not args.quick:
        try:
            all_results['batch_size_comparison'] = compare_batch_sizes(base_config, output_dir)
        except Exception as e:
            print(f"\nBatch size comparison failed: {e}")
            all_results['batch_size_comparison'] = {'error': str(e)}

        try:
            all_results['prefetch_size_comparison'] = compare_prefetch_sizes(base_config, output_dir)
        except Exception as e:
            print(f"\nPrefetch size comparison failed: {e}")
            all_results['prefetch_size_comparison'] = {'error': str(e)}

    # Generate final report
    generate_comparison_report(all_results, output_dir)

    print("\nComparison complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
