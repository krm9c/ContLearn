#!/usr/bin/env python
"""
AWB Performance Benchmarking

Comprehensive performance benchmarks for the AWB pipeline on CIFAR-100.
Measures:
1. JIT compilation time for each step
2. Training throughput (samples/second)
3. Architecture search overhead
4. Memory usage
5. GPU utilization (if available)

Usage:
    python awb_tests/benchmark_performance.py
    python awb_tests/benchmark_performance.py --config awb_tests/configs/awb_test_cifar100.json
    python awb_tests/benchmark_performance.py --output benchmark_results.json
"""

import argparse
import json
import time
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from cl.models.mlp import MLP
from cl.models.cnn import CNN3D
from cl.core.trainer import Trainer
from cl.core.awb import (
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
)
from cl.core.arch_search import search_architecture_grid, search_architecture_bayesian
from cl.datasets.sine import SineDataset
from cl.datasets.cifar import CIFAR100Dataset


def benchmark_jit_compilation(model_type='mlp', verbose=False):
    """Benchmark JIT compilation time for different operations."""
    print("\n" + "=" * 60)
    print("JIT COMPILATION BENCHMARKS")
    print("=" * 60)

    results = {}

    if model_type == 'mlp':
        arch = [10, 128, 128, 5]
        model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
        test_input = jax.random.normal(jax.random.PRNGKey(0), (32, arch[0]))
    else:
        arch = [2304, 512, 256, 100]
        model = CNN3D(
            filter_size=3,
            channel_out=32,
            feed_sizes=arch,
            key=jax.random.PRNGKey(42),
        )
        test_input = jax.random.normal(jax.random.PRNGKey(0), (32, 32, 32, 3))

    # Benchmark 1: Forward pass JIT
    print("\n1. Forward Pass JIT Compilation")

    @jax.jit
    def forward(model, x):
        return jax.vmap(model)(x)

    start = time.time()
    _ = forward(model, test_input).block_until_ready()
    jit_time = time.time() - start
    results['forward_jit'] = jit_time
    print(f"   First call (includes JIT): {jit_time*1000:.2f}ms")

    start = time.time()
    for _ in range(10):
        _ = forward(model, test_input).block_until_ready()
    cached_time = (time.time() - start) / 10
    results['forward_cached'] = cached_time
    print(f"   Cached call avg: {cached_time*1000:.2f}ms")

    # Benchmark 2: Gradient computation JIT
    print("\n2. Gradient Computation JIT")

    def loss_fn(params, static, x, y):
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y) ** 2)

    params, static = eqx.partition(model, eqx.is_array)
    target = jax.random.normal(jax.random.PRNGKey(1), (32, arch[-1] if model_type == 'mlp' else 100))

    grad_fn = jax.jit(jax.grad(loss_fn))

    start = time.time()
    _ = grad_fn(params, static, test_input, target)
    jax.block_until_ready(_)
    jit_time = time.time() - start
    results['grad_jit'] = jit_time
    print(f"   First call (includes JIT): {jit_time*1000:.2f}ms")

    start = time.time()
    for _ in range(10):
        _ = grad_fn(params, static, test_input, target)
        jax.block_until_ready(_)
    cached_time = (time.time() - start) / 10
    results['grad_cached'] = cached_time
    print(f"   Cached call avg: {cached_time*1000:.2f}ms")

    # Benchmark 3: A/B Matrix operations (MLP only)
    if model_type == 'mlp':
        print("\n3. A/B Matrix Operations")

        new_arch = [10, 192, 192, 5]

        start = time.time()
        model_with_ab = set_new_AB_matrices(model, arch, new_arch)
        ab_init_time = time.time() - start
        results['ab_init'] = ab_init_time
        print(f"   A/B initialization: {ab_init_time*1000:.2f}ms")

        start = time.time()
        for _ in range(100):
            _ = compute_V_from_AWB(model_with_ab)
        v_transform_time = (time.time() - start) / 100
        results['v_transform'] = v_transform_time
        print(f"   V transformation avg: {v_transform_time*1000:.3f}ms")

    return results


def benchmark_training_throughput(config, verbose=False):
    """Benchmark training throughput."""
    print("\n" + "=" * 60)
    print("TRAINING THROUGHPUT BENCHMARKS")
    print("=" * 60)

    results = {}

    # Use MLP for fast testing
    arch = [10, 128, 128, 5]
    model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)

    # Create dataset
    sine_config = {
        'batch_size': config.get('batch_size', 64),
        'n_task': 2,
        'debug_mode': True,
        'debug_limit': 1000,
    }
    dataset = SineDataset(sine_config)
    dl_curr, dl_exp = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')

    trainer = Trainer(loss='regression', metric='mse', problem='vectors')
    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adamw(learning_rate=1e-3)
    opt_state = optim.init(params)

    # Warmup
    print("\n1. Standard Training Throughput")
    train_config = {
        'batch_size': 64,
        'problem': 'vectors',
        'data_id': 'sine',
        'flag': [1.0, 1.0],
        'network': 'fcnn',
    }
    record_dict = trainer.initialize_record_dict(config, run_id=0)

    # Warmup JIT
    train_data = (dl_curr, dl_exp, (dl_curr, dl_exp), (dl_curr, dl_exp))
    params, static, opt_state, record_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=2,
        save_iter=1,
        task_id=0,
        config=train_config,
        record_dict=record_dict,
        problem_type='vectors',
        loss_type='regression',
    )

    # Benchmark
    n_epochs = 10
    total_samples = len(list(dl_curr)) * 64 * n_epochs

    start = time.time()
    params, static, opt_state, record_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=n_epochs,
        save_iter=5,
        task_id=0,
        config=train_config,
        record_dict=record_dict,
        problem_type='vectors',
        loss_type='regression',
    )
    elapsed = time.time() - start

    throughput = total_samples / elapsed
    results['standard_throughput'] = throughput
    results['standard_time_per_epoch'] = elapsed / n_epochs
    print(f"   Throughput: {throughput:.0f} samples/sec")
    print(f"   Time per epoch: {elapsed/n_epochs*1000:.2f}ms")

    # Benchmark A/B training
    print("\n2. A/B Training Throughput")

    new_arch = [10, 192, 192, 5]
    model = eqx.combine(params, static)
    model_with_ab = set_new_AB_matrices(model, arch, new_arch)
    params, static = partition_for_AB_training(model_with_ab)
    opt_state = optim.init(params)

    start = time.time()
    params, static, opt_state, record_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=n_epochs,
        save_iter=5,
        task_id=0,
        config=train_config,
        record_dict=record_dict,
        problem_type='vectors',
        loss_type='regression',
        phase='ab_training',
        notABTrain=False,
    )
    elapsed = time.time() - start

    throughput = total_samples / elapsed
    results['ab_throughput'] = throughput
    results['ab_time_per_epoch'] = elapsed / n_epochs
    print(f"   Throughput: {throughput:.0f} samples/sec")
    print(f"   Time per epoch: {elapsed/n_epochs*1000:.2f}ms")

    return results


def benchmark_architecture_search(config, verbose=False):
    """Benchmark architecture search methods."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE SEARCH BENCHMARKS")
    print("=" * 60)

    results = {}

    # Setup
    arch = [10, 128, 128, 5]
    model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)

    sine_config = {
        'batch_size': 64,
        'n_task': 2,
        'debug_mode': True,
        'debug_limit': 200,
    }
    dataset = SineDataset(sine_config)
    dl_curr, dl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='training')
    tl_curr, tl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='testing')

    search_config = config.copy()
    search_config['prob'] = 'regression'
    search_config['loss'] = 'mse'
    search_config['metric'] = 'mse'
    search_config['arch_search_max_iter'] = 2
    search_config['arch_search_epochs'] = 3
    search_config['arch_search_range'] = 1
    search_config['arch_search_mlp_increment'] = 32

    # Grid search benchmark
    print("\n1. Grid Search")
    gc.collect()

    start = time.time()
    grid_arch = search_architecture_grid(
        model=model,
        baseline_arch=arch,
        task_id=1,
        baseline_loss=0.5,
        dataloader_curr=dl_curr,
        dataloader_exp=dl_exp,
        test_loader_curr=tl_curr,
        test_loader_exp=tl_exp,
        config=search_config,
        trainer=None,
        model_type='mlp',
    )
    grid_time = time.time() - start

    results['grid_time'] = grid_time
    results['grid_arch'] = grid_arch
    print(f"   Time: {grid_time:.2f}s")
    print(f"   Result: {grid_arch}")

    # Bayesian search benchmark
    print("\n2. Bayesian Search (Optuna)")
    gc.collect()

    search_config['arch_search_bo_trials'] = 3

    try:
        start = time.time()
        bayesian_arch = search_architecture_bayesian(
            model=model,
            baseline_arch=arch,
            task_id=1,
            baseline_loss=0.5,
            dataloader_curr=dl_curr,
            dataloader_exp=dl_exp,
            test_loader_curr=tl_curr,
            test_loader_exp=tl_exp,
            config=search_config,
            trainer=None,
            model_type='mlp',
        )
        bayesian_time = time.time() - start

        results['bayesian_time'] = bayesian_time
        results['bayesian_arch'] = bayesian_arch
        print(f"   Time: {bayesian_time:.2f}s")
        print(f"   Result: {bayesian_arch}")

        # Comparison
        if 'grid_time' in results and 'bayesian_time' in results:
            speedup = results['grid_time'] / results['bayesian_time']
            results['speedup'] = speedup
            print(f"\n   Speedup: {speedup:.2f}x")

    except ImportError:
        print("   [SKIPPED] Optuna not installed")
        results['bayesian_time'] = None

    return results


def benchmark_memory_usage(verbose=False):
    """Benchmark memory usage for different model sizes."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE BENCHMARKS")
    print("=" * 60)

    results = {}

    # Test different architectures
    architectures = [
        ('small', [10, 64, 64, 5]),
        ('medium', [10, 256, 256, 5]),
        ('large', [10, 512, 512, 5]),
        ('xlarge', [10, 1024, 1024, 5]),
    ]

    for name, arch in architectures:
        model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)

        # Count parameters
        param_count = sum(
            np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(model)
            if hasattr(leaf, 'shape')
        )

        results[name] = {
            'arch': arch,
            'params': param_count,
            'params_mb': param_count * 4 / (1024 * 1024),  # float32
        }

        print(f"\n{name.upper()}: {arch}")
        print(f"   Parameters: {param_count:,}")
        print(f"   Memory (float32): {results[name]['params_mb']:.2f} MB")

        # Test with A/B matrices
        new_arch = [arch[0]] + [int(h * 1.5) for h in arch[1:-1]] + [arch[-1]]
        model_with_ab = set_new_AB_matrices(model, arch, new_arch)

        ab_param_count = sum(
            np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(model_with_ab)
            if hasattr(leaf, 'shape')
        )

        results[name]['with_ab_params'] = ab_param_count
        results[name]['with_ab_params_mb'] = ab_param_count * 4 / (1024 * 1024)
        results[name]['ab_overhead'] = (ab_param_count - param_count) / param_count * 100

        print(f"   With A/B matrices: {ab_param_count:,} ({results[name]['with_ab_params_mb']:.2f} MB)")
        print(f"   A/B overhead: {results[name]['ab_overhead']:.1f}%")

    return results


def run_all_benchmarks(config_path=None, output_path=None, verbose=False):
    """Run all benchmarks and compile results."""

    print("=" * 70)
    print("AWB PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'batch_size': 64,
            'debug_mode': True,
            'debug_limit': 200,
        }

    all_results = {
        'jax_version': jax.__version__,
        'devices': str(jax.devices()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Run benchmarks
    all_results['jit_compilation'] = benchmark_jit_compilation('mlp', verbose)
    all_results['training_throughput'] = benchmark_training_throughput(config, verbose)
    all_results['architecture_search'] = benchmark_architecture_search(config, verbose)
    all_results['memory_usage'] = benchmark_memory_usage(verbose)

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\nJIT Compilation:")
    print(f"   Forward pass: {all_results['jit_compilation']['forward_jit']*1000:.2f}ms (first) / "
          f"{all_results['jit_compilation']['forward_cached']*1000:.2f}ms (cached)")
    print(f"   Gradient: {all_results['jit_compilation']['grad_jit']*1000:.2f}ms (first) / "
          f"{all_results['jit_compilation']['grad_cached']*1000:.2f}ms (cached)")

    print("\nTraining Throughput:")
    print(f"   Standard: {all_results['training_throughput']['standard_throughput']:.0f} samples/sec")
    print(f"   A/B training: {all_results['training_throughput']['ab_throughput']:.0f} samples/sec")

    if all_results['architecture_search'].get('bayesian_time'):
        print("\nArchitecture Search:")
        print(f"   Grid: {all_results['architecture_search']['grid_time']:.2f}s")
        print(f"   Bayesian: {all_results['architecture_search']['bayesian_time']:.2f}s")
        print(f"   Speedup: {all_results['architecture_search'].get('speedup', 'N/A')}x")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWB Performance Benchmarks')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = run_all_benchmarks(args.config, args.output, args.verbose)
