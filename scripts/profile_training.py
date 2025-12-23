#!/usr/bin/env python
"""
Profile training to identify GPU utilization bottlenecks.

Usage:
    python scripts/profile_training.py config/cifar100.json

This script measures time spent in each phase:
1. Data loading (CPU)
2. Data transfer (CPU -> JAX arrays)
3. Hamiltonian computation (GPU)
4. Optimizer update (GPU)
5. Metric computation (GPU)

Run with: python scripts/profile_training.py config/cifar100.json
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def profile_training(config_path: str, num_batches: int = 20):
    """Profile training to identify bottlenecks."""

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Force debug_mode off for realistic profiling
    config['debug_mode'] = False

    print(f"\n{'='*70}")
    print("GPU Training Profiler")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Dataset: {config.get('data', 'unknown')}")
    print(f"Batch size: {config.get('batch_size', 64)}")
    print(f"Profiling {num_batches} batches")
    print(f"{'='*70}\n")

    # Report JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Import after path setup
    from cl.config import load_config
    from cl.datasets import (
        SineDataset, MNISTDataset, PermutedMNISTDataset,
        CIFAR10Dataset, CIFAR100Dataset, SyntheticGraphDataset
    )
    from cl.models.mlp import MLP
    from cl.models.cnn import CNN, CNN3D
    from cl.core.trainer import Trainer
    from cl.core.hamiltonian import _hamiltonian_core_class_standard, _hamiltonian_core_mse_standard

    # Load full config with defaults
    config = load_config(config_path)
    config['debug_mode'] = False  # Ensure full dataset

    # Dataset mapping
    dataset_map = {
        'sine': SineDataset,
        'mnist': MNISTDataset,
        'permuted_mnist': PermutedMNISTDataset,
        'cifar10': CIFAR10Dataset,
        'cifar100': CIFAR100Dataset,
        'synthetic': SyntheticGraphDataset,
    }

    # Initialize dataset
    print("Loading dataset...")
    t0 = time.time()
    data_name = config.get('data', 'mnist')
    if data_name not in dataset_map:
        print(f"Error: Unknown dataset '{data_name}'")
        print(f"Available: {list(dataset_map.keys())}")
        sys.exit(1)
    dataset = dataset_map[data_name](config)
    dataset.load_task(0)
    print(f"Dataset loaded in {time.time() - t0:.2f}s")
    print(f"Training samples: {len(dataset.X_train)}")
    print()

    # Create dataloaders
    batch_size = config.get('batch_size', 64)
    trainloader, exploader = dataset.generate_dataset(0, batch_size, 'training')

    # Model mapping based on network type
    network = config.get('network', 'cnn')
    prob = config.get('prob', 'classification')

    # Initialize model
    print("Initializing model...")
    key = jax.random.PRNGKey(42)

    if network == 'fcnn':
        sizes = config.get('sizes', [784, 256, 128, 10])
        model = MLP(sizes=sizes, key=key)
    elif network == 'cnn':
        feed_sizes = config.get('feed_sizes', [1152, 256, 128, 10])
        filter_size = config.get('filter_size', 5)
        model = CNN(key=key, filter_size=filter_size, feed_sizes=feed_sizes)
    elif network == 'cnn3d':
        # filter_size=3 gives flatten_size=2304 (matches config/cifar100.json)
        # filter_size=4 gives flatten_size=1600
        filter_size = config.get('filter_size', 3)
        channel_in = config.get('channel_in', 3)
        channel_out = config.get('channel_out', 32)
        num_classes = config.get('n_class', 100)
        input_size = config.get('input_size', 32)

        # Compute flatten size: after 2 conv+pool layers
        # Conv1: input_size -> (input_size - filter_size + 1) -> pool /2
        # Conv2: -> (prev - filter_size + 1) -> pool /2
        after_conv1 = (input_size - filter_size + 1) // 2
        after_conv2 = (after_conv1 - filter_size + 1) // 2
        flatten_size = after_conv2 * after_conv2 * (channel_out * 2)

        # Use config feed_sizes if provided, otherwise compute
        feed_sizes = config.get('feed_sizes', None)
        if feed_sizes is None:
            feed_sizes = [flatten_size, 512, 256, num_classes]
        else:
            # Validate feed_sizes[0] matches computed flatten_size
            if feed_sizes[0] != flatten_size:
                print(f"WARNING: feed_sizes[0]={feed_sizes[0]} != computed flatten_size={flatten_size}")
                print(f"  Mismatch! Check filter_size in config.")
                print(f"  filter_size=3 -> flatten=2304, filter_size=4 -> flatten=1600")

        print(f"CNN3D: input={input_size}, filter={filter_size}, flatten_size={flatten_size}")
        print(f"CNN3D: feed_sizes={feed_sizes}")

        model = CNN3D(key=key, filter_size=filter_size, feed_sizes=feed_sizes,
                      input_size=input_size, channel_in=channel_in,
                      channel_out=channel_out, num_classes=num_classes)
    else:
        print(f"Error: Unsupported network type '{network}' for profiling")
        print("Supported: fcnn, cnn, cnn3d")
        sys.exit(1)

    params, static = eqx.partition(model, eqx.is_array)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    print()

    # Select Hamiltonian function based on problem type
    is_regression = (prob == 'regression')
    hamiltonian_fn = _hamiltonian_core_mse_standard if is_regression else _hamiltonian_core_class_standard
    y_dtype = jnp.float64 if is_regression else jnp.int64
    print(f"Problem type: {prob}")
    print(f"Using: {'MSE' if is_regression else 'Classification'} Hamiltonian")
    print()

    # Timing accumulators
    times = {
        'data_loading': [],
        'data_transfer': [],
        'hamiltonian': [],
        'optimizer': [],
        'metric': [],
        'total_batch': [],
    }

    # Warm up JIT (first batch) - must use exact batch_size to avoid recompilation
    print("Warming up JIT compilation...")
    trainiter = iter(trainloader)
    expiter = iter(exploader)

    # Find a batch with exact batch_size (skip partial batches)
    batch = next(trainiter)
    batch_ex = next(expiter)
    while batch[0].shape[0] != batch_size or batch_ex[0].shape[0] != batch_size:
        try:
            batch = next(trainiter)
            batch_ex = next(expiter)
        except StopIteration:
            print(f"ERROR: Could not find batch with size {batch_size}")
            sys.exit(1)

    (x, y) = batch
    (exp_x, exp_y) = batch_ex

    x_jax = jnp.asarray(x.numpy(), dtype=jnp.float64)
    y_jax = jnp.asarray(y.numpy(), dtype=y_dtype)
    exp_x_jax = jnp.asarray(exp_x.numpy(), dtype=jnp.float64)
    exp_y_jax = jnp.asarray(exp_y.numpy(), dtype=y_dtype)
    delta_x = jnp.asarray(np.random.normal(0, 0.01, exp_x_jax.shape))
    print(f"Warmup batch shape: {x_jax.shape}")

    # Initialize optimizer
    import optax
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)

    # JIT warmup call - includes Hamiltonian AND optimizer
    print("Warming up JIT (Hamiltonian + Optimizer)...")
    t0 = time.time()
    grad, losses = hamiltonian_fn(
        params, static, x_jax, y_jax, exp_x_jax, exp_y_jax, delta_x,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(float(param_count)), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    # Also warm up optimizer
    updates, opt_state = optim.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
    jit_time = time.time() - t0
    print(f"JIT compilation time (total): {jit_time:.2f}s")
    print()

    # Profile batches
    print(f"Profiling {num_batches} batches...")
    print("-" * 70)

    # Reinitialize iterators
    trainiter = iter(trainloader)
    expiter = iter(exploader)

    for i in range(num_batches):
        batch_start = time.time()

        # 1. Data loading (get next batch from DataLoader)
        t0 = time.time()
        try:
            batch = next(trainiter)
            batch_ex = next(expiter)
        except StopIteration:
            trainiter = iter(trainloader)
            expiter = iter(exploader)
            batch = next(trainiter)
            batch_ex = next(expiter)
        data_load_time = time.time() - t0
        times['data_loading'].append(data_load_time)

        # 2. Data transfer (PyTorch -> JAX)
        t0 = time.time()
        (x, y) = batch
        (exp_x, exp_y) = batch_ex

        # Skip batches with different sizes to avoid JIT recompilation
        if x.shape[0] != batch_size or exp_x.shape[0] != batch_size:
            continue

        x_jax = jnp.asarray(x.numpy(), dtype=jnp.float64)
        y_jax = jnp.asarray(y.numpy(), dtype=y_dtype)
        exp_x_jax = jnp.asarray(exp_x.numpy(), dtype=jnp.float64)
        exp_y_jax = jnp.asarray(exp_y.numpy(), dtype=y_dtype)
        delta_x = jnp.asarray(np.random.normal(0, 0.01, exp_x_jax.shape))
        # Force transfer to complete
        x_jax.block_until_ready()
        transfer_time = time.time() - t0
        times['data_transfer'].append(transfer_time)

        # 3. Hamiltonian computation
        t0 = time.time()
        grad, losses = hamiltonian_fn(
            params, static, x_jax, y_jax, exp_x_jax, exp_y_jax, delta_x,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(float(param_count)), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        hamiltonian_time = time.time() - t0
        times['hamiltonian'].append(hamiltonian_time)

        # 4. Optimizer update
        t0 = time.time()
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Force update to complete
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
        optimizer_time = time.time() - t0
        times['optimizer'].append(optimizer_time)

        # 5. Metric computation (simplified)
        t0 = time.time()
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(x_jax)
        pred.block_until_ready()
        metric_time = time.time() - t0
        times['metric'].append(metric_time)

        total_batch_time = time.time() - batch_start
        times['total_batch'].append(total_batch_time)

        if i < 5 or i == num_batches - 1:
            print(f"Batch {i+1:3d}: load={data_load_time*1000:6.1f}ms, "
                  f"transfer={transfer_time*1000:6.1f}ms, "
                  f"hamiltonian={hamiltonian_time*1000:6.1f}ms, "
                  f"optim={optimizer_time*1000:6.1f}ms, "
                  f"metric={metric_time*1000:6.1f}ms, "
                  f"total={total_batch_time*1000:6.1f}ms")
        elif i == 5:
            print("...")

    # Summary
    print()
    print("=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    # Calculate averages (skip first batch as it may include extra overhead)
    skip = 1
    for key in times:
        avg = np.mean(times[key][skip:]) * 1000  # ms
        std = np.std(times[key][skip:]) * 1000
        pct = (np.mean(times[key][skip:]) / np.mean(times['total_batch'][skip:])) * 100
        print(f"{key:20s}: {avg:8.2f} ms (±{std:5.2f}) - {pct:5.1f}%")

    print()
    print("-" * 70)
    total_avg = np.mean(times['total_batch'][skip:])
    batches_per_epoch = len(dataset.X_train) // batch_size
    epoch_time = total_avg * batches_per_epoch

    print(f"Average batch time:     {total_avg*1000:.2f} ms")
    print(f"Batches per epoch:      {batches_per_epoch}")
    print(f"Estimated epoch time:   {epoch_time:.2f} s")
    print()

    # Identify bottleneck
    avg_times = {k: np.mean(v[skip:]) for k, v in times.items() if k != 'total_batch'}
    bottleneck = max(avg_times, key=avg_times.get)
    bottleneck_pct = (avg_times[bottleneck] / np.mean(times['total_batch'][skip:])) * 100

    print(f"BOTTLENECK: {bottleneck} ({bottleneck_pct:.1f}% of batch time)")
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 70)
    if bottleneck == 'data_loading':
        print("- Data loading is the bottleneck")
        print("- Cannot use num_workers>0 with JAX (fork incompatibility)")
        print("- Consider: preloading entire dataset to GPU memory")
        print("- Consider: using JAX-native data loading (grain, tf.data)")
    elif bottleneck == 'data_transfer':
        print("- CPU->GPU transfer is the bottleneck")
        print("- Consider: preloading batches to GPU")
        print("- Consider: using float32 instead of float64")
        print("- Consider: larger batch sizes to amortize transfer overhead")
    elif bottleneck == 'hamiltonian':
        print("- Hamiltonian computation is the bottleneck (GOOD!)")
        print("- This means GPU is doing useful work")
        print("- GPU utilization should be high during this phase")
        print("- Consider: larger batch sizes if GPU memory allows")
    elif bottleneck == 'optimizer':
        print("- Optimizer update is the bottleneck")
        print("- This is unusual - check for synchronization issues")

    print("=" * 70)

    return times


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_training.py <config.json> [num_batches]")
        print("Example: python scripts/profile_training.py config/cifar100.json 50")
        sys.exit(1)

    config_path = sys.argv[1]
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    profile_training(config_path, num_batches)
