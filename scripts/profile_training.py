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
    from cl.config.loader import load_and_merge_config
    from cl.datasets import get_dataset
    from cl.models import get_model
    from cl.core.trainer import Trainer
    from cl.core.hamiltonian import _hamiltonian_core_class_standard

    # Load full config with defaults
    config = load_and_merge_config(config_path)
    config['debug_mode'] = False  # Ensure full dataset

    # Initialize dataset
    print("Loading dataset...")
    t0 = time.time()
    dataset = get_dataset(config)
    dataset.load_task(0)
    print(f"Dataset loaded in {time.time() - t0:.2f}s")
    print(f"Training samples: {len(dataset.X_train)}")
    print()

    # Create dataloaders
    batch_size = config.get('batch_size', 64)
    trainloader, exploader = dataset.generate_dataset(0, batch_size, 'training')

    # Initialize model
    print("Initializing model...")
    model = get_model(config)
    params, static = eqx.partition(model, eqx.is_array)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
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

    # Warm up JIT (first batch)
    print("Warming up JIT compilation...")
    trainiter = iter(trainloader)
    expiter = iter(exploader)
    batch = next(trainiter)
    batch_ex = next(expiter)

    (x, y) = batch
    (exp_x, exp_y) = batch_ex
    min_batch = min(exp_x.shape[0], x.shape[0])

    x_jax = jnp.asarray(x.numpy()[:min_batch], dtype=jnp.float64)
    y_jax = jnp.asarray(y.numpy()[:min_batch], dtype=jnp.int64)
    exp_x_jax = jnp.asarray(exp_x.numpy()[:min_batch], dtype=jnp.float64)
    exp_y_jax = jnp.asarray(exp_y.numpy()[:min_batch], dtype=jnp.int64)
    delta_x = jnp.asarray(np.random.normal(0, 0.01, exp_x_jax.shape))

    # JIT warmup call
    t0 = time.time()
    grad, losses = _hamiltonian_core_class_standard(
        params, static, x_jax, y_jax, exp_x_jax, exp_y_jax, delta_x,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(float(param_count)), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    jit_time = time.time() - t0
    print(f"JIT compilation time: {jit_time:.2f}s")
    print()

    # Profile batches
    print(f"Profiling {num_batches} batches...")
    print("-" * 70)

    # Reinitialize iterators
    trainiter = iter(trainloader)
    expiter = iter(exploader)

    # Initialize optimizer for realistic profiling
    import optax
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)

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
        min_batch = min(exp_x.shape[0], x.shape[0])

        x_jax = jnp.asarray(x.numpy()[:min_batch], dtype=jnp.float64)
        y_jax = jnp.asarray(y.numpy()[:min_batch], dtype=jnp.int64)
        exp_x_jax = jnp.asarray(exp_x.numpy()[:min_batch], dtype=jnp.float64)
        exp_y_jax = jnp.asarray(exp_y.numpy()[:min_batch], dtype=jnp.int64)
        delta_x = jnp.asarray(np.random.normal(0, 0.01, exp_x_jax.shape))
        # Force transfer to complete
        x_jax.block_until_ready()
        transfer_time = time.time() - t0
        times['data_transfer'].append(transfer_time)

        # 3. Hamiltonian computation
        t0 = time.time()
        grad, losses = _hamiltonian_core_class_standard(
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
