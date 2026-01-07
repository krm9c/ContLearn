"""AWB Performance Baseline Tests.

Records timing metrics for AWB forward/backward passes before optimization.
Run this to establish baseline, then compare after implementing einsum changes.

Usage:
    pytest tests/test_awb_performance.py -v -s
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

from src.cl.models.mlp import MLP
from src.cl.models.cnn import CNN, CNN3D
from src.cl.models.gcn import GCN


# Output file for baseline metrics
BASELINE_METRICS_FILE = Path(__file__).parent / "awb_baseline_metrics.json"


def profile_function(fn, warmup=3, runs=10, name=""):
    """Profile a function with warmup runs.

    Returns dict with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        result = fn()
        # Block until computation completes
        for leaf in jax.tree_util.tree_leaves(result):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        for leaf in jax.tree_util.tree_leaves(result):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()
        times.append(time.perf_counter() - start)

    return {
        "name": name,
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        "runs": runs
    }


class TestAWBPerformanceBaseline:
    """Performance baseline tests - record timing before optimization."""

    @pytest.fixture
    def jax_key(self):
        return jax.random.PRNGKey(42)

    # =========================================================================
    # MLP Performance
    # =========================================================================

    def test_mlp_awb_timing(self, jax_key):
        """Record timing for MLP AWB forward/backward."""
        original_sizes = [784, 256, 128, 10]
        new_sizes = [784, 320, 160, 10]

        model = MLP(sizes=original_sizes, key=jax_key, awb_enabled=True)
        model = model.with_new_AB_matrices(original_sizes, new_sizes, seed=42)

        x = jax.random.normal(jax_key, (784,))
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(m):
            pred = m.getAWB(x)
            return optax.softmax_cross_entropy_with_integer_labels(pred, target)

        # Forward timing
        forward_stats = profile_function(
            lambda: model.getAWB(x),
            name="MLP AWB forward"
        )

        # Gradient timing
        grad_stats = profile_function(
            lambda: eqx.filter_grad(loss_fn)(model),
            name="MLP AWB gradient"
        )

        print(f"\n{'='*60}")
        print(f"MLP AWB Performance (arch: {original_sizes} -> {new_sizes})")
        print(f"{'='*60}")
        print(f"Forward:  {forward_stats['mean_ms']:.3f}ms ± {forward_stats['std_ms']:.3f}ms")
        print(f"Gradient: {grad_stats['mean_ms']:.3f}ms ± {grad_stats['std_ms']:.3f}ms")

        return {"forward": forward_stats, "gradient": grad_stats}

    # =========================================================================
    # CNN Performance
    # =========================================================================

    def test_cnn_awb_timing(self, jax_key):
        """Record timing for CNN (MNIST) AWB forward/backward."""
        feed_sizes = [432, 64, 10]
        awb_arch = [432, 64, 10]  # Identity to avoid dimension issues

        model = CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=feed_sizes,
            input_size=28,
            channel_in=1,
            channel_out=3,
            awb_arch=awb_arch,
            awb_filter_size=4
        )

        x = jax.random.normal(jax_key, (1, 28, 28))
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(m):
            pred = m.get_AWBT(x)
            return optax.softmax_cross_entropy_with_integer_labels(pred, target)

        forward_stats = profile_function(
            lambda: model.get_AWBT(x),
            name="CNN AWB forward"
        )

        grad_stats = profile_function(
            lambda: eqx.filter_grad(loss_fn)(model),
            name="CNN AWB gradient"
        )

        print(f"\n{'='*60}")
        print(f"CNN AWB Performance (MNIST, feed: {feed_sizes})")
        print(f"{'='*60}")
        print(f"Forward:  {forward_stats['mean_ms']:.3f}ms ± {forward_stats['std_ms']:.3f}ms")
        print(f"Gradient: {grad_stats['mean_ms']:.3f}ms ± {grad_stats['std_ms']:.3f}ms")

        return {"forward": forward_stats, "gradient": grad_stats}

    # =========================================================================
    # CNN3D Performance (CIFAR-10)
    # =========================================================================

    def test_cnn3d_cifar10_awb_timing(self, jax_key):
        """Record timing for CNN3D (CIFAR-10) AWB forward/backward."""
        feed_sizes = [2304, 256, 10]

        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32,
            num_classes=10
        )

        x = jax.random.normal(jax_key, (3, 32, 32))
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(m):
            pred = m.get_AWBT(x)
            return optax.softmax_cross_entropy_with_integer_labels(pred, target)

        forward_stats = profile_function(
            lambda: model.get_AWBT(x),
            name="CNN3D CIFAR-10 AWB forward"
        )

        grad_stats = profile_function(
            lambda: eqx.filter_grad(loss_fn)(model),
            name="CNN3D CIFAR-10 AWB gradient"
        )

        print(f"\n{'='*60}")
        print(f"CNN3D AWB Performance (CIFAR-10, feed: {feed_sizes})")
        print(f"{'='*60}")
        print(f"Forward:  {forward_stats['mean_ms']:.3f}ms ± {forward_stats['std_ms']:.3f}ms")
        print(f"Gradient: {grad_stats['mean_ms']:.3f}ms ± {grad_stats['std_ms']:.3f}ms")

        return {"forward": forward_stats, "gradient": grad_stats}

    # =========================================================================
    # CNN3D Performance (CIFAR-100)
    # =========================================================================

    def test_cnn3d_cifar100_awb_timing(self, jax_key):
        """Record timing for CNN3D (CIFAR-100) AWB forward/backward."""
        feed_sizes = [2304, 256, 100]

        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32,
            num_classes=100
        )

        x = jax.random.normal(jax_key, (3, 32, 32))
        target = jax.random.randint(jax_key, (), 0, 100)

        def loss_fn(m):
            pred = m.get_AWBT(x)
            return optax.softmax_cross_entropy_with_integer_labels(pred, target)

        forward_stats = profile_function(
            lambda: model.get_AWBT(x),
            name="CNN3D CIFAR-100 AWB forward"
        )

        grad_stats = profile_function(
            lambda: eqx.filter_grad(loss_fn)(model),
            name="CNN3D CIFAR-100 AWB gradient"
        )

        print(f"\n{'='*60}")
        print(f"CNN3D AWB Performance (CIFAR-100, feed: {feed_sizes})")
        print(f"{'='*60}")
        print(f"Forward:  {forward_stats['mean_ms']:.3f}ms ± {forward_stats['std_ms']:.3f}ms")
        print(f"Gradient: {grad_stats['mean_ms']:.3f}ms ± {grad_stats['std_ms']:.3f}ms")

        return {"forward": forward_stats, "gradient": grad_stats}

    # =========================================================================
    # GCN Performance
    # =========================================================================

    def test_gcn_awb_timing(self, jax_key):
        """Record timing for GCN AWB forward/backward."""
        model = GCN(
            in_size=5,
            gcn_sizes=[5, 64],
            feed_sizes=[64, 32, 10],
            node_num=20,
            out_size=10,
            SEED=42
        )

        num_nodes = 20
        x = jax.random.normal(jax_key, (num_nodes, 5))
        adj = jax.random.uniform(jax_key, (num_nodes, num_nodes))
        adj = (adj > 0.7).astype(jnp.float32)
        adj = adj + adj.T
        adj = jnp.clip(adj, 0, 1)
        adj = adj + jnp.eye(num_nodes)
        deg = jnp.sum(adj, axis=1, keepdims=True)
        adj = adj / jnp.sqrt(deg) / jnp.sqrt(deg.T)
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        n_nodes = jnp.array([num_nodes])
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(m):
            pred = m.get_AWBT(x, adj, batch, n_nodes)
            return optax.softmax_cross_entropy_with_integer_labels(pred, target)

        forward_stats = profile_function(
            lambda: model.get_AWBT(x, adj, batch, n_nodes),
            name="GCN AWB forward"
        )

        grad_stats = profile_function(
            lambda: eqx.filter_grad(loss_fn)(model),
            name="GCN AWB gradient"
        )

        print(f"\n{'='*60}")
        print(f"GCN AWB Performance (nodes: {num_nodes})")
        print(f"{'='*60}")
        print(f"Forward:  {forward_stats['mean_ms']:.3f}ms ± {forward_stats['std_ms']:.3f}ms")
        print(f"Gradient: {grad_stats['mean_ms']:.3f}ms ± {grad_stats['std_ms']:.3f}ms")

        return {"forward": forward_stats, "gradient": grad_stats}


def save_baseline_metrics():
    """Run all performance tests and save results to JSON."""
    jax_key = jax.random.PRNGKey(42)
    tester = TestAWBPerformanceBaseline()

    # Set up fixture
    class FakeRequest:
        pass

    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jax_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
        "results": {}
    }

    # Run each test
    print("\nCollecting AWB Performance Baseline Metrics...")
    print("=" * 70)

    metrics["results"]["mlp"] = tester.test_mlp_awb_timing(jax_key)
    metrics["results"]["cnn_mnist"] = tester.test_cnn_awb_timing(jax_key)
    metrics["results"]["cnn3d_cifar10"] = tester.test_cnn3d_cifar10_awb_timing(jax_key)
    metrics["results"]["cnn3d_cifar100"] = tester.test_cnn3d_cifar100_awb_timing(jax_key)
    metrics["results"]["gcn"] = tester.test_gcn_awb_timing(jax_key)

    # Save to file
    with open(BASELINE_METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Baseline metrics saved to: {BASELINE_METRICS_FILE}")
    print(f"{'='*70}")

    return metrics


if __name__ == "__main__":
    save_baseline_metrics()
