"""
GPU Performance and Utilization Tests

This module provides tests to measure and report GPU utilization during training.
Tests are marked with @pytest.mark.gpu and can be run with:
    ./run_tests.sh --gpu

Reports are generated in tests/gpu_reports/ directory.

Usage:
    ./run_tests.sh --gpu              # Run GPU performance tests
    ./run_tests.sh --gpu --verbose    # With verbose output
    pytest tests/test_gpu_performance.py -m gpu -v -s  # Direct pytest
"""

import pytest
import time
import os
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx

# Check GPU availability
JAX_BACKEND = jax.default_backend()
JAX_DEVICES = jax.devices()
HAS_GPU = JAX_BACKEND == 'gpu' or any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in JAX_DEVICES)

# Check if nvidia-smi is available for GPU monitoring
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    NVIDIA_SMI_AVAILABLE = result.returncode == 0
except (subprocess.SubprocessError, FileNotFoundError):
    NVIDIA_SMI_AVAILABLE = False


def get_gpu_utilization():
    """Get current GPU utilization using nvidia-smi.

    Returns:
        List of dicts with utilization, memory_used_mb, memory_total_mb per GPU
    """
    if not NVIDIA_SMI_AVAILABLE:
        return None

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                gpu_stats.append({
                    'utilization': float(parts[0].strip()),
                    'memory_used_mb': float(parts[1].strip()),
                    'memory_total_mb': float(parts[2].strip()),
                })
        return gpu_stats
    except Exception:
        return None


def generate_report(test_name, results, report_dir='tests/gpu_reports'):
    """Generate a JSON report for a performance test."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(report_dir) / f'{test_name}_{timestamp}.json'

    report = {
        'test_name': test_name,
        'timestamp': timestamp,
        'jax_backend': JAX_BACKEND,
        'jax_devices': [str(d) for d in JAX_DEVICES],
        'nvidia_smi_available': NVIDIA_SMI_AVAILABLE,
        'results': results,
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    return report_path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def report_dir():
    """Create and return the report directory."""
    report_path = Path('tests/gpu_reports')
    report_path.mkdir(parents=True, exist_ok=True)
    return report_path


@pytest.fixture
def simple_mlp():
    """Create a simple MLP for testing Hamiltonian computation."""
    import jax.random as jrandom

    class SimpleMLP(eqx.Module):
        layers: list

        def __init__(self, key):
            key1, key2, key3 = jrandom.split(key, 3)
            self.layers = [
                eqx.nn.Linear(784, 256, key=key1),
                eqx.nn.Linear(256, 128, key=key2),
                eqx.nn.Linear(128, 10, key=key3),
            ]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = jax.nn.relu(layer(x))
            return self.layers[-1](x)

    return SimpleMLP(jrandom.PRNGKey(42))


# =============================================================================
# GPU Environment Tests
# =============================================================================

@pytest.mark.gpu
class TestGPUEnvironment:
    """Tests to verify GPU environment setup."""

    def test_jax_version(self):
        """Report JAX version and backend."""
        print(f"\nJAX version: {jax.__version__}")
        print(f"JAX backend: {JAX_BACKEND}")
        print(f"JAX devices: {JAX_DEVICES}")
        assert True  # Always pass, just reporting

    def test_gpu_available(self):
        """Test if GPU is available."""
        print(f"\nGPU Available: {HAS_GPU}")
        print(f"Backend: {JAX_BACKEND}")
        if not HAS_GPU:
            pytest.skip("No GPU available - skipping GPU tests")

    def test_nvidia_smi(self):
        """Test nvidia-smi availability and report GPU stats."""
        if not NVIDIA_SMI_AVAILABLE:
            pytest.skip("nvidia-smi not available")

        stats = get_gpu_utilization()
        print(f"\nGPU Stats:")
        for i, gpu in enumerate(stats or []):
            print(f"  GPU {i}: {gpu['utilization']:.1f}% util, "
                  f"{gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB")

    def test_gpu_compute(self):
        """Test GPU can perform computation."""
        if not HAS_GPU:
            pytest.skip("No GPU available")

        x = jnp.ones((5000, 5000))
        y = x @ x
        y.block_until_ready()

        # JAX 0.6+ uses .device (property) not .device() (method)
        device = y.device if hasattr(y, 'device') and not callable(y.device) else y.devices()[0]
        device_str = str(device)
        print(f"\nComputation performed on: {device_str}")
        assert 'gpu' in device_str.lower() or 'cuda' in device_str.lower()


# =============================================================================
# Hamiltonian JIT Performance Tests
# =============================================================================

@pytest.mark.gpu
class TestHamiltonianJITPerformance:
    """Tests for JIT compilation performance of Hamiltonian functions."""

    def test_jit_compilation_speedup(self, simple_mlp, report_dir):
        """Measure JIT compilation speedup for Hamiltonian computation."""
        from cl.core.hamiltonian import _hamiltonian_core_class_standard

        key = jax.random.PRNGKey(0)
        batch_size = 64
        x = jax.random.normal(key, (batch_size, 784))
        y = jax.random.randint(key, (batch_size,), 0, 10)
        exp_x = jax.random.normal(key, (batch_size, 784))
        exp_y = jax.random.randint(key, (batch_size,), 0, 10)
        deltax = jax.random.normal(key, (batch_size, 784)) * 0.01

        params, static = eqx.partition(simple_mlp, eqx.is_array)

        # First call (includes compilation)
        start = time.time()
        grad, losses = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        compile_time = time.time() - start

        # Subsequent calls (already compiled)
        start = time.time()
        for _ in range(10):
            grad, losses = _hamiltonian_core_class_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        exec_time = (time.time() - start) / 10

        speedup = compile_time / exec_time if exec_time > 0 else float('inf')

        results = {
            'compile_time_seconds': compile_time,
            'execution_time_seconds': exec_time,
            'speedup_after_jit': speedup,
            'batch_size': batch_size,
        }

        print(f"\n{'='*60}")
        print("Hamiltonian JIT Compilation Test")
        print(f"{'='*60}")
        print(f"First call (with compilation): {compile_time:.4f}s")
        print(f"Subsequent calls (average):    {exec_time:.4f}s")
        print(f"Speedup after JIT:              {speedup:.1f}x")
        print(f"{'='*60}")

        generate_report('jit_compilation_speedup', results, str(report_dir))

        assert speedup > 1.5, f"Expected JIT speedup > 1.5x, got {speedup:.1f}x"

    def test_gpu_utilization_during_computation(self, simple_mlp, report_dir):
        """Measure GPU utilization during Hamiltonian computation."""
        if not HAS_GPU:
            pytest.skip("No GPU available")
        if not NVIDIA_SMI_AVAILABLE:
            pytest.skip("nvidia-smi not available for monitoring")

        from cl.core.hamiltonian import _hamiltonian_core_class_standard
        import threading

        key = jax.random.PRNGKey(0)
        batch_size = 256
        x = jax.random.normal(key, (batch_size, 784))
        y = jax.random.randint(key, (batch_size,), 0, 10)
        exp_x = jax.random.normal(key, (batch_size, 784))
        exp_y = jax.random.randint(key, (batch_size,), 0, 10)
        deltax = jax.random.normal(key, (batch_size, 784)) * 0.01

        params, static = eqx.partition(simple_mlp, eqx.is_array)

        # Warm up JIT
        grad, _ = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

        # Sample GPU utilization while running
        gpu_samples = []
        stop_sampling = threading.Event()

        def sample_gpu():
            while not stop_sampling.is_set():
                stats = get_gpu_utilization()
                if stats:
                    gpu_samples.append(stats[0]['utilization'])
                time.sleep(0.1)

        sampler = threading.Thread(target=sample_gpu)
        sampler.start()

        # Run iterations
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            grad, _ = _hamiltonian_core_class_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        elapsed = time.time() - start

        stop_sampling.set()
        sampler.join()

        results = {
            'num_iterations': num_iterations,
            'total_time_seconds': elapsed,
            'time_per_iteration_ms': (elapsed / num_iterations) * 1000,
            'batch_size': batch_size,
            'gpu_utilization_min': min(gpu_samples) if gpu_samples else None,
            'gpu_utilization_max': max(gpu_samples) if gpu_samples else None,
            'gpu_utilization_mean': sum(gpu_samples) / len(gpu_samples) if gpu_samples else None,
            'num_samples': len(gpu_samples),
        }

        print(f"\n{'='*60}")
        print("GPU Utilization During Hamiltonian Computation")
        print(f"{'='*60}")
        print(f"Iterations:         {num_iterations}")
        print(f"Total time:         {elapsed:.2f}s")
        print(f"Time per iteration: {results['time_per_iteration_ms']:.2f}ms")
        print(f"Batch size:         {batch_size}")
        if gpu_samples:
            print(f"GPU Utilization:")
            print(f"  Min:  {results['gpu_utilization_min']:.1f}%")
            print(f"  Max:  {results['gpu_utilization_max']:.1f}%")
            print(f"  Mean: {results['gpu_utilization_mean']:.1f}%")
        print(f"{'='*60}")

        generate_report('gpu_utilization', results, str(report_dir))


@pytest.mark.gpu
class TestAllHamiltonianVariants:
    """Test all Hamiltonian function variants compile and run."""

    def test_mse_standard(self, simple_mlp):
        """Test MSE standard Hamiltonian runs."""
        from cl.core.hamiltonian import _hamiltonian_core_mse_standard

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 784))
        y = jax.random.normal(key, (32,))
        exp_x = jax.random.normal(key, (32, 784))
        exp_y = jax.random.normal(key, (32,))
        deltax = jax.random.normal(key, (32, 784)) * 0.01

        # Need MLP that outputs scalar for MSE
        class MSE_MLP(eqx.Module):
            layers: list
            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.layers = [eqx.nn.Linear(784, 64, key=k1), eqx.nn.Linear(64, 1, key=k2)]
            def __call__(self, x):
                for l in self.layers[:-1]:
                    x = jax.nn.relu(l(x))
                return self.layers[-1](x)

        model = MSE_MLP(key)
        params, static = eqx.partition(model, eqx.is_array)

        grad, losses = _hamiltonian_core_mse_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )

        assert grad is not None
        assert len(losses) == 5  # H, V, dV, dV_dtheta, dV_dx
        print(f"\nMSE Standard: H={float(losses[0]):.4f}, V={float(losses[1]):.4f}")

    def test_class_standard(self, simple_mlp):
        """Test classification standard Hamiltonian runs."""
        from cl.core.hamiltonian import _hamiltonian_core_class_standard

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 784))
        y = jax.random.randint(key, (32,), 0, 10)
        exp_x = jax.random.normal(key, (32, 784))
        exp_y = jax.random.randint(key, (32,), 0, 10)
        deltax = jax.random.normal(key, (32, 784)) * 0.01

        params, static = eqx.partition(simple_mlp, eqx.is_array)

        grad, losses = _hamiltonian_core_class_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )

        assert grad is not None
        assert len(losses) == 5
        print(f"\nClass Standard: H={float(losses[0]):.4f}, V={float(losses[1]):.4f}")


@pytest.mark.gpu
def test_generate_summary(report_dir):
    """Generate summary of all GPU performance reports."""
    report_files = list(Path(report_dir).glob('*.json'))

    if not report_files:
        print(f"\nNo GPU reports found in {report_dir}/")
        print("Run GPU tests on a GPU machine first.")
        return

    print(f"\n{'='*60}")
    print("GPU Performance Summary")
    print(f"{'='*60}")

    for f in sorted(report_files)[-5:]:  # Last 5 reports
        with open(f) as fp:
            report = json.load(fp)
        print(f"\n{report['test_name']} ({report['timestamp']})")
        print(f"  Backend: {report['jax_backend']}")
        results = report.get('results', {})
        if 'gpu_utilization_mean' in results and results['gpu_utilization_mean']:
            print(f"  GPU Util: {results['gpu_utilization_mean']:.1f}%")
        if 'speedup_after_jit' in results:
            print(f"  JIT Speedup: {results['speedup_after_jit']:.1f}x")
