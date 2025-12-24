"""
AWB Training Loop GPU Profiling Tests

This module profiles the AWB (Adaptive Weight Basis) training loop phases
to identify GPU bottlenecks and optimization opportunities.

Tests are marked with @pytest.mark.gpu and can be run with:
    ./run_tests.sh --gpu
    pytest tests/test_awb_gpu_profiling.py -m gpu -v -s

Reports are generated in tests/gpu_reports/ directory.

AWB Pipeline Phases Profiled:
    1. Standard CL Training (Task 0)
    2. Preliminary Training (Step 1)
    3. Architecture Search (Step 3a)
    4. A/B Matrix Training (Step 3b)
    5. V Transformation Computation (Step 4)
    6. Final V Training (Step 5)
"""

import pytest
import time
import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

# Check GPU availability
JAX_BACKEND = jax.default_backend()
JAX_DEVICES = jax.devices()
HAS_GPU = JAX_BACKEND == 'gpu' or any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in JAX_DEVICES)

# Check nvidia-smi availability
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    NVIDIA_SMI_AVAILABLE = result.returncode == 0
except (subprocess.SubprocessError, FileNotFoundError):
    NVIDIA_SMI_AVAILABLE = False


# =============================================================================
# Profiling Utilities
# =============================================================================

@dataclass
class PhaseProfile:
    """Profile data for a single AWB phase."""
    phase_name: str
    total_time_seconds: float = 0.0
    num_iterations: int = 0
    time_per_iteration_ms: float = 0.0
    gpu_utilization_samples: List[float] = field(default_factory=list)
    gpu_memory_samples: List[float] = field(default_factory=list)
    jit_compile_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    # Breakdown of time spent in each operation
    hamiltonian_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    data_transfer_time_ms: float = 0.0

    @property
    def gpu_util_mean(self) -> float:
        return np.mean(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0.0

    @property
    def gpu_util_max(self) -> float:
        return max(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0.0

    @property
    def gpu_memory_mean(self) -> float:
        return np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0.0


def get_gpu_stats():
    """Get current GPU utilization and memory using nvidia-smi."""
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


class GPUSampler:
    """Background thread for sampling GPU stats during computation."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.utilization_samples = []
        self.memory_samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            stats = get_gpu_stats()
            if stats:
                self.utilization_samples.append(stats[0]['utilization'])
                self.memory_samples.append(stats[0]['memory_used_mb'])
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def get_samples(self) -> Tuple[List[float], List[float]]:
        return self.utilization_samples.copy(), self.memory_samples.copy()


def generate_awb_report(test_name: str, phases: List[PhaseProfile],
                        config: Dict[str, Any], report_dir: str = 'tests/gpu_reports'):
    """Generate detailed JSON report for AWB profiling."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(report_dir) / f'{test_name}_{timestamp}.json'

    # Compute bottleneck analysis
    total_time = sum(p.total_time_seconds for p in phases)
    bottlenecks = []
    for p in sorted(phases, key=lambda x: x.total_time_seconds, reverse=True):
        pct = (p.total_time_seconds / total_time * 100) if total_time > 0 else 0
        bottlenecks.append({
            'phase': p.phase_name,
            'time_seconds': p.total_time_seconds,
            'percentage': pct,
            'gpu_util_mean': p.gpu_util_mean,
            'is_bottleneck': pct > 25,  # Flag phases taking >25% of time
        })

    report = {
        'test_name': test_name,
        'timestamp': timestamp,
        'jax_backend': JAX_BACKEND,
        'jax_devices': [str(d) for d in JAX_DEVICES],
        'nvidia_smi_available': NVIDIA_SMI_AVAILABLE,
        'config': {k: v for k, v in config.items() if not callable(v)},
        'total_time_seconds': total_time,
        'phases': [
            {
                'name': p.phase_name,
                'total_time_seconds': p.total_time_seconds,
                'num_iterations': p.num_iterations,
                'time_per_iteration_ms': p.time_per_iteration_ms,
                'jit_compile_time_seconds': p.jit_compile_time_seconds,
                'execution_time_seconds': p.execution_time_seconds,
                'gpu_utilization_mean': p.gpu_util_mean,
                'gpu_utilization_max': p.gpu_util_max,
                'gpu_memory_mean_mb': p.gpu_memory_mean,
                'hamiltonian_time_ms': p.hamiltonian_time_ms,
                'optimizer_time_ms': p.optimizer_time_ms,
                'data_transfer_time_ms': p.data_transfer_time_ms,
            }
            for p in phases
        ],
        'bottleneck_analysis': bottlenecks,
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    return report_path, report


def print_bottleneck_report(phases: List[PhaseProfile], title: str = "AWB Pipeline Bottleneck Analysis"):
    """Print formatted bottleneck analysis to console."""
    total_time = sum(p.total_time_seconds for p in phases)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"Total Pipeline Time: {total_time:.2f}s")
    print(f"\n{'Phase':<25} {'Time (s)':<12} {'%':<8} {'GPU Util':<10} {'Iter/ms':<10}")
    print('-' * 70)

    for p in sorted(phases, key=lambda x: x.total_time_seconds, reverse=True):
        pct = (p.total_time_seconds / total_time * 100) if total_time > 0 else 0
        bottleneck_marker = "***" if pct > 25 else ""
        print(f"{p.phase_name:<25} {p.total_time_seconds:<12.3f} {pct:<8.1f} "
              f"{p.gpu_util_mean:<10.1f} {p.time_per_iteration_ms:<10.2f} {bottleneck_marker}")

    print('-' * 70)
    print("\n*** = Bottleneck (>25% of total time)")

    # Print optimization suggestions
    print(f"\n{'='*70}")
    print("Optimization Suggestions:")
    print(f"{'='*70}")

    for p in phases:
        pct = (p.total_time_seconds / total_time * 100) if total_time > 0 else 0
        if pct > 25:
            print(f"\n[{p.phase_name}] ({pct:.1f}% of time)")
            if p.gpu_util_mean < 50:
                print(f"  - Low GPU utilization ({p.gpu_util_mean:.1f}%): Consider larger batch sizes")
            if p.jit_compile_time_seconds > p.execution_time_seconds * 0.5:
                print(f"  - High JIT overhead: Consider caching compiled functions")
            if p.data_transfer_time_ms > p.hamiltonian_time_ms * 0.3:
                print(f"  - Data transfer bottleneck: Consider async data loading")


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
def awb_mlp_model():
    """Create an AWB-enabled MLP for profiling."""
    from cl.models.mlp import MLP

    # Architecture similar to sine regression config
    sizes = [1, 75, 75, 75, 1]
    key = jax.random.PRNGKey(42)
    return MLP(sizes=sizes, key=key, awb_enabled=True)


@pytest.fixture
def awb_cnn_model():
    """Create an AWB-enabled CNN3D for CIFAR profiling."""
    from cl.models.cnn import CNN3D

    filter_size = 3
    feed_sizes = [2304, 512, 256, 100]

    return CNN3D(
        key=jax.random.PRNGKey(42),
        filter_size=filter_size,
        feed_sizes=feed_sizes,
        input_size=32,
        channel_in=3,
        channel_out=32,
        num_classes=100,
        awb_enabled=True
    )


@pytest.fixture
def profiling_config():
    """Configuration for profiling tests."""
    return {
        'n_task': 3,
        'epochs_per_task': 20,
        'batch_size': 64,
        'lr': 1e-4,
        'save_iter': 5,
        'awb_enabled': True,
        'awb_preliminary_epochs': 10,
        'awb_ab_training_epochs': 10,
        'awb_ab_warmup_epochs': 5,
        'awb_ab_max_iterations': 3,
        'awb_averaging_window': 5,
        'flag': [1.0, 1.0],
        'grad_weights': [0.01, 0.98, 0.1],
        'problem': 'vectors',
        'network': 'fcnn',
        'data': 'sine',
        'arch_search_epochs': 10,
        'arch_search_max_iter': 2,
    }


# =============================================================================
# Phase Profiling Functions
# =============================================================================

def profile_hamiltonian_computation(model, batch_size: int, num_iterations: int,
                                    loss_type: str = 'regression') -> PhaseProfile:
    """Profile Hamiltonian gradient computation phase."""
    from cl.core.hamiltonian import _hamiltonian_core_mse_standard, _hamiltonian_core_class_standard

    profile = PhaseProfile(phase_name="Hamiltonian Computation")
    key = jax.random.PRNGKey(0)

    # Generate dummy data based on model type
    if hasattr(model, 'sizes'):  # MLP
        input_shape = (batch_size, model.sizes[0])
        x = jax.random.normal(key, input_shape)
        exp_x = jax.random.normal(key, input_shape)
        deltax = jax.random.normal(key, input_shape) * 0.01
        if loss_type == 'regression':
            y = jax.random.normal(key, (batch_size,))
            exp_y = jax.random.normal(key, (batch_size,))
            hamiltonian_fn = _hamiltonian_core_mse_standard
        else:
            y = jax.random.randint(key, (batch_size,), 0, model.sizes[-1])
            exp_y = jax.random.randint(key, (batch_size,), 0, model.sizes[-1])
            hamiltonian_fn = _hamiltonian_core_class_standard
    else:  # CNN
        input_shape = (batch_size, 3, 32, 32)
        x = jax.random.normal(key, input_shape)
        exp_x = jax.random.normal(key, input_shape)
        deltax = jax.random.normal(key, input_shape) * 0.01
        y = jax.random.randint(key, (batch_size,), 0, 100)
        exp_y = jax.random.randint(key, (batch_size,), 0, 100)
        hamiltonian_fn = _hamiltonian_core_class_standard

    params, static = eqx.partition(model, eqx.is_array)

    # JIT compilation timing
    start = time.time()
    grad, losses = hamiltonian_fn(
        params, static, x, y, exp_x, exp_y, deltax,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(1000.0), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    profile.jit_compile_time_seconds = time.time() - start

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    # Execution timing
    start = time.time()
    for _ in range(num_iterations):
        grad, losses = hamiltonian_fn(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

    profile.execution_time_seconds = time.time() - start

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, profile.gpu_memory_samples = sampler.get_samples()

    profile.total_time_seconds = profile.jit_compile_time_seconds + profile.execution_time_seconds
    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = (profile.execution_time_seconds / num_iterations) * 1000
    profile.hamiltonian_time_ms = profile.time_per_iteration_ms

    return profile


def profile_optimizer_step(model, batch_size: int, num_iterations: int) -> PhaseProfile:
    """Profile optimizer update step phase."""
    profile = PhaseProfile(phase_name="Optimizer Step")

    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)

    # Create dummy gradient with same structure as params
    grad = jax.tree_map(lambda p: jnp.ones_like(p) * 0.01 if p is not None else None, params)

    # JIT compile the optimizer step
    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # JIT compilation timing
    start = time.time()
    params, opt_state = optimizer_step(grad, opt_state, params)
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if a is not None else None, params)
    profile.jit_compile_time_seconds = time.time() - start

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    # Execution timing
    start = time.time()
    for _ in range(num_iterations):
        params, opt_state = optimizer_step(grad, opt_state, params)
        jax.tree_util.tree_map(lambda a: a.block_until_ready() if a is not None else None, params)

    profile.execution_time_seconds = time.time() - start

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, profile.gpu_memory_samples = sampler.get_samples()

    profile.total_time_seconds = profile.jit_compile_time_seconds + profile.execution_time_seconds
    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = (profile.execution_time_seconds / num_iterations) * 1000
    profile.optimizer_time_ms = profile.time_per_iteration_ms

    return profile


def profile_v_transformation(model, num_iterations: int = 100) -> PhaseProfile:
    """Profile V = A @ W @ B.T transformation (AWB Step 4)."""
    from cl.core.awb import compute_V_from_AWB, set_new_AB_matrices

    profile = PhaseProfile(phase_name="V Transformation (Step 4)")

    if not hasattr(model, 'A') or model.A is None:
        # Initialize A/B matrices for the test
        original_arch = model.sizes
        new_arch = [s + 10 for s in original_arch]  # Slightly expanded
        new_arch[0] = original_arch[0]  # Keep input size
        new_arch[-1] = original_arch[-1]  # Keep output size
        model = set_new_AB_matrices(model, original_arch, new_arch)

    # JIT compilation timing
    start = time.time()
    model_transformed = compute_V_from_AWB(model)
    # Force computation
    _ = jax.tree_util.tree_leaves(model_transformed)
    profile.jit_compile_time_seconds = time.time() - start

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    # Execution timing
    start = time.time()
    for _ in range(num_iterations):
        model_transformed = compute_V_from_AWB(model)
        _ = jax.tree_util.tree_leaves(model_transformed)

    profile.execution_time_seconds = time.time() - start

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, profile.gpu_memory_samples = sampler.get_samples()

    profile.total_time_seconds = profile.jit_compile_time_seconds + profile.execution_time_seconds
    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = (profile.execution_time_seconds / num_iterations) * 1000

    return profile


def profile_ab_partitioning(model, num_iterations: int = 1000) -> PhaseProfile:
    """Profile model partitioning for A/B training."""
    from cl.core.awb import partition_for_AB_training, partition_for_standard_training, set_new_AB_matrices

    profile = PhaseProfile(phase_name="Model Partitioning")

    if not hasattr(model, 'A') or model.A is None:
        original_arch = model.sizes
        new_arch = [s + 10 for s in original_arch]
        new_arch[0] = original_arch[0]
        new_arch[-1] = original_arch[-1]
        model = set_new_AB_matrices(model, original_arch, new_arch)

    # Timing (partitioning is not JIT-compiled)
    start = time.time()
    for _ in range(num_iterations):
        diff_model, static_model = partition_for_AB_training(model)
        model_combined = eqx.combine(diff_model, static_model)
        params, static = partition_for_standard_training(model_combined)

    profile.total_time_seconds = time.time() - start
    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = (profile.total_time_seconds / num_iterations) * 1000

    return profile


def profile_data_transfer(batch_size: int, input_shape: Tuple[int, ...],
                          num_iterations: int = 100) -> PhaseProfile:
    """Profile CPU to GPU data transfer overhead."""
    import torch

    profile = PhaseProfile(phase_name="Data Transfer (CPU→GPU)")

    # Create PyTorch tensors (simulating DataLoader output)
    x_torch = torch.randn(batch_size, *input_shape)
    y_torch = torch.randint(0, 10, (batch_size,))

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    start = time.time()
    for _ in range(num_iterations):
        # Simulate the data transfer pattern from loops.py
        x = jnp.asarray(x_torch.numpy(), dtype=jnp.float64)
        y = jnp.asarray(y_torch.numpy(), dtype=jnp.int64)
        # Force transfer to complete
        x.block_until_ready()
        y.block_until_ready()

    profile.total_time_seconds = time.time() - start

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, profile.gpu_memory_samples = sampler.get_samples()

    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = (profile.total_time_seconds / num_iterations) * 1000
    profile.data_transfer_time_ms = profile.time_per_iteration_ms

    return profile


# =============================================================================
# Full Pipeline Profiling
# =============================================================================

def profile_full_training_epoch(model, batch_size: int, num_batches: int,
                                loss_type: str = 'regression') -> PhaseProfile:
    """Profile a complete training epoch with all operations."""
    from cl.core.hamiltonian import _hamiltonian_core_mse_standard, _hamiltonian_core_class_standard

    profile = PhaseProfile(phase_name="Full Training Epoch")
    key = jax.random.PRNGKey(0)

    # Setup based on model type
    if hasattr(model, 'sizes'):  # MLP
        input_dim = model.sizes[0]
        output_dim = model.sizes[-1]
        if loss_type == 'regression':
            hamiltonian_fn = _hamiltonian_core_mse_standard
        else:
            hamiltonian_fn = _hamiltonian_core_class_standard
    else:  # CNN
        input_dim = (3, 32, 32)
        output_dim = 100
        hamiltonian_fn = _hamiltonian_core_class_standard

    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adam(1e-4)
    opt_state = optim.init(params)

    @jax.jit
    def optimizer_step(grad, opt_state, params):
        updates, new_opt_state = optim.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Pre-generate all batch data
    batches = []
    for i in range(num_batches):
        key, subkey = jax.random.split(key)
        if isinstance(input_dim, int):
            x = jax.random.normal(subkey, (batch_size, input_dim))
            exp_x = jax.random.normal(subkey, (batch_size, input_dim))
            deltax = jax.random.normal(subkey, (batch_size, input_dim)) * 0.01
        else:
            x = jax.random.normal(subkey, (batch_size, *input_dim))
            exp_x = jax.random.normal(subkey, (batch_size, *input_dim))
            deltax = jax.random.normal(subkey, (batch_size, *input_dim)) * 0.01

        if loss_type == 'regression':
            y = jax.random.normal(subkey, (batch_size,))
            exp_y = jax.random.normal(subkey, (batch_size,))
        else:
            y = jax.random.randint(subkey, (batch_size,), 0, output_dim)
            exp_y = jax.random.randint(subkey, (batch_size,), 0, output_dim)

        batches.append((x, y, exp_x, exp_y, deltax))

    # Warm up JIT
    x, y, exp_x, exp_y, deltax = batches[0]
    grad, _ = hamiltonian_fn(
        params, static, x, y, exp_x, exp_y, deltax,
        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
        jnp.array(1000.0), jnp.array(1.0)
    )
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
    params, opt_state = optimizer_step(grad, opt_state, params)

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    # Time breakdowns
    hamiltonian_times = []
    optimizer_times = []

    start_total = time.time()
    for x, y, exp_x, exp_y, deltax in batches:
        # Hamiltonian computation
        start = time.time()
        grad, losses = hamiltonian_fn(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        hamiltonian_times.append(time.time() - start)

        # Optimizer step
        start = time.time()
        params, opt_state = optimizer_step(grad, opt_state, params)
        jax.tree_util.tree_map(lambda a: a.block_until_ready() if a is not None else None, params)
        optimizer_times.append(time.time() - start)

    profile.total_time_seconds = time.time() - start_total

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, profile.gpu_memory_samples = sampler.get_samples()

    profile.num_iterations = num_batches
    profile.time_per_iteration_ms = (profile.total_time_seconds / num_batches) * 1000
    profile.hamiltonian_time_ms = np.mean(hamiltonian_times) * 1000
    profile.optimizer_time_ms = np.mean(optimizer_times) * 1000

    return profile


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.gpu
class TestAWBPhasesProfiling:
    """Profile individual AWB pipeline phases."""

    def test_hamiltonian_mse_profiling(self, awb_mlp_model, report_dir):
        """Profile Hamiltonian MSE computation for regression."""
        profile = profile_hamiltonian_computation(
            awb_mlp_model, batch_size=64, num_iterations=100, loss_type='regression'
        )

        print(f"\n{'='*60}")
        print("Hamiltonian MSE Computation Profile")
        print(f"{'='*60}")
        print(f"JIT Compile Time:     {profile.jit_compile_time_seconds:.4f}s")
        print(f"Execution Time:       {profile.execution_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.2f}ms")
        print(f"GPU Utilization Mean: {profile.gpu_util_mean:.1f}%")
        print(f"GPU Utilization Max:  {profile.gpu_util_max:.1f}%")
        print(f"{'='*60}")

    def test_hamiltonian_classification_profiling(self, awb_cnn_model, report_dir):
        """Profile Hamiltonian classification computation for CNN."""
        if not HAS_GPU:
            pytest.skip("No GPU available")

        profile = profile_hamiltonian_computation(
            awb_cnn_model, batch_size=64, num_iterations=50, loss_type='classification'
        )

        print(f"\n{'='*60}")
        print("Hamiltonian Classification (CNN3D) Profile")
        print(f"{'='*60}")
        print(f"JIT Compile Time:     {profile.jit_compile_time_seconds:.4f}s")
        print(f"Execution Time:       {profile.execution_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.2f}ms")
        print(f"GPU Utilization Mean: {profile.gpu_util_mean:.1f}%")
        print(f"GPU Utilization Max:  {profile.gpu_util_max:.1f}%")
        print(f"{'='*60}")

    def test_optimizer_step_profiling(self, awb_mlp_model, report_dir):
        """Profile optimizer update step."""
        profile = profile_optimizer_step(awb_mlp_model, batch_size=64, num_iterations=1000)

        print(f"\n{'='*60}")
        print("Optimizer Step Profile")
        print(f"{'='*60}")
        print(f"JIT Compile Time:     {profile.jit_compile_time_seconds:.4f}s")
        print(f"Execution Time:       {profile.execution_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.3f}ms")
        print(f"GPU Utilization Mean: {profile.gpu_util_mean:.1f}%")
        print(f"{'='*60}")

    def test_v_transformation_profiling(self, awb_mlp_model, report_dir):
        """Profile V = A @ W @ B.T transformation."""
        profile = profile_v_transformation(awb_mlp_model, num_iterations=100)

        print(f"\n{'='*60}")
        print("V Transformation (Step 4) Profile")
        print(f"{'='*60}")
        print(f"JIT Compile Time:     {profile.jit_compile_time_seconds:.4f}s")
        print(f"Execution Time:       {profile.execution_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.3f}ms")
        print(f"{'='*60}")

    def test_partitioning_profiling(self, awb_mlp_model, report_dir):
        """Profile model partitioning operations."""
        profile = profile_ab_partitioning(awb_mlp_model, num_iterations=1000)

        print(f"\n{'='*60}")
        print("Model Partitioning Profile")
        print(f"{'='*60}")
        print(f"Total Time:           {profile.total_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.4f}ms")
        print(f"{'='*60}")

    def test_data_transfer_profiling(self, report_dir):
        """Profile CPU to GPU data transfer."""
        pytest.importorskip("torch")

        profile = profile_data_transfer(batch_size=64, input_shape=(784,), num_iterations=100)

        print(f"\n{'='*60}")
        print("Data Transfer (CPU→GPU) Profile")
        print(f"{'='*60}")
        print(f"Total Time:           {profile.total_time_seconds:.4f}s")
        print(f"Time per Iteration:   {profile.time_per_iteration_ms:.3f}ms")
        print(f"{'='*60}")


@pytest.mark.gpu
class TestFullAWBPipelineProfiling:
    """Profile complete AWB pipeline phases."""

    def test_full_epoch_profiling_mlp(self, awb_mlp_model, profiling_config, report_dir):
        """Profile a complete training epoch for MLP."""
        profile = profile_full_training_epoch(
            awb_mlp_model, batch_size=64, num_batches=50, loss_type='regression'
        )

        print(f"\n{'='*60}")
        print("Full Training Epoch Profile (MLP)")
        print(f"{'='*60}")
        print(f"Total Time:           {profile.total_time_seconds:.4f}s")
        print(f"Time per Batch:       {profile.time_per_iteration_ms:.2f}ms")
        print(f"  Hamiltonian:        {profile.hamiltonian_time_ms:.2f}ms ({profile.hamiltonian_time_ms/profile.time_per_iteration_ms*100:.1f}%)")
        print(f"  Optimizer:          {profile.optimizer_time_ms:.2f}ms ({profile.optimizer_time_ms/profile.time_per_iteration_ms*100:.1f}%)")
        print(f"GPU Utilization Mean: {profile.gpu_util_mean:.1f}%")
        print(f"{'='*60}")

    def test_full_epoch_profiling_cnn(self, awb_cnn_model, profiling_config, report_dir):
        """Profile a complete training epoch for CNN3D."""
        if not HAS_GPU:
            pytest.skip("No GPU available")

        profile = profile_full_training_epoch(
            awb_cnn_model, batch_size=64, num_batches=30, loss_type='classification'
        )

        print(f"\n{'='*60}")
        print("Full Training Epoch Profile (CNN3D)")
        print(f"{'='*60}")
        print(f"Total Time:           {profile.total_time_seconds:.4f}s")
        print(f"Time per Batch:       {profile.time_per_iteration_ms:.2f}ms")
        print(f"  Hamiltonian:        {profile.hamiltonian_time_ms:.2f}ms ({profile.hamiltonian_time_ms/profile.time_per_iteration_ms*100:.1f}%)")
        print(f"  Optimizer:          {profile.optimizer_time_ms:.2f}ms ({profile.optimizer_time_ms/profile.time_per_iteration_ms*100:.1f}%)")
        print(f"GPU Utilization Mean: {profile.gpu_util_mean:.1f}%")
        print(f"{'='*60}")

    def test_awb_pipeline_bottleneck_analysis(self, awb_mlp_model, profiling_config, report_dir):
        """Complete bottleneck analysis of all AWB phases."""
        pytest.importorskip("torch")

        phases = []

        # Phase 1: Standard CL Training (simulated as Task 0)
        print("\nProfiling: Standard CL Training...")
        phases.append(profile_full_training_epoch(
            awb_mlp_model, batch_size=64, num_batches=50, loss_type='regression'
        ))
        phases[-1].phase_name = "Standard CL Training (Task 0)"

        # Phase 2: Preliminary Training (Step 1)
        print("Profiling: Preliminary Training...")
        phases.append(profile_full_training_epoch(
            awb_mlp_model, batch_size=64, num_batches=20, loss_type='regression'
        ))
        phases[-1].phase_name = "Preliminary Training (Step 1)"

        # Phase 3: Architecture Search (Step 3a) - simulated with Hamiltonian evals
        print("Profiling: Architecture Search...")
        phases.append(profile_hamiltonian_computation(
            awb_mlp_model, batch_size=64, num_iterations=30, loss_type='regression'
        ))
        phases[-1].phase_name = "Architecture Search (Step 3a)"

        # Phase 4: A/B Training (Step 3b)
        print("Profiling: A/B Training...")
        phases.append(profile_full_training_epoch(
            awb_mlp_model, batch_size=64, num_batches=20, loss_type='regression'
        ))
        phases[-1].phase_name = "A/B Training (Step 3b)"

        # Phase 5: V Transformation (Step 4)
        print("Profiling: V Transformation...")
        phases.append(profile_v_transformation(awb_mlp_model, num_iterations=10))
        phases[-1].phase_name = "V Transformation (Step 4)"

        # Phase 6: V Training (Step 5)
        print("Profiling: V Training...")
        phases.append(profile_full_training_epoch(
            awb_mlp_model, batch_size=64, num_batches=50, loss_type='regression'
        ))
        phases[-1].phase_name = "V Training (Step 5)"

        # Phase 7: Data Transfer overhead
        print("Profiling: Data Transfer...")
        phases.append(profile_data_transfer(batch_size=64, input_shape=(1,), num_iterations=50))

        # Phase 8: Partitioning overhead
        print("Profiling: Model Partitioning...")
        phases.append(profile_ab_partitioning(awb_mlp_model, num_iterations=100))

        # Print bottleneck report
        print_bottleneck_report(phases, "AWB Pipeline Bottleneck Analysis (MLP)")

        # Generate JSON report
        report_path, report = generate_awb_report(
            'awb_pipeline_bottleneck_mlp', phases, profiling_config, str(report_dir)
        )

        # Verify at least one profile was captured
        assert len(phases) > 0
        assert all(p.total_time_seconds > 0 for p in phases)


@pytest.mark.gpu
class TestBatchSizeScaling:
    """Test how different batch sizes affect GPU utilization."""

    def test_batch_size_scaling_mlp(self, awb_mlp_model, report_dir):
        """Profile MLP performance across different batch sizes."""
        batch_sizes = [16, 32, 64, 128, 256]
        results = []

        print(f"\n{'='*60}")
        print("Batch Size Scaling Analysis (MLP)")
        print(f"{'='*60}")
        print(f"{'Batch':<10} {'Time/Iter (ms)':<15} {'GPU Util %':<12} {'Throughput':<12}")
        print('-' * 60)

        for bs in batch_sizes:
            profile = profile_hamiltonian_computation(
                awb_mlp_model, batch_size=bs, num_iterations=50, loss_type='regression'
            )
            throughput = bs / (profile.time_per_iteration_ms / 1000)  # samples/sec

            results.append({
                'batch_size': bs,
                'time_per_iter_ms': profile.time_per_iteration_ms,
                'gpu_util': profile.gpu_util_mean,
                'throughput': throughput,
            })

            print(f"{bs:<10} {profile.time_per_iteration_ms:<15.2f} {profile.gpu_util_mean:<12.1f} {throughput:<12.0f}")

        print(f"{'='*60}")

        # Find optimal batch size (highest throughput)
        optimal = max(results, key=lambda x: x['throughput'])
        print(f"\nOptimal batch size: {optimal['batch_size']} (throughput: {optimal['throughput']:.0f} samples/sec)")

    def test_batch_size_scaling_cnn(self, awb_cnn_model, report_dir):
        """Profile CNN3D performance across different batch sizes."""
        if not HAS_GPU:
            pytest.skip("No GPU available")

        batch_sizes = [8, 16, 32, 64, 128]
        results = []

        print(f"\n{'='*60}")
        print("Batch Size Scaling Analysis (CNN3D)")
        print(f"{'='*60}")
        print(f"{'Batch':<10} {'Time/Iter (ms)':<15} {'GPU Util %':<12} {'Throughput':<12}")
        print('-' * 60)

        for bs in batch_sizes:
            try:
                profile = profile_hamiltonian_computation(
                    awb_cnn_model, batch_size=bs, num_iterations=20, loss_type='classification'
                )
                throughput = bs / (profile.time_per_iteration_ms / 1000)

                results.append({
                    'batch_size': bs,
                    'time_per_iter_ms': profile.time_per_iteration_ms,
                    'gpu_util': profile.gpu_util_mean,
                    'throughput': throughput,
                })

                print(f"{bs:<10} {profile.time_per_iteration_ms:<15.2f} {profile.gpu_util_mean:<12.1f} {throughput:<12.0f}")
            except Exception as e:
                print(f"{bs:<10} OOM or error: {str(e)[:30]}")

        print(f"{'='*60}")

        if results:
            optimal = max(results, key=lambda x: x['throughput'])
            print(f"\nOptimal batch size: {optimal['batch_size']} (throughput: {optimal['throughput']:.0f} samples/sec)")


@pytest.mark.gpu
def test_generate_awb_profiling_summary(report_dir):
    """Generate summary of all AWB profiling reports."""
    report_files = list(Path(report_dir).glob('awb_*.json'))

    if not report_files:
        print(f"\nNo AWB profiling reports found in {report_dir}/")
        print("Run AWB profiling tests first with: pytest tests/test_awb_gpu_profiling.py -m gpu -v -s")
        return

    print(f"\n{'='*70}")
    print("AWB GPU Profiling Summary")
    print(f"{'='*70}")

    for f in sorted(report_files)[-5:]:
        with open(f) as fp:
            report = json.load(fp)

        print(f"\n{report['test_name']} ({report['timestamp']})")
        print(f"  Backend: {report['jax_backend']}")
        print(f"  Total Time: {report['total_time_seconds']:.2f}s")

        # Print bottlenecks
        bottlenecks = report.get('bottleneck_analysis', [])
        if bottlenecks:
            print("  Bottlenecks:")
            for b in bottlenecks[:3]:
                marker = "***" if b.get('is_bottleneck') else ""
                print(f"    {b['phase']}: {b['percentage']:.1f}% {marker}")
