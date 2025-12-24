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

    # CNN3D doesn't have awb_enabled parameter - AWB matrices are always initialized
    return CNN3D(
        key=jax.random.PRNGKey(42),
        filter_size=filter_size,
        feed_sizes=feed_sizes,
        input_size=32,
        channel_in=3,
        channel_out=32,
        num_classes=100
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
            # Added by Claude: MLP outputs (batch, 1) but gets squeezed to (batch,) in hamiltonian
            # So y needs shape (batch,) to match after squeezing
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
    grad = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01 if p is not None else None, params)

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

    # Always initialize A/B matrices properly for transformation test
    # The default A/B matrices have wrong shapes for V = A @ W @ B.T
    original_arch = model.sizes
    # New arch with expanded hidden layers (input/output same)
    new_arch = [original_arch[0]] + [s + 10 for s in original_arch[1:-1]] + [original_arch[-1]]
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

    # Always initialize A/B matrices properly for partitioning test
    original_arch = model.sizes
    new_arch = [original_arch[0]] + [s + 10 for s in original_arch[1:-1]] + [original_arch[-1]]
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
# Architecture Search Profiling (Real JIT Costs)
# =============================================================================

@dataclass
class ArchSearchProfile:
    """Detailed profile for architecture search with JIT recompilation costs."""
    total_time_seconds: float = 0.0
    num_candidates: int = 0
    num_iterations: int = 0

    # Per-candidate breakdown
    model_creation_times: List[float] = field(default_factory=list)
    weight_init_times: List[float] = field(default_factory=list)
    partitioning_times: List[float] = field(default_factory=list)
    jit_compile_times: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)

    # Aggregated stats
    total_jit_time: float = 0.0
    total_training_time: float = 0.0
    total_model_creation_time: float = 0.0

    gpu_utilization_samples: List[float] = field(default_factory=list)

    @property
    def avg_jit_per_candidate(self) -> float:
        return np.mean(self.jit_compile_times) if self.jit_compile_times else 0.0

    @property
    def avg_training_per_candidate(self) -> float:
        return np.mean(self.training_times) if self.training_times else 0.0

    @property
    def jit_percentage(self) -> float:
        return (self.total_jit_time / self.total_time_seconds * 100) if self.total_time_seconds > 0 else 0.0


def profile_model_creation(base_arch: List[int], num_candidates: int = 4) -> PhaseProfile:
    """Profile model creation for different architectures.

    This measures the cost of creating new MLP models with different architectures,
    which happens for each candidate during architecture search.
    """
    from cl.models.mlp import MLP

    profile = PhaseProfile(phase_name="Model Creation")

    # Generate candidate architectures (similar to generate_search_candidates)
    candidates = []
    for i in range(num_candidates):
        # Expand hidden layers by increments
        new_arch = [base_arch[0]] + [h + 15 * (i + 1) for h in base_arch[1:-1]] + [base_arch[-1]]
        candidates.append(new_arch)

    creation_times = []
    start_total = time.time()

    for arch in candidates:
        start = time.time()
        model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
        # Force any lazy initialization
        _ = jax.tree_util.tree_leaves(model)
        creation_times.append(time.time() - start)

    profile.total_time_seconds = time.time() - start_total
    profile.num_iterations = num_candidates
    profile.time_per_iteration_ms = np.mean(creation_times) * 1000

    return profile


def profile_weight_reinitialization(model, num_iterations: int = 10) -> PhaseProfile:
    """Profile weight reinitialization cost."""
    from cl.models.mlp import MLP

    profile = PhaseProfile(phase_name="Weight Reinitialization")

    reinit_times = []
    start_total = time.time()

    for i in range(num_iterations):
        start = time.time()
        # Reinitialize weights (as done in arch_search.py)
        initializer = jax.nn.initializers.glorot_uniform()
        new_model = model
        for j in range(len(model.sizes) - 1):
            in_size = model.sizes[j]
            out_size = model.sizes[j + 1]
            weight = initializer(jax.random.PRNGKey(i + j), (out_size, in_size))
            bias = initializer(jax.random.PRNGKey(i + j + 100), (1, out_size))
            new_model = eqx.tree_at(lambda x, idx=j: x.layers[idx].weight, new_model, weight)
            new_model = eqx.tree_at(lambda x, idx=j: x.layers[idx].bias, new_model, bias)
        # Force computation
        _ = jax.tree_util.tree_leaves(new_model)
        reinit_times.append(time.time() - start)

    profile.total_time_seconds = time.time() - start_total
    profile.num_iterations = num_iterations
    profile.time_per_iteration_ms = np.mean(reinit_times) * 1000

    return profile


def profile_jit_recompilation_per_architecture(base_arch: List[int],
                                                num_candidates: int = 4,
                                                batch_size: int = 64) -> ArchSearchProfile:
    """Profile JIT recompilation cost for each new architecture.

    This is the KEY bottleneck test - it measures the real cost of JIT recompilation
    that happens when architecture changes during search.
    """
    from cl.models.mlp import MLP
    from cl.core.hamiltonian import _hamiltonian_core_mse_standard

    profile = ArchSearchProfile()
    key = jax.random.PRNGKey(0)

    # Generate candidate architectures
    candidates = []
    for i in range(num_candidates):
        new_arch = [base_arch[0]] + [h + 15 * (i + 1) for h in base_arch[1:-1]] + [base_arch[-1]]
        candidates.append(new_arch)

    profile.num_candidates = num_candidates

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    start_total = time.time()

    for arch in candidates:
        # Step 1: Model Creation
        start = time.time()
        model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
        _ = jax.tree_util.tree_leaves(model)
        profile.model_creation_times.append(time.time() - start)

        # Step 2: Weight Reinitialization
        start = time.time()
        initializer = jax.nn.initializers.glorot_uniform()
        for j in range(len(model.sizes) - 1):
            in_size = model.sizes[j]
            out_size = model.sizes[j + 1]
            weight = initializer(jax.random.PRNGKey(j), (out_size, in_size))
            bias = initializer(jax.random.PRNGKey(j + 100), (1, out_size))
            model = eqx.tree_at(lambda x, idx=j: x.layers[idx].weight, model, weight)
            model = eqx.tree_at(lambda x, idx=j: x.layers[idx].bias, model, bias)
        profile.weight_init_times.append(time.time() - start)

        # Step 3: Partitioning
        start = time.time()
        params, static = eqx.partition(model, eqx.is_array)
        if model.awb_enabled and model.A is not None:
            static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
            params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))
        profile.partitioning_times.append(time.time() - start)

        # Step 4: JIT Compilation (first call to Hamiltonian with this architecture)
        # Generate data matching this architecture
        x = jax.random.normal(key, (batch_size, arch[0]))
        y = jax.random.normal(key, (batch_size,))
        exp_x = jax.random.normal(key, (batch_size, arch[0]))
        exp_y = jax.random.normal(key, (batch_size,))
        deltax = jax.random.normal(key, (batch_size, arch[0])) * 0.01

        start = time.time()
        grad, losses = _hamiltonian_core_mse_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        profile.jit_compile_times.append(time.time() - start)

        # Step 5: Training iterations (already compiled)
        num_training_iters = 10  # Simulate a few epochs
        start = time.time()
        for _ in range(num_training_iters):
            grad, losses = _hamiltonian_core_mse_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        profile.training_times.append(time.time() - start)

    profile.total_time_seconds = time.time() - start_total

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        profile.gpu_utilization_samples, _ = sampler.get_samples()

    # Compute aggregated stats
    profile.total_jit_time = sum(profile.jit_compile_times)
    profile.total_training_time = sum(profile.training_times)
    profile.total_model_creation_time = sum(profile.model_creation_times)
    profile.num_iterations = num_candidates

    return profile


def profile_full_architecture_search_simulation(base_arch: List[int],
                                                  search_iterations: int = 2,
                                                  candidates_per_iter: int = 4,
                                                  epochs_per_candidate: int = 10,
                                                  batch_size: int = 64,
                                                  batches_per_epoch: int = 10) -> Dict[str, Any]:
    """Simulate and profile a complete architecture search.

    This profiles the ACTUAL architecture search loop as it runs in production,
    including all JIT recompilation costs for each candidate.

    Args:
        base_arch: Starting architecture [input, h1, h2, ..., output]
        search_iterations: Number of search iterations (default: 2)
        candidates_per_iter: Candidates per iteration (default: 4, from search_range=2)
        epochs_per_candidate: Training epochs per candidate (default: 10)
        batch_size: Batch size for training
        batches_per_epoch: Batches per epoch

    Returns:
        Dict with detailed profiling breakdown
    """
    from cl.models.mlp import MLP
    from cl.core.hamiltonian import _hamiltonian_core_mse_standard

    key = jax.random.PRNGKey(0)

    results = {
        'total_time': 0.0,
        'total_candidates': 0,
        'per_candidate': [],
        'phase_totals': {
            'model_creation': 0.0,
            'weight_init': 0.0,
            'partitioning': 0.0,
            'jit_compilation': 0.0,
            'training': 0.0,
            'optimizer_init': 0.0,
        },
        'gpu_samples': [],
    }

    # Start GPU sampling
    sampler = GPUSampler()
    if NVIDIA_SMI_AVAILABLE:
        sampler.start()

    start_total = time.time()

    for iteration in range(search_iterations):
        # Generate candidates for this iteration
        for c in range(candidates_per_iter):
            candidate_profile = {
                'iteration': iteration,
                'candidate': c,
                'arch': None,
                'times': {},
            }

            # Generate candidate architecture
            expansion = 15 * (iteration * candidates_per_iter + c + 1)
            new_arch = [base_arch[0]] + [h + expansion for h in base_arch[1:-1]] + [base_arch[-1]]
            candidate_profile['arch'] = new_arch

            # 1. Model Creation
            start = time.time()
            model = MLP(sizes=new_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
            _ = jax.tree_util.tree_leaves(model)
            candidate_profile['times']['model_creation'] = time.time() - start
            results['phase_totals']['model_creation'] += candidate_profile['times']['model_creation']

            # 2. Weight Reinitialization
            start = time.time()
            initializer = jax.nn.initializers.glorot_uniform()
            for j in range(len(model.sizes) - 1):
                in_size = model.sizes[j]
                out_size = model.sizes[j + 1]
                weight = initializer(jax.random.PRNGKey(c + j), (out_size, in_size))
                bias = initializer(jax.random.PRNGKey(c + j + 100), (1, out_size))
                model = eqx.tree_at(lambda x, idx=j: x.layers[idx].weight, model, weight)
                model = eqx.tree_at(lambda x, idx=j: x.layers[idx].bias, model, bias)
            candidate_profile['times']['weight_init'] = time.time() - start
            results['phase_totals']['weight_init'] += candidate_profile['times']['weight_init']

            # 3. Partitioning
            start = time.time()
            params, static = eqx.partition(model, eqx.is_array)
            if model.awb_enabled and model.A is not None:
                static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
                params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))
            candidate_profile['times']['partitioning'] = time.time() - start
            results['phase_totals']['partitioning'] += candidate_profile['times']['partitioning']

            # 4. Optimizer Initialization
            start = time.time()
            optim = optax.adam(1e-4)
            opt_state = optim.init(params)
            candidate_profile['times']['optimizer_init'] = time.time() - start
            results['phase_totals']['optimizer_init'] += candidate_profile['times']['optimizer_init']

            # Generate training data
            x = jax.random.normal(key, (batch_size, new_arch[0]))
            y = jax.random.normal(key, (batch_size,))
            exp_x = jax.random.normal(key, (batch_size, new_arch[0]))
            exp_y = jax.random.normal(key, (batch_size,))
            deltax = jax.random.normal(key, (batch_size, new_arch[0])) * 0.01

            # 5. First call (JIT Compilation)
            start = time.time()
            grad, losses = _hamiltonian_core_mse_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
            candidate_profile['times']['jit_compilation'] = time.time() - start
            results['phase_totals']['jit_compilation'] += candidate_profile['times']['jit_compilation']

            # 6. Training loop (already compiled)
            @jax.jit
            def optimizer_step(grad, opt_state, params):
                updates, new_opt_state = optim.update(grad, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state

            # Warm up optimizer JIT
            params, opt_state = optimizer_step(grad, opt_state, params)
            jax.tree_util.tree_map(lambda a: a.block_until_ready() if a is not None else None, params)

            start = time.time()
            for epoch in range(epochs_per_candidate):
                for batch in range(batches_per_epoch):
                    grad, losses = _hamiltonian_core_mse_standard(
                        params, static, x, y, exp_x, exp_y, deltax,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        jnp.array(1000.0), jnp.array(1.0)
                    )
                    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
                    params, opt_state = optimizer_step(grad, opt_state, params)
                    jax.tree_util.tree_map(lambda a: a.block_until_ready() if a is not None else None, params)

            candidate_profile['times']['training'] = time.time() - start
            results['phase_totals']['training'] += candidate_profile['times']['training']

            candidate_profile['times']['total'] = sum(candidate_profile['times'].values()) - candidate_profile['times'].get('total', 0)
            results['per_candidate'].append(candidate_profile)
            results['total_candidates'] += 1

    results['total_time'] = time.time() - start_total

    if NVIDIA_SMI_AVAILABLE:
        sampler.stop()
        results['gpu_samples'], _ = sampler.get_samples()

    return results


def print_arch_search_report(results: Dict[str, Any]):
    """Print detailed architecture search profiling report."""
    print(f"\n{'='*80}")
    print("ARCHITECTURE SEARCH PROFILING REPORT")
    print(f"{'='*80}")

    print(f"\nTotal Time: {results['total_time']:.2f}s")
    print(f"Total Candidates Evaluated: {results['total_candidates']}")
    print(f"Avg Time per Candidate: {results['total_time'] / results['total_candidates']:.2f}s")

    # Phase breakdown
    print(f"\n{'Phase Breakdown':^80}")
    print('-' * 80)
    print(f"{'Phase':<25} {'Total (s)':<12} {'% of Total':<12} {'Avg/Candidate (ms)':<20}")
    print('-' * 80)

    for phase, total in sorted(results['phase_totals'].items(), key=lambda x: x[1], reverse=True):
        pct = (total / results['total_time'] * 100) if results['total_time'] > 0 else 0
        avg_per_cand = (total / results['total_candidates'] * 1000) if results['total_candidates'] > 0 else 0
        marker = "***" if pct > 20 else ""
        print(f"{phase:<25} {total:<12.3f} {pct:<12.1f} {avg_per_cand:<20.1f} {marker}")

    print('-' * 80)

    # JIT vs Training breakdown
    jit_total = results['phase_totals']['jit_compilation']
    training_total = results['phase_totals']['training']
    overhead_total = results['total_time'] - jit_total - training_total

    print(f"\n{'Time Category Breakdown':^80}")
    print('-' * 80)
    print(f"JIT Compilation:        {jit_total:8.2f}s  ({jit_total/results['total_time']*100:5.1f}%)")
    print(f"Actual Training:        {training_total:8.2f}s  ({training_total/results['total_time']*100:5.1f}%)")
    print(f"Overhead (setup/init):  {overhead_total:8.2f}s  ({overhead_total/results['total_time']*100:5.1f}%)")
    print('-' * 80)

    # Per-candidate details (first few)
    print(f"\n{'Per-Candidate Details (first 4)':^80}")
    print('-' * 80)
    for cand in results['per_candidate'][:4]:
        arch_str = f"{cand['arch'][0]}→{cand['arch'][1]}→...→{cand['arch'][-1]}"
        print(f"Iter {cand['iteration']}, Cand {cand['candidate']}: {arch_str}")
        print(f"  JIT: {cand['times']['jit_compilation']*1000:.1f}ms, "
              f"Train: {cand['times']['training']*1000:.1f}ms, "
              f"Total: {cand['times']['total']*1000:.1f}ms")

    # GPU utilization
    if results['gpu_samples']:
        print(f"\nGPU Utilization: Mean={np.mean(results['gpu_samples']):.1f}%, "
              f"Max={max(results['gpu_samples']):.1f}%")

    # Optimization recommendations
    print(f"\n{'='*80}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}")

    jit_pct = jit_total / results['total_time'] * 100
    if jit_pct > 30:
        print(f"\n[CRITICAL] JIT compilation is {jit_pct:.1f}% of total time!")
        print("  Recommendations:")
        print("  1. Cache JIT-compiled functions across similar architectures")
        print("  2. Use architecture families with fixed PyTree structure")
        print("  3. Reduce search_range to evaluate fewer candidates")
        print("  4. Implement progressive architecture search (start small)")

    model_creation_pct = results['phase_totals']['model_creation'] / results['total_time'] * 100
    if model_creation_pct > 10:
        print(f"\n[WARNING] Model creation is {model_creation_pct:.1f}% of total time")
        print("  Recommendations:")
        print("  1. Reuse model structure and only update weights")
        print("  2. Use lazy initialization for unused layers")

    training_pct = training_total / results['total_time'] * 100
    if training_pct < 50:
        print(f"\n[INFO] Only {training_pct:.1f}% of time spent on actual training")
        print("  The search is overhead-dominated, not compute-dominated")


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
            # MLP outputs (batch, 1), so y needs shape (batch, 1) to match
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
class TestArchitectureSearchProfiling:
    """Profile architecture search with JIT recompilation costs - the REAL bottleneck."""

    def test_jit_recompilation_per_architecture(self, report_dir):
        """Profile JIT recompilation cost for each new architecture.

        This is the KEY test - measures the real cost of creating and JIT-compiling
        each candidate architecture during search.
        """
        base_arch = [1, 75, 75, 75, 1]  # Sine regression architecture

        profile = profile_jit_recompilation_per_architecture(
            base_arch=base_arch,
            num_candidates=4,
            batch_size=64
        )

        print(f"\n{'='*70}")
        print("JIT Recompilation per Architecture Profile")
        print(f"{'='*70}")
        print(f"Total Candidates:    {profile.num_candidates}")
        print(f"Total Time:          {profile.total_time_seconds:.2f}s")
        print(f"Avg per Candidate:   {profile.total_time_seconds / profile.num_candidates:.2f}s")
        print(f"\nBreakdown:")
        print(f"  Model Creation:    {profile.total_model_creation_time*1000:.1f}ms total "
              f"({profile.total_model_creation_time / profile.total_time_seconds * 100:.1f}%)")
        print(f"  JIT Compilation:   {profile.total_jit_time:.3f}s total "
              f"({profile.jit_percentage:.1f}%)")
        print(f"  Training (post-JIT): {profile.total_training_time:.3f}s total "
              f"({profile.total_training_time / profile.total_time_seconds * 100:.1f}%)")
        print(f"\nPer-candidate JIT times:")
        for i, jit_time in enumerate(profile.jit_compile_times):
            print(f"  Candidate {i}: {jit_time*1000:.1f}ms")
        print(f"{'='*70}")

        # Assert JIT is measured
        assert profile.total_jit_time > 0

    def test_full_architecture_search_simulation(self, report_dir):
        """Profile a complete architecture search simulation.

        This simulates the ACTUAL search loop as it runs in production,
        with 2 iterations × 4 candidates each = 8 total candidate evaluations.
        """
        base_arch = [1, 75, 75, 75, 1]

        results = profile_full_architecture_search_simulation(
            base_arch=base_arch,
            search_iterations=2,
            candidates_per_iter=4,
            epochs_per_candidate=10,
            batch_size=64,
            batches_per_epoch=10
        )

        # Print detailed report
        print_arch_search_report(results)

        # Save results to file
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(report_dir) / f'arch_search_simulation_{timestamp}.json'

        with open(report_path, 'w') as f:
            # Convert non-serializable items
            serializable_results = {
                k: v for k, v in results.items()
                if k != 'per_candidate'
            }
            serializable_results['per_candidate_summary'] = [
                {
                    'iteration': c['iteration'],
                    'candidate': c['candidate'],
                    'arch': c['arch'],
                    'jit_time_ms': c['times']['jit_compilation'] * 1000,
                    'training_time_ms': c['times']['training'] * 1000,
                }
                for c in results['per_candidate']
            ]
            json.dump(serializable_results, f, indent=2)

        print(f"\nReport saved to: {report_path}")

        # Assertions
        assert results['total_candidates'] == 8
        assert results['phase_totals']['jit_compilation'] > 0

    def test_model_creation_overhead(self, report_dir):
        """Profile just model creation overhead."""
        base_arch = [1, 75, 75, 75, 1]

        profile = profile_model_creation(base_arch=base_arch, num_candidates=8)

        print(f"\n{'='*60}")
        print("Model Creation Overhead Profile")
        print(f"{'='*60}")
        print(f"Candidates Created:  {profile.num_iterations}")
        print(f"Total Time:          {profile.total_time_seconds*1000:.1f}ms")
        print(f"Time per Model:      {profile.time_per_iteration_ms:.2f}ms")
        print(f"{'='*60}")

    def test_weight_reinit_overhead(self, awb_mlp_model, report_dir):
        """Profile weight reinitialization overhead."""
        profile = profile_weight_reinitialization(awb_mlp_model, num_iterations=20)

        print(f"\n{'='*60}")
        print("Weight Reinitialization Overhead Profile")
        print(f"{'='*60}")
        print(f"Reinits Performed:   {profile.num_iterations}")
        print(f"Total Time:          {profile.total_time_seconds*1000:.1f}ms")
        print(f"Time per Reinit:     {profile.time_per_iteration_ms:.2f}ms")
        print(f"{'='*60}")


@pytest.mark.gpu
class TestOptimizationOpportunities:
    """Tests to identify specific optimization opportunities for architecture search."""

    def test_fixed_pytree_structure_benefit(self, report_dir):
        """Compare JIT cost: varying architectures vs fixed PyTree with masked weights.

        This tests a potential optimization: instead of creating new models with
        different architectures (triggering JIT recompilation), use a large fixed
        model and mask unused parameters.
        """
        from cl.models.mlp import MLP
        from cl.core.hamiltonian import _hamiltonian_core_mse_standard

        key = jax.random.PRNGKey(0)
        batch_size = 64

        # Strategy 1: Varying architectures (current approach)
        varying_architectures = [
            [1, 75, 75, 75, 1],
            [1, 90, 90, 90, 1],
            [1, 105, 105, 105, 1],
            [1, 120, 120, 120, 1],
        ]

        print(f"\n{'='*70}")
        print("Fixed PyTree Structure Optimization Analysis")
        print(f"{'='*70}")

        # Measure varying architecture approach
        varying_times = []
        for arch in varying_architectures:
            model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
            params, static = eqx.partition(model, eqx.is_array)

            x = jax.random.normal(key, (batch_size, arch[0]))
            y = jax.random.normal(key, (batch_size,))
            exp_x = jax.random.normal(key, (batch_size, arch[0]))
            exp_y = jax.random.normal(key, (batch_size,))
            deltax = jax.random.normal(key, (batch_size, arch[0])) * 0.01

            start = time.time()
            grad, _ = _hamiltonian_core_mse_standard(
                params, static, x, y, exp_x, exp_y, deltax,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
            varying_times.append(time.time() - start)

        total_varying = sum(varying_times)

        # Strategy 2: Fixed large architecture (potential optimization)
        # Use largest architecture and train same number of times
        fixed_arch = [1, 120, 120, 120, 1]  # Largest
        fixed_model = MLP(sizes=fixed_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
        params_fixed, static_fixed = eqx.partition(fixed_model, eqx.is_array)

        x_fixed = jax.random.normal(key, (batch_size, fixed_arch[0]))
        y_fixed = jax.random.normal(key, (batch_size,))
        exp_x_fixed = jax.random.normal(key, (batch_size, fixed_arch[0]))
        exp_y_fixed = jax.random.normal(key, (batch_size,))
        deltax_fixed = jax.random.normal(key, (batch_size, fixed_arch[0])) * 0.01

        # First call = JIT compilation
        start = time.time()
        grad, _ = _hamiltonian_core_mse_standard(
            params_fixed, static_fixed, x_fixed, y_fixed, exp_x_fixed, exp_y_fixed, deltax_fixed,
            jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
        fixed_jit_time = time.time() - start

        # Subsequent calls (no recompilation)
        fixed_times = []
        for _ in range(len(varying_architectures) - 1):
            start = time.time()
            grad, _ = _hamiltonian_core_mse_standard(
                params_fixed, static_fixed, x_fixed, y_fixed, exp_x_fixed, exp_y_fixed, deltax_fixed,
                jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                jnp.array(1000.0), jnp.array(1.0)
            )
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)
            fixed_times.append(time.time() - start)

        total_fixed = fixed_jit_time + sum(fixed_times)

        print(f"\nVarying Architectures (current approach):")
        for i, (arch, t) in enumerate(zip(varying_architectures, varying_times)):
            print(f"  Arch {i}: {arch[1]}-{arch[2]}-{arch[3]} -> {t*1000:.1f}ms (includes JIT)")
        print(f"  Total: {total_varying*1000:.1f}ms")

        print(f"\nFixed Large Architecture (optimization):")
        print(f"  JIT (once):  {fixed_jit_time*1000:.1f}ms")
        print(f"  Subsequent:  {sum(fixed_times)*1000:.1f}ms ({len(fixed_times)} calls)")
        print(f"  Total: {total_fixed*1000:.1f}ms")

        speedup = total_varying / total_fixed if total_fixed > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time Saved: {(total_varying - total_fixed)*1000:.1f}ms ({(1 - total_fixed/total_varying)*100:.1f}%)")
        print(f"{'='*70}")

    def test_parallel_candidate_evaluation(self, report_dir):
        """Test if candidates can be evaluated in parallel using vmap/pmap.

        Architecture search currently evaluates candidates sequentially.
        This tests whether JAX's vectorization can help.
        """
        from cl.models.mlp import MLP

        print(f"\n{'='*70}")
        print("Parallel Candidate Evaluation Analysis")
        print(f"{'='*70}")

        # Create multiple models and evaluate in parallel using vmap
        # Note: This is a conceptual test - actual implementation would need
        # batched model application which isn't trivial with equinox

        # For now, measure baseline sequential performance
        base_arch = [1, 75, 75, 75, 1]
        num_candidates = 4
        batch_size = 64
        key = jax.random.PRNGKey(0)

        # Sequential evaluation (current)
        sequential_times = []
        for i in range(num_candidates):
            arch = [base_arch[0]] + [h + 15*i for h in base_arch[1:-1]] + [base_arch[-1]]
            model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)

            x = jax.random.normal(key, (batch_size, arch[0]))

            start = time.time()
            _ = model(x)
            jax.block_until_ready(_)
            sequential_times.append(time.time() - start)

        total_sequential = sum(sequential_times)

        print(f"Sequential evaluation: {total_sequential*1000:.1f}ms for {num_candidates} candidates")
        print(f"Average per candidate: {np.mean(sequential_times)*1000:.2f}ms")

        # Note: True parallel evaluation would require:
        # 1. Batched model application (vectorizing over model parameters)
        # 2. Or multi-GPU pmap distribution
        # Both are non-trivial with Equinox's Module structure

        print("\nPotential Optimizations:")
        print("  1. Multi-GPU pmap: Evaluate N candidates on N GPUs")
        print("  2. Batched vmap: Vectorize over parameter variations (complex)")
        print("  3. Async evaluation: Overlap JIT compilation with training")
        print(f"{'='*70}")

    def test_progressive_search_benefit(self, report_dir):
        """Test progressive/hierarchical search vs exhaustive search.

        Instead of evaluating all candidates equally, use progressive refinement:
        - Quick evaluation with few epochs to eliminate bad candidates
        - Full evaluation only for promising candidates
        """
        from cl.models.mlp import MLP
        from cl.core.hamiltonian import _hamiltonian_core_mse_standard

        print(f"\n{'='*70}")
        print("Progressive Search Analysis")
        print(f"{'='*70}")

        base_arch = [1, 75, 75, 75, 1]
        num_candidates = 8
        batch_size = 64
        key = jax.random.PRNGKey(0)

        # Simulate exhaustive search (current approach)
        # All candidates get full evaluation
        exhaustive_epochs = 10
        batches_per_epoch = 10

        exhaustive_time = 0
        exhaustive_evals = []

        for i in range(num_candidates):
            arch = [base_arch[0]] + [h + 15*i for h in base_arch[1:-1]] + [base_arch[-1]]
            model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
            params, static = eqx.partition(model, eqx.is_array)

            x = jax.random.normal(key, (batch_size, arch[0]))
            y = jax.random.normal(key, (batch_size,))
            exp_x = jax.random.normal(key, (batch_size, arch[0]))
            exp_y = jax.random.normal(key, (batch_size,))
            deltax = jax.random.normal(key, (batch_size, arch[0])) * 0.01

            start = time.time()
            # JIT + training
            for epoch in range(exhaustive_epochs):
                for batch in range(batches_per_epoch):
                    grad, losses = _hamiltonian_core_mse_standard(
                        params, static, x, y, exp_x, exp_y, deltax,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        jnp.array(1000.0), jnp.array(1.0)
                    )
                    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

            eval_time = time.time() - start
            exhaustive_time += eval_time
            exhaustive_evals.append(eval_time)

        # Simulate progressive search
        # Phase 1: Quick evaluation (2 epochs) for all candidates
        # Phase 2: Full evaluation (8 epochs) for top 25%
        quick_epochs = 2
        full_epochs = 8

        progressive_time = 0
        quick_times = []
        full_times = []

        # Quick evaluation for all
        for i in range(num_candidates):
            arch = [base_arch[0]] + [h + 15*i for h in base_arch[1:-1]] + [base_arch[-1]]
            model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
            params, static = eqx.partition(model, eqx.is_array)

            x = jax.random.normal(key, (batch_size, arch[0]))
            y = jax.random.normal(key, (batch_size,))
            exp_x = jax.random.normal(key, (batch_size, arch[0]))
            exp_y = jax.random.normal(key, (batch_size,))
            deltax = jax.random.normal(key, (batch_size, arch[0])) * 0.01

            start = time.time()
            for epoch in range(quick_epochs):
                for batch in range(batches_per_epoch):
                    grad, losses = _hamiltonian_core_mse_standard(
                        params, static, x, y, exp_x, exp_y, deltax,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        jnp.array(1000.0), jnp.array(1.0)
                    )
                    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

            quick_time = time.time() - start
            quick_times.append(quick_time)
            progressive_time += quick_time

        # Full evaluation for top 25% (2 candidates)
        top_candidates = num_candidates // 4
        for i in range(top_candidates):
            arch = [base_arch[0]] + [h + 15*i for h in base_arch[1:-1]] + [base_arch[-1]]
            model = MLP(sizes=arch, key=jax.random.PRNGKey(42), awb_enabled=True)
            params, static = eqx.partition(model, eqx.is_array)

            x = jax.random.normal(key, (batch_size, arch[0]))
            y = jax.random.normal(key, (batch_size,))
            exp_x = jax.random.normal(key, (batch_size, arch[0]))
            exp_y = jax.random.normal(key, (batch_size,))
            deltax = jax.random.normal(key, (batch_size, arch[0])) * 0.01

            start = time.time()
            # Note: JIT already cached from quick phase
            for epoch in range(full_epochs):
                for batch in range(batches_per_epoch):
                    grad, losses = _hamiltonian_core_mse_standard(
                        params, static, x, y, exp_x, exp_y, deltax,
                        jnp.array(0.01), jnp.array(0.98), jnp.array(0.1),
                        jnp.array(1000.0), jnp.array(1.0)
                    )
                    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad)

            full_time = time.time() - start
            full_times.append(full_time)
            progressive_time += full_time

        print(f"Exhaustive Search ({num_candidates} candidates × {exhaustive_epochs} epochs):")
        print(f"  Total time: {exhaustive_time:.2f}s")
        print(f"  Avg per candidate: {np.mean(exhaustive_evals)*1000:.1f}ms")

        print(f"\nProgressive Search:")
        print(f"  Phase 1: {num_candidates} candidates × {quick_epochs} epochs = {sum(quick_times):.2f}s")
        print(f"  Phase 2: {top_candidates} candidates × {full_epochs} epochs = {sum(full_times):.2f}s")
        print(f"  Total time: {progressive_time:.2f}s")

        speedup = exhaustive_time / progressive_time if progressive_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time Saved: {exhaustive_time - progressive_time:.2f}s ({(1 - progressive_time/exhaustive_time)*100:.1f}%)")
        print(f"{'='*70}")


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def format_memory(mb: float) -> str:
    """Format memory in appropriate units."""
    if mb < 1:
        return f"{mb * 1024:.1f}KB"
    elif mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb / 1024:.2f}GB"


def display_gpu_report(report: dict, verbose: bool = False):
    """Display a GPU profiling report in formatted output.

    Args:
        report: Dictionary containing the profiling report
        verbose: If True, show detailed metrics for each phase
    """
    # Header
    print(f"\n{'='*70}")
    print(f"{'AWB GPU Profiling Report':^70}")
    print(f"{'='*70}")

    # Basic info
    print(f"\nTest Name: {report.get('test_name', 'Unknown')}")
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"JAX Backend: {report.get('jax_backend', 'N/A')}")
    print(f"Devices: {', '.join(report.get('jax_devices', ['N/A']))}")
    print(f"NVIDIA SMI Available: {report.get('nvidia_smi_available', False)}")
    print(f"Total Time: {format_time(report.get('total_time_seconds', 0))}")

    # Configuration (if verbose)
    if verbose and 'config' in report:
        print(f"\n{'-'*50}")
        print("Configuration")
        print(f"{'-'*50}")
        config = report['config']
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")

    # Phase breakdown
    phases = report.get('phases', [])
    if phases:
        print(f"\n{'-'*50}")
        print("Phase Breakdown")
        print(f"{'-'*50}")

        # Calculate total for percentage
        total_time = sum(p.get('total_time_seconds', 0) for p in phases)

        # Header row
        print(f"{'Phase':<35} {'Time':>10} {'%':>6} {'Iter':>6} {'Per Iter':>10}")
        print("-" * 70)

        for phase in phases:
            name = phase.get('name', 'Unknown')[:34]
            time = phase.get('total_time_seconds', 0)
            pct = (time / total_time * 100) if total_time > 0 else 0
            iterations = phase.get('num_iterations', 0)
            per_iter = phase.get('time_per_iteration_ms', 0)

            print(f"{name:<35} {format_time(time):>10} {pct:>5.1f}% {iterations:>6} {per_iter:>9.2f}ms")

        print("-" * 70)
        print(f"{'TOTAL':<35} {format_time(total_time):>10} {'100.0%':>6}")

    # Detailed metrics (if verbose)
    if verbose and phases:
        print(f"\n{'-'*50}")
        print("Detailed Phase Metrics")
        print(f"{'-'*50}")
        for phase in phases:
            print(f"\n  {phase.get('name', 'Unknown')}:")
            print(f"    Total Time: {format_time(phase.get('total_time_seconds', 0))}")
            print(f"    Iterations: {phase.get('num_iterations', 0)}")
            print(f"    Time per Iteration: {phase.get('time_per_iteration_ms', 0):.3f}ms")

            if phase.get('jit_compile_time_seconds', 0) > 0:
                print(f"    JIT Compile Time: {format_time(phase.get('jit_compile_time_seconds', 0))}")
            if phase.get('hamiltonian_time_ms', 0) > 0:
                print(f"    Hamiltonian Time: {phase.get('hamiltonian_time_ms', 0):.3f}ms")
            if phase.get('optimizer_time_ms', 0) > 0:
                print(f"    Optimizer Time: {phase.get('optimizer_time_ms', 0):.3f}ms")
            if phase.get('gpu_utilization_mean', 0) > 0:
                print(f"    GPU Utilization: {phase.get('gpu_utilization_mean', 0):.1f}% (max: {phase.get('gpu_utilization_max', 0):.1f}%)")
            if phase.get('gpu_memory_mean_mb', 0) > 0:
                print(f"    GPU Memory: {format_memory(phase.get('gpu_memory_mean_mb', 0))}")

    # Bottleneck analysis
    bottlenecks = report.get('bottleneck_analysis', [])
    if bottlenecks:
        print(f"\n{'-'*50}")
        print("Bottleneck Analysis (Sorted by Time)")
        print(f"{'-'*50}")

        # Header row
        print(f"{'Rank':<5} {'Phase':<35} {'Time':>10} {'%':>7} {'Status':>10}")
        print("-" * 70)

        for i, bn in enumerate(bottlenecks, 1):
            name = bn.get('phase', 'Unknown')[:34]
            time = bn.get('time_seconds', 0)
            pct = bn.get('percentage', 0)
            is_bn = "BOTTLENECK" if bn.get('is_bottleneck', False) else ""

            print(f"{i:<5} {name:<35} {format_time(time):>10} {pct:>6.1f}% {is_bn:>10}")

    # Summary
    print(f"\n{'-'*50}")
    print("Summary")
    print(f"{'-'*50}")
    if bottlenecks:
        top_bottleneck = bottlenecks[0] if bottlenecks else None
        if top_bottleneck:
            print(f"  Slowest Phase: {top_bottleneck.get('phase', 'N/A')} ({top_bottleneck.get('percentage', 0):.1f}%)")

    gpu_phases = [p for p in phases if p.get('gpu_utilization_mean', 0) > 0]
    if gpu_phases:
        avg_gpu = sum(p.get('gpu_utilization_mean', 0) for p in gpu_phases) / len(gpu_phases)
        print(f"  Average GPU Utilization: {avg_gpu:.1f}%")
    else:
        print(f"  GPU Utilization: Not measured (running on CPU)")

    print(f"{'='*70}\n")


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

        # Use the new display function for detailed output
        display_gpu_report(report, verbose=False)


# Command-line interface for displaying reports
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Display AWB GPU Profiling Reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_awb_gpu_profiling.py                      # Display latest report
  python tests/test_awb_gpu_profiling.py --verbose            # With detailed metrics
  python tests/test_awb_gpu_profiling.py --file report.json   # Display specific file
  python tests/test_awb_gpu_profiling.py --all                # Display all reports
        """
    )
    parser.add_argument('--file', '-f', type=str, help='Path to specific report file')
    parser.add_argument('--all', '-a', action='store_true', help='Display all reports')
    parser.add_argument('--dir', '-d', type=str, default='tests/gpu_reports',
                        help='Report directory (default: tests/gpu_reports)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed metrics')

    args = parser.parse_args()

    report_dir = Path(args.dir)

    if args.file:
        # Display specific file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Report file not found: {args.file}")
            exit(1)
        with open(file_path) as f:
            report = json.load(f)
        display_gpu_report(report, verbose=args.verbose)

    elif args.all:
        # Display all reports
        report_files = sorted(report_dir.glob('awb_*.json'))
        if not report_files:
            print(f"No AWB profiling reports found in {report_dir}/")
            exit(1)
        print(f"Found {len(report_files)} report(s)")
        for f in report_files:
            with open(f) as fp:
                report = json.load(fp)
            display_gpu_report(report, verbose=args.verbose)

    else:
        # Display latest report
        report_files = sorted(report_dir.glob('awb_*.json'))
        if not report_files:
            print(f"No AWB profiling reports found in {report_dir}/")
            print("Run AWB profiling tests first with:")
            print("  pytest tests/test_awb_gpu_profiling.py -m gpu -v -s")
            exit(1)
        latest = report_files[-1]
        print(f"Loading latest report: {latest}")
        with open(latest) as f:
            report = json.load(f)
        display_gpu_report(report, verbose=args.verbose)
