"""
JAX Profiler - Industry-standard profiling toolkit for JAX applications.

This toolkit provides:
1. Standard profiling tool wrappers (TensorBoard, Nsight, nvidia-smi)
2. Custom hook infrastructure for domain-specific profiling
3. Adapters for specific projects (e.g., ContLearn)

Usage:
    # Standard profiling
    from jax_profiler import TensorBoardProfiler, GPUMonitor

    with TensorBoardProfiler("/tmp/traces"):
        # ... training code ...

    # GPU monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    # ... training code ...
    stats = monitor.stop()

    # Custom hooks
    from jax_profiler.hooks import HookRegistry, TimingHook

    registry = HookRegistry()
    registry.register("train_step", TimingHook())
"""

__version__ = "0.1.0"

from .standard.tensorboard import TensorBoardProfiler
from .standard.gpu_monitor import GPUMonitor, get_gpu_stats
from .standard.nsight import NsightWrapper
from .hooks.base import Hook, TimingHook
from .hooks.registry import HookRegistry
from .hooks.awb import AWBPipelineHooks

__all__ = [
    "TensorBoardProfiler",
    "GPUMonitor",
    "get_gpu_stats",
    "NsightWrapper",
    "Hook",
    "TimingHook",
    "HookRegistry",
    "AWBPipelineHooks",
]
