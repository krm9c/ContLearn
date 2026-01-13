# JAX Profiler

Industry-standard profiling toolkit for JAX applications.

## Overview

This toolkit provides a **stable, external profiling infrastructure** that wraps industry-standard tools:

- **TensorBoard Profiler** - JAX's built-in profiler with XLA trace visualization
- **NVIDIA Nsight Systems** - GPU kernel-level profiling
- **nvidia-smi monitoring** - Real-time GPU utilization tracking
- **Custom hooks** - Domain-specific profiling for ML frameworks

## Installation

```bash
pip install -e /path/to/jax-profiler

# With TensorBoard support
pip install -e /path/to/jax-profiler[tensorboard]
```

## Quick Start

### Command Line

```bash
# Show GPU status
jax-profile gpu

# Monitor GPU for 60 seconds
jax-profile monitor -d 60 -o stats.json

# Profile with TensorBoard
jax-profile tensorboard /tmp/traces train.py config.json

# Profile with Nsight Systems
jax-profile nsight train.py config.json --stats
```

### Python API

```python
from jax_profiler import TensorBoardProfiler, GPUMonitor

# TensorBoard profiling
with TensorBoardProfiler("/tmp/traces"):
    # ... training code ...
# View: tensorboard --logdir=/tmp/traces

# GPU monitoring
monitor = GPUMonitor(interval=0.5)
monitor.start()
# ... training code ...
stats = monitor.stop()
print(f"GPU utilization: {stats.utilization_mean:.1f}%")
```

## Standard Tools

### 1. TensorBoard Profiler

JAX's built-in profiler provides:
- XLA compilation traces
- Memory allocation tracking
- Kernel execution timing
- Device placement visualization

```python
from jax_profiler import TensorBoardProfiler

# Context manager
with TensorBoardProfiler("/tmp/traces"):
    # Training code here
    pass

# Or manual control
profiler = TensorBoardProfiler("/tmp/traces")
profiler.start()
# ... code ...
profiler.stop()

# View results
# tensorboard --logdir=/tmp/traces
```

**Enable JIT compilation logging:**
```python
from jax_profiler.standard.tensorboard import enable_compile_logging
enable_compile_logging()  # Prints on each JIT compilation
```

### 2. GPU Monitor (nvidia-smi)

Real-time GPU utilization monitoring:

```python
from jax_profiler import GPUMonitor, get_gpu_stats

# One-time snapshot
stats = get_gpu_stats()
print(f"GPU: {stats['name']}")
print(f"Utilization: {stats['utilization']}%")
print(f"Memory: {stats['memory_used_mb']}/{stats['memory_total_mb']}MB")

# Continuous monitoring during training
monitor = GPUMonitor(interval=0.5)
monitor.start()

# ... training loop ...

stats = monitor.stop()
print(f"Avg utilization: {stats.utilization_mean:.1f}%")
print(f"Peak memory: {stats.memory_max_mb:.0f}MB")
```

### 3. Nsight Systems

GPU kernel-level profiling:

```bash
# Basic profiling
nsys profile -o report python train.py

# View statistics
nsys stats report.nsys-rep

# Open GUI (if available)
nsys-ui report.nsys-rep
```

```python
from jax_profiler import NsightWrapper

nsight = NsightWrapper(output_dir="/tmp/nsight")
report = nsight.profile_script("train.py", args=["config.json"])
stats = nsight.generate_stats(report)
print(stats)
```

## Custom Hooks

For domain-specific profiling without modifying core code:

```python
from jax_profiler.hooks import HookRegistry, TimingHook, MetricHook

# Create registry
registry = HookRegistry()

# Register hooks
registry.register("train_step", TimingHook("train_step"))
registry.register("eval", TimingHook("eval"))
registry.register("losses", MetricHook("losses"))

# Use in training loop
for batch in dataloader:
    with registry.hook("train_step"):
        loss = train_step(batch)
    registry.trigger("losses", "after", {"loss": loss})

# Get statistics
registry.print_summary()
```

### AWB Pipeline Hooks

The AWB (Adaptive Weight Basis) pipeline causes poor GPU utilization due to:
- **A/B matrix training**: Computes A @ W @ B.T inside gradient (cannot be cached)
- **Architecture search**: Evaluates multiple configurations
- **JIT recompilation**: Mode switches between standard/AWB training

```python
from jax_profiler.hooks import AWBPipelineHooks

# Create hooks for AWB pipeline profiling
hooks = AWBPipelineHooks(gpu_monitoring=True)
hooks.start()

# These would be called from within the AWB pipeline
hooks.on_task_start(task_id=1)
hooks.on_phase_start("preliminary")
# ... preliminary training ...
hooks.on_phase_end("preliminary")

hooks.on_phase_start("ab_training")
# ... A/B training (the bottleneck) ...
hooks.on_hamiltonian_call(is_awb=True, duration_ms=150.0)
hooks.on_phase_end("ab_training")

# Print detailed analysis
hooks.print_analysis()
```

**Profile AWB pipeline externally (no code modification):**

```bash
# From jax-profiler directory
python scripts/profile_awb.py /path/to/ContLearn/tests/training/configs/mnist_awb.json

# Or with TensorBoard
jax-profile tensorboard /tmp/awb_traces python /path/to/ContLearn/run.py config.json
```

### ContLearn Adapter

Specialized adapter for the ContLearn continual learning framework:

```python
from jax_profiler.adapters import ContLearnProfiler

profiler = ContLearnProfiler(gpu_monitoring=True)
profiler.start()

# Training calls these
profiler.on_task_start(task_id=0)
profiler.on_phase_start("preliminary")
# ... training ...
profiler.on_phase_end("preliminary")
profiler.on_task_end(task_id=0, test_accuracy=0.95)

# Get report
report = profiler.stop()
profiler.print_summary()
```

## Integration Guide

### Minimal Integration (Recommended)

Add profiling without modifying core training code:

```python
# profile_training.py
import sys
sys.path.insert(0, "/path/to/your/project")

from jax_profiler import TensorBoardProfiler, GPUMonitor

# Start monitoring
monitor = GPUMonitor(interval=0.5)
monitor.start()

# Import and run your training
with TensorBoardProfiler("/tmp/traces"):
    from your_project import train_model
    train_model(config)

# Report
stats = monitor.stop()
print(f"GPU utilization: {stats.utilization_mean:.1f}%")
```

### Hook-Based Integration

For more detailed profiling:

```python
from jax_profiler.hooks import HookRegistry, TimingHook

# Global registry
_registry = HookRegistry(enabled=False)  # Disabled by default

def enable_profiling():
    _registry.enabled = True
    _registry.register("hamiltonian", TimingHook())
    _registry.register("optimizer", TimingHook())

# In your training loop (minimal changes)
def train_step(params, batch):
    with _registry.hook("hamiltonian"):
        grad, loss = compute_gradient(params, batch)

    with _registry.hook("optimizer"):
        params = update_params(params, grad)

    return params, loss
```

## Best Practices

### 1. Use Standard Tools First

Before adding custom profiling:
1. Run `jax-profile gpu` to check GPU status
2. Use `jax-profile monitor` during training
3. Profile with TensorBoard for XLA traces
4. Use Nsight for kernel-level analysis

### 2. Keep Profiling External

Don't embed profiling in core code:
```python
# BAD - profiling embedded in training
def train_step(batch):
    start = time.time()  # Don't do this
    ...

# GOOD - external profiling wrapper
with registry.hook("train_step"):
    train_step(batch)
```

### 3. Disable by Default

Hooks should have zero overhead when disabled:
```python
registry = HookRegistry(enabled=False)  # Default off
# Enable only when profiling
registry.enabled = True
```

## File Structure

```
jax-profiler/
├── pyproject.toml
├── README.md
├── jax_profiler/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── standard/           # Industry-standard tool wrappers
│   │   ├── tensorboard.py  # JAX TensorBoard profiler
│   │   ├── gpu_monitor.py  # nvidia-smi monitoring
│   │   └── nsight.py       # Nsight Systems wrapper
│   ├── hooks/              # Custom hook infrastructure
│   │   ├── base.py         # Hook base classes
│   │   └── registry.py     # Hook registry
│   └── adapters/           # Project-specific adapters
│       └── contlearn.py    # ContLearn adapter
└── scripts/
    └── profile.py          # Profiling scripts
```

## Requirements

- Python >= 3.9
- JAX >= 0.4.0
- NVIDIA GPU with CUDA (for GPU profiling)
- nvidia-smi (for GPU monitoring)
- Nsight Systems (optional, for kernel profiling)

## License

MIT
