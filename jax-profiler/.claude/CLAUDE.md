# CLAUDE.md - JAX Profiler Toolkit

This document provides guidance for Claude Code when working with this profiling toolkit.

## Purpose

This is an **external, reusable profiling toolkit** for JAX applications. It was extracted from the ContLearn repository to:
1. Keep profiling infrastructure separate from research code
2. Use industry-standard tools (TensorBoard, Nsight, nvidia-smi)
3. Provide stable, well-documented profiling that doesn't change frequently

## Key Principles

### 1. Use Standard Tools First
Always prefer industry-standard tools over custom profiling:
- **TensorBoard** for XLA traces and memory profiling
- **Nsight Systems** for GPU kernel analysis
- **nvidia-smi** for utilization monitoring
- **JAX built-in** profiler for compilation tracking

### 2. Custom Hooks for Domain-Specific Only
Use custom hooks only when standard tools can't provide the insight:
- Phase boundaries in pipelines (e.g., AWB phases)
- Domain-specific metrics (e.g., continual learning BWT/FWT)
- Semantic groupings (e.g., "hamiltonian" vs just "gradient")

### 3. Zero Overhead When Disabled
All hooks should have negligible overhead when profiling is off:
```python
registry = HookRegistry(enabled=False)  # Default
```

### 4. External Integration
Don't modify core training code for profiling. Wrap externally:
```python
# GOOD - External wrapper
with TensorBoardProfiler(log_dir):
    train_model(config)

# BAD - Embedded profiling
def train_step():
    start = time.time()  # Don't do this
```

## Directory Structure

```
jax-profiler/
├── jax_profiler/
│   ├── __init__.py         # Main exports
│   ├── cli.py              # Command-line interface
│   ├── standard/           # Industry-standard wrappers
│   │   ├── tensorboard.py  # JAX TensorBoard profiler
│   │   ├── gpu_monitor.py  # nvidia-smi monitoring
│   │   └── nsight.py       # Nsight Systems wrapper
│   ├── hooks/              # Custom hook infrastructure
│   │   ├── base.py         # Hook base classes (TimingHook, MetricHook, etc.)
│   │   └── registry.py     # Hook registration and triggering
│   └── adapters/           # Project-specific adapters
│       └── contlearn.py    # ContLearn-specific hooks
├── .claude/
│   └── CLAUDE.md           # This file
├── README.md               # User documentation
└── pyproject.toml          # Package configuration
```

## Common Tasks

### Adding a New Standard Tool Wrapper

1. Create file in `jax_profiler/standard/`
2. Follow existing patterns (context manager, start/stop methods)
3. Export from `jax_profiler/standard/__init__.py`
4. Add CLI command if needed in `cli.py`

### Adding a New Hook Type

1. Inherit from `Hook` base class in `hooks/base.py`
2. Implement: `before()`, `after()`, `get_stats()`, `reset()`
3. Export from `jax_profiler/hooks/__init__.py`

### Adding a Project Adapter

1. Create file in `jax_profiler/adapters/`
2. Use `HookRegistry` and standard hooks
3. Provide project-specific hook points and metrics
4. Export from `jax_profiler/adapters/__init__.py`

## Usage Examples

### Quick GPU Check
```bash
jax-profile gpu
```

### Monitor During Training
```python
from jax_profiler import GPUMonitor

monitor = GPUMonitor(interval=0.5)
monitor.start()
# ... training ...
stats = monitor.stop()
print(f"Avg GPU: {stats.utilization_mean:.1f}%")
```

### TensorBoard Profiling
```python
from jax_profiler import TensorBoardProfiler

with TensorBoardProfiler("/tmp/traces"):
    # ... training code ...
# View: tensorboard --logdir=/tmp/traces
```

### Custom Hooks for ContLearn
```python
from jax_profiler.adapters import ContLearnProfiler

profiler = ContLearnProfiler()
profiler.start()
profiler.on_task_start(0)
profiler.on_phase_start("ab_training")
# ... AWB training ...
profiler.on_phase_end("ab_training")
profiler.print_summary()
```

## Testing

```bash
# From jax-profiler directory
pip install -e ".[dev]"
pytest tests/
```

## Integration with ContLearn

The ContLearn repository should NOT contain profiling code. To profile ContLearn:

```bash
# Option 1: CLI monitoring
jax-profile monitor -d 300 &  # Background monitor
python run.py config.json
jax-profile stop

# Option 2: TensorBoard
jax-profile tensorboard /tmp/cl_traces python run.py config.json

# Option 3: Python wrapper
from jax_profiler import TensorBoardProfiler, GPUMonitor
from jax_profiler.adapters import ContLearnProfiler

monitor = GPUMonitor()
monitor.start()

with TensorBoardProfiler("/tmp/traces"):
    # Import and run ContLearn
    import sys
    sys.path.insert(0, "/path/to/ContLearn")
    from cl.runners import train_model
    train_model(config)

stats = monitor.stop()
```

## DO NOT

1. **DO NOT** add profiling code to ContLearn core files
2. **DO NOT** create custom timing code when standard tools work
3. **DO NOT** add debug prints to production code
4. **DO NOT** make hooks enabled by default

## References

- JAX Profiling: https://jax.readthedocs.io/en/latest/profiling.html
- TensorBoard Profiler: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- nvidia-smi: https://developer.nvidia.com/nvidia-system-management-interface
