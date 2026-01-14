# CLAUDE.md - JAX Profiler Toolkit

This document provides guidance for Claude Code when working with this profiling toolkit.

## Purpose

This is an **external, reusable profiling toolkit** for JAX applications. It was extracted from the ContLearn repository to:
1. Keep profiling infrastructure separate from research code
2. Use industry-standard tools (TensorBoard, Nsight, nvidia-smi)
3. Provide non-intrusive profiling via hooks (no core code modifications)

---

## Directory Structure

```
jax-profiler/
├── jax_profiler/
│   ├── __init__.py              # Main exports (GPUMonitor, TensorBoardProfiler, etc.)
│   ├── cli.py                   # Command-line interface
│   ├── standard/                # Industry-standard tool wrappers
│   │   ├── __init__.py
│   │   ├── tensorboard.py       # JAX TensorBoard profiler wrapper
│   │   ├── gpu_monitor.py       # nvidia-smi background monitoring
│   │   └── nsight.py            # NVIDIA Nsight Systems wrapper
│   ├── hooks/                   # Custom hook infrastructure
│   │   ├── __init__.py
│   │   ├── base.py              # Hook, TimingHook, CounterHook, MetricHook
│   │   ├── registry.py          # HookRegistry for managing multiple hooks
│   │   └── awb.py               # AWBPipelineHooks for ContLearn AWB profiling
│   └── adapters/                # Project-specific adapters
│       ├── __init__.py
│       └── contlearn.py         # ContLearn-specific integration
├── scripts/
│   └── profile_awb.py           # AWB profiling script
├── run_awb_profile.py           # Main profiling entry point
├── mnist.json                   # Latest MNIST profiling results (A40)
├── .claude/
│   └── CLAUDE.md                # This file
└── README.md                    # User documentation
```

---

## Latest Profiling Results

### MNIST AWB (Condition 4) - A40 GPU

**File**: `mnist.json`

| Metric | Value |
|--------|-------|
| Config | `mnist_condition4_awb_full.json` |
| Total Time | 2.9 hours (10,451 sec) |
| GPU | NVIDIA A40 (48GB GDDR6) |

**Configuration:**
- 10 tasks × 200 epochs/task
- Batch size: 512
- AWB preliminary: 1 epoch
- AWB A/B training: 200 epochs

**GPU Statistics:**
| Metric | Value |
|--------|-------|
| Mean Utilization | 67.8% |
| Min/Max | 0% - 100% |
| Std Dev | 25.4% |
| Memory Used | ~34.4 GB |
| Memory Peak | ~34.5 GB |

**Observations:**
1. High GPU utilization variance (std=25.4%) indicates alternating compute-heavy and CPU-bound phases
2. 0% minimum suggests idle periods during data loading or phase transitions
3. 200 A/B training epochs dominates the total runtime

---

## AWB Pipeline Bottleneck Analysis

### AWB 7 Phases (tracked by AWBPipelineHooks)

1. **preliminary** - Initial training to estimate task difficulty
2. **arch_decision** - Compare losses to decide if architecture change needed
3. **arch_search** - Evaluate multiple architectures (expensive)
4. **ab_training** - Train A/B transformation matrices (PRIMARY BOTTLENECK)
5. **v_transform** - Compute V = A @ W @ B.T
6. **v_warmup** - Warmup training with transformed weights
7. **v_training** - Final training with V

### Root Cause of AWB Slowdown

**The primary bottleneck is `ab_training` phase:**
- Computes `A @ W @ B.T` inside gradient computation
- Cannot be cached because gradients flow through A and B matrices
- Each iteration recomputes the full matrix multiplication chain
- JIT compilation helps but the operation is inherently expensive

### Performance Comparison (MNIST, A40)

| Condition | Time | GPU Util | Throughput |
|-----------|------|----------|------------|
| Baseline (Cond 1) | 7.9s | 71.4% | 9,076 samples/sec |
| AWB (Cond 4) | 460s | 5.5% | ~170 samples/sec |

**58x slowdown** due to A/B training overhead.

---

## Key Principles

### 1. Use Standard Tools First
Always prefer industry-standard tools over custom profiling:
- **TensorBoard** for XLA traces and memory profiling
- **Nsight Systems** for GPU kernel analysis
- **nvidia-smi** for utilization monitoring
- **JAX built-in** profiler for compilation tracking

### 2. Custom Hooks for Domain-Specific Only
Use custom hooks only when standard tools can't provide the insight:
- Phase boundaries in AWB pipeline
- Hamiltonian gradient timing (standard vs AWB mode)
- Semantic groupings (e.g., "ab_training" vs generic "gradient")

### 3. Zero Overhead When Disabled
All hooks have negligible overhead when profiling is off:
```python
registry = HookRegistry(enabled=False)  # Default - no overhead
```

### 4. External Integration
Don't modify core training code for profiling. Wrap externally:
```python
# GOOD - External wrapper
monitor = GPUMonitor()
monitor.start()
train_model(config)
stats = monitor.stop()

# BAD - Embedded profiling
def train_step():
    start = time.time()  # Don't add to core code
```

---

## Usage Examples

### Quick GPU Check
```python
from jax_profiler import get_gpu_stats

stats = get_gpu_stats()
print(f"GPU: {stats['name']}, Utilization: {stats['utilization']}%")
```

### Background GPU Monitoring
```python
from jax_profiler import GPUMonitor

monitor = GPUMonitor(interval=0.5)
monitor.start()
# ... training code ...
stats = monitor.stop()
print(f"Avg utilization: {stats.utilization_mean:.1f}%")
print(f"Memory peak: {stats.memory_max_mb:.0f}MB")
```

### AWB Pipeline Profiling (Full)
```bash
# From ContLearn directory
python jax-profiler/run_awb_profile.py --config runs__/configs/mnist_condition4_awb_full.json

# Quick mode (fewer epochs for testing)
python jax-profiler/run_awb_profile.py --quick
```

### AWB Pipeline Hooks (Programmatic)
```python
from jax_profiler.hooks import AWBPipelineHooks

hooks = AWBPipelineHooks()
hooks.start()
hooks.on_task_start(0)

# Profile each AWB phase
hooks.on_phase_start("ab_training")
# ... A/B training code ...
hooks.on_hamiltonian_call(is_awb=True, duration_ms=50.5)
hooks.on_phase_end("ab_training")

# Get analysis
hooks.print_analysis()
report = hooks.get_report()
```

### Hook Registry (Generic)
```python
from jax_profiler.hooks import HookRegistry, TimingHook

registry = HookRegistry(enabled=True)
registry.register("train_step", TimingHook("train_step"))

# In training loop
with registry.hook("train_step"):
    # ... training step ...

registry.print_summary()
```

---

## Integration with ContLearn

### Profiling Config Files

ContLearn has dedicated profiling configs in `runs__/configs/`:
- `mnist_condition1_profiling.json` - Baseline with profiling
- `mnist_condition4_profiling.json` - AWB with profiling

These configs have:
- `profiling_enabled: true`
- `detailed_profiling: true`
- Reduced epochs for faster profiling

### SLURM Script for KKT Cluster

```bash
# Run profiling benchmark on H200
cd ContLearn
sbatch runs__/experiments/profiling/submit_profiling_benchmark.slurm
```

This runs Condition 1 and Condition 4 in parallel on 2 GPUs.

### ContLearn Core Profiling Module

**Location**: `ContLearn/src/cl/core/profiling.py`

Provides:
- `set_xla_flags()` - XLA optimization flags (call BEFORE importing JAX)
- `configure_jax_for_gpu()` - JAX GPU configuration
- `ProfileCollector` - Thread-safe timing collection
- `GPUMonitor` - Background nvidia-smi monitoring
- `@profile`, `@timed` decorators
- `timed_section` context manager

---

## Hardware Benchmarks

### Tested GPUs

| GPU | Memory | Memory BW | Status |
|-----|--------|-----------|--------|
| A40 | 48 GB GDDR6 | 696 GB/s | Benchmarked |
| H200 | 141 GB HBM3e | 4.8 TB/s | Pending |

### Expected H200 Improvements

H200's 7x memory bandwidth should improve:
- Matrix operations (`A @ W @ B.T`)
- Data transfer overhead
- Overall GPU utilization

---

## Recommendations by Utilization

### Low GPU Utilization (<30%)

**Likely causes:**
1. A/B training phase overhead (AWB specific)
2. Data loading bottleneck
3. Excessive JIT recompilation

**Solutions:**
- Reduce `awb_ab_training_epochs`
- Increase `batch_size`
- Use `use_jax_prefetch: true`
- Consider Condition 3 (`awb_skip_transfer: true`)

### Moderate GPU Utilization (30-60%)

**Likely causes:**
1. Frequent test evaluation
2. Suboptimal batch size

**Solutions:**
- Increase `eval_interval` (e.g., 50 instead of 10)
- Tune `batch_size` for GPU memory

### Good GPU Utilization (>60%)

Training is compute-bound. Further optimization requires:
- Algorithm-level changes
- Multi-GPU scaling (`jax.pmap`)

---

## DO NOT

1. **DO NOT** add profiling code to ContLearn core files (`losses.py`, `hamiltonian.py`, `awb.py`)
2. **DO NOT** create custom timing when standard tools work
3. **DO NOT** make hooks enabled by default
4. **DO NOT** modify gradient computation for profiling

---

## Files Quick Reference

| File | Purpose |
|------|---------|
| `gpu_monitor.py` | nvidia-smi wrapper, `GPUMonitor` class |
| `tensorboard.py` | JAX TensorBoard profiler wrapper |
| `base.py` | `Hook`, `TimingHook`, `CounterHook`, `MetricHook` |
| `registry.py` | `HookRegistry` for managing hooks |
| `awb.py` | `AWBPipelineHooks` for AWB phase profiling |
| `run_awb_profile.py` | CLI entry point for AWB profiling |
| `mnist.json` | Latest profiling results |

---

## References

- JAX Profiling: https://jax.readthedocs.io/en/latest/profiling.html
- TensorBoard Profiler: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- nvidia-smi: https://developer.nvidia.com/nvidia-system-management-interface

---

**Last Updated**: 2026-01-14
**Status**: A40 benchmarks complete, H200 benchmarks pending
