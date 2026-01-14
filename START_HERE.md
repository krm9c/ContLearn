# START HERE - Claude Context Guide

This file provides orientation for Claude Code sessions working on the ContLearn codebase.

---

## Quick Context Loading Order

**Read these files in order for full context:**

1. **Project Overview**: `.claude/CLAUDE.md`
   - Directory structure, core architecture, config parameters
   - Entry point flow, trainer mixins, AWB pipeline
   - Testing commands, benchmark results

2. **Performance Optimization**: `.claude/profiling_context.md`
   - **CRITICAL**: Read before any performance work
   - Safe vs unsafe optimization areas
   - Completed optimizations, known issues
   - XLA flags, fused train step, JAX prefetching

3. **Profiling Toolkit**: `jax-profiler/.claude/CLAUDE.md`
   - External profiling infrastructure
   - GPU monitoring, AWB pipeline hooks
   - Latest benchmark results (A40, H200 pending)
   - Usage examples

---

## All Context Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `.claude/CLAUDE.md` | Main project guide | Always first |
| `.claude/profiling_context.md` | Performance optimization | Before optimization work |
| `.claude/awb_refactoring_context.md` | AWB architecture details | When modifying AWB |
| `.claude/LAYER_AWB_ARCHITECTURE.md` | Layer-level AWB design | When modifying AWB models |
| `.claude/RECORDING_RESTRUCTURE_PLAN.md` | Recording mixin refactor | When modifying recording |
| `jax-profiler/.claude/CLAUDE.md` | Profiling toolkit guide | When profiling |
| `../CLAUDE.md` (parent) | Code organization preferences | For style guidance |

---

## Critical Rules

### DO NOT MODIFY (breaks accuracy)
- `src/cl/core/losses.py` - Loss functions
- `src/cl/core/hamiltonian.py` - Gradient computation
- `src/cl/core/awb.py` - AWB weight transformations

### SAFE TO MODIFY
- Data loading (`datasets/`)
- Logging/recording (`core/recording.py`)
- Config defaults (`config/constants.py`)
- Profiling (`core/profiling.py`)

---

## Current Branch: `profiling`

This branch focuses on GPU profiling and performance optimization.

### Key Files
- `runs__/experiments/profiling/submit_profiling_benchmark.slurm` - H200 benchmark script
- `runs__/configs/mnist_condition1_profiling.json` - Baseline profiling config
- `runs__/configs/mnist_condition4_profiling.json` - AWB profiling config
- `jax-profiler/mnist.json` - Latest A40 profiling results

### Pending Work
- H200 profiling benchmarks (SLURM script ready)
- Compare A40 vs H200 GPU utilization

---

## Quick Commands

```bash
# Run experiment
python run.py kkt_run/configs/mnist_condition1_baseline.json

# Run tests
pytest -m unit -v        # Fast (~30 sec)
pytest -m training -v    # Full (~5 min)

# Monitor GPU
watch -n 0.5 nvidia-smi

# Profile AWB
python jax-profiler/run_awb_profile.py --quick

# Submit to KKT cluster
sbatch runs__/experiments/profiling/submit_profiling_benchmark.slurm
```

---

## Project Structure Overview

```
ContLearn/
├── .claude/                  # Claude context files (READ FIRST)
├── src/cl/                   # Core source code
│   ├── core/                 # Trainer mixins, profiling
│   ├── models/               # MLP, CNN, GCN
│   ├── datasets/             # Data loaders
│   └── config/               # Constants, params
├── runs__/                   # Experiment configs and scripts
│   ├── configs/              # 20 production + profiling configs
│   └── experiments/          # SLURM scripts
├── jax-profiler/             # External profiling toolkit
├── tests/                    # Test suite
└── run.py                    # Main entry point
```

---

## AWB Pipeline Overview

When `awb_enabled: true`, training follows 7 phases:
1. **preliminary** - Initial training
2. **arch_decision** - Decide if architecture change needed
3. **arch_search** - Evaluate architectures
4. **ab_training** - Train A/B matrices (PRIMARY BOTTLENECK)
5. **v_transform** - Compute V = A @ W @ B.T
6. **v_warmup** - Warmup with new weights
7. **v_training** - Final training

**Key insight**: AWB is 58x slower than baseline due to `A @ W @ B.T` computation inside gradient.

---

## Latest Benchmarks (A40)

| Condition | Time | GPU Util | Notes |
|-----------|------|----------|-------|
| 1 (Baseline) | 7.9s | 71.4% | Good |
| 4 (AWB Full) | 460s | 5.5% | A/B training bottleneck |

Full MNIST AWB (10 tasks, 200 epochs): 2.9 hours, 67.8% GPU util

---

**Last Updated**: 2026-01-14
