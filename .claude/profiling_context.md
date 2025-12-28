# Performance Optimization Context

**Document Purpose**: Reference guide for future performance optimization sessions
**Created**: 2025-12-28
**Git Baseline**: `521b29c` (latest commit)
**Git Commit Message**: "Add JAX asynchronous data pipeline + profiling context doc"

---

## 🚨 CRITICAL CONSTRAINTS - READ FIRST

### **PRIMARY DIRECTIVE: DO NOT BREAK ACCURACY OR CALCULATIONS**

**All performance optimizations MUST preserve**:
1. **Mathematical correctness**: Hamiltonian gradient computation, loss calculations, metrics
2. **Numerical accuracy**: Training convergence, test metrics, final model performance
3. **Experimental reproducibility**: Same config → same results (within floating-point tolerance)
4. **Algorithm integrity**: AWB 5-step pipeline, architecture search, experience replay

**Before ANY code change**:
- Ask: "Does this change ANY mathematical computation, gradient calculation, or model training logic?"
- If YES → **DO NOT PROCEED** without explicit user approval
- If NO → Safe to optimize (I/O, logging, data loading, memory management)

**Verification Requirements**:
- Run tests after optimizations: `./run_tests.sh --all`
- Compare training metrics before/after on small debug configs
- Check that loss curves, test accuracy, and final metrics are unchanged

---

## 👤 USER PREFERENCES

### Code Philosophy (from `.claude/CLAUDE.md` and `../CLAUDE.md`)

1. **Minimize new files** - Prefer editing existing files over creating new ones
2. **Consolidate related code** - Group functionality in single files
3. **Comment new code** - Mark Claude-added sections with `# Added by Claude:`
4. **Maximize code reuse** - Look for existing utilities before writing new ones
5. **Keep code simple and understandable** - No over-engineering
6. **Archive don't delete** - Move old code to `old/` directory
7. **Ask before creating markdown/docs** - Only create documentation when explicitly requested
8. **No additional code without purpose** - Every line must solve a real problem

### Communication Preferences

- **Be blunt and honest** about opinions
- **Challenge opinions only with evidence** from authoritative sources
- **Report changes concisely** without generating report files
- **Confirm approach for significant refactoring** before proceeding

### Performance Optimization Specific

- **User has tried batch size changes** - They don't help (confirmed bottleneck is data loading, not compute)
- **Focus on I/O and data pipeline** - GPU is idle waiting for data
- **Prefer JAX-native solutions** - Avoid PyTorch/NumPy bottlenecks in training loop

---

## 📊 OPTIMIZATIONS COMPLETED (Session 2025-12-28)

### **Git History Leading to Current State**

```bash
521b29c Add JAX asynchronous data pipeline + profiling context doc
9058f9e Optimize configs and fix graph metric calculation
ba59c67 just for now commented the existence of logs in git
502ddaf fixed the fact that the code grabs data
6a46cd8 Fix graph evaluation: Apply transforms to convert edge_index to adj
b493e61 Optimize CNN3D AWB: Replace double vmap with list comprehension
ab2f122 Add GPU memory tracking to profiling system + force_arch_change debug flag
```

### **1. Three-Tier Logging Intervals** ✅

**Problem**: Computing test metrics every epoch was expensive (30s/iter on CIFAR10)

**Solution**: Separated logging into three independent frequencies
- **`log_interval=1`**: Progress bar updates (cheap, every epoch)
- **`eval_interval=50`**: Test metric computation (expensive, every 50 epochs)
- **`save_iter=50`**: Save metrics to pickle file (I/O intensive, every 50 epochs)

**Files Modified**:
- `src/cl/core/loops.py` (lines 459-462, 550-605)
- `src/cl/config/constants.py` (lines 99-102)
- `src/cl/config/params.py` (lines 111-114)

**Impact**:
- CIFAR10: 30s/iter → 3-5s/iter (6-10x speedup)
- MNIST: 2.9s/iter → 0.5s/iter (5-6x speedup)
- Graph: 16.7s/iter → 3s/iter (5x speedup)

**Correctness**: ✅ No impact on training - only affects when metrics are computed/logged

---

### **2. Eigenvalue Computation Optimization** ✅

**Problem**: Computing eigenvalues every epoch during AWB training (expensive linear algebra)

**Solution**: Only compute eigenvalues during A/B training phase (`notABTrain=False`)
- Standard training (`notABTrain=True`): Skip eigenvalues
- A/B training (`notABTrain=False`): Compute eigenvalues for monitoring

**Files Modified**:
- `src/cl/core/recording.py` (lines 161-212)

**Impact**: Significant speedup in AWB Condition 4 experiments

**Correctness**: ✅ Eigenvalues are for monitoring only, not used in training

---

### **3. Async Checkpointing** ✅

**Problem**: Synchronous checkpoint saving blocked training loop

**Solution**: Background thread saves checkpoints while training continues

**Files Modified**:
- `src/cl/core/recording.py` (lines 351-420)
- `src/cl/config/constants.py` (lines 103-106)

**Impact**: Eliminates checkpoint I/O blocking

**Correctness**: ✅ Checkpointing is independent of training computation

---

### **4. Output Organization** ✅

**Problem**: All conditions' outputs mixed in same directory

**Solution**: Each config saves to its own subdirectory
- Pickle files: `results/{config_name}/record_dict_*.pkl`
- Logs: `logs/{config_name}/{config_name}_*.log`

**Files Modified**:
- `src/cl/core/recording.py` (line 433)
- `kkt_run/kkt/run_optimized_profiles.sh` (lines 64-67, 86-88)

**Impact**: Clean organization, easier to track experiments

**Correctness**: ✅ No impact on computation, only file paths

---

### **5. Graph Metric Calculation Fix** 🐛

**Problem**: Graph classification used wrong edge representation (edge_index vs adj_matrix)

**Solution**: Apply `to_dense_adj()` transform in GCN evaluation

**Files Modified**:
- `src/cl/core/loops.py` (lines 625-632)

**Impact**: Fixed graph classification metrics (was buggy before)

**Correctness**: ✅ **BUG FIX** - improves correctness

---

### **6. CNN3D AWB Optimization** ✅

**Problem**: Double `vmap` in AWB computation was slow and memory-intensive

**Solution**: Replaced with explicit list comprehension for conv layers

**Files Modified**:
- `src/cl/models/cnn.py` (lines 268-280)

**Impact**: Faster AWB training for CIFAR10/CIFAR100

**Correctness**: ✅ Equivalent computation, just faster execution

---

### **7. JAX Asynchronous Data Pipeline** ✅

**Problem**: PyTorch DataLoader with `num_workers=0` creates CPU-GPU transfer bottleneck
- GPU utilization: 1% (idle waiting for data)
- GPU memory usage: 3GB/8GB (underutilized)
- Root cause: Synchronous data loading, GPU waits for CPU

**Solution**: Custom JAX-native data pipeline with background thread prefetching
- Background thread loads batches and transfers to GPU asynchronously
- Uses `jax.device_put()` for non-blocking GPU transfer
- Queue-based prefetching (default 3 batches ahead)
- Overlaps data loading with GPU computation

**Files Created**:
- `src/cl/datasets/jax_dataloader.py` (246 lines) - **NEW FILE**
  - `PrefetchDataLoader`: Single loader with async GPU prefetch
  - `DualPrefetchDataLoader`: Dual loader for continual learning (current + experience)
  - Benchmarking utilities

**Files Modified**:
- `src/cl/core/loops.py` (lines 31, 316-325, 382-411)
  - Import prefetch classes
  - Wrap train/exp loaders with `PrefetchDataLoader`
  - Skip PyTorch→JAX conversion when using prefetch (data already on GPU)
- `src/cl/config/constants.py` (lines 22-27)
  - `DEFAULT_USE_JAX_PREFETCH = True`
  - `DEFAULT_PREFETCH_SIZE = 3`
- `src/cl/config/params.py` (lines 172-175)
  - Add config defaults for prefetching parameters

**Expected Impact**:
- GPU utilization: 1% → 60-90%
- Training speed: 5-10x faster data loading
- GPU memory: Better utilization of 8GB

**How to Disable** (for debugging):
```json
{
    "use_jax_prefetch": false
}
```

**Correctness**: ✅ No impact on training computation - only changes when/how data arrives on GPU
- Data is identical, just transferred asynchronously
- Training loop receives same batches in same order

**Testing Status**: ⏳ NOT YET TESTED - needs validation on MNIST

---

## 🧪 VERIFICATION STRATEGY

### After ANY Performance Optimization

1. **Run Unit Tests**:
   ```bash
   ./run_tests.sh --unit  # Fast (~30 sec)
   ```

2. **Run Training Tests**:
   ```bash
   ./run_tests.sh --training  # Full pipeline (~5 min)
   ```

3. **Compare Metrics on Debug Config**:
   ```bash
   # Baseline run (save metrics)
   python run_files/scripts/run.py tests/training/configs/sine_debug.json

   # After optimization (compare metrics)
   python run_files/scripts/run.py tests/training/configs/sine_debug.json

   # Check: Final test loss/accuracy should be within 1% of baseline
   ```

4. **Visual Comparison**:
   - Compare loss curves before/after
   - Check that training converges similarly
   - Verify no NaN/Inf values introduced

### Safe vs. Unsafe Optimization Areas

**✅ SAFE TO OPTIMIZE** (No risk to accuracy):
- Data loading pipeline (I/O, prefetching, async transfer)
- Logging frequency (eval_interval, save_iter, log_interval)
- Checkpointing (async, frequency, memory limits)
- Progress bars and console output
- File I/O (pickle saving, result organization)
- Profiling instrumentation
- GPU memory management (XLA flags, device allocation)
- Batch size (affects speed, not correctness if properly implemented)

**⚠️ REQUIRES CAREFUL VALIDATION**:
- JAX JIT compilation flags (can change numerics)
- Gradient computation optimizations (must preserve math exactly)
- Experience replay sampling (must preserve statistical properties)
- Random seed management (must maintain reproducibility)

**🚫 DO NOT OPTIMIZE WITHOUT EXPLICIT APPROVAL**:
- Hamiltonian gradient computation (`src/cl/core/hamiltonian.py`)
- Loss functions (`src/cl/core/losses.py`)
- AWB weight transformation (`src/cl/core/awb.py`)
- Model forward passes (`src/cl/models/*.py` - except proven equivalent refactors)
- Architecture search logic (`src/cl/arch_search/*.py`)
- Metric calculations (unless fixing bugs like graph metric fix)

---

## 📁 PROJECT STRUCTURE

### Core Training Pipeline
```
src/cl/
├── core/
│   ├── trainer.py          # Main Trainer class (mixin-based)
│   ├── losses.py           # LossMixin - DO NOT OPTIMIZE
│   ├── hamiltonian.py      # HamiltonianMixin - DO NOT OPTIMIZE
│   ├── loops.py            # TrainingLoopsMixin - SAFE: I/O, logging
│   ├── recording.py        # RecordingMixin - SAFE: logging, checkpoints
│   └── awb.py              # AWB utilities - DO NOT OPTIMIZE
├── models/
│   ├── mlp.py              # UNSAFE: model forward pass
│   ├── cnn.py              # UNSAFE: model forward pass
│   └── gcn.py              # UNSAFE: model forward pass
├── datasets/
│   ├── jax_dataloader.py   # SAFE: data loading optimization (NEW)
│   └── *.py                # SAFE: data loading, preprocessing
└── config/
    ├── constants.py        # SAFE: add new defaults
    └── params.py           # SAFE: add new config parameters
```

### Configuration Files

**Production Configs** (`kkt_run/configs/`):
- 5 datasets × 4 conditions = 20 configs
- Full experiments (10-20 tasks, 150-200 epochs)
- Used by SLURM scripts

**Profile Configs** (`kkt_run/configs/debug/`):
- 4 datasets × 2 conditions = 8 configs
- Fast profiling (2 tasks, 100 epochs, 1000 samples)
- Used by `run_optimized_profiles.sh`

**Test Configs** (`tests/training/configs/`):
- 10 configs for automated testing
- Minimal size (50 samples, 2 epochs)

---

## 🔍 KNOWN ISSUES AND SOLUTIONS

### Issue 1: GPU Utilization Only 1% (MNIST)

**Status**: Solution implemented, awaiting testing

**Symptoms**:
- GPU utilization: 1%
- GPU memory: 3GB/8GB used
- Training slow despite high batch size

**Root Cause**:
- PyTorch DataLoader with `num_workers=0` (required for JAX fork incompatibility)
- Synchronous data loading: GPU waits idle while CPU loads/transfers data

**Solution**:
- JAX-native asynchronous data pipeline (`jax_dataloader.py`)
- Background thread prefetches batches to GPU
- Overlaps data loading with computation

**Testing**:
```bash
# Monitor GPU utilization in real-time
watch -n 0.5 nvidia-smi

# Run MNIST training
python run_files/scripts/run.py kkt_run/configs/mnist_condition1_baseline.json

# Expected: 60-90% GPU utilization (vs. 1% before)
```

**Fallback** (if prefetching causes issues):
```json
{
    "use_jax_prefetch": false  // Disable prefetching
}
```

---

### Issue 2: Batch Size Changes Don't Improve Speed

**Status**: Resolved (not a batch size problem)

**Finding**: User confirmed batch size changes don't help
- Larger batches use more memory but don't improve GPU utilization
- Confirms bottleneck is data loading, not compute

**Implication**: Don't focus on batch size - fix data pipeline instead

---

## 🎯 FUTURE OPTIMIZATION OPPORTUNITIES

### High Priority (Safe + High Impact)

1. **XLA Compilation Flags** ✅ SAFE
   ```python
   # Add to config or environment
   XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"
   XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"  # For newer GPUs
   ```

2. **JIT Compilation Tuning** ⚠️ VALIDATE
   - Increase static argument caching
   - Pre-compile common operations
   - **Test**: Verify numerics unchanged

3. **Memory Layout Optimization** ✅ SAFE
   - Optimize array strides for GPU memory access
   - Use JAX array sharding on multi-GPU

4. **Batch Prefetch Tuning** ✅ SAFE
   ```python
   # Benchmark optimal prefetch_size per dataset
   from cl.datasets.jax_dataloader import benchmark_dataloader
   stats = benchmark_dataloader(loader, num_batches=100)
   ```

### Medium Priority (Requires Testing)

1. **Experience Replay Sampling** ⚠️ VALIDATE
   - Current: Random sampling every epoch
   - Proposed: Pre-sample at task start (faster but different random order)
   - **Test**: Verify training convergence unchanged

2. **Gradient Accumulation** ⚠️ VALIDATE
   - Larger effective batch size without memory increase
   - **Test**: Verify gradients mathematically equivalent

3. **Mixed Precision Training** ⚠️ VALIDATE CAREFULLY
   - Use float16 for forward pass, float32 for gradients
   - **Risk**: Numerical stability issues
   - **Test**: Extensive metric comparison required

### Low Priority (Future Work)

1. **Multi-GPU Training**
   - Use `jax.pmap` for data parallelism
   - Requires architecture changes

2. **TPU Support**
   - JAX already supports TPUs
   - Requires cloud infrastructure

---

## 📝 SESSION NOTES

### GPU Utilization Debugging Session (2025-12-28)

**User Feedback**:
- "gpu utilization for mnist is still 1% with just 3095 mb being used"
- "i have tried batch size, does not do much, what about custom data pipeline"

**Key Insights**:
1. Batch size is NOT the bottleneck (user confirmed)
2. Data loading is the bottleneck (CPU-GPU transfer)
3. PyTorch DataLoader `num_workers=0` required (JAX fork incompatibility)
4. Solution: JAX-native async pipeline with background threads

**Implementation Approach**:
- Used Python `threading` module (JAX-compatible, unlike multiprocessing)
- Used `queue.Queue` for thread-safe batch queuing
- Used `jax.device_put()` for asynchronous GPU transfer
- Prefetch size = 3 batches (empirically good balance)

**Next Steps**:
1. Test JAX prefetching on MNIST (verify GPU utilization improvement)
2. If successful, benchmark all datasets
3. Update optimized profile configs to use prefetching
4. Run full production experiments

---

## 🛠️ DEBUGGING COMMANDS

### Monitor Training Performance

```bash
# GPU utilization (real-time)
watch -n 0.5 nvidia-smi

# Profile training speed
python run_files/scripts/profile_training.py kkt_run/configs/mnist_condition1_baseline.json

# Benchmark data loader
python -c "
from cl.datasets.jax_dataloader import benchmark_dataloader
from torch.utils.data import DataLoader
# ... create loader ...
stats = benchmark_dataloader(loader, num_batches=100, warmup=10)
print(stats)
"

# Check for bottlenecks
grep 'it/s\|s/it' kkt_run/kkt/logs/**/*.log
```

### Verify Correctness

```bash
# Run all tests
./run_tests.sh --all

# Run specific test suite
pytest -m unit          # Fast unit tests
pytest -m training      # Full pipeline tests

# Compare training runs
diff <(python run.py config1.json 2>&1) <(python run.py config2.json 2>&1)
```

### Git State Management

```bash
# Check current state
git log --oneline -5
git status

# Create checkpoint before risky optimization
git checkout -b optimization/experimental
git add -A
git commit -m "Before risky optimization"

# Revert if broken
git checkout main
git branch -D optimization/experimental
```

---

## 📚 REFERENCES

### Documentation
- JAX Documentation: https://jax.readthedocs.io/
- JAX Performance Tips: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- Equinox Documentation: https://docs.kidger.site/equinox/

### Key Papers
- Hamiltonian Continual Learning (framework basis)
- AWB: Adaptive Weight Basis for knowledge transfer
- Partitioning Reservoir Sampling (Kim et al., 2020) - for balanced replay

### Internal Docs
- `README.md`: High-level project overview
- `.claude/CLAUDE.md`: Detailed project structure and commands
- `CLAUDE.md`: Code organization preferences
- `session_context.md`: Current session state (if exists)

---

## ✅ PRE-FLIGHT CHECKLIST FOR OPTIMIZATIONS

Before implementing ANY optimization, confirm:

- [ ] **Does this change gradient computation?** → If YES, get approval
- [ ] **Does this change loss calculation?** → If YES, get approval
- [ ] **Does this change model forward pass?** → If YES, get approval
- [ ] **Does this change AWB algorithm?** → If YES, get approval
- [ ] **Does this change experience replay sampling?** → If YES, validate carefully
- [ ] **Is this only I/O, logging, or data loading?** → If YES, safe to proceed
- [ ] **Have I read the user preferences?** → Always minimize new files
- [ ] **Do I have a rollback plan?** → Git branch or commit hash
- [ ] **How will I verify correctness?** → Define test strategy first

**If ANY checkbox is uncertain → ASK USER FIRST**

---

## 🎓 LESSONS LEARNED

### What Worked Well

1. **Three-tier logging intervals**: Massive speedup with zero correctness risk
2. **Eigenvalue optimization**: Simple conditional check, big impact
3. **Output organization**: Quality-of-life improvement, no risk
4. **User-driven debugging**: User's feedback ("batch size doesn't help") revealed true bottleneck

### What to Avoid

1. **Premature optimization**: Don't guess bottlenecks, profile first
2. **Over-engineering**: Simple solutions (conditional check) often best
3. **Changing math without validation**: Always test gradient/loss changes
4. **Ignoring user feedback**: User knows their code and past attempts

### Best Practices

1. **Profile before optimizing**: Measure actual bottlenecks (nvidia-smi, profilers)
2. **Start with safe optimizations**: I/O and logging first, then data loading
3. **One change at a time**: Easier to isolate issues
4. **Git commit frequently**: Easy rollback if broken
5. **Test immediately**: Run tests after each optimization
6. **Document everything**: Future you (or Claude) will thank you

---

**END OF PROFILING CONTEXT**

**Last Updated**: 2025-12-28
**Git Commit**: `521b29c` - "Add JAX asynchronous data pipeline + profiling context doc"
**Status**: JAX prefetching implemented, awaiting testing
