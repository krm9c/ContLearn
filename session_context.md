# Session Context for Continual Learning Framework

**Last Updated**: December 25, 2024

## Project Overview

JAX/Equinox-based continual learning framework implementing Hamiltonian-based gradient computation with optional AWB (Adaptive Weight Basis) for architecture morphing during lifelong learning.

## Recent Critical Fixes

### 1. CNN3D AB Training Hang (Dec 25, 2024)
**Problem**: AB training phase hung during JAX JIT compilation for CNN3D models.

**Root Cause**: `@eqx.filter_jit` decorator on `_hamiltonian_core_class_awb()` caused eager JIT compilation of complex AWB transformations (96 matrix multiplications for first conv layer alone).

**Solution**:
- Removed `@eqx.filter_jit` from `_hamiltonian_core_class_awb()` in `src/cl/core/hamiltonian.py`
- Let `jax.grad` and `jax.linearize` handle JIT compilation lazily (like old framework)
- Kept vmap optimization in `CNN3D.get_AWBT()` for better compilation efficiency

**Files Modified**:
- `src/cl/core/hamiltonian.py:304` - Removed eager JIT decorator

**Expected Behavior**:
- First batch: ~30-60s compilation (normal for AWB)
- Subsequent batches: Fast (using JIT cache)

### 2. Architecture Search Candidate Generation (Dec 25, 2024)
**Problem**: Architecture search found same candidate repeatedly because `iteration * (j + 1) = 0` when iteration=0.

**Solution**:
- Center search around current best with expanding radius
- Use `(j - range//2) * step` scaled by `(iteration + 1)`
- Provides variety at iteration=0, expands in later iterations

**Files Modified**:
- `src/cl/models/cnn.py:272-291` (CNN)
- `src/cl/models/cnn.py:591-610` (CNN3D)

### 3. Graph Dataset Support in AWB (Dec 25, 2024)
**Problem**: `create_balanced_validation_set()` failed on graph datasets (unpacking error).

**Solution**:
- Auto-detect loader type (vector vs graph) by checking first batch
- Handle `torch_geometric.data.Batch` objects correctly
- Extract individual graphs from batches for balanced sampling

**Files Modified**:
- `src/cl/core/awb_pipeline.py:50-150`

### 4. Condition 1 Task Warmup (Dec 25, 2024)
**Problem**: Condition 1 (baseline) had task warmup enabled by default, causing unintended smoothing between tasks.

**Solution**:
- Added `"task_warmup_enabled": false` to all condition 1 configs
- Ensures true baseline with no inter-task smoothing

**Files Modified**:
- All `*_condition1_baseline.json` configs in `kkt_run/configs/`

## Experimental Infrastructure

### JLSE Run Scripts (Created Dec 25, 2024)

Located in `kkt_run/`, these scripts run experiments with conda environment setup and absolute path resolution:

**Individual Dataset Scripts**:
- `run_sine.sh` - SINE regression (10 tasks, 500 epochs/task)
- `run_mnist.sh` - MNIST classification (10 tasks, 500 epochs/task)
- `run_cifar10.sh` - CIFAR-10 classification (10 tasks, 500 epochs/task)
- `run_cifar100.sh` - CIFAR-100 classification (20 tasks, 500 epochs/task)
- `run_synthetic_graph.sh` - Synthetic graph classification (10 tasks, 500 epochs/task)
- `run_all_conditions.sh` - Runs all datasets sequentially (20 total conditions)

**Each script**:
- Initializes conda environment (`jax__kkt`)
- Uses absolute paths (works from any directory)
- Runs 4 conditions sequentially per dataset
- Logs to `kkt_run/logs/{dataset}_{condition}_{timestamp}.log`

**Key Features**:
```bash
# Auto-resolves paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RUN_SCRIPT="$REPO_ROOT/run_files/scripts/run.py"

# Logs always go to kkt_run/logs/
logfile="$SCRIPT_DIR/logs/{dataset}_{condition}_{timestamp}.log"

# Configs always found
python "$RUN_SCRIPT" "$SCRIPT_DIR/configs/$config"
```

### Experimental Conditions

**4 Conditions per Dataset**:

1. **Condition 1 - Baseline (No Smoothing)**:
   - Fixed architecture
   - Constant LR (no schedule)
   - Constant grad_weights `[0.01, 0.98, 0.1]`
   - **No task warmup** (`task_warmup_enabled: false`)
   - No AWB
   - True baseline with no inter-task smoothing

2. **Condition 2 - Heuristics**:
   - Uses heuristics/warmup mechanisms
   - Task warmup enabled (default 5 epochs)

3. **Condition 3 - Architecture Search (No Transfer)**:
   - Architecture search enabled
   - No weight transfer/AWB

4. **Condition 4 - AWB Full**:
   - Full AWB pipeline with architecture morphing
   - Weight transfer via A/B matrices

### Configuration Locations

- **Configs**: `kkt_run/configs/`
- **Logs**: `kkt_run/logs/` (auto-created)
- **Results**: `outputs/` (pickle files, figures)
- **Test Configs**: `tests/training/configs/` (debug mode, 50 samples, 2 epochs)

## Key Implementation Details

### Hamiltonian Gradient Computation

Core CL algorithm computes gradients as:
```
grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
```

Where:
- `delta_theta`: Current task loss gradient
- `grad_V`: Experience replay gradient
- `grad_dV`: Regularization term

Configurable via `grad_weights: [alpha, beta, gamma]` in config JSON.

### AWB 5-Step Pipeline

When `awb_enabled: true`, tasks 1+ follow:
1. **STEP 1**: Preliminary training on new task
2. **STEP 2**: Decide if architecture change needed
3. **STEP 3a**: Architecture search for optimal dimensions
4. **STEP 3b**: Train A/B matrices with W frozen (`notABTrain=False`)
5. **STEP 4**: Compute V = A @ W @ B.T
6. **STEP 5**: Train V with A/B frozen (`notABTrain=True`)

**Critical**: AWB architecture search now uses balanced validation sets (20% of experience replay) instead of full training data.

### Task Warmup (New Context)

**Default Behavior** (if not disabled):
- `DEFAULT_TASK_WARMUP_ENABLED = True`
- `DEFAULT_TASK_WARMUP_EPOCHS = 5`
- `DEFAULT_WARMUP_GRAD_WEIGHTS = [1.0, 0.0, 0.0]` (current task only)

**Disable for baselines**:
```json
"task_warmup_enabled": false
```

Located in: `src/cl/runners/generic_runner.py:952-993`

## Common Issues & Solutions

### Issue: "can't open file '../run_files/scripts/run.py'"
**Solution**: Run scripts now use absolute paths. Scripts work from any directory.

### Issue: AB training hangs at 0% progress
**Cause**: JAX JIT compilation of complex AWB transformations
**Solution**: Fixed by removing eager JIT. First batch takes 30-60s (normal), subsequent batches are fast.

### Issue: Architecture search finds same candidate repeatedly
**Cause**: `iteration * (j+1) = 0` at iteration=0
**Solution**: Use centered offsets scaled by `(iteration + 1)`

### Issue: Graph dataset AWB fails
**Cause**: Validation set creation didn't handle `torch_geometric.data.Batch`
**Solution**: Auto-detect loader type and handle graph data correctly

### Issue: Condition 1 has unintended smoothing
**Cause**: Task warmup enabled by default
**Solution**: Add `"task_warmup_enabled": false` to condition 1 configs

## Testing

### Quick Tests (30 seconds):
```bash
./run_tests.sh --unit
```

### Full Training Tests (5 minutes):
```bash
./run_tests.sh --training
```

### Test Organization:
- **Unit tests** (`tests/*.py`): 195 tests, ~30 sec
- **Training tests** (`tests/training/`): 11 tests, ~5 min
- Test configs use `debug_mode: true`, `debug_limit: 50-100`

## Running Experiments

From `kkt_run/`:
```bash
# Single dataset
./run_sine.sh

# All datasets and conditions (20 total)
./run_all_conditions.sh
```

Logs: `kkt_run/logs/{dataset}_{condition}_{timestamp}.log`
Results: `outputs/{dataset}_{network}_run{N}_records.pkl`

## Git Workflow

```bash
# View changes
git status
git diff

# Commit
git add <files>
git commit -m "message"
git push origin main

# Restore to last pull
git reset --hard origin/main
git clean -fd  # if needed
```

## Important Code Locations

**Training Pipeline**:
- `src/cl/core/trainer.py` - Main Trainer class (mixin-based)
- `src/cl/core/hamiltonian.py` - Hamiltonian gradient computation
- `src/cl/core/loops.py` - Unified training loop
- `src/cl/core/awb_pipeline.py` - AWB 5-step orchestration

**Models**:
- `src/cl/models/mlp.py` - Fully connected networks
- `src/cl/models/cnn.py` - CNN and CNN3D (lines 476-522: get_AWBT)
- `src/cl/models/gcn.py` - Graph convolutional networks

**Runners**:
- `src/cl/runners/generic_runner.py` - Base runner with warmup logic (line 952)
- `src/cl/runners/classification.py` - MNIST/CIFAR runner
- `src/cl/runners/regression.py` - Sine regression runner
- `src/cl/runners/graph_classification.py` - Graph runner

**Configuration**:
- `src/cl/config/constants.py` - Default hyperparameters
- `kkt_run/configs/` - Production configs
- `tests/training/configs/` - Debug configs

## Known Limitations

1. **MaxPool Warning**: JAX warns about reduced precision for second-order derivatives in max-pooling. This is expected and harmless.

2. **First Batch Compilation**: AWB training first batch takes 30-60 seconds for JIT compilation. This is normal.

3. **Architecture Search**: Can be slow on large datasets. Use balanced validation set (20% of data) to speed up.

## Environment

- **Conda env**: `jax__kkt`
- **Python**: 3.9
- **Framework**: JAX + Equinox
- **Graph**: torch_geometric (for synthetic_graph)

## Completion Status

✅ **Completed**:
- AWB architecture search debugging (Dec 25, 2024)
- AWB pipeline validation (runs without hanging)
- All 4 critical fixes implemented and tested
- SINE, MNIST, CIFAR-10, CIFAR-100 experiments working
- Synthetic graph config fixes (Dec 25, 2024):
  - Fixed `"data": "synthetic_graph"` → `"synthetic"` in all 4 condition configs
  - Added missing `"problem": "graph"` and `"network": "gcn"` fields
- Graph evaluation fix (Dec 25, 2024):
  - Fixed unpacking error in `return_metric()` for graph data
  - Now handles both training mode (batch, batch_ex) and evaluation mode (single batch)

✅ **GPU Utilization Optimization (Dec 25, 2024)**:
- Diagnosed root cause: batch_size=1024 too small for H200 (processes in 0.14ms)
- **REVISED** to conservative settings after OOM errors:
  - Initial attempt (100x dataset increase) caused CPU RAM exhaustion
  - Synthetic graphs OOM killed, CIFAR-100 took 60s/iteration
- **Final conservative configuration**:
  - Dataset sizes (10x original):
    - Sine: 4,000 → 40,000 samples
    - Synthetic Graph: 1,000 → 10,000 graphs
  - Batch sizes (uniform 2x improvement):
    - All datasets: 2,048 (was 1,024)
- Expected improvement: 8% → 15-25% GPU utilization
- Tradeoff: Conservative but stable (avoids OOM, fast iterations)

✅ **CPU Bottleneck Analysis & Phase 1-3 Optimizations (Dec 25, 2024)**:
- **Bottleneck diagnosis**: GPU at 0% utilization, code is CPU-bound not GPU-bound
- **Root causes identified**:
  1. PyTorch → JAX tensor conversion (10,000× per task) - CRITICAL
  2. CPU-based NumPy random generation (10,000× per task) - CRITICAL
  3. Eigenvalue computation O(n³) blocking loop (10× per task) - MODERATE
  4. Full dataset variance computation at task start (1× per task) - MODERATE

**Phase 1 fixes** (commit 7a75264):
- Fix #2: Sample variance from subset (max_batches=5 instead of ~20 batches)
- Fix #3: Reduce eigenvalue frequency (every 100 epochs instead of every 50)
- Impact: Reduces CPU blocking at task start and during recording

**Phase 2 fixes** (commit 385b264):
- Fix #1: Replace NumPy random with JAX random (GPU-accelerated)
- Eliminates 10,000+ CPU random calls per task
- Deterministic seeding: `config['random_seed'] + task_id * 1000`
- Impact: GPU-accelerated perturbation generation

**Phase 3 fixes** (commit cb147cf):
- Fix #4: Pre-convert train/exp loaders to JAX (ONCE per task)
- Eliminates 10,000+ PyTorch→JAX conversions per task
- Vector data only (graphs use torch_geometric, unchanged)
- Memory analysis: < 1% GPU usage (exp replay capped at 200k samples)
- Impact: Removes all PyTorch→JAX conversion overhead from training loop

**Files modified**:
- `src/cl/core/loops.py`: All three phases (variance, JAX random, pre-conversion)
- `src/cl/core/recording.py`: Phase 1 eigenvalue optimization

**Expected cumulative impact**: 25-40% GPU utilization (up from 0%)

**Actual results (Dec 25, 2024)**:
- ✅ Performance improved: 60s/iteration → 12s/iteration (**5x speedup**)
- ✅ System now GPU-bound (computing Hamiltonian) instead of CPU-bound (data movement)
- ✅ All optimizations preserve deterministic results (same random seeds)
- ✅ Memory usage: < 1% GPU (safe on H200 144GB)
- **Conclusion**: Further optimization would require algorithm changes. Current performance is optimal for the Hamiltonian CL algorithm (3 gradients + Hessian-vector product per iteration).

## 🔴 ACTIVE: CIFAR-10 AWB Performance Issue (Dec 26, 2024)

**Problem**: CIFAR-10 Conditions 3 & 4 with AWB enabled take ~2 hours for first epoch of A/B training (0% GPU utilization).
- Condition 3: Takes ~25 hours (should be ~5 hours)
- Condition 4: Takes ~50+ hours (should be ~10-15 hours)

**Root Causes**:
1. `awb_skip_transfer` flag exists in configs but NOT implemented in code
   - Condition 3 runs 100 epochs of A/B training when it should skip entirely
2. JAX JIT compilation hang during A/B training first epoch
   - Takes ~7200s (2 hours) instead of expected 30-180s
   - Likely in `hamiltonian.py:305` (_hamiltonian_core_class_awb)

**Solution Plan** (see `PROFILING_PLAN.md` for full details):
- **Part 1**: Implement `awb_skip_transfer` flag check in `awb_pipeline.py:302`
- **Part 2**: Create non-intrusive profiling system with progressive testing
  - Profiling decorator module: `src/cl/core/profiling.py`
  - Debug configs: `kkt_run/configs/debug/*.json` (Sine, MNIST, CIFAR-10)
  - Test progression: Simple → Complex to isolate bottleneck
- **Part 3**: Apply targeted optimization based on profiling data
  - Hypothesis 1: Phase 3 pre-conversion causing memory pressure
  - Hypothesis 2: Experience replay (200k samples) overwhelming GPU
  - Hypothesis 3: Complex Hamiltonian causing excessive JIT compilation
  - Hypothesis 4: A/B matrix structure incompatible with JIT

**Status**: Implementation ready to begin. See `PROFILING_PLAN.md` for complete handoff documentation.

**Files**:
- `PROFILING_PLAN.md` - Comprehensive implementation plan with code snippets
- `/Users/kraghavan/.claude/plans/synthetic-stargazing-lighthouse.md` - Original plan

## Next Steps / TODO

**PRIORITY: CIFAR-10 AWB Performance Fix**
- [ ] Implement Part 1: `awb_skip_transfer` flag + profiling system (see `PROFILING_PLAN.md`)
- [ ] Run progressive profiling tests (Sine → MNIST → CIFAR-10)
- [ ] Analyze profiling data to identify bottleneck
- [ ] Implement targeted optimization (Part 3)
- [ ] Verify Condition 3 completes in ~5 hours (not 25)
- [ ] Verify Condition 4 completes in ~10-15 hours (not 50+)

**Experimental Validation**
- [ ] Test synthetic graph experiments (should now work with config fixes)
- [ ] Monitor condition 1 runs to ensure no smoothing occurs
- [ ] Run all 4 conditions across 5 datasets (after AWB fix)
- [ ] Generate comparative plots for paper

## References

- See `CLAUDE.md` and `.claude/CLAUDE.md` for additional project documentation
- See `kkt_run/jlse/README.md` for JLSE run script details
- See `run_tests.sh` for testing workflow
