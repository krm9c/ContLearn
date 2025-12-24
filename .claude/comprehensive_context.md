# ContLearn - Comprehensive Project Context

**Last Updated**: 2025-12-24

---

## Project Overview

**ContLearn** is a JAX/Equinox-based continual learning framework that implements the theoretical approach from the paper "The Effect of Architecture During Continual Learning" by Allyson Hahn and Krishnan Raghavan (Argonne National Laboratory).

**Core Innovation**: Jointly optimizes both neural network architecture and weights during continual learning to prevent catastrophic forgetting when data distributions shift across tasks.

---

## Theoretical Foundation (from Paper)

### Problem Formulation

The framework models continual learning in **Sobolev function spaces** $W^{k,p}(\Omega)$:

- **Neural Network**: $\hat{f}(\mathbf{w}(t), \psi(t)) \in W^{k,p}$
  - $\mathbf{w}(t)$: Weight parameters (Euclidean space)
  - $\psi(t)$: Architecture parameters (discrete/categorical)
  - $t \in \{0, 1, \ldots, T\}$: Task instances

- **Forgetting Loss**: $J(t) = \int_0^t \ell(\hat{f}(\mathbf{w}(\tau), \psi(\tau)), \mathcal{X}(\tau)) d\tau$

### Key Theoretical Results

1. **Capacity Divergence Theorem**: When architecture is fixed, the model's capacity to represent tasks diverges as the number of tasks increases (if tasks introduce distributional changes).

2. **Necessary Condition for CL Solution**: The forgetting loss $J(t)$ must be absolutely continuous with respect to the task measure $\mu$ for a continual learning solution to exist.

3. **Continuity w.r.t. Measure**: Defined via set symmetric difference $\mu(A \triangle B)$ - measures task similarity in function space.

4. **Main Implication**: **Weights alone are insufficient** when data distribution shifts across tasks - **architecture must be changed on the fly**.

### Bilevel Optimization

**Upper Level** - Architecture Search:
```
ψ*(t) = argmin_{ψ ∈ Ψ} J(w(t), ψ, X(t))
```

**Lower Level** - Weight Optimization via Dynamic Programming:
```
V*(t, w(t)) = min_{w ∈ W(ψ*(t))} V(t, w(t))
where V(t, w(t)) = ∫_t^T J(w(τ), ψ*(t), X(τ)) dτ
```

---

## Algorithm: Low-Rank Transfer for Architecture Morphing

### Paper Algorithm 3 (Main Training Loop)

**Step 1**: Standard CL training of $\mathbf{w}(t)$ on current task $t$

**Step 2**: Architecture search via NDDS (Neighborhood Directional Direct Search)
- Derivative-free local search using finite differences
- Mimics weak derivatives from Sobolev space framework
- Searches: number of neurons per layer, filter sizes (CNN/GCN)

**Step 3-6**: **Low-Rank Transfer** (Core Innovation)
- **Problem**: New architecture $\psi^*(t)$ has different parameter space than $\psi(t-1)$
- **Solution**: Transfer knowledge via low-rank decomposition

1. Initialize random tensors $\mathbf{A}(t)$, $\mathbf{B}(t)$:
   - $\mathbf{A}_i(t) \in \mathbb{R}^{a_i \times r_i}$ (new × old dimensions)
   - $\mathbf{B}_i(t) \in \mathbb{R}^{b_i \times s_i}$ (new × old dimensions)

2. Compute transformed weights: $C(t) = \mathbf{A}(t) \mathbf{w}(t) \mathbf{B}^T(t)$
   - $C_i(t) \in \mathbb{R}^{a_i \times b_i}$ (matches new architecture)

3. **Train A/B, freeze W**: Learn transformation matrices
   - Freeze $\mathbf{w}(t)$ (preserve old knowledge)
   - Train $\mathbf{A}(t), \mathbf{B}(t)$ for fixed epochs
   - Ensures knowledge transfer to new parameter space

4. Set new weights: $\mathbf{w}(t+1) = C^*(t)$, $\psi(t+1) = \psi^*(t)$

**Step 7**: Standard CL training with new architecture and transferred weights

---

## Mapping: Paper Theory → ContLearn Implementation

### AWB (Adaptive Weight Basis) = Low-Rank Transfer

| Paper Notation | Code Implementation | Description |
|---------------|---------------------|-------------|
| $\mathbf{A}(t)$ | `model.A` | Left transformation matrix |
| $\mathbf{B}(t)$ | `model.B` | Right transformation matrix |
| $\mathbf{w}(t)$ | `W` (old weights) | Previous architecture weights |
| $C(t) = AWB^T$ | `V = A @ W @ B.T` | New architecture weights |
| "Train A/B, freeze W" | `notABTrain=False` | A/B training phase |
| "Train V, freeze A/B" | `notABTrain=True` | Final training phase |

### 5-Step AWB Pipeline in Code

When `awb_enabled: true`, tasks $t \geq 1$ follow:

**STEP 1**: Preliminary Training (`awb_preliminary_epochs`)
- Train on new task with current architecture
- Record preliminary loss for decision making

**STEP 2**: Architecture Change Decision
- Check if `loss_ratio > threshold` AND `loss increased`
- If YES → proceed to architecture search
- If NO → continue standard training

**STEP 3a**: Architecture Search
- Use `src/cl/arch_search/` modules (MLP, CNN, GCN)
- Search optimal hidden layer dimensions
- Evaluate candidates on validation data

**STEP 3b**: A/B Matrix Training (`awb_ab_training_epochs`)
- Initialize A/B matrices for new architecture
- **Freeze W** (old weights), **train A/B** with `notABTrain=False`
- A/B learn transformation to new architecture

**STEP 4**: Weight Transformation
- Compute `V = A @ W @ B.T`
- V becomes new weight matrix in expanded architecture

**STEP 5**: Final Training (remaining epochs)
- **Freeze A/B** matrices, **train V** with `notABTrain=True`
- V now trainable in new architecture space

---

## Core Training Pipeline

### Hamiltonian-Based Gradient Computation

The core CL algorithm computes gradients as weighted combination:

```
grad = α · δθ + β · grad_V + γ · grad_dV
```

Where:
- `δθ` (delta_theta): Current task loss gradient
- `grad_V`: Experience replay gradient (loss on past data)
- `grad_dV`: Regularization term (change in loss due to perturbations)

Configurable via `grad_weights: [α, β, γ]` in config JSON.

### Mixin-Based Trainer Architecture

`src/cl/core/trainer.py` combines:
- **LossMixin** (`losses.py`): Loss and metric computation (MSE, cross-entropy)
- **HamiltonianMixin** (`hamiltonian.py`): Hamiltonian-based gradient computation
- **TrainingLoopsMixin** (`loops.py`): Unified training loop for all problem types
- **RecordingMixin** (`recording.py`): Metric recording with eigenvalue tracking

### Model Partitioning Pattern (Equinox)

```python
# Standard partitioning
params, static = eqx.partition(model, eqx.is_array)

# AWB: Freeze A/B matrices (move to static)
static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))
```

---

## Directory Structure

```
ContLearn/
├── src/cl/              # Core framework source code
│   ├── arch_search/     # Architecture search modules
│   │   ├── mlp_search.py       # Search optimal MLP dimensions
│   │   ├── cnn_search.py       # Search optimal CNN dimensions
│   │   └── gcn_search.py       # Search optimal GCN dimensions
│   ├── config/          # Configuration parameters
│   │   ├── constants.py        # Default hyperparameters, AWB settings
│   │   └── params.py           # Config parsing and validation
│   ├── core/            # Core training components (mixins)
│   │   ├── trainer.py          # Main Trainer class (combines mixins)
│   │   ├── losses.py           # LossMixin - loss/metric computation
│   │   ├── hamiltonian.py      # HamiltonianMixin - Hamiltonian gradients
│   │   ├── loops.py            # TrainingLoopsMixin - unified training loop
│   │   ├── recording.py        # RecordingMixin - metric recording
│   │   ├── awb.py              # AWB utilities (partitioning, matrix ops)
│   │   └── arch_search.py      # Architecture search orchestration
│   ├── datasets/        # Dataset implementations
│   │   ├── base.py             # Base dataset with experience replay
│   │   ├── sine.py             # Sine wave regression
│   │   ├── mnist.py            # MNIST digit classification
│   │   ├── cifar.py            # CIFAR-10/100 classification
│   │   └── synthetic_graph.py  # Synthetic graph classification
│   ├── models/          # Neural network architectures
│   │   ├── mlp.py              # Fully connected networks
│   │   ├── cnn.py              # Convolutional networks (CNN, CNN3D)
│   │   ├── gcn.py              # Graph convolutional networks
│   │   └── layers.py           # Custom layer implementations
│   └── runners/         # Problem-specific orchestration
│       ├── generic_runner.py   # Base runner with common logic
│       ├── regression.py       # Sine wave regression runner
│       ├── classification.py   # MNIST/CIFAR classification runner
│       └── graph_classification.py  # Graph classification runner
├── run_files/           # Execution scripts
│   └── scripts/         # Main execution scripts
│       ├── run.py              # Main training script
│       ├── plot_results.py     # Plot generation from results
│       ├── compare_runs.py     # Multi-run comparison
│       ├── profile_training.py # GPU/CPU profiling
│       └── run_*.sh            # Convenience scripts per config
├── kkt_run/             # KKT cluster-specific runs
│   ├── configs/         # Production config files (.json)
│   │   ├── sine.json, sine_awb.json
│   │   ├── mnist.json, mnist_awb.json
│   │   ├── cifar10.json, cifar10_awb.json
│   │   ├── cifar100.json, cifar100_awb.json
│   │   └── synthetic_graph.json, synthetic_graph_awb.json
│   ├── logs/            # Training logs
│   ├── results/         # Training outputs
│   └── *.sh             # Slurm/parallel execution scripts
├── tests/               # Test suite
│   ├── training/        # Full pipeline training tests (11 tests, ~5 min)
│   │   └── configs/     # Debug configs (50 samples, 2 epochs)
│   ├── test_*.py        # Unit tests (195 tests, ~30 sec)
│   └── conftest.py      # Pytest fixtures
├── data/                # Dataset storage (MNIST, CIFAR, etc.)
└── docs/                # Documentation
```

---

## Configuration System

### Key Config Fields

```json
{
    "prob": "regression|classification",
    "problem": "vectors|graph",
    "data": "sine|mnist|permuted_mnist|cifar10|cifar100|synthetic",
    "network": "fcnn|cnn|gcn",

    // AWB settings
    "awb_enabled": false,
    "awb_preliminary_epochs": 50,
    "awb_ab_training_epochs": 100,
    "awb_loss_ratio_threshold": 1.1,

    // Hamiltonian gradient weights [α, β, γ]
    "grad_weights": [0.01, 0.98, 0.1],

    // Training settings
    "lr_schedule": "constant|step|exponential|cosine|linear",
    "optimizer": "adam|adamw|sgd|rmsprop",
    "flag": [1.0, 1.0],

    // Debug settings
    "debug_mode": false,
    "debug_limit": 100
}
```

### Available Configs

**Standard Configs** (fixed architecture):
- `sine.json` - Sine wave regression (MLP)
- `mnist.json` - MNIST digit classification (CNN)
- `cifar10.json` - CIFAR-10 classification (CNN)
- `cifar100.json` - CIFAR-100 classification (CNN)
- `synthetic_graph.json` - Graph classification (GCN)

**AWB Configs** (adaptive architecture):
- `*_awb.json` - Same tasks with AWB enabled

**Test Configs** (`tests/training/configs/`):
- Debug settings: 50 samples, 2 epochs for fast validation

---

## Core Code Patterns

### Dataset Interface

All datasets implement:
```python
def generate_dataset(task_id, batch_size, phase):
    """Returns (current_loader, experience_loader) tuples"""

def append_to_experience(task_id):
    """Manages experience replay buffer"""
```

### Training Loop Signature

```python
params, static, opt_state, record_dict = trainer.train__CL(
    train__=(trainloader, exploader, valloader, testloader),
    params=params,
    static=static,
    opt_state=opt_state,
    optim=optim,
    n_iter=epochs,
    task_id=i,
    config=config,
    record_dict=record_dict,
    notABTrain=True,      # False for AWB A/B training
    problem_type='vectors',  # or 'graph'
    loss_type='regression'   # or 'classification'
)
```

### AWB Utility Functions (`src/cl/core/awb.py`)

```python
# Decision logic
should_change_arch(loss_ratio, threshold, loss_increased) -> bool

# Matrix initialization
set_new_AB_matrices(model, new_dims, old_dims, key) -> model

# Weight transformation
compute_V_from_AWB(model) -> model  # V = A @ W @ B.T

# Model partitioning
partition_for_AB_training(model) -> (params, static)  # Freeze W, train A/B
partition_for_standard_training(model) -> (params, static)  # Freeze A/B, train V
```

---

## Loss Components (Recorded Metrics)

During training, multiple loss values are recorded:
- **H**: Total Hamiltonian = V + dV
- **V**: Experience replay loss (loss on past data)
- **dV**: Regularization term (change in loss due to perturbations)
- **dV/dx**: Sensitivity to input perturbations
- **dV/dtheta**: Sensitivity to parameter perturbations
- **grad_norm**: L2 norm of total gradient

---

## Test Organization

### Two-Tier Testing Strategy

**Unit Tests** (`tests/*.py`) - **195 tests, ~30 seconds**
- Model architecture tests (`test_models.py`, `test_cnn.py`, `test_graph.py`)
- Layer implementation tests (`test_layers.py`)
- Dataset tests (`test_datasets.py`, `test_mnist.py`)
- Loss and metric tests (`test_losses.py`)
- AWB utility tests (`test_awb.py`)
- Recording tests (`test_recording.py`)
- Component integration tests (`test_integration.py`)

**Training Tests** (`tests/training/`) - **11 tests, ~5 minutes**
- Full pipeline tests for all 10 configs
- Test configs with debug settings (50 samples, 2 epochs)
- Validates end-to-end training workflow
- Outputs logged to `SCRIPT_TEST_RESULTS.md`

### Pytest Markers

```bash
# Run specific test tiers
pytest -m unit           # Fast unit tests
pytest -m training       # Slow training tests

# Run specific test categories
./run_tests.sh --models      # Model tests only
./run_tests.sh --datasets    # Dataset tests only
./run_tests.sh --awb         # AWB utility tests only
```

---

## Quick Commands

### Running Experiments

```bash
# Basic run
python run_files/scripts/run.py kkt_run/configs/sine.json

# Multiple runs with plots
python run_files/scripts/run.py kkt_run/configs/sine.json --runs 3

# Skip plot generation
python run_files/scripts/run.py kkt_run/configs/sine.json --no-plots

# Custom figures output
python run_files/scripts/run.py kkt_run/configs/sine.json --figures-dir outputs/figures

# Using convenience scripts
cd run_files/scripts/
./run_sine.sh              # Run sine regression
./run_mnist.sh             # Run MNIST classification
./run_cifar10.sh           # Run CIFAR-10 classification
./run_sine_awb.sh          # Run sine with AWB enabled
```

### Testing

```bash
# Using run_tests.sh (recommended)
./run_tests.sh --unit             # Fast unit tests (~30 sec)
./run_tests.sh --training         # Full training tests (~5 min)
./run_tests.sh --all              # All tests (~5-10 min)
./run_tests.sh --verbose          # Verbose output
./run_tests.sh -k regression      # Run tests matching pattern
./run_tests.sh --all --cov        # Run with coverage report

# Using pytest directly
pytest -m unit                    # Unit tests only
pytest -m training                # Training tests only
pytest tests/test_models.py       # Run specific test file
pytest --cov=src/cl               # Run with coverage
pytest -v --tb=short              # Verbose output
```

### KKT Cluster Runs

```bash
cd kkt_run/
./run_parallel_standard.sh        # Run all standard configs in parallel
./run_parallel_awb.sh              # Run all AWB configs in parallel
./run_single.sh <config_name>      # Run single config

# Slurm submission
sbatch submit_kkt.slurm            # Submit standard jobs
sbatch submit_kkt_awb_gpu.slurm    # Submit AWB GPU jobs
```

### Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Format code
black src/ tests/ run_files/scripts/
isort src/ tests/ run_files/scripts/

# Type checking
mypy src/
```

---

## Plot Generation

After training, four types of plots are auto-generated:

1. **Losses** (`*_losses.png`): All loss components (H, V, dV, dV/dx, dV/dtheta, gradient norm)
2. **Metrics** (`*_metrics.png`): Train and test metrics over time
3. **Eigenvalues** (`*_eigenvalues.png`):
   - Standard mode: Weight matrix eigenvalues
   - AWB mode: A and B matrix eigenvalues
4. **Overview** (`*_overview.png`): Combined visualization

Manual plot generation:
```bash
python run_files/scripts/plot_results.py <records_file.pkl> --output-dir figures
```

---

## Empirical Results (from Paper)

### Datasets

1. **Regression**: Sine wave dataset (varying amplitude/phase)
2. **Classification**: Omniglot (handwritten characters)
3. **Graph Classification**: Synthetic graphs (PyTorch Geometric FakeDataset)

### Key Findings

**Performance Improvements**:
- Up to **2 orders of magnitude** improvement in training loss
- Architecture change + transfer consistently beats:
  1. Fixed architecture (baseline)
  2. Architecture change without transfer (random re-init)

**Robustness**:
- Maintains performance under noisy data distributions
- Scales from 5 to 10+ tasks
- Works across MLP, CNN, GCN architectures seamlessly

**Saturation Analysis**:
- Fixed architecture saturates after ~1 task (red curves in paper)
- AWB continues improving through ~6 tasks before saturation
- Saturation due to architecture search limitations (not theoretical)

---

## Code Preferences (from CLAUDE.md)

### File Management
- **Minimize new files** - prefer editing existing files
- **Consolidate related code** - group functionality in single files
- **Split only when necessary** - files > 500-800 lines with clear divisions
- Archive old code to `old/` rather than deletion

### Code Quality
- **Always comment new code** - mark with `# Added by Claude:` when significant
- **Maximize code reuse** - check for existing utilities first
- Use inheritance and mixins for shared behavior
- **Keep code simple and understandable**

### Import Patterns
```python
# Prefer absolute imports with package prefix
from package.module import Class

# Use relative imports within same module
from .submodule import helper
```

### Documentation
- **Do not create summary/documentation markdown files automatically**
- **Ask before creating** any new markdown or summary files
- Prefer inline comments over separate documentation files
- Document APIs in docstrings, not separate files

### Workflow
- Make minimal, focused changes
- Avoid over-engineering or unrequested features
- Test changes before considering complete
- Update existing files rather than creating new ones
- **Be direct** - challenge opinions only with evidence

---

## Related Literature

**Neural Architecture Search + CL**:
- CLEAS (Continual Learning with Efficient Architecture Search) - neuron-level NAS
- CAS (Continual Architecture Search) - weight-sharing strategy
- SEAL (Searching Expandable Architectures) - capacity estimation metric

**Dynamic Networks**:
- Progressive Neural Networks (PNNs) - add layer per task
- Dynamically Expandable Networks (DENs) - selective neuron training + pruning

**Parameter-Efficient Fine-Tuning + CL**:
- CoLoRA - new LoRA adapter per task (high compute)
- CLoRA - single LoRA adapter (reduced compute)

**ContLearn Novelty**:
- First to combine optimal architecture search with low-rank transfer for CL
- First theoretical framework (Sobolev space) supporting architecture+weight optimization
- Enables transfer across mismatched parameter spaces (impossible in prior work)

---

## Technical Infrastructure

**Hardware**: MacBook Pro M4 Pro (14-core CPU, 20-core GPU, 24GB RAM)
**OS**: macOS Sequoia 15.6.1
**Acceleration**: Metal Performance Shaders (MPS) for Apple Silicon
**Frameworks**: JAX, Equinox, PyTorch Geometric
**Code**: https://github.com/krm9c/ContLearn.git

---

## Key Files Reference

### Core Training
- `src/cl/core/trainer.py` - Main trainer combining all mixins
- `src/cl/core/hamiltonian.py` - Hamiltonian gradient computation
- `src/cl/core/loops.py` - Training loops
- `src/cl/core/awb.py` - AWB utilities (low-rank transfer implementation)

### Models
- `src/cl/models/mlp.py` - MLP architecture
- `src/cl/models/cnn.py` - CNN architectures
- `src/cl/models/gcn.py` - GCN architecture

### Runners
- `src/cl/runners/regression.py` - Sine wave regression
- `src/cl/runners/classification.py` - MNIST/CIFAR classification
- `src/cl/runners/graph_classification.py` - Graph classification

### Configuration
- `src/cl/config/constants.py` - Default hyperparameters, AWB settings
- `src/cl/config/params.py` - Config parsing

### Scripts
- `run_files/scripts/run.py` - Main training entry point
- `run_files/scripts/plot_results.py` - Plot generation

---

## Git Status (2025-12-24)

Current branch: `main`

Modified files:
- `.claude/session_context.md`

Recent commits:
- `aac94c3` comprehensive test suite and corresponding context
- `dd0f4c2` Fix remaining test failures - shape mismatches and PyTorch tensor issues
- `c76ff35` Fix 4 test failures
- `ce2d763` comprehensive test suite
- `e3bc44c` moved unnecessary markdowns to .claude

---

## Summary

ContLearn implements the first continual learning framework that:
1. **Jointly optimizes architecture and weights** using bilevel optimization
2. **Provides theoretical guarantees** via Sobolev space formulation
3. **Enables knowledge transfer** across mismatched parameter spaces via AWB
4. **Demonstrates empirical success** across regression, classification, and graph tasks

The AWB (Adaptive Weight Basis) mechanism is the production implementation of the paper's low-rank transfer method, allowing neural networks to dynamically expand/contract architecture while preserving learned knowledge through the $V = A \cdot W \cdot B^T$ transformation.

---

## Validation Experiments (2025-12-24)

### Core Research Finding

**Main Finding**: When you introduce **smoothness/continuity in the forgetting cost function J(t)** across tasks, the model remembers better and forgets less.

**Two Mechanisms for Smoothness**:
1. **Heuristics**: Warm starts + learning rate schedules (traditional approach)
2. **Architecture Change + Transfer**: AWB mechanism (paper's contribution)

The paper proves that absolute continuity of J(t) is **necessary** for a continual learning solution to exist. When tasks differ greatly:
- Intersection space in function space shrinks
- J(t) becomes discontinuous
- Catastrophic forgetting occurs

**AWB provides smoothness via**:
- **Architecture expansion** → larger intersection in function space (structural smoothness)
- **Low-rank transfer (A@W@B^T)** → warm start in new parameter space (transfer smoothness)

### Experimental Validation Plan

**Goal**: Validate codebase by proving smoothness mechanisms work across all datasets/architectures

**Infrastructure**: 4× NVIDIA H200 GPUs available for parallel experiments

### Four Experimental Conditions

#### Condition 1: Baseline Hamiltonian (No Smoothness)
- Fixed architecture throughout all tasks
- Continuous training (weights persist, no reset)
- **Constant learning rate** (no schedule)
- **No warm start** (no warmup period)
- Pure Hamiltonian CL with experience replay

**Config**:
```json
{
  "awb_enabled": false,
  "lr_schedule": "constant",
  "warmup_epochs": 0
}
```

#### Condition 2: Smoothness via Heuristics (Traditional)
- Fixed architecture throughout all tasks
- Continuous training (weights persist - natural warm start)
- **Learning rate schedule** (cosine/exponential decay)
- **Explicit warm start** with warmup period at task transitions
- Hamiltonian CL with smoothness heuristics

**Config**:
```json
{
  "awb_enabled": false,
  "lr_schedule": "cosine",
  "warmup_epochs": 50,
  "lr_warmup_factor": 0.1
}
```

#### Condition 3: Architecture Change, No Transfer
- **Architecture change** at decision points (same as AWB)
- **Random initialization** after architecture change (no A/B transfer)
- Architecture search enabled
- Tests if capacity expansion alone helps without knowledge transfer

**Config**:
```json
{
  "awb_enabled": true,
  "awb_skip_transfer": true,  // Skip A/B training
  "lr_schedule": "constant",
  "warmup_epochs": 0
}
```

#### Condition 4: Full AWB (Architecture + Transfer)
- **Architecture change** at decision points
- **Low-rank transfer** via A/B training
- Architecture search enabled
- Full smoothness mechanism

**Config**:
```json
{
  "awb_enabled": true,
  "awb_skip_transfer": false,
  "lr_schedule": "constant",
  "warmup_epochs": 0
}
```

### Key Comparisons

| Comparison | Tests Hypothesis |
|------------|------------------|
| Cond 2 vs Cond 1 | Heuristics (LR + warmup) improve smoothness |
| Cond 3 vs Cond 1 | Architecture expansion helps (without transfer) |
| Cond 4 vs Cond 3 | AWB transfer is critical (not just arch expansion) |
| Cond 4 vs Cond 2 | Architecture change beats heuristics alone |
| Cond 4 vs All | Full AWB is best overall smoothness mechanism |

### Validation Datasets

| Dataset | Architecture | # Tasks | Purpose |
|---------|--------------|---------|---------|
| Sine | MLP | 10 | Regression baseline |
| MNIST | CNN | 10 | Image classification |
| Permuted MNIST | CNN | 10 | Distribution shift |
| CIFAR-10 | CNN | 10 | Complex images |
| CIFAR-100 | CNN | 20 | Many-task scaling |
| Synthetic Graph | GCN | 10 | Graph domain |

**Total**: 6 datasets × 4 conditions × 5 runs = **120 experiments**

### Metrics (Literature-Validated)

**Primary Metrics**:
1. **Average Accuracy/Loss**: $\frac{1}{T}\sum_{i=1}^T L_{test}^i(T)$
2. **Average Forgetting**: $F = \frac{1}{T-1}\sum_{i=1}^{T-1} \max_{j \in [i,T]} (L_i^j - L_i^i)$
3. **Backward Transfer**: $BT = \frac{1}{T-1}\sum_{i=1}^{T-1}(L_i^T - L_i^i)$
4. **Forward Transfer**: $FT = \frac{1}{T-1}\sum_{i=2}^T(L_i^{i-1} - L_i^{random})$

**Smoothness Metrics** (Novel):
5. **Loss Jump at Task Boundaries**: $\Delta_t = |J(t_{end}) - J(t+1_{start})|$
6. **Gradient Norm Continuity**: Track $\|\nabla J(t)\|$ at task boundaries
7. **Loss Variance**: $\text{Var}(L_1^T, \ldots, L_T^T)$ - uniformity across tasks

**Architecture Metrics** (AWB-specific):
8. Architecture evolution over tasks
9. Number of architecture changes triggered
10. A/B matrix conditioning: $\kappa(A), \kappa(B)$
11. Transfer quality: Loss before/after A/B training

### Code Modifications Required

**New Flags**:
- `awb_skip_transfer`: Skip A/B training in Condition 3
- `warmup_epochs`: Number of warmup epochs for Condition 2
- `lr_warmup_factor`: LR multiplier during warmup

**New Recording**:
- Task boundary markers
- Loss jumps at boundaries
- Gradient norms at boundaries
- Architecture change events

### Experiment Organization

**Directory Structure**:
```
experiments/
├── configs/           # All experimental configs
│   ├── sine/
│   │   ├── condition1_baseline.json
│   │   ├── condition2_heuristics.json
│   │   ├── condition3_arch_no_transfer.json
│   │   └── condition4_awb_full.json
│   └── [mnist, permuted_mnist, cifar10, cifar100, synthetic_graph]/
├── slurm/            # Slurm submission scripts for H200
├── results/          # Experimental outputs
└── analysis/         # Analysis scripts and outputs
```

**Data Recording**: All metrics saved to `results/[dataset]/[condition]/run_[id]/records.pkl` for transfer back to local machine for analysis.

### Expected Timeline

- **Implementation**: 2-3 days (configs, code mods, scripts)
- **Phase 1 (Quick)**: 1 day (Sine + MNIST, 3 runs each)
- **Phase 2 (Full)**: 2-3 days (all datasets, 5 runs each, parallel on 4 H200s)
- **Analysis**: 2-3 days (metrics, plots, statistical tests)

### Success Criteria

1. ✅ Condition 4 (AWB) achieves lowest forgetting across all datasets
2. ✅ Condition 2 (heuristics) beats Condition 1 (baseline)
3. ✅ Condition 4 beats Condition 3 (proves transfer is critical)
4. ✅ Loss jumps smallest for Condition 4 (smoothness validated)
5. ✅ Results consistent with paper's theoretical predictions
