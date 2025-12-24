# Standard Continual Learning Implementation

This document provides comprehensive implementation details for the Hamiltonian Continual Learning (HCL) framework with Adaptive Weight Basis (AWB).

## Table of Contents

- [Overview](#overview)
- [Algorithm Structure](#algorithm-structure)
- [Main Algorithms](#main-algorithms)
  - [1. Complete HCL Pipeline](#1-complete-hcl-pipeline)
  - [2. Task 0 Training](#2-task-0-training)
  - [3. Continual Learning with AWB](#3-continual-learning-with-awb)
  - [4. Preliminary Training](#4-preliminary-training)
  - [5. Continue Training](#5-continue-training)
  - [6. AWB Pipeline](#6-awb-pipeline)
  - [7. Train A/B Matrices](#7-train-ab-matrices)
  - [8. Train V](#8-train-v)
- [Supporting Algorithms](#supporting-algorithms)
  - [9. Hamiltonian Gradient Computation](#9-hamiltonian-gradient-computation)
  - [10. Task Warmup](#10-task-warmup)
  - [11. Adaptive Hyperparameters](#11-adaptive-hyperparameters)
  - [12. Balanced Experience Replay](#12-balanced-experience-replay)
  - [13. AWB Decision](#13-awb-decision)
  - [14. Architecture Search](#14-architecture-search)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Implementation Notes](#implementation-notes)
- [References](#references)

## Overview

The standard CL implementation consists of:

- **Task 0**: Standard Hamiltonian-based training establishing the initial model
- **Tasks t ≥ 1**: Full AWB pipeline with adaptive architecture morphing
- **Core Heuristics**: Task warmup, adaptive learning rates, adaptive gradient weights, balanced experience replay, and gradient normalization

### Notation

Throughout this document, we use the following notation:

- **w(t)**: Weights at task t
- **ψ(t)**: Architecture at task t
- **X(t)**: Task data at task t
- **f̂(w, ψ)**: Neural network function
- **J(w(t), ψ(t), X(t))**: Forgetting loss
- **V(t, w(t))**: Cumulative loss (value function)
- **A(t), B(t)**: Low-rank transfer matrices
- **ℓ**: Loss function
- **E**: Experience replay buffer
- **O**: Optimizer state

## Algorithm Structure

The implementation is organized into **14 modular algorithms**:

### Main Algorithms (High-Level Orchestration)

1. **Complete HCL Pipeline** - Top-level orchestrator
2. **Task 0 Training** - Initial model training
3. **Continual Learning with AWB** - Pipeline for tasks t ≥ 1
4. **Preliminary Training** - Initial training on new task
5. **Continue Training** - Training without architecture change
6. **AWB Pipeline** - Architecture morphing workflow
7. **Train A/B Matrices** - Low-rank transfer matrix training
8. **Train V** - Final training with new architecture

### Supporting Algorithms (Core Components)

9. **Hamiltonian Gradient Computation** - Three-component gradient
10. **Task Warmup** - Smooth task transition
11. **Adaptive Hyperparameters** - Dynamic LR and gradient weights
12. **Balanced Experience Replay** - Strategic buffer sampling
13. **AWB Decision** - Architecture change criteria
14. **Architecture Search** - NDDS optimization

---

## Main Algorithms

### 1. Complete HCL Pipeline

**Purpose**: Top-level orchestrator that manages the entire continual learning process.

**Input:**
- Initial weights w(0), architecture ψ(0)
- Sequence of tasks {X(0), X(1), ..., X(T)}

**Output:**
- Final weights w(T) and architecture ψ(T)

**Algorithm:**

```
Initialize:
  - Experience buffer E ← ∅
  - Optimizer state O
  - Default gradient weights: [α₀, β₀, γ₀] = [0.01, 0.98, 0.1]
  - Default learning rate: η₀ = 10⁻⁴

For each task t = 0 to T:
  If t = 0:
    // Task 0: Initialize with standard training
    w(t), ψ(t), E ← Task0_Training(w(0), ψ(0), X(0), E)
                     [Algorithm 2]
  Else:
    // Tasks t ≥ 1: Full pipeline with AWB
    w(t), ψ(t), E ← CL_with_AWB(w(t-1), ψ(t-1), X(t), E, t)
                     [Algorithm 3]

Return w(T), ψ(T)
```

**Key Features:**
- Separates Task 0 (initialization) from subsequent tasks
- Maintains experience buffer across all tasks
- Delegates to specialized algorithms for each phase

---

### 2. Task 0 Training

**Purpose**: Establish the initial model without experience replay or architecture morphing.

**Input:**
- Initial weights w(0), architecture ψ(0)
- Task data X(0)
- Experience buffer E

**Output:**
- Trained weights w(0), architecture ψ(0), updated buffer E

**Algorithm:**

```
For each epoch e = 1 to N_epochs:
  // Sample current task batch
  B_curr ← sample(X(0))

  // Compute gradient (current task only, no experience replay)
  ∇_w H ← α · ∇_w ℓ(f̂(w(0), ψ(0)), B_curr)

  // Gradient clipping
  If ||∇_w H||₂ > 1.0:
    ∇_w H ← ∇_w H / ||∇_w H||₂

  // Update weights
  w(0) ← optimizer_step(w(0), ∇_w H, η, O)

  // Apply learning rate schedule
  η ← lr_schedule(η, e, N_epochs, 10⁻⁶)

// Add task data to experience buffer
E ← E ∪ {samples from X(0)}

Return w(0), ψ(0), E
```

**Key Features:**
- Simple gradient descent on current task
- No experience replay (buffer is empty)
- Establishes baseline performance

**Default Hyperparameters:**
- Learning rate η = 10⁻⁴
- Gradient weight α = 0.01
- Gradient clip threshold = 1.0

---

### 3. Continual Learning with AWB

**Purpose**: Orchestrate the full continual learning pipeline for tasks t ≥ 1.

**Input:**
- Previous weights w(t-1), architecture ψ(t-1)
- New task X(t)
- Experience buffer E
- Task ID t

**Output:**
- Updated weights w(t), architecture ψ(t), buffer E

**Algorithm:**

```
// Step 1: Task warmup
w(t) ← w(t-1), ψ(t) ← ψ(t-1)
w(t) ← Task_Warmup(w(t), ψ(t), X(t))
       [Algorithm 10]

// Step 2: Compute adaptive hyperparameters
J_prev ← J(w(t-1), ψ(t-1), X(t-1))
[α, β, γ], η_min ← Adaptive_Hyperparams(J(w(t), ψ(t), X(t)), J_prev)
                    [Algorithm 11]

// Step 3: Preliminary training
w(t), J_new ← Preliminary_Training(w(t), ψ(t), X(t), E, t, [α, β, γ], η_min)
              [Algorithm 4]

// Step 4: Check if architecture change needed
change_arch ← Should_Change_Architecture(J_prev, J_new)
              [Algorithm 13]

If change_arch:
  // Step 5: AWB pipeline
  w(t), ψ(t) ← AWB_Pipeline(w(t), ψ(t), X(t), E, t, [α, β, γ], η_min)
               [Algorithm 6]
Else:
  // Continue standard training
  w(t) ← Continue_Training(w(t), ψ(t), X(t), E, t, [α, β, γ], η_min)
         [Algorithm 5]

// Add task data to experience buffer (max 200k samples)
E ← E ∪ {samples from X(t)}

Return w(t), ψ(t), E
```

**Key Features:**
- Multi-step pipeline with decision points
- Adaptive hyperparameters based on task difficulty
- Conditional architecture morphing based on loss criteria

**Flow:**
1. Warmup → Adapt → Train → Decide → (Morph OR Continue) → Update Buffer

---

### 4. Preliminary Training

**Purpose**: Initial training phase on new task to assess difficulty and prepare for potential architecture change.

**Input:**
- Weights w, architecture ψ
- Task X(t), buffer E, task ID t
- Gradient weights [α, β, γ], η_min

**Output:**
- Trained weights w, final loss J_new

**Algorithm:**

```
J_start ← J(w, ψ, X(t))
η ← 10⁻⁴

For each epoch e = 1 to N_prelim = 100:
  // Sample batches
  B_curr ← sample(X(t))
  B_exp ← Balanced_Replay(E, t)
          [Algorithm 12]

  // Compute full Hamiltonian gradient
  ∇_w H ← Hamiltonian_Gradient(w, ψ, B_curr, B_exp, [α, β, γ], t)
          [Algorithm 9]

  // Gradient clipping
  If ||∇_w H||₂ > 1.0:
    ∇_w H ← ∇_w H / ||∇_w H||₂

  // Update weights and learning rate
  w ← optimizer_step(w, ∇_w H, η, O)
  η ← lr_schedule(η, e, N_prelim, η_min)

J_new ← J(w, ψ, X(t))
Return w, J_new
```

**Key Features:**
- Fixed 100 epochs for consistent assessment
- Full Hamiltonian gradient (current + experience + regularization)
- Returns final loss for decision making

**Purpose of Preliminary Training:**
- Assess task difficulty via loss change
- Warm up model on new task distribution
- Provide baseline for architecture decision

---

### 5. Continue Training

**Purpose**: Continue training without architecture change when current architecture is sufficient.

**Input:**
- Weights w, architecture ψ
- Task X(t), buffer E, task ID t
- Gradient weights [α, β, γ], η_min

**Output:**
- Trained weights w

**Algorithm:**

```
η ← 10⁻⁴
N_remaining ← N_epochs - N_prelim

For each epoch e = 1 to N_remaining:
  // Sample batches
  B_curr ← sample(X(t))
  B_exp ← Balanced_Replay(E, t)

  // Compute full Hamiltonian gradient
  ∇_w H ← Hamiltonian_Gradient(w, ψ, B_curr, B_exp, [α, β, γ], t)

  // Gradient clipping
  If ||∇_w H||₂ > 1.0:
    ∇_w H ← ∇_w H / ||∇_w H||₂

  // Update
  w ← optimizer_step(w, ∇_w H, η, O)
  η ← lr_schedule(η, e, N_remaining, η_min)

Return w
```

**Key Features:**
- Same training procedure as preliminary phase
- Trains for remaining epochs (total - preliminary)
- No architecture modification

---

### 6. AWB Pipeline

**Purpose**: Execute the full Adaptive Weight Basis pipeline for architecture morphing.

**Input:**
- Weights w(t), architecture ψ(t)
- Task X(t), buffer E, task ID t
- Gradient weights [α, β, γ], η_min

**Output:**
- New weights w(t+1), new architecture ψ(t+1)

**Algorithm:**

```
// Step 1: Architecture Search
ψ*(t) ← NDDS_Search(ψ(t), X(t), w(t))
        [Algorithm 14]

// Step 2: Initialize A, B matrices
For i = 1 to d:
  A_i(t), B_i(t) ← init_AB(a_i, b_i, r_i, s_i)  // Glorot initialization

// Step 3: Train A/B matrices (W frozen)
A(t), B(t) ← Train_AB_Matrices(w(t), A(t), B(t), ψ*(t), X(t), t)
             [Algorithm 7]

// Step 4: Compute new weights via low-rank transfer
w(t+1) ← A(t) · w(t) · B^T(t)
ψ(t+1) ← ψ*(t)

// Step 5: Train V with A/B frozen
w(t+1) ← Train_V(w(t+1), ψ(t+1), A(t), B(t), X(t), E, t, [α, β, γ], η_min)
         [Algorithm 8]

Return w(t+1), ψ(t+1)
```

**Key Features:**
- 5-step process for architecture morphing
- Low-rank transfer via A · W · B^T
- Preserves knowledge while expanding capacity

**AWB Steps Explained:**
1. **Search**: Find optimal architecture dimensions
2. **Initialize**: Create transfer matrices A and B
3. **Train AB**: Learn transformation with W frozen
4. **Transfer**: Compute V = A · W · B^T
5. **Train V**: Fine-tune in new architecture space

---

### 7. Train A/B Matrices

**Purpose**: Train transfer matrices A and B to map old weights to new architecture.

**Input:**
- Frozen weights w
- Matrices A(t), B(t)
- New architecture ψ*
- Task X(t), task ID t

**Output:**
- Trained matrices A(t), B(t)

**Algorithm:**

```
ε_AB ← 0.01 · (1 + 0.1 · t)⁻¹  // Dynamic convergence threshold
η ← 10⁻⁴

For each epoch e = 1 to N_AB = 50:
  // Sample batch
  B_curr ← sample(X(t))

  // Compute transformed weights
  C(t) ← A(t) · w · B^T(t)

  // Gradient w.r.t. A and B only (W frozen)
  ∇_{A,B} ← ∇_{A,B} ℓ(f̂(C(t), ψ*), B_curr)

  // Update A and B
  A(t), B(t) ← optimizer_step(A(t), B(t), ∇_{A,B}, η, O)

  // Check convergence
  If loss converged within ε_AB:
    break

Return A(t), B(t)
```

**Key Features:**
- W (old weights) completely frozen
- Only A and B are trainable
- Dynamic convergence threshold: tighter for later tasks
- Early stopping when converged

**Convergence Threshold:**
- Task 1: ε_AB = 0.01 / 1.1 ≈ 0.0091
- Task 5: ε_AB = 0.01 / 1.5 ≈ 0.0067
- Task 10: ε_AB = 0.01 / 2.0 = 0.0050

---

### 8. Train V

**Purpose**: Final training phase with new architecture and frozen transfer matrices.

**Input:**
- Weights V, architecture ψ
- Frozen A, B matrices
- Task X(t), buffer E, task ID t
- Gradient weights [α, β, γ], η_min

**Output:**
- Trained weights V

**Algorithm:**

```
N_remaining ← N_epochs - N_prelim - N_AB
η ← 10⁻⁴

For each epoch e = 1 to N_remaining:
  // Sample batches
  B_curr ← sample(X(t))
  B_exp ← Balanced_Replay(E, t)

  // Compute full Hamiltonian gradient on new architecture
  ∇_V H ← Hamiltonian_Gradient(V, ψ, B_curr, B_exp, [α, β, γ], t)

  // Gradient clipping
  If ||∇_V H||₂ > 1.0:
    ∇_V H ← ∇_V H / ||∇_V H||₂

  // Update V only (A and B frozen)
  V ← optimizer_step(V, ∇_V H, η, O)
  η ← lr_schedule(η, e, N_remaining, η_min)

Return V
```

**Key Features:**
- A and B completely frozen
- V trained in expanded architecture space
- Full Hamiltonian gradient for continual learning

**Epoch Budget:**
- Total epochs: N_epochs (e.g., 500)
- Preliminary: 100 epochs
- A/B training: 50 epochs
- V training: 350 epochs (remaining)

---

## Supporting Algorithms

### 9. Hamiltonian Gradient Computation

**Purpose**: Compute the three-component Hamiltonian gradient.

**Input:**
- Weights w, architecture ψ
- Current batch B_curr, experience batch B_exp
- Gradient weights [α, β, γ]
- Task ID t

**Output:**
- Hamiltonian gradient ∇_w H

**Algorithm:**

```
// Component 1: Current task gradient
∇_w ℓ_curr ← ∇_w ℓ(f̂(w, ψ), B_curr)

// Component 2: Experience replay gradient (value function)
∇_w V ← ∇_w ℓ(f̂(w, ψ), B_exp)

// Component 3: Regularization term (perturbation sensitivity)
σ²_x ← 10⁻⁴, σ²_w ← 10⁻⁸

// Input perturbations
V₀ ← ℓ(f̂(w, ψ), B_exp)
For k = 1 to 5:
  B_exp^(k) ← B_exp + ε_x^(k), where ε_x^(k) ~ N(0, σ²_x I)
  V_k ← ℓ(f̂(w, ψ), B_exp^(k))
∇_x V ← (1/5) Σ(V_k - V₀) / σ_x

// Parameter perturbations
For k = 1 to 5:
  ε_w^(k) ~ N(0, σ²_w I)
  w^(k) ← w + ε_w^(k)
  V_k^w ← ℓ(f̂(w^(k), ψ), B_exp)
∇_w V_perturb ← (1/5) Σ(V_k^w - V₀) / σ_w

∇_w δV ← ∇_x V + ∇_w V_perturb

// Normalize dV by task count
dV_norm ← ||∇_w δV||₂ / (t + 1)

// Combine all components
∇_w H ← α · ∇_w ℓ_curr + β · ∇_w V + γ · dV_norm

Return ∇_w H
```

**Mathematical Form:**
```
∇_w H = α · δθ + β · ∇V + γ · ∇δV
```

Where:
- **δθ**: Current task gradient
- **∇V**: Experience replay gradient
- **∇δV**: Regularization (perturbation sensitivity)

**Key Features:**
- Finite difference approximation for perturbations (5 samples each)
- Task-normalized regularization: prevents dominance as t grows
- Three-component balance controlled by [α, β, γ]

**Default Values:**
- α = 0.01 (current task: 1%)
- β = 0.98 (experience: 98%)
- γ = 0.1 (regularization: 10%)
- σ²_x = 10⁻⁴ (input perturbation)
- σ²_w = 10⁻⁸ (parameter perturbation)

---

### 10. Task Warmup

**Purpose**: Smooth transition to new task with reduced learning rate and current-task-only gradients.

**Input:**
- Weights w(t), architecture ψ(t)
- New task X(t)

**Output:**
- Warmed-up weights w(t)

**Algorithm:**

```
// Reduce learning rate for warmup
η_warmup ← 0.1 · 10⁻⁴ = 10⁻⁵

// Focus only on current task (disable experience replay)
[α, β, γ] ← [1.0, 0.0, 0.0]

For e = 1 to N_warmup = 5:
  B_curr ← sample(X(t))
  ∇_w ℓ_curr ← ∇_w ℓ(f̂(w(t), ψ(t)), B_curr)
  w(t) ← optimizer_step(w(t), ∇_w ℓ_curr, η_warmup, O)

Return w(t)
```

**Key Features:**
- Very short duration (5 epochs)
- 10x reduced learning rate
- No experience replay interference
- Helps model adapt to new distribution

**Rationale:**
- Prevents large weight changes at task boundaries
- Gives model time to adjust to new data
- Reduces initial loss variance

---

### 11. Adaptive Hyperparameters

**Purpose**: Dynamically adjust learning rate minimum and gradient weights based on task difficulty.

**Input:**
- Current loss J_curr
- Previous loss J_prev

**Output:**
- Adapted gradient weights [α, β, γ]
- LR minimum η_min

**Algorithm:**

```
// Compute loss ratio
r_loss ← J_curr / J_prev

// Adaptive learning rate minimum
η_min ← min(10⁻⁶, 0.1 · 10⁻⁴ / r_loss)
      = min(10⁻⁶, 10⁻⁵ / r_loss)

// Adaptive gradient weights based on task difficulty
w_curr ← min(1.0, r_loss)     // Weight for current task
w_exp ← 1.0 - w_curr          // Weight for experience

α ← w_curr · 0.01             // Current task gradient weight
β ← w_exp · 0.98              // Experience gradient weight
γ ← 0.1                       // Regularization weight (fixed)

Return [α, β, γ], η_min
```

**Adaptive Behavior:**

| Scenario | r_loss | w_curr | w_exp | α | β | Interpretation |
|----------|--------|--------|-------|---|---|----------------|
| Easy task | 0.5 | 0.5 | 0.5 | 0.005 | 0.49 | More experience |
| Similar difficulty | 1.0 | 1.0 | 0.0 | 0.01 | 0.0 | All current |
| Hard task | 1.5 | 1.0 | 0.0 | 0.01 | 0.0 | All current |

**Learning Rate Minimum:**
- Easy task (r_loss = 0.5): η_min = min(10⁻⁶, 2×10⁻⁵) = 10⁻⁶
- Hard task (r_loss = 2.0): η_min = min(10⁻⁶, 5×10⁻⁶) = 10⁻⁶

**Key Insight:** When task is harder (high r_loss), focus more on current task by increasing α relative to β.

---

### 12. Balanced Experience Replay

**Purpose**: Strategic sampling from experience buffer to balance recency, history, and diversity.

**Input:**
- Experience buffer E
- Current task ID t
- Batch size B

**Output:**
- Experience batch B_exp

**Algorithm:**

```
// Compute sampling quotas
n_recent ← ⌊0.1 · B⌋     // 10% from recent task
n_older ← ⌊0.8 · B⌋      // 80% from older tasks
n_random ← B - n_recent - n_older  // 10% uniform random

// Sample from recent task (t-1)
If t > 0:
  S_recent ← sample(E_{t-1}, n_recent)
Else:
  S_recent ← ∅

// Sample from older tasks (0 to t-2) uniformly
If t > 1:
  n_per_task ← ⌈n_older / (t-1)⌉
  S_older ← ∪_{i=0}^{t-2} sample(E_i, n_per_task)
Else:
  S_older ← ∅

// Sample uniformly from all tasks
S_random ← sample(E, n_random)

// Combine samples
B_exp ← S_recent ∪ S_older ∪ S_random

Return B_exp
```

**Sampling Strategy (10-80-10):**

For batch size B = 128:
- **Recent (13 samples)**: From task t-1 to capture recent knowledge
- **Older (102 samples)**: Distributed across tasks 0 to t-2 to prevent forgetting
- **Random (13 samples)**: Uniform from all tasks for diversity

**Example at Task 5:**
- Recent: 13 samples from task 4
- Older: 102 samples → ~26 from each of tasks 0, 1, 2, 3
- Random: 13 samples from tasks 0-4 uniformly

**Key Features:**
- Prevents recency bias (80% from older tasks)
- Maintains recent knowledge (10% from t-1)
- Adds diversity (10% random)
- Buffer size capped at 200k samples

---

### 13. AWB Decision

**Purpose**: Decide whether to change architecture based on loss criteria.

**Input:**
- Loss before preliminary training J_prev
- Loss after preliminary training J_new

**Output:**
- Decision: change architecture (True/False)

**Algorithm:**

```
// Compute loss ratio and change
r_loss ← J_new / J_prev
ΔJ ← J_new - J_prev

// Architecture change criteria
θ_high ← 0.9

If (r_loss > θ_high) AND (ΔJ > 0):
  Return True   // Loss increased significantly, change architecture
Else:
  Return False  // Continue with current architecture
```

**Decision Logic:**

| Condition | r_loss | ΔJ | Decision | Rationale |
|-----------|--------|-------|----------|-----------|
| Loss decreased | 0.8 | -0.2 | Keep | Model is learning |
| Loss stable | 1.0 | ~0 | Keep | Adequate capacity |
| Loss increased slightly | 0.95 | +0.05 | Keep | Within threshold |
| Loss increased significantly | 1.2 | +0.2 | **Change** | Need more capacity |

**Key Features:**
- Two criteria: BOTH must be true
  1. Relative increase: r_loss > 0.9
  2. Absolute increase: ΔJ > 0
- Conservative threshold (0.9) prevents unnecessary changes
- Based on preliminary training performance

---

### 14. Architecture Search

**Purpose**: Find optimal architecture using derivative-free NDDS (Neighborhood Directional Direct Search).

**Input:**
- Current architecture ψ(t)
- Task data X(t)
- Weights w(t)

**Output:**
- Optimal architecture ψ*(t)

**Algorithm:**

```
// Extract current dimensions
{d₁, d₂, ..., d_L} ← extract_dims(ψ(t))

// Define search space (neighborhood)
S ← {d_i ± k·s : i ∈ {1, ..., L}, k ∈ {1, 2, 3}}
where s = 16 is step size

// Neighborhood Directional Direct Search
ψ* ← ψ(t), J_best ← ∞

For each candidate architecture ψ' ∈ S:
  // Evaluate on validation subset
  w' ← init_weights(ψ')  // Transfer from w(t) if possible
  J' ← quick_eval(w', ψ', X_val(t))  // 50 samples, 10 epochs

  If J' < J_best:
    J_best ← J'
    ψ* ← ψ'

Return ψ*
```

**Search Space Example:**

For MLP with current dims [128, 64, 32]:
- Layer 1 candidates: 112, 128, 144, 160, 176 (128 ± {16, 32, 48})
- Layer 2 candidates: 48, 64, 80, 96, 112
- Layer 3 candidates: 16, 32, 48, 64, 80

Total candidates: 5 × 5 × 5 = 125 architectures

**Quick Evaluation:**
- Uses small validation subset (50 samples)
- Short training (10 epochs)
- Designed for speed, not final performance

**Key Features:**
- Derivative-free (works with discrete spaces)
- Local search (preserves similarity to current architecture)
- Step size s = 16 balances exploration vs. exploitation
- Efficient with quick evaluations

---

## Hyperparameter Reference

### Gradient Computation

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| α | 0.01 | Current task gradient weight |
| β | 0.98 | Experience replay gradient weight |
| γ | 0.1 | Regularization gradient weight |
| σ²_x | 10⁻⁴ | Input perturbation variance |
| σ²_w | 10⁻⁸ | Parameter perturbation variance |
| Perturbation samples | 5 | Samples for finite difference |

### Optimization

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| η₀ | 10⁻⁴ | Base learning rate |
| η_min | 10⁻⁶ | Minimum learning rate |
| Optimizer | Adam | Default optimizer (β₁=0.9, β₂=0.999) |
| Gradient clip | 1.0 | Maximum gradient L2 norm |

### Task Warmup

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| N_warmup | 5 epochs | Warmup duration |
| η_warmup | 10⁻⁵ | Warmup learning rate (0.1 × η₀) |
| Warmup gradients | [1.0, 0.0, 0.0] | Current task only |

### Experience Replay

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Buffer size | 200,000 | Maximum samples in buffer |
| Recent quota | 10% | Samples from task t-1 |
| Older quota | 80% | Samples from tasks 0 to t-2 |
| Random quota | 10% | Uniform random samples |

### AWB Pipeline

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| N_prelim | 100 epochs | Preliminary training duration |
| N_AB | 50 epochs | A/B matrix training duration |
| θ_high | 0.9 | Architecture change threshold |
| ε_AB | 0.01 · (1 + 0.1t)⁻¹ | A/B convergence threshold |
| Search step | 16 | Dimension search step size |
| Quick eval samples | 50 | Validation samples for arch search |
| Quick eval epochs | 10 | Training epochs for arch search |

### Training Epochs

| Phase | Epochs | Description |
|-------|--------|-------------|
| Task 0 | N_epochs | Full training (e.g., 500) |
| Warmup (t ≥ 1) | 5 | Task transition |
| Preliminary | 100 | Initial training on new task |
| A/B training | 50 | Transfer matrix learning |
| V training | N_epochs - 150 | Final training (e.g., 350) |
| Continue training | N_epochs - 100 | No arch change (e.g., 400) |

---

## Implementation Notes

### Gradient Normalization

The ∇_w δV component is normalized by task count:

```
dV_norm = ||∇_w δV||₂ / (t + 1)
```

**Rationale:** Without normalization, the regularization term would grow unbounded as more tasks are learned, eventually dominating the other gradient components.

**Effect:**
- Task 1: dV_norm = ||∇_w δV||₂ / 2
- Task 5: dV_norm = ||∇_w δV||₂ / 6
- Task 10: dV_norm = ||∇_w δV||₂ / 11

This ensures balanced contributions throughout learning.

### Learning Rate Schedules

The framework supports multiple LR schedules:

1. **Constant**: η(e) = η₀
2. **Step**: η(e) = η₀ · 0.1^⌊e/100⌋
3. **Exponential**: η(e) = η₀ · exp(-0.01 · e)
4. **Cosine**: η(e) = η_min + 0.5(η₀ - η_min)(1 + cos(πe/N_epochs))
5. **Linear**: η(e) = η₀ · (1 - e/N_epochs)

**Default:** Cosine annealing with warm restarts.

**Example (Cosine with η₀ = 10⁻⁴, η_min = 10⁻⁶, N = 500):**
- Epoch 0: η = 10⁻⁴
- Epoch 125: η ≈ 5 × 10⁻⁵
- Epoch 250: η = 10⁻⁶ (minimum)
- Epoch 375: η ≈ 5 × 10⁻⁵
- Epoch 500: η = 10⁻⁴ (restart)

### JAX Implementation

All gradient computations are JIT-compiled using JAX:

**Problem-Specific Variants (8 total):**

| Variant | Loss Type | Mode | Problem | Use Case |
|---------|-----------|------|---------|----------|
| 1 | MSE | Standard | Vector | Regression tasks |
| 2 | MSE | AWB | Vector | Regression with AWB |
| 3 | Cross-entropy | Standard | Vector | Classification tasks |
| 4 | Cross-entropy | AWB | Vector | Classification with AWB |
| 5 | MSE | Standard | Graph | Graph regression |
| 6 | MSE | AWB | Graph | Graph regression with AWB |
| 7 | Cross-entropy | Standard | Graph | Graph classification |
| 8 | Cross-entropy | AWB | Graph | Graph classification with AWB |

**JIT Compilation Benefits:**
- 10-100x speedup on GPU/TPU
- Automatic parallelization
- Memory optimization
- Device-agnostic code

### Model Partitioning (Equinox)

Equinox models are partitioned into trainable and static components:

#### Standard Training
```python
params, static = eqx.partition(model, eqx.is_array)
# All weights trainable
```

#### A/B Training (AWB Step 3)
```python
# Freeze W, train only A and B
static = eqx.tree_at(lambda x: x.W, static, replace=model.W)
params = eqx.tree_at(lambda x: x.W, params, replace=None)
# Now: A, B trainable; W frozen
```

#### V Training (AWB Step 5)
```python
# Freeze A and B, train only V
static = eqx.tree_at(lambda x: (x.A, x.B), static,
                      replace=(model.A, model.B))
params = eqx.tree_at(lambda x: (x.A, x.B), params,
                      replace=(None, None))
# Now: V trainable; A, B frozen
```

### Experience Buffer Implementation

Task-indexed dictionary structure:

```python
experience_buffer = {
    'task_0': {
        'inputs': jax.Array[...],
        'targets': jax.Array[...],
        'count': N0
    },
    'task_1': {
        'inputs': jax.Array[...],
        'targets': jax.Array[...],
        'count': N1
    },
    ...
}
```

**Features:**
- Efficient task-wise access for balanced sampling
- FIFO replacement within tasks when full
- JAX arrays for GPU acceleration
- Metadata tracking (count, task ID)

### Testing and Validation

Comprehensive test suite with 206 tests:

**Unit Tests (~30 seconds):**
- Model architecture tests (MLP, CNN, GCN)
- Layer implementation tests
- Dataset tests (MNIST, CIFAR, synthetic)
- Loss and metric tests
- AWB utility tests
- Experience buffer tests
- Integration tests

**Training Tests (~5 minutes):**
- Full pipeline tests for 10 configurations
- Sine, MNIST, CIFAR-10, CIFAR-100, synthetic graph
- Standard and AWB variants
- End-to-end validation

**Running Tests:**
```bash
./run_tests.sh --unit          # Fast unit tests
./run_tests.sh --training      # Full training tests
./run_tests.sh --all           # All tests with coverage
```

---

## References

### Core Algorithms

1. **Pontryagin, L. S., et al.** (2018). *Mathematical Theory of Optimal Processes*. Hamiltonian formulation for gradient computation.

2. **Kingma, D. P., & Ba, J.** (2014). *Adam: A Method for Stochastic Optimization*. ICLR 2015.

3. **Pascanu, R., et al.** (2013). *On the difficulty of training recurrent neural networks*. ICML 2013. Gradient clipping.

4. **Loshchilov, I., & Hutter, F.** (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017.

### Continual Learning

5. **Rolnick, D., et al.** (2019). *Experience Replay for Continual Learning*. NeurIPS 2019.

6. **Kirkpatrick, J., et al.** (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS 2017. (EWC - Elastic Weight Consolidation)

7. **Zenke, F., et al.** (2017). *Continual Learning Through Synaptic Intelligence*. ICML 2017.

8. **Li, Z., & Hoiem, D.** (2017). *Learning without Forgetting*. IEEE TPAMI 2017.

### Architecture and Optimization

9. **Glorot, X., & Bengio, Y.** (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS 2010.

10. **Goyal, P., et al.** (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. Learning rate warmup.

11. **Kolda, T. G., et al.** (2003). *Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods*. SIAM Review. NDDS algorithm.

12. **Bishop, C. M.** (1995). *Training with Noise is Equivalent to Tikhonov Regularization*. Neural Computation. Perturbation analysis.

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{hahn2024architecture,
  title={The Effect of Architecture During Continual Learning},
  author={Hahn, Allyson and Raghavan, Krishnan},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

---

## Contact

**Allyson Hahn**: ahahn2813@gmail.com
**Krishnan Raghavan** (Corresponding Author): kraghavan@anl.gov

Mathematics and Computer Science Division
Argonne National Laboratory
