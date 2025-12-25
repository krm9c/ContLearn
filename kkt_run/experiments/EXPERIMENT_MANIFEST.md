# Validation Experiments Manifest

**Generated**: 2025-12-24
**Total Experiments**: 120 (6 datasets × 4 conditions × 5 runs)

---

## Experimental Conditions

| ID | Name | AWB | LR Schedule | Warmup | Description |
|----|------|-----|-------------|--------|-------------|
| 1 | Baseline | No | Constant | No | Pure Hamiltonian, no smoothness |
| 2 | Heuristics | No | Cosine | Yes (50 epochs) | Traditional smoothness via LR + warm start |
| 3 | Arch No Transfer | Yes | Constant | No | Arch search, random init (skip A/B) |
| 4 | AWB Full | Yes | Constant | No | Full AWB: arch search + A/B transfer |

---

## Dataset Specifications

| Dataset | Data | Tasks | Epochs/Task | Batch Size | Est. Time/Run | Slurm Script |
|---------|------|-------|-------------|------------|---------------|--------------|
| Sine | sine | 10 | 500 | 128 | ~30 min | submit_sine.slurm |
| MNIST | mnist | 10 | 500 | 256 | ~1 hr | submit_mnist.slurm |
| Permuted MNIST | permuted_mnist | 10 | 500 | 256 | ~1 hr | submit_permuted_mnist.slurm |
| CIFAR-10 | cifar10 | 10 | 500 | 128 | ~2 hr | submit_cifar10.slurm |
| CIFAR-100 | cifar100 | 20 | 500 | 128 | ~4 hr | submit_cifar100.slurm |
| Synthetic Graph | synthetic | 10 | 500 | 128 | ~1.5 hr | submit_synthetic_graph.slurm |

---

## Complete Experiment List

### Phase 1: Quick Validation (24 experiments)
Datasets: Sine, MNIST
Runs: 3 per condition

#### Sine (12 experiments)
1. sine/condition1_baseline/run_0
2. sine/condition1_baseline/run_1
3. sine/condition1_baseline/run_2
4. sine/condition2_heuristics/run_0
5. sine/condition2_heuristics/run_1
6. sine/condition2_heuristics/run_2
7. sine/condition3_arch_no_transfer/run_0
8. sine/condition3_arch_no_transfer/run_1
9. sine/condition3_arch_no_transfer/run_2
10. sine/condition4_awb_full/run_0
11. sine/condition4_awb_full/run_1
12. sine/condition4_awb_full/run_2

#### MNIST (12 experiments)
13-24. [Same pattern as Sine]

**Phase 1 Total**: 24 experiments
**Estimated Time**: ~6-8 hours (with 4 GPUs in parallel)

### Phase 2: Full Validation (120 experiments)
All 6 datasets, 5 runs per condition

#### Sine (20 experiments)
Experiments 1-20

#### MNIST (20 experiments)
Experiments 21-40

#### Permuted MNIST (20 experiments)
Experiments 41-60

#### CIFAR-10 (20 experiments)
Experiments 61-80

#### CIFAR-100 (20 experiments)
Experiments 81-100

#### Synthetic Graph (20 experiments)
Experiments 101-120

**Phase 2 Total**: 120 experiments
**Estimated Time**: ~48-60 hours total (distributed across 6 slurm jobs)

---

## Execution Plan

### Submit All Datasets (Parallel)

```bash
cd experiments/slurm

# Submit all datasets at once (6 separate jobs)
sbatch submit_sine.slurm
sbatch submit_mnist.slurm
sbatch submit_permuted_mnist.slurm
sbatch submit_cifar10.slurm
sbatch submit_cifar100.slurm
sbatch submit_synthetic_graph.slurm
```

### Monitor Progress

```bash
# Check slurm queue
squeue -u $USER

# Check logs
tail -f experiments/logs/sine_*.out
tail -f experiments/logs/mnist_*.out

# Check completion status
python experiments/scripts/check_completion.py
```

### Transfer Results

```bash
# From cluster to local machine
rsync -avz --progress \
    user@cluster:~/ContLearn/experiments/results/ \
    ./experiments/results/
```

---

## Output Structure

Each experiment produces:

```
experiments/results/{dataset}/{condition}/run_{id}/
├── records.pkl              # All metrics and training history
├── config.json              # Config used for this run
├── {prob}_{data}_{net}[_awb]_run{id}_records.pkl  # Duplicate for compatibility
└── .success                 # Marker file (if successful)
```

---

## Metrics Recorded

### Per-Iteration Metrics (saved at each save_iter)
- H, V, dV, dV/dx, dV/dtheta, grad_norm
- train_metric, test_current, test_experience
- Eigenvalues (A/B matrices for AWB, weights for standard)

### Per-Task Metrics
- Architecture changes (if AWB)
- Preliminary phase summary (if AWB)
- AB training history (if AWB with transfer)

### Continual Learning Metrics (Computed in Analysis)
From Lopez-Paz & Ranzato (2017), Chaudhry et al. (2018):
- Average Accuracy: ACC = (1/T) Σ A_{T,i}
- Backward Transfer: BWT = (1/(T-1)) Σ (A_{T,i} - A_{i,i})
- Average Forgetting: F = (1/(T-1)) Σ max(A_{i,i} - A_{i,j})
- Forward Transfer: FWT = (1/(T-1)) Σ (A_{i,i-1} - A_{random})

### Smoothness Metrics (Novel)
- Loss jump at task boundaries: Δ_t = |J(t_end) - J(t+1_start)|
- Gradient norm continuity
- Loss variance across tasks

---

## Success Criteria

1. ✅ All 120 experiments complete successfully
2. ✅ Condition 4 (AWB) achieves lowest forgetting across datasets
3. ✅ Condition 2 (heuristics) beats Condition 1 (baseline)
4. ✅ Condition 4 beats Condition 3 (proves A/B transfer critical)
5. ✅ Loss jumps smallest for Condition 4
6. ✅ Results statistically significant (p < 0.05)

---

## References

**CL Metrics**:
- Lopez-Paz & Ranzato. "Gradient Episodic Memory for Continual Learning." NeurIPS 2017.
- Chaudhry et al. "Riemannian Walk for Incremental Learning." ECCV 2018.

**Analysis Tools**:
- `experiments/analysis/compute_metrics.py` - Compute all CL metrics
- `experiments/analysis/generate_plots.py` - Create comparison plots
- `experiments/analysis/generate_tables.py` - Create LaTeX tables
