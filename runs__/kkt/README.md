# KKT Server Experiment Execution

This directory contains SLURM scripts for running all experimental conditions on the KKT cluster.

## Directory Structure

```
runs__/kkt/
├── logs/                          # SLURM output logs
├── results/                       # Experimental results (organized by condition)
├── submit_all.sh                  # Master script (submits all jobs)
├── submit_sine_mnist_cond12.slurm # SINE & MNIST - Conditions 1&2
├── submit_sine_mnist_cond34.slurm # SINE & MNIST - Conditions 3&4
├── submit_synthetic_graph_cond12.slurm # Synthetic Graph - Conditions 1&2
├── submit_synthetic_graph_cond34.slurm # Synthetic Graph - Conditions 3&4
├── run_parallel.sh                # Parallel job distributor (non-SLURM)
└── run_single.sh                  # Single job wrapper
```

## Quick Start

### Submit All Experiments (Recommended)

From the ContLearn root directory:

```bash
./runs__/kkt/submit_all.sh
```

This submits 4 SLURM jobs covering all 12 experiments (3 datasets × 4 conditions).

### Submit Individual Datasets

```bash
# SINE & MNIST - Conditions 1&2 (4 GPUs)
sbatch runs__/kkt/submit_sine_mnist_cond12.slurm

# SINE & MNIST - Conditions 3&4 (4 GPUs)
sbatch runs__/kkt/submit_sine_mnist_cond34.slurm

# SYNTHETIC GRAPH - Conditions 1&2 (2 GPUs)
sbatch runs__/kkt/submit_synthetic_graph_cond12.slurm

# SYNTHETIC GRAPH - Conditions 3&4 (2 GPUs)
sbatch runs__/kkt/submit_synthetic_graph_cond34.slurm
```

## Experimental Conditions

### Condition 1: Baseline
- Fixed architecture, constant LR, no warmup
- Static gradient weights: [0.4, 0.4, 0.1]

### Condition 2: Heuristics
- Fixed architecture, cosine LR, task warmup
- Adaptive gradient weights based on loss ratio

### Condition 3: Architecture Search (No Transfer)
- AWB enabled, architecture search
- Random A/B initialization (no knowledge transfer)

### Condition 4: AWB Full
- AWB enabled, architecture search
- Learned A/B transfer (100 epochs training)

## Configuration Files

All configs are in `runs__/configs/`:
- `sine_condition{1-4}_{descriptor}.json`
- `mnist_condition{1-4}_{descriptor}.json`
- `synthetic_graph_condition{1-4}_{descriptor}.json`

Results are saved to `runs__/kkt/results/{dataset}_condition{1-4}/`

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor in real-time
watch -n 5 squeue -u $USER

# View logs (live)
tail -f runs__/kkt/logs/*.out

# View specific job log
tail -f runs__/kkt/logs/sine_mnist_cond12_<job_id>.out
```

## GPU Allocation

- **SINE & MNIST C1&C2**: 4 GPUs (4 experiments in parallel)
- **SINE & MNIST C3&4**: 4 GPUs (4 experiments in parallel)
- **SYNTHETIC GRAPH C1&C2**: 2 GPUs (2 experiments in parallel)
- **SYNTHETIC GRAPH C3&4**: 2 GPUs (2 experiments in parallel)

Total: 12 experiments across 4 SLURM jobs

## Results

After completion, results are organized as:

```
runs__/kkt/results/
├── sine_condition1/
│   ├── training_record.pkl
│   ├── model_checkpoints/
│   └── plots/
├── sine_condition2/
├── ...
├── mnist_condition1/
├── ...
└── synthetic_graph_condition4/
```

## Time Estimates

- **Condition 1 & 2**: ~12-24 hours per dataset
- **Condition 3 & 4**: ~24-48 hours per dataset (includes A/B training)

## Troubleshooting

### Job Failed

1. Check logs: `runs__/kkt/logs/<dataset>_<condition>_<timestamp>.log`
2. Verify config: `runs__/configs/<dataset>_<condition>.json`
3. Check GPU availability: `squeue`

### Out of Memory

Adjust batch size in config:
- MNIST: 1024 (default)
- SINE: 1024 (default)
- SYNTHETIC GRAPH: 256 (default for graphs)

### Config Not Found

Ensure you're running from ContLearn root directory:
```bash
cd /path/to/ContLearn
sbatch runs__/kkt/submit_*.slurm
```

## Contact

For issues, check `.claude/CLAUDE.md` or contact the maintainer.
