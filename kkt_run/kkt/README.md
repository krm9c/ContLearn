# KKT Parallel Experiments

This directory contains scripts for running continual learning experiments on the KKT cluster using 4 GPUs in parallel.

## Overview

Each dataset has a dedicated SLURM script that runs all 4 experimental conditions simultaneously:
- **Condition 1**: Baseline (no smoothing)
- **Condition 2**: Heuristics (with warmup)
- **Condition 3**: Architecture search (no transfer)
- **Condition 4**: AWB Full (architecture morphing + transfer)

Each condition runs on a dedicated GPU (4 GPUs total).

## Directory Structure

```
kkt/
├── logs/                       # SLURM and experiment logs
├── results/                    # Experiment results and figures
│   ├── sine/
│   ├── mnist/
│   ├── cifar10/
│   ├── cifar100/
│   └── synthetic_graph/
├── run_dataset_parallel.sh     # Helper script to run 4 conditions in parallel
├── submit_sine.slurm           # SLURM script for SINE
├── submit_mnist.slurm          # SLURM script for MNIST
├── submit_cifar10.slurm        # SLURM script for CIFAR-10
├── submit_cifar100.slurm       # SLURM script for CIFAR-100
├── submit_synthetic_graph.slurm # SLURM script for synthetic graph
└── submit_all_datasets.sh      # Submit all datasets at once
```

## Usage

### Submit a Single Dataset

```bash
# Submit from repository root
sbatch kkt_run/kkt/submit_sine.slurm
sbatch kkt_run/kkt/submit_mnist.slurm
sbatch kkt_run/kkt/submit_cifar10.slurm
sbatch kkt_run/kkt/submit_cifar100.slurm
sbatch kkt_run/kkt/submit_synthetic_graph.slurm
```

### Submit All Datasets

```bash
# Submit all 5 datasets at once
bash kkt_run/kkt/submit_all_datasets.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel a job
scancel <job_id>
```

### Check Logs and Results

```bash
# SLURM output logs
ls kkt_run/kkt/logs/

# Experiment logs (detailed, one per condition)
ls kkt_run/kkt/logs/*_condition*.log

# Results (pickled records and figures)
ls kkt_run/kkt/results/
```

## Resource Allocation

Each dataset SLURM job requests:
- **GPUs**: 4 (one per condition)
- **Memory**: 128GB
- **Time limits**:
  - SINE: 24 hours
  - MNIST: 24 hours
  - CIFAR-10: 36 hours
  - CIFAR-100: 48 hours
  - Synthetic Graph: 24 hours

## How It Works

1. **SLURM script** (`submit_<dataset>.slurm`):
   - Allocates 4 GPUs
   - Sets up conda environment
   - Calls `run_dataset_parallel.sh` with dataset name

2. **Parallel runner** (`run_dataset_parallel.sh`):
   - Launches 4 conditions in parallel as background processes
   - Assigns one GPU per condition (CUDA_VISIBLE_DEVICES)
   - Waits for all conditions to complete
   - Reports success/failure summary

3. **Main script** (`run.py`):
   - Runs individual experiment
   - Saves results to `kkt_run/kkt/results/<dataset>/`
   - Generates plots in `kkt_run/kkt/results/<dataset>/figures/`

## Output Files

For each dataset run, you'll get:

```
kkt_run/kkt/
├── logs/
│   ├── <dataset>_<job_id>.out              # SLURM stdout
│   ├── <dataset>_<job_id>.err              # SLURM stderr
│   ├── <dataset>_condition1_*.log          # Condition 1 detailed log
│   ├── <dataset>_condition2_*.log          # Condition 2 detailed log
│   ├── <dataset>_condition3_*.log          # Condition 3 detailed log
│   └── <dataset>_condition4_*.log          # Condition 4 detailed log
└── results/
    └── <dataset>/
        ├── *_run0_records.pkl              # Pickled training records
        ├── *_run0.eqx                      # Saved model weights
        └── figures/                        # Generated plots
            ├── *_losses.png
            ├── *_metrics.png
            ├── *_eigenvalues.png
            └── *_overview.png
```

## Configuration Files

All experiment configurations are in `kkt_run/configs/`:
- `<dataset>_condition1_baseline.json`
- `<dataset>_condition2_heuristics.json`
- `<dataset>_condition3_arch_no_transfer.json`
- `<dataset>_condition4_awb_full.json`

## Troubleshooting

**Job fails immediately:**
- Check SLURM logs in `kkt_run/kkt/logs/<dataset>_<job_id>.err`
- Verify conda environment exists: `conda env list | grep jax__kkt`

**Condition fails:**
- Check detailed log: `kkt_run/kkt/logs/<dataset>_condition<N>_*.log`
- Verify config file exists: `ls kkt_run/configs/<dataset>_condition*.json`

**Out of memory:**
- Increase `#SBATCH --mem` in the SLURM script
- Reduce batch size in config file

**GPU not found:**
- Check GPU allocation: `echo $CUDA_VISIBLE_DEVICES`
- Verify GPUs available: `nvidia-smi`
