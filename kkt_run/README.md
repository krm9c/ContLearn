# KKT Server Parallel Execution Scripts

This directory contains scripts for running all continual learning experiments in parallel on the KKT server using SLURM.

## Directory Structure

```
kkt_run/
├── submit_kkt.slurm         # SLURM job submission script
├── run_parallel.sh          # Distributes jobs across GPUs
├── run_single.sh            # Wrapper for single config execution
├── setup_env.sh             # Environment setup helper
├── download_datasets.py     # Pre-download datasets
├── clear_resume.sh          # Clear success markers for re-run
├── configs/                 # All experiment configs (11 total)
├── results/                 # Output directory (created at runtime)
│   ├── cifar10/            # cifar10.json + cifar10_awb.json outputs
│   ├── cifar100/           # cifar100.json + cifar100_awb.json outputs
│   ├── mnist/              # mnist.json + mnist_awb.json outputs
│   ├── sine/               # sine.json + sine_awb.json + sine_awb_test.json
│   └── synthetic_graph/    # synthetic_graph.json + synthetic_graph_awb.json
└── logs/                    # Execution logs (created at runtime)
```

## Hardware Configuration

- **GPUs**: 4 (configurable in submit_kkt.slurm)
- **Memory**: 128G
- **Conda Environment**: jax__kkt
- **Conda Path**: /home/kraghavan/miniconda3/condabin/conda

## Execution Plan

With 4 GPUs and 11 configs, jobs run in 3 batches:

### Batch 1 (4 jobs in parallel)
1. cifar10.json → GPU 0
2. cifar10_awb.json → GPU 1
3. cifar100.json → GPU 2
4. cifar100_awb.json → GPU 3

### Batch 2 (4 jobs in parallel - after Batch 1 completes)
5. mnist.json → GPU 0
6. mnist_awb.json → GPU 1
7. sine.json → GPU 2
8. sine_awb.json → GPU 3

### Batch 3 (3 jobs in parallel - after Batch 2 completes)
9. sine_awb_test.json → GPU 0
10. synthetic_graph.json → GPU 1
11. synthetic_graph_awb.json → GPU 2

## Resume Functionality

The scripts automatically **skip configs that completed successfully** in previous runs. Success markers are stored in `kkt_run/logs/*.success`.

**Key Features:**
- Automatically resumes from where it left off if job times out or fails
- Skips already-completed configs (saves time and compute)
- Shows which configs are skipped vs. running
- Works across multiple SLURM job submissions

**To force re-run all configs:**
```bash
bash kkt_run/clear_resume.sh
# or manually:
rm -f kkt_run/logs/*.success
```

**To re-run specific config:**
```bash
rm kkt_run/logs/cifar10.success  # Force re-run cifar10.json
```

## Usage

### 1. Setup Environment (First Time)

Ensure the conda environment exists:
```bash
source kkt_run/setup_env.sh
```

If the environment doesn't exist, create it:
```bash
/home/kraghavan/miniconda3/condabin/conda create -n jax__kkt python=3.10
/home/kraghavan/miniconda3/condabin/conda activate jax__kkt
pip install -e .[dev]
```

### 2. Submit Job to SLURM

From the project root directory:
```bash
/usr/local/bin/sbatch kkt_run/submit_kkt.slurm
```

You should see output like: `Submitted batch job 9`

### 3. Monitor Job

```bash
# Check job status
/usr/local/bin/squeue -u $USER

# Monitor output (replace JOBID with actual job ID)
tail -f kkt_run/logs/contlearn_kkt_JOBID.out

# Check error log
tail -f kkt_run/logs/contlearn_kkt_JOBID.err
```

### 4. Check Individual Job Logs

```bash
# View log for specific config
cat kkt_run/logs/cifar10.log

# Monitor in real-time
tail -f kkt_run/logs/mnist_awb.log
```

### 5. View Summary

After job completes:
```bash
cat kkt_run/logs/job_summary.txt
```

The summary shows:
- ✓ Completed jobs
- ✗ Failed jobs (if any)
- Total statistics

## Output Files

For each dataset (e.g., `cifar10`), outputs are saved to `results/cifar10/`:

```
results/cifar10/
├── cifar10_run0_records.pkl          # Standard CL results
├── cifar10_awb_run0_records.pkl      # AWB variant results
├── cifar10_run0.eqx                  # Trained model (standard)
├── cifar10_awb_run0.eqx              # Trained model (AWB)
└── figures/
    ├── cifar10_run0_losses.png
    ├── cifar10_run0_metrics.png
    ├── cifar10_run0_eigenvalues.png
    ├── cifar10_run0_overview.png
    ├── cifar10_awb_run0_losses.png
    └── ... (AWB plots)
```

## SLURM Commands Reference

```bash
# Submit job
/usr/local/bin/sbatch kkt_run/submit_kkt.slurm

# Check queue
/usr/local/bin/squeue -u $USER

# Cancel job
/usr/local/bin/scancel <JOBID>

# View job info
/usr/local/bin/scontrol show job <JOBID>
```

## Troubleshooting

### Job Failed

1. Check the job summary:
   ```bash
   cat kkt_run/logs/job_summary.txt
   ```

2. Check individual log for failed config:
   ```bash
   cat kkt_run/logs/<failed_config>.log
   ```

3. Rerun single config manually:
   ```bash
   source kkt_run/setup_env.sh
   bash kkt_run/run_single.sh <config_name>.json
   ```

### Environment Issues

If conda activation fails:
```bash
# Check if conda is available
/home/kraghavan/miniconda3/condabin/conda info

# List environments
/home/kraghavan/miniconda3/condabin/conda env list

# Create environment if missing
/home/kraghavan/miniconda3/condabin/conda create -n jax__kkt python=3.10
```

### GPU Not Available

Check that CUDA is working:
```bash
source kkt_run/setup_env.sh
python -c "import jax; print(jax.devices())"
```

## Files Generated

- `kkt_run/logs/contlearn_kkt_JOBID.out` - SLURM stdout
- `kkt_run/logs/contlearn_kkt_JOBID.err` - SLURM stderr
- `kkt_run/logs/<config>.log` - Individual config logs
- `kkt_run/logs/<config>.success` - Success markers for resume
- `kkt_run/logs/job_summary.txt` - Final summary report
- `results/<dataset>/*.pkl` - Training records
- `results/<dataset>/*.eqx` - Trained models
- `results/<dataset>/figures/*.png` - Plots

## Notes

- Standard and AWB variants of the same dataset share the same output folder
- The `run_single.sh` script automatically extracts the base dataset name
- Jobs run independently and can fail without affecting others
- Summary is always generated showing which jobs succeeded/failed
- SLURM commands are in `/usr/local/bin/` - ensure this is in your PATH
