# Polaris Parallel Execution Scripts

This directory contains scripts for running all continual learning experiments in parallel on Polaris (ALCF).

## Directory Structure

```
polaris_run/
├── submit_polaris.pbs      # PBS job submission script
├── run_parallel.sh          # Distributes jobs across GPUs
├── run_single.sh            # Wrapper for single config execution
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

- **Nodes**: 2
- **GPUs per node**: 4
- **Total GPUs**: 8
- **Allocation**: FOUND4CHEM
- **Walltime**: 24 hours

## Execution Plan

### Batch 1 (8 jobs in parallel)
1. cifar10.json → GPU 0
2. cifar10_awb.json → GPU 1
3. cifar100.json → GPU 2
4. cifar100_awb.json → GPU 3
5. mnist.json → GPU 4
6. mnist_awb.json → GPU 5
7. sine.json → GPU 6
8. sine_awb.json → GPU 7

### Batch 2 (3 jobs in parallel - after Batch 1 completes)
9. sine_awb_test.json → GPU 0
10. synthetic_graph.json → GPU 1
11. synthetic_graph_awb.json → GPU 2

## Usage

### 1. Submit Job to Polaris

From the project root directory:

```bash
qsub polaris_run/submit_polaris.pbs
```

### 2. Monitor Job

```bash
# Check job status
qstat -u $USER

# Monitor output (replace JOBID)
tail -f polaris_run/logs/job_JOBID.out

# Check error log
tail -f polaris_run/logs/job_JOBID.err
```

### 3. Check Individual Job Logs

```bash
# View log for specific config
cat polaris_run/logs/cifar10.log

# Monitor in real-time
tail -f polaris_run/logs/mnist_awb.log
```

### 4. View Summary

After job completes:

```bash
cat polaris_run/logs/job_summary.txt
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

## Job Tracking

Each job logs its execution status. The parallel script tracks:
- Job start time
- GPU assignment
- Exit codes
- Success/failure status

At the end, a summary is generated showing which jobs completed successfully and which failed.

## Troubleshooting

### Job Failed

1. Check the job summary:
   ```bash
   cat polaris_run/logs/job_summary.txt
   ```

2. Check individual log for failed config:
   ```bash
   cat polaris_run/logs/<failed_config>.log
   ```

3. Rerun single config manually:
   ```bash
   bash polaris_run/run_single.sh <config_name>.json
   ```

### Environment Issues

If dependencies are missing, the PBS script will create and install them in `venvs/`.
To manually rebuild:

```bash
rm -rf venvs/
qsub polaris_run/submit_polaris.pbs
```

### GPU Not Available

The script uses `CUDA_VISIBLE_DEVICES` to assign GPUs. Check that JAX detects GPUs:

```bash
python -c "import jax; print(jax.devices())"
```

## Files Generated

- `polaris_run/logs/job_JOBID.out` - PBS stdout
- `polaris_run/logs/job_JOBID.err` - PBS stderr
- `polaris_run/logs/<config>.log` - Individual config logs
- `polaris_run/logs/job_summary.txt` - Final summary report
- `results/<dataset>/*.pkl` - Training records
- `results/<dataset>/*.eqx` - Trained models
- `results/<dataset>/figures/*.png` - Plots

## Notes

- Standard and AWB variants of the same dataset share the same output folder
- The `run_single.sh` script automatically extracts the base dataset name
- Jobs run independently and can fail without affecting others
- Summary is always generated showing which jobs succeeded/failed
