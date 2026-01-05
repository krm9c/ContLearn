#!/bin/bash
### KKT Server - SINE Multiple Runs (all 4 conditions)
### Runs multiple runs of all 4 sine conditions in parallel on 4 GPUs
### USAGE: sbatch runs__/experiments/sine/submit_sine_multiple_runs.slurm <num_runs>
###        Default: 3 runs per condition (12 total experiments)

#SBATCH --job-name=sine_multi_runs
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=runs__/experiments/sine/logs/sine_multi_runs_%j.out
#SBATCH --error=runs__/experiments/sine/logs/sine_multi_runs_%j.err

# Get number of runs from command line argument (default: 3)
NUM_RUNS=${1:-3}

echo ""
echo "=========================================================================================================="
echo "SINE EXPERIMENTS - MULTIPLE RUNS (ALL 4 CONDITIONS)"
echo "=========================================================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Number of runs per condition: ${NUM_RUNS}"
echo "Total experiments: $((4 * ${NUM_RUNS}))"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Change to submission directory (should be ContLearn root)
cd ${SLURM_SUBMIT_DIR}

# Ensure we're in the ContLearn directory
if [ ! -f "run.py" ]; then
    echo "ERROR: Must be run from ContLearn directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Add /usr/local/bin to PATH for SLURM commands
export PATH="/usr/local/bin:$PATH"

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"
if [ -f "${CONDA_PATH}" ]; then
    eval "$(${CONDA_PATH} shell.bash hook)"
    conda activate jax__kkt
else
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
    echo ""
fi

# Create necessary directories
mkdir -p runs__/experiments/sine/logs
mkdir -p runs__/experiments/sine/results

# Run the parallel execution script
echo "=========================================="
echo "Launching parallel execution script..."
echo "=========================================="
echo ""

bash runs__/experiments/sine/run_parallel_multiple_runs.sh ${NUM_RUNS}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
else
    echo "SOME EXPERIMENTS FAILED (exit code: ${EXIT_CODE})"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
