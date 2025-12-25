#!/bin/bash
### KKT Server - SINE Regression (4 conditions in parallel)
### Runs all 4 conditions on 4 GPUs simultaneously

#SBATCH --job-name=sine_4cond
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=kkt_run/kkt/logs/sine_%j.out
#SBATCH --error=kkt_run/kkt/logs/sine_%j.err

echo ""
echo "=========================================================================================================="
echo "SINE REGRESSION - 4 Conditions Parallel"
echo "=========================================================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Change to submission directory
cd ${SLURM_SUBMIT_DIR}

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
echo ""

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
    echo ""
fi

# Create directories
mkdir -p kkt_run/kkt/logs
mkdir -p kkt_run/kkt/results

# Run all 4 conditions in parallel
bash kkt_run/kkt/run_dataset_parallel.sh sine

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
