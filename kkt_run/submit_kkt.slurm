#!/bin/bash
### KKT Server SLURM Job Submission Script
### Runs all continual learning configs in parallel across 4 GPUs

#SBATCH --job-name=contlearn_kkt
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --output=kkt_run/logs/%x_%j.out
#SBATCH --error=kkt_run/logs/%x_%j.err

echo ""
echo "=========================================================================================================="
echo "NEW SLURM JOB STARTED"
echo "=========================================================================================================="
echo ""
echo "=========================================="
echo "KKT Continual Learning Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Change to submission directory
cd ${SLURM_SUBMIT_DIR}
echo "Working directory: $(pwd)"

# Add /usr/local/bin to PATH for SLURM commands
export PATH="/usr/local/bin:$PATH"

# Set up environment
echo "Setting up environment..."

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"
if [ -f "${CONDA_PATH}" ]; then
    echo "Initializing conda from: ${CONDA_PATH}"
    eval "$(${CONDA_PATH} shell.bash hook)"
else
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    exit 1
fi

# Activate the jax__kkt environment
echo "Activating conda environment: jax__kkt"
conda activate jax__kkt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment jax__kkt"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Detect and display GPU information
echo "GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "  Total GPUs detected: ${NUM_GPUS}"
    echo "  GPU details:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
else
    echo "  nvidia-smi not available"
fi
echo ""

# Create results and logs directories
mkdir -p kkt_run/results
mkdir -p kkt_run/logs

# Pre-download datasets to avoid parallel download conflicts
echo "Pre-downloading datasets..."
python kkt_run/download_datasets.py
echo ""

# Run parallel jobs
echo "Launching parallel jobs..."
bash kkt_run/run_parallel.sh

# Print summary
echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
cat kkt_run/logs/job_summary.txt

echo ""
echo "Job completed at: $(date)"
