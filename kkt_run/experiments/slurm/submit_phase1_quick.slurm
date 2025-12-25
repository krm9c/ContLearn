#!/bin/bash
### H200 Cluster SLURM Job Submission Script
### Phase 1: Quick Validation (Sine + MNIST, 24 experiments)

#SBATCH --job-name=contlearn_val_quick
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/phase1_quick_%j.out
#SBATCH --error=experiments/logs/phase1_quick_%j.err

echo ""
echo "=========================================================================================================="
echo "VALIDATION EXPERIMENTS - PHASE 1 (QUICK)"
echo "=========================================================================================================="
echo ""
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
mkdir -p experiments/results
mkdir -p experiments/logs

# Run Phase 1 experiments
echo "Launching Phase 1 experiments (quick validation)..."
bash experiments/slurm/run_parallel_validation.sh quick

# Print summary
echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
cat experiments/logs/job_summary.txt

echo ""
echo "Job completed at: $(date)"
