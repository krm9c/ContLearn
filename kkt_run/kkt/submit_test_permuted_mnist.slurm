#!/bin/bash
#SBATCH --job-name=test_permuted_mnist
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --output=kkt_run/logs/test_permuted_mnist_%j.out
#SBATCH --error=kkt_run/logs/test_permuted_mnist_%j.err

echo "=========================================="
echo "Testing PERMUTED_MNIST - All 4 Conditions"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo ""

cd ${SLURM_SUBMIT_DIR}

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"
eval "$(${CONDA_PATH} shell.bash hook)"
conda activate jax__kkt

# Run test
bash kkt_run/test_permuted_mnist.sh
