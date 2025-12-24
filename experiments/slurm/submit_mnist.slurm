#!/bin/bash
### H200 Cluster SLURM - MNIST Dataset Validation
### 4 conditions × 5 runs = 20 experiments

#SBATCH --job-name=val_mnist
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=experiments/logs/mnist_%j.out
#SBATCH --error=experiments/logs/mnist_%j.err

cd ${SLURM_SUBMIT_DIR}
export PATH="/usr/local/bin:$PATH"

# Initialize conda
eval "$(${CONDA_PATH:-/home/kraghavan/miniconda3/condabin/conda} shell.bash hook)"
conda activate jax__kkt

mkdir -p experiments/results experiments/logs

# Run MNIST experiments
bash experiments/slurm/run_dataset_validation.sh mnist 5

echo "MNIST validation completed at: $(date)"
