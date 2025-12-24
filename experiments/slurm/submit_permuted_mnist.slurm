#!/bin/bash
### H200 Cluster SLURM - Permuted MNIST Dataset Validation
### 4 conditions × 5 runs = 20 experiments

#SBATCH --job-name=val_pmnist
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=experiments/logs/permuted_mnist_%j.out
#SBATCH --error=experiments/logs/permuted_mnist_%j.err

cd ${SLURM_SUBMIT_DIR}
export PATH="/usr/local/bin:$PATH"

# Initialize conda
eval "$(${CONDA_PATH:-/home/kraghavan/miniconda3/condabin/conda} shell.bash hook)"
conda activate jax__kkt

mkdir -p experiments/results experiments/logs

# Run Permuted MNIST experiments
bash experiments/slurm/run_dataset_validation.sh permuted_mnist 5

echo "Permuted MNIST validation completed at: $(date)"
