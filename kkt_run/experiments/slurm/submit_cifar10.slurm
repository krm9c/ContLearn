#!/bin/bash
### H200 Cluster SLURM - CIFAR-10 Dataset Validation
### 4 conditions × 5 runs = 20 experiments

#SBATCH --job-name=val_cifar10
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/cifar10_%j.out
#SBATCH --error=experiments/logs/cifar10_%j.err

cd ${SLURM_SUBMIT_DIR}
export PATH="/usr/local/bin:$PATH"

# Initialize conda
eval "$(${CONDA_PATH:-/home/kraghavan/miniconda3/condabin/conda} shell.bash hook)"
conda activate jax__kkt

mkdir -p experiments/results experiments/logs

# Run CIFAR-10 experiments
bash experiments/slurm/run_dataset_validation.sh cifar10 5

echo "CIFAR-10 validation completed at: $(date)"
