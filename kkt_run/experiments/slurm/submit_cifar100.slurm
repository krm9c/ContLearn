#!/bin/bash
### H200 Cluster SLURM - CIFAR-100 Dataset Validation
### 4 conditions × 5 runs = 20 experiments (20 tasks per run)

#SBATCH --job-name=val_cifar100
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/logs/cifar100_%j.out
#SBATCH --error=experiments/logs/cifar100_%j.err

cd ${SLURM_SUBMIT_DIR}
export PATH="/usr/local/bin:$PATH"

# Initialize conda
eval "$(${CONDA_PATH:-/home/kraghavan/miniconda3/condabin/conda} shell.bash hook)"
conda activate jax__kkt

mkdir -p experiments/results experiments/logs

# Run CIFAR-100 experiments
bash experiments/slurm/run_dataset_validation.sh cifar100 5

echo "CIFAR-100 validation completed at: $(date)"
