#!/bin/bash
### H200 Cluster SLURM - Synthetic Graph Dataset Validation
### 4 conditions × 5 runs = 20 experiments

#SBATCH --job-name=val_graph
#SBATCH --gpus=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=192G
#SBATCH --time=20:00:00
#SBATCH --output=experiments/logs/synthetic_graph_%j.out
#SBATCH --error=experiments/logs/synthetic_graph_%j.err

cd ${SLURM_SUBMIT_DIR}
export PATH="/usr/local/bin:$PATH"

# Initialize conda
eval "$(${CONDA_PATH:-/home/kraghavan/miniconda3/condabin/conda} shell.bash hook)"
conda activate jax__kkt

mkdir -p experiments/results experiments/logs

# Run Synthetic Graph experiments
bash experiments/slurm/run_dataset_validation.sh synthetic_graph 5

echo "Synthetic Graph validation completed at: $(date)"
