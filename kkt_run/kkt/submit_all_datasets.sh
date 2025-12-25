#!/bin/bash
# Submit all 5 datasets to KKT cluster
# Each dataset runs 4 conditions in parallel on 4 GPUs

echo "=========================================="
echo "Submitting all datasets to KKT cluster"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(
    "sine"
    "mnist"
    "cifar10"
    "cifar100"
    "synthetic_graph"
)

for dataset in "${DATASETS[@]}"; do
    slurm_file="${SCRIPT_DIR}/submit_${dataset}.slurm"

    if [ -f "${slurm_file}" ]; then
        echo "Submitting ${dataset}..."
        JOB_ID=$(sbatch "${slurm_file}" | awk '{print $NF}')
        echo "  Job ID: ${JOB_ID}"
        echo ""
    else
        echo "Warning: SLURM file not found: ${slurm_file}"
        echo ""
    fi
done

echo "=========================================="
echo "All datasets submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: kkt_run/kkt/logs/"
echo "Check results in: kkt_run/kkt/results/"
