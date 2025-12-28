#!/bin/bash
# Submit both SLURM jobs for sine and mnist (conditions 1-4)
# This submits 2 jobs that together run all 8 experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting SINE & MNIST Experiments"
echo "=========================================="
echo ""

# Submit conditions 1&2
echo "Submitting Conditions 1&2 (Baseline + Heuristics)..."
JOB1=$(sbatch "${SCRIPT_DIR}/submit_sine_mnist_cond12.slurm")
JOB1_ID=$(echo ${JOB1} | awk '{print $NF}')
echo "  Job ID: ${JOB1_ID}"
echo ""

# Submit conditions 3&4
echo "Submitting Conditions 3&4 (Arch Search + AWB Full)..."
JOB2=$(sbatch "${SCRIPT_DIR}/submit_sine_mnist_cond34.slurm")
JOB2_ID=$(echo ${JOB2} | awk '{print $NF}')
echo "  Job ID: ${JOB2_ID}"
echo ""

echo "=========================================="
echo "Both jobs submitted!"
echo ""
echo "Job IDs:"
echo "  Conditions 1&2: ${JOB1_ID}"
echo "  Conditions 3&4: ${JOB2_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f kkt_run/kkt/logs/sine_mnist_cond12_${JOB1_ID}.out"
echo "  tail -f kkt_run/kkt/logs/sine_mnist_cond34_${JOB2_ID}.out"
echo ""
echo "Results will be saved according to config 'model_path':"
echo "  Check config files for output directories"
echo "  Typically: kkt_run/results/<dataset>_condition<N>/"
echo "=========================================="
