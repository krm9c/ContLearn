#!/bin/bash
# Master submission script for all experiments
# Submits all datasets and all conditions via SLURM
# USAGE: From ContLearn directory, run: ./runs__/kkt/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================================================================="
echo "SUBMITTING ALL EXPERIMENTS"
echo "=========================================================================================================="
echo "Date: $(date)"
echo ""

# Ensure we're in the ContLearn directory
if [ ! -f "run.py" ]; then
    echo "ERROR: Must be run from ContLearn directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create results directory
mkdir -p runs__/kkt/logs
mkdir -p runs__/kkt/results

# Submit all jobs
echo "Submitting SLURM jobs..."
echo ""

# SINE & MNIST - Conditions 1&2
echo "[1/6] Submitting: SINE & MNIST - Conditions 1&2 (Baseline + Heuristics)"
SINE_MNIST_C12=$(sbatch runs__/kkt/submit_sine_mnist_cond12.slurm | awk '{print $4}')
echo "  Job ID: ${SINE_MNIST_C12}"
echo ""

# SINE & MNIST - Conditions 3&4
echo "[2/6] Submitting: SINE & MNIST - Conditions 3&4 (Arch Search + AWB Full)"
SINE_MNIST_C34=$(sbatch runs__/kkt/submit_sine_mnist_cond34.slurm | awk '{print $4}')
echo "  Job ID: ${SINE_MNIST_C34}"
echo ""

# SYNTHETIC GRAPH - Conditions 1&2
echo "[3/6] Submitting: SYNTHETIC GRAPH - Conditions 1&2 (Baseline + Heuristics)"
SYNTH_C12=$(sbatch runs__/kkt/submit_synthetic_graph_cond12.slurm | awk '{print $4}')
echo "  Job ID: ${SYNTH_C12}"
echo ""

# SYNTHETIC GRAPH - Conditions 3&4
echo "[4/6] Submitting: SYNTHETIC GRAPH - Conditions 3&4 (Arch Search + AWB Full)"
SYNTH_C34=$(sbatch runs__/kkt/submit_synthetic_graph_cond34.slurm | awk '{print $4}')
echo "  Job ID: ${SYNTH_C34}"
echo ""

echo "=========================================================================================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================================================================================="
echo ""
echo "Summary:"
echo "  SINE & MNIST C1&C2: Job ${SINE_MNIST_C12}"
echo "  SINE & MNIST C3&C4: Job ${SINE_MNIST_C34}"
echo "  SYNTHETIC GRAPH C1&C2: Job ${SYNTH_C12}"
echo "  SYNTHETIC GRAPH C3&C4: Job ${SYNTH_C34}"
echo ""
echo "Total: 4 SLURM jobs covering 12 experiments (3 datasets × 4 conditions)"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f runs__/kkt/logs/*.out"
echo ""
echo "Results will be saved to: runs__/kkt/results/"
echo ""
