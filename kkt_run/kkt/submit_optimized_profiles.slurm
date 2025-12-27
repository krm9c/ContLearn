#!/bin/bash
### KKT Server - OPTIMIZED PROFILE BENCHMARK
### Tests optimized logging/evaluation intervals on all 4 datasets
### Runs Condition 1 (baseline) + Condition 4 (AWB) for each dataset on 4 GPUs in parallel

#SBATCH --job-name=optimized_profile
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=kkt_run/kkt/logs/optimized_profile_%j.out
#SBATCH --error=kkt_run/kkt/logs/optimized_profile_%j.err

echo ""
echo "=========================================================================================================="
echo "OPTIMIZED PROFILE BENCHMARK - 4 Datasets × 2 Conditions"
echo "=========================================================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Date: $(date)"
echo ""
echo "Datasets: sine, mnist, cifar10, synthetic_graph"
echo "Conditions: 1 (baseline), 4 (AWB)"
echo "Total configs: 8 (2 per dataset)"
echo ""
echo "Optimizations under test:"
echo "  - eval_interval=50 (test eval every 50 epochs, not every epoch)"
echo "  - eigenvalues only in AB phase (not main training)"
echo "  - save_iter=50 (5x less I/O)"
echo ""
echo "Expected speedups:"
echo "  CIFAR10: 30s/iter → 3-5s/iter (6-10x)"
echo "  MNIST: 2.9s/iter → 0.5s/iter (5-6x)"
echo "  Graph: 16.7s/iter → 3s/iter (5x)"
echo "  SINE: Validate no regression"
echo "=========================================="
echo ""

# Change to submission directory
cd ${SLURM_SUBMIT_DIR}

# Add /usr/local/bin to PATH for SLURM commands
export PATH="/usr/local/bin:$PATH"

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"
if [ -f "${CONDA_PATH}" ]; then
    eval "$(${CONDA_PATH} shell.bash hook)"
    conda activate jax__kkt
else
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
    echo ""
fi

# Create directories
mkdir -p kkt_run/kkt/logs
mkdir -p kkt_run/kkt/results

# Run optimized profiles (4 datasets in parallel on 4 GPUs)
bash kkt_run/kkt/run_optimized_profiles.sh

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="
echo ""
echo "Results location:"
echo "  Logs: kkt_run/kkt/logs/*_optimized_*.log"
echo "  Pattern: <dataset>_condition<1|4>_optimized_<timestamp>.log"
echo ""
echo "To analyze speedups:"
echo "  grep 'it/s\\|s/it' kkt_run/kkt/logs/*_optimized_*.log | head -50"
echo ""

exit ${EXIT_CODE}
