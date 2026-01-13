#!/bin/bash
### KKT Server - Profiling Benchmark (H200)
### Runs profiling for Condition 1 (baseline) and Condition 4 (AWB) on H200
### USAGE: Submit from ContLearn directory: sbatch runs__/experiments/profiling/submit_profiling_benchmark.slurm
###
### Purpose: Compare GPU utilization and timing between baseline and AWB training
### Expected output: JSON profiling reports in runs__/profiling/results/

#SBATCH --job-name=profile_bench
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=runs__/experiments/profiling/logs/profile_bench_%j.out
#SBATCH --error=runs__/experiments/profiling/logs/profile_bench_%j.err

echo ""
echo "=========================================================================================================="
echo "PROFILING BENCHMARK - Condition 1 (Baseline) vs Condition 4 (AWB)"
echo "=========================================================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Change to submission directory (should be ContLearn root)
cd ${SLURM_SUBMIT_DIR}

# Ensure we're in the ContLearn directory
if [ ! -f "run.py" ]; then
    echo "ERROR: Must be run from ContLearn directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

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
echo "Working directory: $(pwd)"
echo ""

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
    echo ""
fi

# Create directories
mkdir -p runs__/experiments/profiling/logs
mkdir -p runs__/profiling/results

# Profiling configs (Condition 1 and Condition 4)
JOBS=(
    "mnist_condition1_profiling:0"
    "mnist_condition4_profiling:1"
)

echo "=========================================="
echo "Running profiling benchmark:"
echo "  GPU 0: Condition 1 (Baseline) - profiling enabled"
echo "  GPU 1: Condition 4 (AWB Full) - profiling enabled"
echo "=========================================="
echo ""

# Track background processes
declare -A PIDS
declare -A STATUS
declare -A START_TIMES

# Launch both jobs in parallel
for JOB in "${JOBS[@]}"; do
    IFS=':' read -r CONFIG_NAME GPU_ID <<< "$JOB"

    CONFIG_PATH="runs__/configs/${CONFIG_NAME}.json"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="runs__/experiments/profiling/logs/${CONFIG_NAME}_${TIMESTAMP}.log"

    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "ERROR: Config not found: ${CONFIG_PATH}"
        continue
    fi

    echo "[$(date)] Starting: ${CONFIG_NAME} on GPU ${GPU_ID}"
    echo "  Config: ${CONFIG_PATH}"
    echo "  Log: ${LOG_FILE}"

    START_TIMES[${CONFIG_NAME}]=$(date +%s)

    # Run in background with dedicated GPU
    (
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        # GPU optimization flags for H200
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_PREALLOCATE=true
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform
        # XLA flags for better GPU utilization
        export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"
        export TF_GPU_THREAD_MODE=gpu_private
        export TF_GPU_THREAD_COUNT=2

        python run.py "${CONFIG_PATH}" > "${LOG_FILE}" 2>&1
    ) &

    PIDS[${CONFIG_NAME}]=$!
    echo "  PID: ${PIDS[${CONFIG_NAME}]}"
    echo ""
done

# Wait for all jobs to complete
echo "Waiting for profiling jobs to complete..."
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

for JOB in "${JOBS[@]}"; do
    IFS=':' read -r CONFIG_NAME GPU_ID <<< "$JOB"

    if [ -n "${PIDS[${CONFIG_NAME}]}" ]; then
        PID=${PIDS[${CONFIG_NAME}]}

        if wait ${PID}; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIMES[${CONFIG_NAME}]))
            STATUS[${CONFIG_NAME}]="SUCCESS (${DURATION}s)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "[$(date)] ✓ SUCCESS: ${CONFIG_NAME} - Duration: ${DURATION}s"
        else
            STATUS[${CONFIG_NAME}]="FAILED"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "[$(date)] ✗ FAILED: ${CONFIG_NAME}"
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "PROFILING BENCHMARK COMPLETE"
echo "=========================================="
echo "Successful: ${SUCCESS_COUNT}/2"
echo "Failed: ${FAILED_COUNT}/2"
echo "Completed at: $(date)"
echo ""

# Print status for each job
echo "Job Status Summary:"
for JOB in "${JOBS[@]}"; do
    IFS=':' read -r CONFIG_NAME GPU_ID <<< "$JOB"
    echo "  ${CONFIG_NAME}: ${STATUS[${CONFIG_NAME}]}"
done
echo ""

# Print location of results
echo "Results Location:"
echo "  runs__/profiling/results/mnist_condition1_profile/"
echo "  runs__/profiling/results/mnist_condition4_profile/"
echo ""

echo "To analyze results, check the profiling JSON files and compare:"
echo "  - GPU utilization (mean, min, max)"
echo "  - Time breakdown (hamiltonian, optimizer, data_prep)"
echo "  - AWB-specific metrics (A/B training overhead)"
echo ""

# Return non-zero if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi

exit 0
