#!/bin/bash
# Run all optimized profile configs in parallel on 4 GPUs
# Tests Condition 1 (baseline) and Condition 4 (AWB) across all datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/../configs"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Profile configs grouped by dataset (run both conditions for each dataset in parallel)
DATASETS=(
    "sine"
    "mnist"
    "cifar10"
    "synthetic_graph"
)

# GPU assignment (4 GPUs available)
GPUS=(0 1 2 3)

echo "=========================================="
echo "OPTIMIZED PROFILE BENCHMARK"
echo "=========================================="
echo "Running 4 datasets in parallel (Condition 1 & 4 each)"
echo "Each dataset uses 1 GPU for both conditions sequentially"
echo ""
echo "Optimizations:"
echo "  - eval_interval=50 (test eval every 50 epochs, not every epoch)"
echo "  - eigenvalues only in AB phase (Condition 4)"
echo "  - save_iter=50 (5x less I/O)"
echo ""
echo "Expected speedups:"
echo "  CIFAR10: 30s/iter → 3-5s/iter (6-10x)"
echo "  MNIST: 2.9s/iter → 0.5s/iter (5-6x)"
echo "  Graph: 16.7s/iter → 3s/iter (5x)"
echo "  SINE: Validate no regression"
echo "=========================================="
echo ""

# Track background processes
declare -A PIDS
declare -A LOGS

# Function to run both conditions for a dataset on one GPU
run_dataset_conditions() {
    local dataset=$1
    local gpu_id=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)

    # Set GPU
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform

    cd "${PROJECT_ROOT}"

    # Run Condition 1 (baseline)
    local config1="debug/${dataset}_condition1_optimized_profile.json"
    # Fixed by Claude: Create subdirectory for each config to organize logs
    local config_name1="${dataset}_condition1_optimized"
    local log_subdir1="${LOG_DIR}/${config_name1}"
    mkdir -p "${log_subdir1}"
    local log1="${log_subdir1}/${config_name1}_${timestamp}.log"

    echo "[GPU ${gpu_id}] [$(date)] Starting: ${dataset} Condition 1 (Baseline)"
    echo "  Config: ${config1}"
    echo "  Log: ${log1}"

    python run.py "${CONFIG_DIR}/${config1}" > "${log1}" 2>&1
    local exit1=$?

    if [ ${exit1} -eq 0 ]; then
        echo "[GPU ${gpu_id}] [$(date)] ✓ SUCCESS: ${dataset} Condition 1"
    else
        echo "[GPU ${gpu_id}] [$(date)] ✗ FAILED: ${dataset} Condition 1 (exit ${exit1})"
    fi

    # Run Condition 4 (AWB)
    local config4="debug/${dataset}_condition4_optimized_profile.json"
    # Fixed by Claude: Create subdirectory for each config to organize logs
    local config_name4="${dataset}_condition4_optimized"
    local log_subdir4="${LOG_DIR}/${config_name4}"
    mkdir -p "${log_subdir4}"
    local log4="${log_subdir4}/${config_name4}_${timestamp}.log"

    echo "[GPU ${gpu_id}] [$(date)] Starting: ${dataset} Condition 4 (AWB)"
    echo "  Config: ${config4}"
    echo "  Log: ${log4}"

    python run.py "${CONFIG_DIR}/${config4}" > "${log4}" 2>&1
    local exit4=$?

    if [ ${exit4} -eq 0 ]; then
        echo "[GPU ${gpu_id}] [$(date)] ✓ SUCCESS: ${dataset} Condition 4"
    else
        echo "[GPU ${gpu_id}] [$(date)] ✗ FAILED: ${dataset} Condition 4 (exit ${exit4})"
    fi

    # Return non-zero if either failed
    if [ ${exit1} -ne 0 ] || [ ${exit4} -ne 0 ]; then
        return 1
    fi
    return 0
}

# Launch all 4 datasets in parallel, one per GPU
for i in {0..3}; do
    DATASET="${DATASETS[$i]}"
    GPU_ID="${GPUS[$i]}"

    echo "[$(date)] Launching: ${DATASET} on GPU ${GPU_ID} (Condition 1 + 4)"

    # Run in background
    run_dataset_conditions "${DATASET}" "${GPU_ID}" &
    PIDS[${DATASET}]=$!

    echo "  PID: ${PIDS[${DATASET}]}"
    echo ""
done

# Wait for all datasets to complete
echo "========================================"
echo "Waiting for all 4 datasets to complete..."
echo "========================================"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

for DATASET in "${DATASETS[@]}"; do
    if [ -n "${PIDS[${DATASET}]}" ]; then
        PID=${PIDS[${DATASET}]}

        if wait ${PID}; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "[$(date)] ✓ COMPLETED: ${DATASET} (both conditions)"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "[$(date)] ✗ FAILED: ${DATASET} (one or both conditions)"
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "PROFILE BENCHMARK COMPLETE"
echo "=========================================="
echo "Datasets successful: ${SUCCESS_COUNT}/4"
echo "Datasets failed: ${FAILED_COUNT}/4"
echo ""
echo "Logs saved to organized subdirectories in: ${LOG_DIR}/"
echo "  Each config has its own subdirectory"
echo ""
echo "Log directory structure (8 configs total):"
for DATASET in "${DATASETS[@]}"; do
    echo "  ${LOG_DIR}/${DATASET}_condition1_optimized/${DATASET}_condition1_optimized_*.log"
    echo "  ${LOG_DIR}/${DATASET}_condition4_optimized/${DATASET}_condition4_optimized_*.log"
done
echo ""
echo "To analyze speedups:"
echo "  grep 'it/s\\|s/it' ${LOG_DIR}/*/*optimized*.log | head -50"
echo "=========================================="

# Return non-zero if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
