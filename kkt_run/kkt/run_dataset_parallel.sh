#!/bin/bash
# Run all 4 conditions for a single dataset in parallel on 4 GPUs
# Usage: ./run_dataset_parallel.sh <dataset_name>
# Example: ./run_dataset_parallel.sh sine

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 sine"
    exit 1
fi

DATASET="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results"

cd "${PROJECT_ROOT}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}/${DATASET}"

# Define the 4 conditions
CONDITIONS=(
    "condition1_baseline"
    "condition2_heuristics"
    "condition3_arch_no_transfer"
    "condition4_awb_full"
)

# GPU assignment (4 GPUs available in KKT)
GPUS=(0 1 2 3)

echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Running 4 conditions in parallel on 4 GPUs"
echo "=========================================="
echo ""

# Track background processes
declare -A PIDS
declare -A STATUS

# Launch 4 conditions in parallel, one per GPU
for i in {0..3}; do
    CONDITION="${CONDITIONS[$i]}"
    GPU_ID="${GPUS[$i]}"
    CONFIG_FILE="${DATASET}_${CONDITION}.json"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
    LOG_FILE="${LOG_DIR}/${DATASET}_${CONDITION}_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "Warning: Config not found: ${CONFIG_PATH}"
        continue
    fi

    echo "[$(date)] Starting: ${DATASET} - ${CONDITION} on GPU ${GPU_ID}"
    echo "  Config: ${CONFIG_PATH}"
    echo "  Log: ${LOG_FILE}"

    # Run in background with dedicated GPU
    (
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        # GPU optimization flags (conservative, deterministic)
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_PREALLOCATE=true
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform
        python "${PROJECT_ROOT}/run.py" "${CONFIG_PATH}" \
            --output-dir "${RESULTS_DIR}/${DATASET}" \
            --figures-dir "${RESULTS_DIR}/${DATASET}/figures" \
            > "${LOG_FILE}" 2>&1
    ) &

    PIDS[${CONDITION}]=$!
    echo "  PID: ${PIDS[${CONDITION}]}"
    echo ""
done

# Wait for all 4 conditions to complete
echo "Waiting for all 4 conditions to complete..."
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

for CONDITION in "${CONDITIONS[@]}"; do
    if [ -n "${PIDS[${CONDITION}]}" ]; then
        PID=${PIDS[${CONDITION}]}

        if wait ${PID}; then
            STATUS[${CONDITION}]="SUCCESS"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "[$(date)] ✓ SUCCESS: ${DATASET} - ${CONDITION}"
        else
            STATUS[${CONDITION}]="FAILED"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "[$(date)] ✗ FAILED: ${DATASET} - ${CONDITION}"
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Dataset ${DATASET} completed!"
echo "Successful: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"
echo "=========================================="
echo ""

# Return non-zero if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
