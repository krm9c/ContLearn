#!/bin/bash
# Test script for SYNTHETIC_GRAPH dataset - all 4 conditions in parallel on 4 GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

DATASET="synthetic_graph"

# All 4 conditions for this dataset
CONFIGS=(
    "${DATASET}_condition1_baseline.json"
    "${DATASET}_condition2_heuristics.json"
    "${DATASET}_condition3_arch_no_transfer.json"
    "${DATASET}_condition4_awb_full.json"
)

echo "=========================================="
echo "Testing ${DATASET} - All 4 Conditions"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Launch all 4 conditions in parallel, one per GPU
declare -A PIDS
for i in {0..3}; do
    config="${CONFIGS[$i]}"
    gpu=$i
    echo "[GPU ${gpu}] Starting: ${config}"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${SCRIPT_DIR}/run_single.sh" "${config}" > "${LOG_DIR}/${config%.json}.log" 2>&1 &
    PIDS[$i]=$!
    echo "  PID: ${PIDS[$i]}"
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo ""

# Wait for all jobs and track results
SUCCESS_COUNT=0
FAILED_COUNT=0
for i in {0..3}; do
    config="${CONFIGS[$i]}"
    if wait ${PIDS[$i]}; then
        echo "[$(date)] ✓ ${config} - SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[$(date)] ✗ ${config} - FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "${DATASET} Test Complete"
echo "=========================================="
echo "Success: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"
echo "Finished: $(date)"
echo ""

# Exit with error if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
