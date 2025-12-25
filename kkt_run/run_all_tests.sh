#!/bin/bash
# Comprehensive test script for KKT server
# Tests all 24 configs: 6 datasets × 4 conditions
# Purpose: Verify all code paths work correctly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/job_summary.txt"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Initialize summary file
echo "" > "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "KKT COMPREHENSIVE TEST RUN: $(date)" >> "${SUMMARY_FILE}"
echo "Testing all 24 configs (6 datasets × 4 conditions)" >> "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# All datasets
DATASETS=(
    "sine"
    "mnist"
    "permuted_mnist"
    "cifar10"
    "cifar100"
    "synthetic_graph"
)

# All conditions
CONDITIONS=(
    "condition1_baseline"
    "condition2_heuristics"
    "condition3_arch_no_transfer"
    "condition4_awb_full"
)

# Build full config list
CONFIGS=()
for dataset in "${DATASETS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        CONFIGS+=("${dataset}_${condition}.json")
    done
done

echo "Total configs to test: ${#CONFIGS[@]}"
echo ""

# Track job status
declare -A JOB_STATUS
declare -A JOB_PIDS

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "Detected ${NUM_GPUS} GPUs via nvidia-smi"
else
    NUM_GPUS=4
    echo "nvidia-smi not available, assuming ${NUM_GPUS} GPUs"
fi
echo ""

# Function to run a single job
run_job() {
    local config_file=$1
    local gpu_id=$2
    echo "[$(date)] Starting: ${config_file} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} bash "${SCRIPT_DIR}/run_single.sh" "${config_file}" > "${LOG_DIR}/${config_file%.json}.log" 2>&1 &
    local pid=$!
    JOB_PIDS[${config_file}]=${pid}
    echo "  PID: ${pid}"
}

# Function to wait for a job
wait_for_job() {
    local config_file=$1
    local pid=${JOB_PIDS[${config_file}]}
    if wait ${pid}; then
        JOB_STATUS[${config_file}]="SUCCESS"
        echo "[$(date)] Completed: ${config_file} ✓"
    else
        JOB_STATUS[${config_file}]="FAILED"
        echo "[$(date)] Failed: ${config_file} ✗"
    fi
}

# Check for previously completed jobs
echo "=========================================="
echo "Checking for completed jobs..."
echo "=========================================="
SKIPPED_COUNT=0
TO_RUN=()

for config in "${CONFIGS[@]}"; do
    SUCCESS_FILE="${LOG_DIR}/${config%.json}.success"
    if [ -f "${SUCCESS_FILE}" ]; then
        echo "  ✓ ${config} - Already completed"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        JOB_STATUS[${config}]="SKIPPED"
    else
        TO_RUN+=("${config}")
    fi
done

echo ""
echo "Summary: ${SKIPPED_COUNT} completed, ${#TO_RUN[@]} to run"
echo ""

CONFIGS_TO_RUN=("${TO_RUN[@]}")

if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
    echo "All configs already completed!"
    exit 0
fi

# Launch jobs in batches
echo "Launching ${#CONFIGS_TO_RUN[@]} jobs across ${NUM_GPUS} GPUs..."
echo ""

TOTAL_CONFIGS=${#CONFIGS_TO_RUN[@]}
CONFIG_IDX=0

while [ ${CONFIG_IDX} -lt ${TOTAL_CONFIGS} ]; do
    BATCH_SIZE=$((TOTAL_CONFIGS - CONFIG_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "=========================================="
    echo "BATCH: Launching ${BATCH_SIZE} jobs (configs ${CONFIG_IDX}-$((CONFIG_IDX + BATCH_SIZE - 1)))"
    echo "=========================================="

    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        run_job "${CONFIGS_TO_RUN[${ACTUAL_IDX}]}" ${i}
    done

    echo ""
    echo "Waiting for batch to complete..."
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        wait_for_job "${CONFIGS_TO_RUN[${ACTUAL_IDX}]}"
    done

    CONFIG_IDX=$((CONFIG_IDX + BATCH_SIZE))
    echo ""
done

# Generate summary by dataset and condition
echo "" >> "${SUMMARY_FILE}"
echo "=========================================="  >> "${SUMMARY_FILE}"
echo "RESULTS BY DATASET"  >> "${SUMMARY_FILE}"
echo "=========================================="  >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

for dataset in "${DATASETS[@]}"; do
    echo "${dataset}:" >> "${SUMMARY_FILE}"
    for condition in "${CONDITIONS[@]}"; do
        config="${dataset}_${condition}.json"
        status="${JOB_STATUS[${config}]}"
        if [ "${status}" == "SUCCESS" ]; then
            echo "  ✓ ${condition}" >> "${SUMMARY_FILE}"
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
        elif [ "${status}" == "SKIPPED" ]; then
            echo "  → ${condition} (skipped, already completed)" >> "${SUMMARY_FILE}"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        else
            echo "  ✗ ${condition}" >> "${SUMMARY_FILE}"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
    done
    echo "" >> "${SUMMARY_FILE}"
done

echo "=========================================="  >> "${SUMMARY_FILE}"
echo "OVERALL SUMMARY"  >> "${SUMMARY_FILE}"
echo "=========================================="  >> "${SUMMARY_FILE}"
echo "Total configs: 24 (6 datasets × 4 conditions)" >> "${SUMMARY_FILE}"
echo "Successful: ${TOTAL_SUCCESS}" >> "${SUMMARY_FILE}"
echo "Skipped (already completed): ${TOTAL_SKIPPED}" >> "${SUMMARY_FILE}"
echo "Failed: ${TOTAL_FAILED}" >> "${SUMMARY_FILE}"
echo "Finished: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

if [ ${TOTAL_FAILED} -eq 0 ]; then
    echo "✓ ALL TESTS PASSED!" >> "${SUMMARY_FILE}"
else
    echo "✗ SOME TESTS FAILED - See logs above" >> "${SUMMARY_FILE}"
fi

echo ""
echo "=========================================="
echo "All jobs completed!"
echo "=========================================="
echo "Successful: ${TOTAL_SUCCESS}"
echo "Skipped: ${TOTAL_SKIPPED}"
echo "Failed: ${TOTAL_FAILED}"
echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""

# Exit with error if any failed
if [ ${TOTAL_FAILED} -gt 0 ]; then
    exit 1
fi
