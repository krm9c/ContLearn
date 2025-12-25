#!/bin/bash
# Run all conditions for a single dataset
# Usage: ./run_dataset_validation.sh <dataset_name> <num_runs>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <num_runs>"
    echo "Example: $0 sine 5"
    exit 1
fi

DATASET="$1"
NUM_RUNS="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../.."
# Use shared kkt_run/configs/ folder (flat structure)
CONFIG_DIR="${SCRIPT_DIR}/../../configs"
LOG_DIR="${SCRIPT_DIR}/../logs"

cd "${PROJECT_DIR}"

CONDITIONS=(
    "condition1_baseline"
    "condition2_heuristics"
    "condition3_arch_no_transfer"
    "condition4_awb_full"
)

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
else
    NUM_GPUS=4
fi

echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Runs per condition: ${NUM_RUNS}"
echo "GPUs available: ${NUM_GPUS}"
echo "=========================================="
echo ""

# Build experiment list
EXPERIMENTS=()
for condition in "${CONDITIONS[@]}"; do
    for run_id in $(seq 0 $((NUM_RUNS - 1))); do
        # Use flat config naming: dataset_condition.json
        config_path="${CONFIG_DIR}/${DATASET}_${condition}.json"
        if [ -f "${config_path}" ]; then
            EXPERIMENTS+=("${condition}:${run_id}:${config_path}")
        fi
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments: ${TOTAL}"
echo ""

# Track jobs
declare -A JOB_PIDS
declare -A JOB_STATUS

# Run experiments in batches
EXP_IDX=0

while [ ${EXP_IDX} -lt ${TOTAL} ]; do
    BATCH_SIZE=$((TOTAL - EXP_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "BATCH: Launching ${BATCH_SIZE} experiments..."

    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((EXP_IDX + i))
        GPU_ID=$i

        IFS=':' read -r condition run_id config_path <<< "${EXPERIMENTS[${ACTUAL_IDX}]}"
        exp_name="${DATASET}/${condition}/run_${run_id}"
        log_file="${LOG_DIR}/${DATASET}_${condition}_run${run_id}.log"

        echo "[$(date)] Starting: ${exp_name} on GPU ${GPU_ID}"

        bash "${SCRIPT_DIR}/run_single_validation.sh" \
            "${config_path}" \
            "${run_id}" \
            "${GPU_ID}" \
            > "${log_file}" 2>&1 &

        local pid=$!
        JOB_PIDS[${exp_name}]=${pid}
        echo "  PID: ${pid}"
    done

    echo "Waiting for batch to complete..."
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((EXP_IDX + i))
        IFS=':' read -r condition run_id config_path <<< "${EXPERIMENTS[${ACTUAL_IDX}]}"
        exp_name="${DATASET}/${condition}/run_${run_id}"

        local pid=${JOB_PIDS[${exp_name}]}
        if wait ${pid}; then
            JOB_STATUS[${exp_name}]="SUCCESS"
            echo "[$(date)] Completed: ${exp_name} ✓"
        else
            JOB_STATUS[${exp_name}]="FAILED"
            echo "[$(date)] Failed: ${exp_name} ✗"
        fi
    done

    EXP_IDX=$((EXP_IDX + BATCH_SIZE))
    echo ""
done

# Summary
FAILED_COUNT=0
SUCCESS_COUNT=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r condition run_id config_path <<< "${exp}"
    exp_name="${DATASET}/${condition}/run_${run_id}"

    if [ "${JOB_STATUS[${exp_name}]}" == "SUCCESS" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "=========================================="
echo "Dataset ${DATASET} completed!"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"
echo "=========================================="
