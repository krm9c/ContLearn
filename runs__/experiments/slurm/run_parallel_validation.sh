#!/bin/bash
# Parallel job distributor for validation experiments
# Runs experiments across 4 H200 GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../.."
CONFIG_DIR="${SCRIPT_DIR}/../configs"
LOG_DIR="${SCRIPT_DIR}/../logs"
SUMMARY_FILE="${LOG_DIR}/job_summary.txt"

# Change to project root
cd "${PROJECT_DIR}"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Initialize summary file
echo "" > "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "VALIDATION EXPERIMENTS: $(date)" >> "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "Detected ${NUM_GPUS} GPUs via nvidia-smi"
else
    NUM_GPUS=4
    echo "nvidia-smi not available, assuming ${NUM_GPUS} GPUs (H200)"
fi
echo ""

# Number of runs per config
RUNS_PER_CONFIG=5

# Phase selection (from command line argument)
PHASE="${1:-full}"

if [ "${PHASE}" == "quick" ]; then
    echo "=========================================="
    echo "PHASE 1: Quick Validation"
    echo "Datasets: sine, mnist"
    echo "Runs per config: 3"
    echo "=========================================="
    DATASETS=("sine" "mnist")
    RUNS_PER_CONFIG=3
else
    echo "=========================================="
    echo "PHASE 2: Full Validation"
    echo "Datasets: all (6)"
    echo "Runs per config: 5"
    echo "=========================================="
    DATASETS=("sine" "mnist" "permuted_mnist" "cifar10" "cifar100" "synthetic_graph")
    RUNS_PER_CONFIG=5
fi

CONDITIONS=(
    "condition1_baseline"
    "condition2_heuristics"
    "condition3_arch_no_transfer"
    "condition4_awb_full"
)

# Build list of all experiments to run
EXPERIMENTS=()
for dataset in "${DATASETS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        for run_id in $(seq 0 $((RUNS_PER_CONFIG - 1))); do
            config_path="${CONFIG_DIR}/${dataset}/${condition}.json"
            if [ -f "${config_path}" ]; then
                EXPERIMENTS+=("${dataset}:${condition}:${run_id}:${config_path}")
            fi
        done
    done
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo ""

# Track job status
declare -A JOB_STATUS
declare -A JOB_PIDS

# Function to run a single experiment
run_experiment() {
    local exp_spec=$1
    local gpu_id=$2

    IFS=':' read -r dataset condition run_id config_path <<< "${exp_spec}"

    exp_name="${dataset}/${condition}/run_${run_id}"
    log_file="${LOG_DIR}/${dataset}_${condition}_run${run_id}.log"

    echo "[$(date)] Starting: ${exp_name} on GPU ${gpu_id}"

    bash "${SCRIPT_DIR}/run_single_validation.sh" \
        "${config_path}" \
        "${run_id}" \
        "${gpu_id}" \
        > "${log_file}" 2>&1 &

    local pid=$!
    JOB_PIDS[${exp_name}]=${pid}
    echo "  PID: ${pid}"
}

# Function to wait for an experiment
wait_for_experiment() {
    local exp_spec=$1
    IFS=':' read -r dataset condition run_id config_path <<< "${exp_spec}"
    exp_name="${dataset}/${condition}/run_${run_id}"

    local pid=${JOB_PIDS[${exp_name}]}
    if wait ${pid}; then
        JOB_STATUS[${exp_name}]="SUCCESS"
        echo "[$(date)] Completed: ${exp_name} ✓"
    else
        JOB_STATUS[${exp_name}]="FAILED"
        echo "[$(date)] Failed: ${exp_name} ✗"
    fi
}

# Launch experiments in batches
echo "Launching experiments across ${NUM_GPUS} GPUs..."
echo ""

EXP_IDX=0

while [ ${EXP_IDX} -lt ${TOTAL_EXPERIMENTS} ]; do
    BATCH_SIZE=$((TOTAL_EXPERIMENTS - EXP_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "BATCH: Launching ${BATCH_SIZE} experiments..."

    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((EXP_IDX + i))
        GPU_ID=$i
        run_experiment "${EXPERIMENTS[${ACTUAL_IDX}]}" ${GPU_ID}
    done

    echo "Waiting for batch to complete..."
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((EXP_IDX + i))
        wait_for_experiment "${EXPERIMENTS[${ACTUAL_IDX}]}"
    done

    EXP_IDX=$((EXP_IDX + BATCH_SIZE))
    echo ""
done

# Generate summary
echo "" >> "${SUMMARY_FILE}"
echo "RESULTS:" >> "${SUMMARY_FILE}"
FAILED_COUNT=0
SUCCESS_COUNT=0

for exp_spec in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r dataset condition run_id config_path <<< "${exp_spec}"
    exp_name="${dataset}/${condition}/run_${run_id}"

    if [ "${JOB_STATUS[${exp_name}]}" == "SUCCESS" ]; then
        echo "  ✓ ${exp_name}" >> "${SUMMARY_FILE}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ✗ ${exp_name}" >> "${SUMMARY_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "Finished: $(date)" >> "${SUMMARY_FILE}"
echo "Successful: ${SUCCESS_COUNT}" >> "${SUMMARY_FILE}"
echo "Failed: ${FAILED_COUNT}" >> "${SUMMARY_FILE}"

echo ""
echo "All experiments completed!"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"
echo "Summary: ${SUMMARY_FILE}"
