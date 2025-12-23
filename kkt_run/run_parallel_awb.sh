#!/bin/bash
# Parallel job distributor for KKT server - AWB configs only
# Runs: mnist_awb, cifar10_awb, cifar100_awb, sine_awb, synthetic_graph_awb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/job_summary_awb.txt"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Initialize summary file
echo "" > "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "NEW RUN (AWB): $(date)" >> "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# AWB config files only
CONFIGS=(
    "mnist_awb.json"
    "cifar10_awb.json"
    "cifar100_awb.json"
    "sine_awb.json"
    "synthetic_graph_awb.json"
)

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
echo "Running AWB configs"
echo "=========================================="
SKIPPED_COUNT=0
TO_RUN=()

for config in "${CONFIGS[@]}"; do
    SUCCESS_FILE="${LOG_DIR}/${config%.json}.success"
    if [ -f "${SUCCESS_FILE}" ]; then
        echo "  ✓ ${config} - Already completed"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    else
        TO_RUN+=("${config}")
    fi
done

echo ""
echo "Summary: ${SKIPPED_COUNT} completed, ${#TO_RUN[@]} to run"
echo ""

CONFIGS=("${TO_RUN[@]}")

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "All AWB configs already completed!"
    exit 0
fi

# Launch jobs in batches
echo "Launching ${#CONFIGS[@]} jobs across ${NUM_GPUS} GPUs..."
echo ""

TOTAL_CONFIGS=${#CONFIGS[@]}
CONFIG_IDX=0

while [ ${CONFIG_IDX} -lt ${TOTAL_CONFIGS} ]; do
    BATCH_SIZE=$((TOTAL_CONFIGS - CONFIG_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "BATCH: Launching ${BATCH_SIZE} jobs..."

    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        run_job "${CONFIGS[${ACTUAL_IDX}]}" ${i}
    done

    echo "Waiting for batch to complete..."
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        wait_for_job "${CONFIGS[${ACTUAL_IDX}]}"
    done

    CONFIG_IDX=$((CONFIG_IDX + BATCH_SIZE))
    echo ""
done

# Generate summary
echo "" >> "${SUMMARY_FILE}"
echo "RESULTS:" >> "${SUMMARY_FILE}"
FAILED_COUNT=0
for config in "${CONFIGS[@]}"; do
    if [ "${JOB_STATUS[${config}]}" == "SUCCESS" ]; then
        echo "  ✓ ${config}" >> "${SUMMARY_FILE}"
    else
        echo "  ✗ ${config}" >> "${SUMMARY_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "Finished: $(date)" >> "${SUMMARY_FILE}"
echo "Successful: $((${#CONFIGS[@]} - ${FAILED_COUNT}))" >> "${SUMMARY_FILE}"
echo "Failed: ${FAILED_COUNT}" >> "${SUMMARY_FILE}"

echo ""
echo "All AWB jobs completed!"
echo "Summary: ${SUMMARY_FILE}"
