#!/bin/bash
# Parallel job distributor for KKT server - ALL configs (standard + AWB)
# Distributes 10 configs across available GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/job_summary.txt"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Initialize summary file
echo "" > "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "NEW RUN (ALL CONFIGS): $(date)" >> "${SUMMARY_FILE}"
echo "==========================================================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# All config files (standard + AWB)
CONFIGS=(
    "mnist.json"
    "mnist_awb.json"
    "cifar10.json"
    "cifar10_awb.json"
    "cifar100.json"
    "cifar100_awb.json"
    "sine.json"
    "sine_awb.json"
    "synthetic_graph.json"
    "synthetic_graph_awb.json"
)

# Count existing successful completions
EXISTING_SUCCESS=0
for config in "${CONFIGS[@]}"; do
    if [ -f "${LOG_DIR}/${config%.json}.success" ]; then
        EXISTING_SUCCESS=$((EXISTING_SUCCESS + 1))
    fi
done

if [ ${EXISTING_SUCCESS} -gt 0 ]; then
    echo "Resume mode: ${EXISTING_SUCCESS} configs already completed" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
fi

# Track job status
declare -A JOB_STATUS
declare -A JOB_PIDS

# Automatically detect number of GPUs available
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

# Function to wait for a job and update status
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
echo "Checking for previously completed jobs..."
echo "=========================================="
SKIPPED_COUNT=0
TO_RUN=()

for config in "${CONFIGS[@]}"; do
    SUCCESS_FILE="${LOG_DIR}/${config%.json}.success"
    if [ -f "${SUCCESS_FILE}" ]; then
        echo "  ✓ ${config} - Already completed ($(cat ${SUCCESS_FILE} | grep Timestamp | cut -d' ' -f2-))"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    else
        TO_RUN+=("${config}")
    fi
done

echo ""
echo "Summary: ${SKIPPED_COUNT} completed, ${#TO_RUN[@]} to run"
echo ""

# Update config list to only include jobs that need to run
CONFIGS=("${TO_RUN[@]}")

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "All configs already completed! Nothing to run."
    echo "To force re-run, delete .success files in ${LOG_DIR}/"
    echo "COMPLETED_SUCCESSFULLY" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    echo "All jobs were already completed in previous runs." >> "${SUMMARY_FILE}"
    exit 0
fi

# Launch jobs in batches
echo "=========================================="
echo "Launching ${#CONFIGS[@]} jobs across ${NUM_GPUS} GPUs..."
echo "=========================================="
echo ""

# Calculate number of batches needed
TOTAL_CONFIGS=${#CONFIGS[@]}
BATCH_NUM=0

# Process configs in batches of NUM_GPUS
CONFIG_IDX=0
while [ ${CONFIG_IDX} -lt ${TOTAL_CONFIGS} ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_SIZE=$((TOTAL_CONFIGS - CONFIG_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "BATCH ${BATCH_NUM}: Launching ${BATCH_SIZE} jobs..."

    # Launch jobs for this batch
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        run_job "${CONFIGS[${ACTUAL_IDX}]}" ${i}
    done

    # Wait for batch to complete
    echo ""
    echo "Waiting for batch ${BATCH_NUM} to complete..."
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((CONFIG_IDX + i))
        wait_for_job "${CONFIGS[${ACTUAL_IDX}]}"
    done

    CONFIG_IDX=$((CONFIG_IDX + BATCH_SIZE))
    echo ""
done

# Generate summary
echo "" >> "${SUMMARY_FILE}"
echo "COMPLETED JOBS:" >> "${SUMMARY_FILE}"
echo "---------------" >> "${SUMMARY_FILE}"
for config in "${CONFIGS[@]}"; do
    if [ "${JOB_STATUS[${config}]}" == "SUCCESS" ]; then
        echo "  ✓ ${config}" >> "${SUMMARY_FILE}"
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "FAILED JOBS:" >> "${SUMMARY_FILE}"
echo "------------" >> "${SUMMARY_FILE}"
FAILED_COUNT=0
for config in "${CONFIGS[@]}"; do
    if [ "${JOB_STATUS[${config}]}" == "FAILED" ]; then
        echo "  ✗ ${config}" >> "${SUMMARY_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

if [ ${FAILED_COUNT} -eq 0 ]; then
    echo "  None - All jobs completed successfully!" >> "${SUMMARY_FILE}"
fi

echo "" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "Finished: $(date)" >> "${SUMMARY_FILE}"
echo "Total configs: ${#CONFIGS[@]}" >> "${SUMMARY_FILE}"
echo "Successful: $((${#CONFIGS[@]} - ${FAILED_COUNT}))" >> "${SUMMARY_FILE}"
echo "Failed: ${FAILED_COUNT}" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"

echo ""
echo "All jobs completed!"
echo "Summary written to: ${SUMMARY_FILE}"
