#!/bin/bash
# Parallel job distributor for Polaris
# Distributes 11 configs across 8 GPUs (2 nodes × 4 GPUs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/job_summary.txt"

# Create logs directory
mkdir -p "${LOG_DIR}"

# Initialize summary file
echo "Continual Learning Job Summary" > "${SUMMARY_FILE}"
echo "Started: $(date)" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

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

# All config files
CONFIGS=(
    "cifar10.json"
    "cifar10_awb.json"
    "cifar100.json"
    "cifar100_awb.json"
    "mnist.json"
    "mnist_awb.json"
    "sine.json"
    "sine_awb.json"
    "sine_awb_test.json"
    "synthetic_graph.json"
    "synthetic_graph_awb.json"
)

# Track job status
declare -A JOB_STATUS
declare -A JOB_PIDS

# Number of GPUs available
NUM_GPUS=8

# Function to run a single job
run_job() {
    local config_file=$1
    local gpu_id=$2

    echo "[$(date)] Starting: ${config_file} on GPU ${gpu_id}"

    # Run job in background and capture PID
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

# Determine batch sizes
BATCH1_SIZE=$((${#CONFIGS[@]} < ${NUM_GPUS} ? ${#CONFIGS[@]} : ${NUM_GPUS}))
BATCH2_SIZE=$((${#CONFIGS[@]} - ${BATCH1_SIZE}))

# First batch: Launch up to 8 jobs in parallel
if [ ${BATCH1_SIZE} -gt 0 ]; then
    echo "BATCH 1: Launching first ${BATCH1_SIZE} jobs..."
    for i in $(seq 0 $((BATCH1_SIZE - 1))); do
        run_job "${CONFIGS[$i]}" $i
    done
fi

# Wait for first batch to complete
if [ ${BATCH1_SIZE} -gt 0 ]; then
    echo ""
    echo "Waiting for batch 1 to complete..."
    for i in $(seq 0 $((BATCH1_SIZE - 1))); do
        wait_for_job "${CONFIGS[$i]}"
    done
fi

# Second batch: Launch remaining jobs
if [ ${BATCH2_SIZE} -gt 0 ]; then
    echo ""
    echo "BATCH 2: Launching remaining ${BATCH2_SIZE} jobs..."
    for i in $(seq ${BATCH1_SIZE} $((${#CONFIGS[@]} - 1))); do
        gpu_id=$((i - BATCH1_SIZE))  # Use GPUs starting from 0
        run_job "${CONFIGS[$i]}" ${gpu_id}
    done

    # Wait for second batch to complete
    echo ""
    echo "Waiting for batch 2 to complete..."
    for i in $(seq ${BATCH1_SIZE} $((${#CONFIGS[@]} - 1))); do
        wait_for_job "${CONFIGS[$i]}"
    done
fi

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
