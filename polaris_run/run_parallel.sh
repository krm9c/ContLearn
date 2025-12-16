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

# Launch jobs in batches
echo "=========================================="
echo "Launching jobs across ${NUM_GPUS} GPUs..."
echo "=========================================="
echo ""

# First batch: Launch 8 jobs in parallel
echo "BATCH 1: Launching first 8 jobs..."
for i in {0..7}; do
    if [ $i -lt ${#CONFIGS[@]} ]; then
        run_job "${CONFIGS[$i]}" $i
    fi
done

# Wait for first batch to complete
echo ""
echo "Waiting for batch 1 to complete..."
for i in {0..7}; do
    if [ $i -lt ${#CONFIGS[@]} ]; then
        wait_for_job "${CONFIGS[$i]}"
    fi
done

# Second batch: Launch remaining 3 jobs
echo ""
echo "BATCH 2: Launching remaining jobs..."
for i in {8..10}; do
    if [ $i -lt ${#CONFIGS[@]} ]; then
        gpu_id=$((i - 8))  # Use GPUs 0-2 for remaining jobs
        run_job "${CONFIGS[$i]}" ${gpu_id}
    fi
done

# Wait for second batch to complete
echo ""
echo "Waiting for batch 2 to complete..."
for i in {8..10}; do
    if [ $i -lt ${#CONFIGS[@]} ]; then
        wait_for_job "${CONFIGS[$i]}"
    fi
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
