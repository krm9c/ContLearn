#!/bin/bash
# Parallel Sine Noise experiments runner - runs multiple runs of all 4 conditions
# Distributes experiments across 4 GPUs intelligently
#
# Usage: run_parallel_multiple_runs.sh <num_runs>
#
# Example: run_parallel_multiple_runs.sh 5
#          Will run 5 runs (run0-run4) of each of the 4 conditions = 20 total experiments

NUM_RUNS=${1:-3}  # Default to 3 runs if not specified
NUM_GPUS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/run_summary_$(date +%Y%m%d_%H%M%S).txt"

# Create logs directory
mkdir -p "${LOG_DIR}"

# All Sine Noise config files (4 conditions)
CONFIGS=(
    "sine_noise_condition1_baseline.json"
    "sine_noise_condition2_heuristics.json"
    "sine_noise_condition3_arch_no_transfer.json"
    "sine_noise_condition4_awb_full.json"
)

echo ""
echo "=========================================================================================================="
echo "SINE NOISE EXPERIMENTS - MULTIPLE RUNS ACROSS 4 CONDITIONS"
echo "=========================================================================================================="
echo "Number of runs per condition: ${NUM_RUNS}"
echo "Number of conditions: ${#CONFIGS[@]}"
echo "Total experiments: $((${NUM_RUNS} * ${#CONFIGS[@]}))"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Initialize summary file
echo "SINE NOISE EXPERIMENTS - MULTIPLE RUNS" > "${SUMMARY_FILE}"
echo "Started: $(date)" >> "${SUMMARY_FILE}"
echo "Runs per condition: ${NUM_RUNS}" >> "${SUMMARY_FILE}"
echo "Total experiments: $((${NUM_RUNS} * ${#CONFIGS[@]}))" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Build job list: config:run_id pairs
JOBS=()
for CONFIG in "${CONFIGS[@]}"; do
    for RUN_ID in $(seq 0 $((NUM_RUNS - 1))); do
        JOBS+=("${CONFIG}:${RUN_ID}")
    done
done

# Count existing successful completions
echo "Checking for previously completed experiments..."
EXISTING_SUCCESS=0
SKIPPED_JOBS=()
TO_RUN=()

for JOB in "${JOBS[@]}"; do
    IFS=':' read -r CONFIG RUN_ID <<< "$JOB"
    CONFIG_BASE=$(basename "${CONFIG}" .json)
    SUCCESS_FILE="${LOG_DIR}/${CONFIG_BASE}_run${RUN_ID}.success"

    if [ -f "${SUCCESS_FILE}" ]; then
        echo "  ✓ ${CONFIG} (run${RUN_ID}) - Already completed"
        EXISTING_SUCCESS=$((EXISTING_SUCCESS + 1))
        SKIPPED_JOBS+=("${JOB}")
    else
        TO_RUN+=("${JOB}")
    fi
done

echo ""
echo "Summary: ${EXISTING_SUCCESS} completed, ${#TO_RUN[@]} to run"
echo ""

# Update jobs list to only include jobs that need to run
JOBS=("${TO_RUN[@]}")

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "All experiments already completed! Nothing to run."
    echo "To force re-run, delete .success files in ${LOG_DIR}/"
    echo "COMPLETED_SUCCESSFULLY" >> "${SUMMARY_FILE}"
    echo "All jobs were already completed in previous runs." >> "${SUMMARY_FILE}"
    exit 0
fi

# Track job status
declare -A JOB_STATUS
declare -A JOB_PIDS

# Function to run a single job
run_job() {
    local job_spec=$1
    local gpu_id=$2

    IFS=':' read -r config run_id <<< "$job_spec"
    local config_base=$(basename "${config}" .json)

    echo "[$(date)] Starting: ${config} (run${run_id}) on GPU ${gpu_id}"

    bash "${SCRIPT_DIR}/run_single.sh" "${config}" "${run_id}" "${gpu_id}" &
    local pid=$!

    local job_key="${config_base}_run${run_id}"
    JOB_PIDS[${job_key}]=${pid}
    echo "  PID: ${pid}"
    echo ""
}

# Function to wait for a job and update status
wait_for_job() {
    local job_spec=$1

    IFS=':' read -r config run_id <<< "$job_spec"
    local config_base=$(basename "${config}" .json)
    local job_key="${config_base}_run${run_id}"
    local pid=${JOB_PIDS[${job_key}]}

    if wait ${pid}; then
        JOB_STATUS[${job_key}]="SUCCESS"
        echo "[$(date)] ✓ Completed: ${config} (run${run_id})"
    else
        JOB_STATUS[${job_key}]="FAILED"
        echo "[$(date)] ✗ Failed: ${config} (run${run_id})"
    fi
}

# Launch jobs in batches of NUM_GPUS (4 GPUs at a time)
echo "=========================================="
echo "Launching ${#JOBS[@]} experiments across ${NUM_GPUS} GPUs..."
echo "=========================================="
echo ""

TOTAL_JOBS=${#JOBS[@]}
JOB_IDX=0
BATCH_NUM=0

# Process jobs in batches of NUM_GPUS
while [ ${JOB_IDX} -lt ${TOTAL_JOBS} ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_SIZE=$((TOTAL_JOBS - JOB_IDX))
    if [ ${BATCH_SIZE} -gt ${NUM_GPUS} ]; then
        BATCH_SIZE=${NUM_GPUS}
    fi

    echo "=========================================="
    echo "BATCH ${BATCH_NUM}: Launching ${BATCH_SIZE} jobs..."
    echo "=========================================="

    # Launch jobs for this batch
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((JOB_IDX + i))
        GPU_ID=${i}
        run_job "${JOBS[${ACTUAL_IDX}]}" ${GPU_ID}
    done

    # Wait for batch to complete
    echo "Waiting for batch ${BATCH_NUM} to complete..."
    echo ""
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        ACTUAL_IDX=$((JOB_IDX + i))
        wait_for_job "${JOBS[${ACTUAL_IDX}]}"
    done

    JOB_IDX=$((JOB_IDX + BATCH_SIZE))
    echo ""
done

# Generate summary
echo "" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "RESULTS SUMMARY" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Count successes and failures by condition
for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_BASE=$(basename "${CONFIG}" .json)
    echo "Condition: ${CONFIG_BASE}" >> "${SUMMARY_FILE}"

    SUCCESS_COUNT=0
    FAILED_COUNT=0

    for RUN_ID in $(seq 0 $((NUM_RUNS - 1))); do
        JOB_KEY="${CONFIG_BASE}_run${RUN_ID}"

        if [ "${JOB_STATUS[${JOB_KEY}]}" == "SUCCESS" ]; then
            echo "  ✓ run${RUN_ID}: SUCCESS" >> "${SUMMARY_FILE}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        elif [ "${JOB_STATUS[${JOB_KEY}]}" == "FAILED" ]; then
            echo "  ✗ run${RUN_ID}: FAILED" >> "${SUMMARY_FILE}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        else
            # Was skipped (already completed)
            echo "  - run${RUN_ID}: SKIPPED (already completed)" >> "${SUMMARY_FILE}"
        fi
    done

    echo "  Summary: ${SUCCESS_COUNT}/${NUM_RUNS} successful" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
done

# Overall summary
TOTAL_SUCCESS=0
TOTAL_FAILED=0
for KEY in "${!JOB_STATUS[@]}"; do
    if [ "${JOB_STATUS[${KEY}]}" == "SUCCESS" ]; then
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
done

echo "========================================" >> "${SUMMARY_FILE}"
echo "OVERALL SUMMARY" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"
echo "Total experiments run: ${#JOBS[@]}" >> "${SUMMARY_FILE}"
echo "Successful: ${TOTAL_SUCCESS}" >> "${SUMMARY_FILE}"
echo "Failed: ${TOTAL_FAILED}" >> "${SUMMARY_FILE}"
echo "Skipped (previously completed): ${EXISTING_SUCCESS}" >> "${SUMMARY_FILE}"
echo "Finished: $(date)" >> "${SUMMARY_FILE}"
echo "========================================" >> "${SUMMARY_FILE}"

echo ""
echo "=========================================================================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================================================================================="
echo "Total run: ${#JOBS[@]}"
echo "Successful: ${TOTAL_SUCCESS}"
echo "Failed: ${TOTAL_FAILED}"
echo "Skipped: ${EXISTING_SUCCESS}"
echo ""
echo "Summary written to: ${SUMMARY_FILE}"
echo "Results saved to: ${SCRIPT_DIR}/results/"
echo "=========================================="
echo ""

# Print summary to terminal
cat "${SUMMARY_FILE}"

# Return non-zero if any failed
if [ ${TOTAL_FAILED} -gt 0 ]; then
    exit 1
fi

exit 0