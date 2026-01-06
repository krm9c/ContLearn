#!/bin/bash
# Single sine experiment runner - runs one config with one run_id
# Usage: run_single.sh <config_file> <run_id> [gpu_id]
#
# Example: run_single.sh sine_condition1_baseline.json 0 0
#          Will run Condition 1, run0 on GPU 0

CONFIG_FILE=$1
RUN_ID=$2
GPU_ID=${3:-0}  # Default to GPU 0 if not specified

if [ -z "${CONFIG_FILE}" ] || [ -z "${RUN_ID}" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: run_single.sh <config_file> <run_id> [gpu_id]"
    echo ""
    echo "Example: run_single.sh sine_condition1_baseline.json 0 0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/runs__/configs/${CONFIG_FILE}"
CONFIG_BASE=$(basename "${CONFIG_FILE}" .json)
LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Extract condition name from config filename (e.g., "sine_condition1_baseline" -> "condition1")
CONDITION=$(echo "${CONFIG_BASE}" | sed -E 's/sine_//' | sed -E 's/_.*//')

# Create output directory for this condition and run
OUTPUT_DIR="${RESULTS_DIR}/${CONFIG_BASE}_run${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# Check if this run already completed successfully
SUCCESS_MARKER="${LOG_DIR}/${CONFIG_BASE}_run${RUN_ID}.success"
if [ -f "${SUCCESS_MARKER}" ]; then
    echo "=========================================="
    echo "SKIPPING: ${CONFIG_FILE} (run${RUN_ID})"
    echo "Reason: Already completed successfully"
    echo "Completed at: $(cat ${SUCCESS_MARKER} | grep Timestamp)"
    echo "=========================================="
    echo ""
    echo "To force re-run, delete: ${SUCCESS_MARKER}"
    exit 0
fi

# Log file for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${CONFIG_BASE}_run${RUN_ID}_${TIMESTAMP}.log"

echo ""
echo "=========================================================================================================="
echo "SINE EXPERIMENT - NEW RUN"
echo "=========================================================================================================="
echo "Config: ${CONFIG_FILE}"
echo "Run ID: ${RUN_ID}"
echo "GPU: ${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Run the experiment
cd "${PROJECT_ROOT}"

# GPU optimization flags
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run with run_id specified via --runs parameter
python run.py "${CONFIG_PATH}" \
    --runs ${RUN_ID} \
    --output-dir "${OUTPUT_DIR}" \
    --model-suffix "${CONFIG_BASE}_run${RUN_ID}" \
    --figures-dir "${OUTPUT_DIR}/figures" \
    > "${LOG_FILE}" 2>&1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: ${CONFIG_FILE} (run${RUN_ID})"
    # Write success marker
    echo "COMPLETED_SUCCESSFULLY" > "${SUCCESS_MARKER}"
    echo "Timestamp: $(date)" >> "${SUCCESS_MARKER}"
    echo "Config: ${CONFIG_FILE}" >> "${SUCCESS_MARKER}"
    echo "Run ID: ${RUN_ID}" >> "${SUCCESS_MARKER}"
    echo "Output: ${OUTPUT_DIR}" >> "${SUCCESS_MARKER}"
else
    echo "FAILED: ${CONFIG_FILE} (run${RUN_ID}) - exit code: ${EXIT_CODE}"
    # Remove success marker if it exists
    rm -f "${SUCCESS_MARKER}"
fi
echo "=========================================="

exit ${EXIT_CODE}
