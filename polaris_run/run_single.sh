#!/bin/bash
# Single job wrapper for running one config file
# Usage: run_single.sh <config_file>

CONFIG_FILE=$1

if [ -z "${CONFIG_FILE}" ]; then
    echo "Error: No config file provided"
    echo "Usage: run_single.sh <config_file>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/${CONFIG_FILE}"
CONFIG_BASE=$(basename "${CONFIG_FILE}" .json)
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Check if this config already completed successfully
SUCCESS_MARKER="${LOG_DIR}/${CONFIG_BASE}.success"
if [ -f "${SUCCESS_MARKER}" ]; then
    echo "=========================================="
    echo "SKIPPING: ${CONFIG_FILE}"
    echo "Reason: Already completed successfully"
    echo "Completed at: $(cat ${SUCCESS_MARKER} | grep Timestamp)"
    echo "=========================================="
    echo ""
    echo "To force re-run, delete: ${SUCCESS_MARKER}"
    exit 0
fi

# Extract base dataset name (remove _awb, _test suffixes)
DATASET_NAME=$(echo "${CONFIG_BASE}" | sed -E 's/_(awb|test).*$//')

# Create output directory for this dataset
OUTPUT_DIR="${SCRIPT_DIR}/results/${DATASET_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/figures"

echo "=========================================="
echo "Running: ${CONFIG_FILE}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Run the experiment
cd "${PROJECT_ROOT}"

python scripts/run.py "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-suffix "${CONFIG_BASE}" \
    --figures-dir "${OUTPUT_DIR}/figures"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: ${CONFIG_FILE}"
    # Write success marker for resume functionality
    echo "COMPLETED_SUCCESSFULLY" > "${SUCCESS_MARKER}"
    echo "Timestamp: $(date)" >> "${SUCCESS_MARKER}"
    echo "Config: ${CONFIG_FILE}" >> "${SUCCESS_MARKER}"
    echo "Output: ${OUTPUT_DIR}" >> "${SUCCESS_MARKER}"
else
    echo "FAILED: ${CONFIG_FILE} (exit code: ${EXIT_CODE})"
    # Remove success marker if it exists
    rm -f "${SUCCESS_MARKER}"
fi
echo "=========================================="

exit ${EXIT_CODE}
