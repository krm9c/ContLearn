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

# Extract base dataset name (remove _awb, _test suffixes)
CONFIG_BASE=$(basename "${CONFIG_FILE}" .json)
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
else
    echo "FAILED: ${CONFIG_FILE} (exit code: ${EXIT_CODE})"
fi
echo "=========================================="

exit ${EXIT_CODE}
