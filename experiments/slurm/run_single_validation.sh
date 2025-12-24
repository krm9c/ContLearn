#!/bin/bash
# Run a single validation experiment
# Usage: ./run_single_validation.sh <config_path> <run_id> <gpu_id>

if [ $# -lt 3 ]; then
    echo "Usage: $0 <config_path> <run_id> <gpu_id>"
    echo "Example: $0 ../configs/sine/condition1_baseline.json 0 0"
    exit 1
fi

CONFIG_PATH="$1"
RUN_ID="$2"
GPU_ID="$3"

# Get config name without extension
CONFIG_NAME=$(basename "${CONFIG_PATH}" .json)
DATASET=$(basename $(dirname "${CONFIG_PATH}"))

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Output directory
OUTPUT_DIR="experiments/results/${DATASET}/${CONFIG_NAME}/run_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# Copy config to output dir for reproducibility
cp "${CONFIG_PATH}" "${OUTPUT_DIR}/config.json"

# Set random seed based on run_id
SEED=$((42 + RUN_ID))

# Run experiment
echo "[$(date)] Starting: ${DATASET}/${CONFIG_NAME}/run_${RUN_ID} on GPU ${GPU_ID}"
echo "  Config: ${CONFIG_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Seed: ${SEED}"

python run_files/scripts/run.py \
    "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --no-plots

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date)] Completed: ${DATASET}/${CONFIG_NAME}/run_${RUN_ID} ✓"
    touch "${OUTPUT_DIR}/.success"
else
    echo "[$(date)] Failed: ${DATASET}/${CONFIG_NAME}/run_${RUN_ID} ✗ (exit code: ${EXIT_CODE})"
    touch "${OUTPUT_DIR}/.failed"
fi

exit ${EXIT_CODE}
