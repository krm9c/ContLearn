#!/bin/bash
# Local CPU testing script with DEBUG configs - very fast testing
# Runs 2 tasks, 50 samples, 2 epochs per condition
# Completes in ~5-10 minutes total on CPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Dataset to test (only sine has debug configs for now)
DATASET="sine"

echo "=========================================="
echo "Local CPU Test (DEBUG MODE) - ${DATASET}"
echo "=========================================="
echo "Date: $(date)"
echo "Mode: 2 tasks, 50 samples, 2 epochs"
echo "Running 4 conditions sequentially on CPU"
echo ""

# All 4 conditions with debug suffix
CONDITIONS=(
    "condition1_baseline_debug"
    "condition2_heuristics_debug"
    "condition3_arch_no_transfer_debug"
    "condition4_awb_full_debug"
)

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_CONFIGS=()
START_TIME=$(date +%s)

for condition in "${CONDITIONS[@]}"; do
    config="${DATASET}_${condition}.json"

    echo "=========================================="
    echo "Testing: ${config}"
    echo "=========================================="
    COND_START=$(date +%s)

    # Run without GPU (JAX will use CPU automatically)
    python run_files/scripts/run.py "kkt_run/configs/${config}" --no-plots > "${LOG_DIR}/${DATASET}_${condition}.log" 2>&1

    COND_END=$(date +%s)
    COND_TIME=$((COND_END - COND_START))

    if [ $? -eq 0 ]; then
        echo "✓ ${config} - SUCCESS (${COND_TIME}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Find the most recent records file
        RECORDS_FILE=$(ls -t kkt_run/results/*records.pkl 2>/dev/null | head -1)

        if [ -n "$RECORDS_FILE" ]; then
            echo "  Generating plots from: $(basename ${RECORDS_FILE})"
            PLOT_DIR="kkt_run/results/${DATASET}_${condition}_plots"
            python kkt_run/scripts/plot_condition.py \
                --records "${RECORDS_FILE}" \
                --output-dir "${PLOT_DIR}" >> "${LOG_DIR}/${DATASET}_${condition}.log" 2>&1

            if [ $? -eq 0 ]; then
                echo "  ✓ Plots saved to: ${PLOT_DIR}"
            else
                echo "  ✗ Plotting failed (check log)"
            fi
        else
            echo "  ⚠ Could not find records file"
        fi
    else
        echo "✗ ${config} - FAILED (${COND_TIME}s)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_CONFIGS+=("${config}")
        echo "  See log: ${LOG_DIR}/${DATASET}_${condition}.log"
    fi
    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "Test Summary - ${DATASET} (DEBUG)"
echo "=========================================="
echo "Success: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"
echo "Total time: ${TOTAL_TIME}s (~$((TOTAL_TIME / 60))m)"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo ""
    echo "Failed configs:"
    for cfg in "${FAILED_CONFIGS[@]}"; do
        echo "  - ${cfg}"
    done
fi

echo ""
echo "Logs saved to: ${LOG_DIR}"
echo "Finished: $(date)"

# Exit with error if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
