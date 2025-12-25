#!/bin/bash
# Local CPU testing script - runs all conditions for a dataset sequentially
# No GPU required, uses debug configs for fast testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Dataset to test (pass as argument or default to sine)
DATASET=${1:-sine}

echo "=========================================="
echo "Local CPU Test - ${DATASET}"
echo "=========================================="
echo "Date: $(date)"
echo "Running 4 conditions sequentially on CPU"
echo ""

# All 4 conditions
CONDITIONS=(
    "condition1_baseline"
    "condition2_heuristics"
    "condition3_arch_no_transfer"
    "condition4_awb_full"
)

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_CONFIGS=()

for condition in "${CONDITIONS[@]}"; do
    config="${DATASET}_${condition}.json"

    echo "=========================================="
    echo "Testing: ${config}"
    echo "=========================================="

    # Run without GPU (JAX will use CPU automatically)
    python run_files/scripts/run.py "kkt_run/configs/${config}" --no-plots > "${LOG_DIR}/${DATASET}_${condition}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ ${config} - SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Find the generated records file
        RECORDS_FILE=$(find kkt_run/results -name "*${DATASET}*${condition}*records.pkl" -o -name "*${DATASET}*run0_records.pkl" 2>/dev/null | head -1)

        if [ -z "$RECORDS_FILE" ]; then
            # Try generic pattern
            RECORDS_FILE=$(find kkt_run/results -name "*records.pkl" -newer "${LOG_DIR}/${DATASET}_${condition}.log" 2>/dev/null | head -1)
        fi

        if [ -n "$RECORDS_FILE" ]; then
            echo "  Generating plots..."
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
            echo "  ⚠ Could not find records file for plotting"
        fi
    else
        echo "✗ ${config} - FAILED"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_CONFIGS+=("${config}")
        echo "  See log: ${LOG_DIR}/${DATASET}_${condition}.log"
    fi
    echo ""
done

echo "=========================================="
echo "Test Summary - ${DATASET}"
echo "=========================================="
echo "Success: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"

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
