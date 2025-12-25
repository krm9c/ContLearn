#!/bin/bash
# Forced AWB Test - Verifies architecture search and A/B training occur
# Low threshold (1.01) ensures architecture changes happen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

CONFIG="sine_awb_forced_debug.json"
LOG_FILE="${LOG_DIR}/awb_forced_test.log"

echo "=========================================="
echo "FORCED AWB Test"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Date: $(date)"
echo ""
echo "This test FORCES architecture changes by:"
echo "  - Low threshold: 1.01 (vs normal 1.1)"
echo "  - 3 tasks to see multiple changes"
echo "  - Full A/B transfer enabled"
echo ""
echo "Expected behavior:"
echo "  Task 0: Standard training"
echo "  Task 1: Architecture search + A/B training"
echo "  Task 2: Architecture search + A/B training"
echo "=========================================="
echo ""

# Run the test
echo "Running test..."
python run_files/scripts/run.py "kkt_run/configs/${CONFIG}" --no-plots > "${LOG_FILE}" 2>&1

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test PASSED"
    echo ""

    # Extract key information from log
    echo "Checking for AWB behavior..."
    echo ""

    # Check for architecture searches
    ARCH_SEARCHES=$(grep -c "Architecture search" "${LOG_FILE}" || echo "0")
    echo "  Architecture searches found: ${ARCH_SEARCHES}"

    # Check for A/B training
    AB_TRAINING=$(grep -c "STEP 3b - A/B Matrix Training" "${LOG_FILE}" || echo "0")
    echo "  A/B training phases found: ${AB_TRAINING}"

    # Check for V transformation
    V_TRANSFORM=$(grep -c "STEP 4 - Weight Transformation" "${LOG_FILE}" || echo "0")
    echo "  V transformations found: ${V_TRANSFORM}"

    # Check for architecture changes
    ARCH_CHANGES=$(grep -c "Architecture changed" "${LOG_FILE}" || echo "0")
    echo "  Architecture changes detected: ${ARCH_CHANGES}"

    echo ""

    # Verify AWB actually happened
    if [ ${ARCH_SEARCHES} -ge 2 ] && [ ${AB_TRAINING} -ge 2 ]; then
        echo "✓✓ AWB PIPELINE VERIFIED - Architecture search and A/B training occurred!"
    elif [ ${ARCH_SEARCHES} -ge 1 ]; then
        echo "⚠ PARTIAL AWB - Some architecture search occurred, but may not be on all tasks"
    else
        echo "✗ WARNING - No architecture search detected. Check log for details."
    fi

    echo ""
    echo "Architecture history:"
    grep -A 2 "hidden_sizes" "${LOG_FILE}" | head -20 || echo "  (Not found in log)"

    echo ""

    # Find records file
    RECORDS_FILE=$(ls -t kkt_run/results/*records.pkl 2>/dev/null | head -1)

    if [ -n "$RECORDS_FILE" ]; then
        echo "Records saved: ${RECORDS_FILE}"

        # Generate plots
        PLOT_DIR="kkt_run/results/sine_awb_forced_plots"
        echo "Generating plots to: ${PLOT_DIR}"
        python kkt_run/scripts/plot_condition.py \
            --records "${RECORDS_FILE}" \
            --output-dir "${PLOT_DIR}" >> "${LOG_FILE}" 2>&1

        if [ $? -eq 0 ]; then
            echo "✓ Plots generated"
            echo ""
            echo "Check plots to see:"
            echo "  - Eigenvalue changes (A and B matrices)"
            echo "  - Loss jumps at architecture changes"
            echo "  - Performance improvements"
        fi
    fi

else
    echo "✗ Test FAILED"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "${LOG_FILE}"
fi

echo ""
echo "=========================================="
echo "Full log: ${LOG_FILE}"
echo "=========================================="

exit ${EXIT_CODE}
