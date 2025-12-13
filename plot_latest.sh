#!/bin/bash

# Quick plotting script for the most recent training results
# Usage: ./plot_latest.sh [output_directory]

OUTPUT_DIR="${1:-figures}"

echo "======================================"
echo "ContLearn Quick Plotting Script"
echo "======================================"
echo ""

# Find the most recent allruns file
LATEST_ALLRUNS=$(find logdir/model -name "*_allruns.pkl" -type f -print0 | xargs -0 ls -t | head -n 1)

if [ -z "$LATEST_ALLRUNS" ]; then
    echo "No *_allruns.pkl files found in logdir/model/"
    echo ""
    echo "Looking for individual run files..."

    # Try to find most recent individual run file
    LATEST_RUN=$(find logdir/model -name "*_run*_records.pkl" -type f -print0 | xargs -0 ls -t | head -n 1)

    if [ -z "$LATEST_RUN" ]; then
        echo "No training results found in logdir/model/"
        echo "Please run training first: python run.py train <runs> <config.json>"
        exit 1
    else
        echo "Found: $LATEST_RUN"
        echo ""
        echo "Generating plots..."
        python3 plot_results.py "$LATEST_RUN" --output-dir "$OUTPUT_DIR"
    fi
else
    echo "Found: $LATEST_ALLRUNS"
    echo ""
    echo "Generating plots..."
    python3 plot_results.py "$LATEST_ALLRUNS" --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "======================================"
echo "Done! Plots saved to: $OUTPUT_DIR/"
echo "======================================"
echo ""
echo "Generated plots:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null | tail -10
