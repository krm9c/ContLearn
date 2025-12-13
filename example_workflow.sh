#!/bin/bash

# Example workflow: Train and visualize results
# This script demonstrates a complete workflow for running experiments and generating plots

echo "========================================"
echo "ContLearn Example Workflow"
echo "========================================"
echo ""

# Configuration
NUM_RUNS=1
CONFIG="param_sine.json"
OUTPUT_DIR="figures/example_workflow"

echo "Configuration:"
echo "  - Number of runs: $NUM_RUNS"
echo "  - Config file: $CONFIG"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Run training
echo "Step 1: Running training..."
echo "----------------------------------------"
python run.py train $NUM_RUNS "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo ""
echo "Training completed successfully!"
echo ""

# Step 2: Generate plots
echo "Step 2: Generating plots..."
echo "----------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find the most recent results file
LATEST_FILE=$(find logdir/model -name "*_allruns.pkl" -o -name "*_run*_records.pkl" | xargs ls -t | head -n 1)

if [ -z "$LATEST_FILE" ]; then
    echo "No results file found!"
    exit 1
fi

echo "Plotting results from: $LATEST_FILE"
python plot_results.py "$LATEST_FILE" --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Plotting failed!"
    exit 1
fi

echo ""
echo "Plots generated successfully!"
echo ""

# Step 3: Summary
echo "========================================"
echo "Workflow Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Model checkpoint: $(find logdir/model -name "*.eqx" | xargs ls -t | head -n 1)"
echo "  - Records file: $LATEST_FILE"
echo "  - Figures: $OUTPUT_DIR/"
echo ""
echo "Generated plots:"
ls -lh "$OUTPUT_DIR"/*.png | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "View plots:"
echo "  open $OUTPUT_DIR/"
echo ""
