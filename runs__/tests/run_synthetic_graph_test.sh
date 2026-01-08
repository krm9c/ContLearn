#!/bin/bash
# Run all 4 synthetic graph task-shift conditions

# Get the directory of this script and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
RUN_SCRIPT="$REPO_ROOT/run.py"
CONFIG_DIR="$REPO_ROOT/runs__/configs"

# Create output directories
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/results"

# Function to run a config
run_config() {
    local config=$1
    local name=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logfile="$SCRIPT_DIR/logs/${name}_${timestamp}.log"
    echo "========================================"
    echo "Running: $name"
    echo "Config: $config"
    echo "Log: $logfile"
    echo "Started at: $(date)"

    cd "$REPO_ROOT"
    python "$RUN_SCRIPT" "$CONFIG_DIR/$config" > "$logfile" 2>&1
    local status=$?

    if [ $status -eq 0 ]; then
        echo "SUCCESS: $name ($(date))"
        echo "$name" >> "$SCRIPT_DIR/logs/${name}.success"
    else
        echo "FAILED: $name ($(date))"
        echo "  Check log: $logfile"
        echo "$name" >> "$SCRIPT_DIR/logs/${name}.failed"
    fi
    echo ""
    return $status
}

echo "========================================"
echo "Synthetic Graph Task-Shift Test (All 4 Conditions)"
echo "Started at: $(date)"
echo ""

# Run all 4 conditions sequentially
run_config "synthetic_graph_2task_condition1_baseline.json" "synthetic_2task_C1_baseline"
run_config "synthetic_graph_2task_condition2_heuristics.json" "synthetic_2task_C2_heuristics"
run_config "synthetic_graph_2task_condition3_arch_no_transfer.json" "synthetic_2task_C3_arch_no_transfer"
run_config "synthetic_graph_2task_condition4_awb_full.json" "synthetic_2task_C4_awb_full"

echo "========================================"
echo "All Synthetic Graph Tests Completed"
echo "Finished at: $(date)"
echo ""
echo "Check logs in: $SCRIPT_DIR/logs/"
