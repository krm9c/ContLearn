#!/bin/bash
# Run all 4 conditions for Synthetic Graph (2-task)
# Output directed to logs/

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/kraghavan/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/kraghavan/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/kraghavan/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/kraghavan/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate jax__kkt

# Get the directory of this script and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR"
RUN_SCRIPT="$REPO_ROOT/run.py"
CONFIG_DIR="$REPO_ROOT/runs__/configs"

# Create output directories
mkdir -p "$SCRIPT_DIR/logs"

# Function to run a config
run_config() {
    local config=$1
    local condition=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logfile="$SCRIPT_DIR/logs/synthetic_graph_${condition}_${timestamp}.log"

    echo "========================================"
    echo "Running: Synthetic Graph - $condition"
    echo "Config: $config"
    echo "Log: $logfile"
    echo "Started at: $(date)"

    # GPU optimization flags (conservative, deterministic)
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform

    python "$RUN_SCRIPT" "$CONFIG_DIR/$config" > "$logfile" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: Synthetic Graph - $condition ($(date))"
    else
        echo "✗ FAILED: Synthetic Graph - $condition ($(date))"
        echo "  Check log: $logfile"
    fi
    echo ""
}

echo "========================================"
echo "Synthetic Graph - 2-task Experiment (All 4 Conditions)"
echo "Started at: $(date)"
echo ""

# Run all 4 conditions
run_config "synthetic_graph_2task_condition1_baseline.json" "condition1_baseline"
run_config "synthetic_graph_2task_condition2_heuristics.json" "condition2_heuristics"
run_config "synthetic_graph_2task_condition3_arch_no_transfer.json" "condition3_arch_no_transfer"
run_config "synthetic_graph_2task_condition4_awb_full.json" "condition4_awb_full"

echo "========================================"
echo "Synthetic Graph - All conditions completed"
echo "Finished at: $(date)"