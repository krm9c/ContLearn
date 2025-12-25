#!/bin/bash
# Run all 4 conditions for MNIST
# Output directed to kkt_run/jlse/logs/

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
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RUN_SCRIPT="$REPO_ROOT/run_files/scripts/run.py"

# Create output directories
mkdir -p "$SCRIPT_DIR/logs"

# Function to run a config
run_config() {
    local config=$1
    local condition=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logfile="$SCRIPT_DIR/logs/mnist_${condition}_${timestamp}.log"

    echo "========================================"
    echo "Running: MNIST - $condition"
    echo "Config: $config"
    echo "Log: $logfile"
    echo "Started at: $(date)"
    echo "========================================"

    python "$RUN_SCRIPT" "$SCRIPT_DIR/configs/$config" > "$logfile" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: MNIST - $condition ($(date))"
    else
        echo "✗ FAILED: MNIST - $condition ($(date))"
        echo "  Check log: $logfile"
    fi
    echo ""
}

echo "========================================"
echo "MNIST - All Conditions"
echo "Started at: $(date)"
echo "========================================"
echo ""

run_config "mnist_condition1_baseline.json" "condition1_baseline"
run_config "mnist_condition2_heuristics.json" "condition2_heuristics"
run_config "mnist_condition3_arch_no_transfer.json" "condition3_arch_no_transfer"
run_config "mnist_condition4_awb_full.json" "condition4_awb_full"

echo "========================================"
echo "MNIST - All conditions completed"
echo "Finished at: $(date)"
echo "========================================"
