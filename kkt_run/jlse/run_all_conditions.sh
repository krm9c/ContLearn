#!/bin/bash
# Sequential run of all conditions for sine, mnist, cifar10, cifar100
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
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
RUN_SCRIPT="$REPO_ROOT/run.py"
CONFIG_DIR="$SCRIPT_DIR/../configs"
# Create output directories
mkdir -p "$SCRIPT_DIR/logs"
# Function to run a config
run_config() {
    local config=$1
    local dataset=$2
    local condition=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logfile="$SCRIPT_DIR/logs/${dataset}_${condition}_${timestamp}.log"
    echo "========================================"
    echo "Running: $dataset - $condition"
    echo "Config: $config"
    echo "Log: $logfile"
    echo "Started at: $(date)"
    python "$RUN_SCRIPT" "$CONFIG_DIR/$config" > "$logfile" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $dataset - $condition ($(date))"
        echo "✗ FAILED: $dataset - $condition ($(date))"
        echo "  Check log: $logfile"
    echo ""
}
# Start time
echo "========================================"
echo "Starting sequential runs"
echo "Started at: $(date)"
echo ""
# Sine regression
echo "### SINE REGRESSION ###"
run_config "sine_condition1_baseline.json" "sine" "condition1_baseline"
run_config "sine_condition2_heuristics.json" "sine" "condition2_heuristics"
run_config "sine_condition3_arch_no_transfer.json" "sine" "condition3_arch_no_transfer"
run_config "sine_condition4_awb_full.json" "sine" "condition4_awb_full"
# MNIST
echo "### MNIST ###"
run_config "mnist_condition1_baseline.json" "mnist" "condition1_baseline"
run_config "mnist_condition2_heuristics.json" "mnist" "condition2_heuristics"
run_config "mnist_condition3_arch_no_transfer.json" "mnist" "condition3_arch_no_transfer"
run_config "mnist_condition4_awb_full.json" "mnist" "condition4_awb_full"
# CIFAR-10
echo "### CIFAR-10 ###"
run_config "cifar10_condition1_baseline.json" "cifar10" "condition1_baseline"
run_config "cifar10_condition2_heuristics.json" "cifar10" "condition2_heuristics"
run_config "cifar10_condition3_arch_no_transfer.json" "cifar10" "condition3_arch_no_transfer"
run_config "cifar10_condition4_awb_full.json" "cifar10" "condition4_awb_full"
# CIFAR-100
echo "### CIFAR-100 ###"
run_config "cifar100_condition1_baseline.json" "cifar100" "condition1_baseline"
run_config "cifar100_condition2_heuristics.json" "cifar100" "condition2_heuristics"
run_config "cifar100_condition3_arch_no_transfer.json" "cifar100" "condition3_arch_no_transfer"
run_config "cifar100_condition4_awb_full.json" "cifar100" "condition4_awb_full"
# End time
echo "All runs completed"
echo "Finished at: $(date)"
echo "Logs are in: jlse/logs/"
echo "Results are in: ../outputs/"
