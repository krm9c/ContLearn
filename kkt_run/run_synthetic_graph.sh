#!/bin/bash
# Run all 4 conditions for Synthetic Graph
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

# Create output directories
mkdir -p jlse/logs

# Function to run a config
run_config() {
    local config=$1
    local condition=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logfile="jlse/logs/synthetic_graph_${condition}_${timestamp}.log"

    echo "========================================"
    echo "Running: Synthetic Graph - $condition"
    echo "Config: $config"
    echo "Log: $logfile"
    echo "Started at: $(date)"
    echo "========================================"

    python ../run_files/scripts/run.py configs/$config > $logfile 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: Synthetic Graph - $condition ($(date))"
    else
        echo "✗ FAILED: Synthetic Graph - $condition ($(date))"
        echo "  Check log: $logfile"
    fi
    echo ""
}

echo "========================================"
echo "SYNTHETIC GRAPH - All Conditions"
echo "Started at: $(date)"
echo "========================================"
echo ""

run_config "synthetic_graph_condition1_baseline.json" "condition1_baseline"
run_config "synthetic_graph_condition2_heuristics.json" "condition2_heuristics"
run_config "synthetic_graph_condition3_arch_no_transfer.json" "condition3_arch_no_transfer"
run_config "synthetic_graph_condition4_awb_full.json" "condition4_awb_full"

echo "========================================"
echo "Synthetic Graph - All conditions completed"
echo "Finished at: $(date)"
echo "========================================"
