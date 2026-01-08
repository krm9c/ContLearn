#!/bin/bash
# Run MNIST AWB test with minimal epochs

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
CONFIG_DIR="$SCRIPT_DIR/configs"

# Create output directories
mkdir -p "$SCRIPT_DIR/logs"

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

    # GPU optimization flags (conservative, deterministic)
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform

    python "$RUN_SCRIPT" "$CONFIG_DIR/$config" > "$logfile" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $name ($(date))"
    else
        echo "✗ FAILED: $name ($(date))"
        echo "  Check log: $logfile"
    fi
    echo ""
}

echo "========================================"
echo "MNIST AWB Test"
echo "Started at: $(date)"
echo ""

run_config "mnist_test_awb.json" "mnist_test_awb"

echo "MNIST AWB Test completed"
echo "Finished at: $(date)"
