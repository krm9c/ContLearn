#!/bin/bash
# Environment setup script for KKT server
# Initializes conda and activates the jax__kkt environment

# Add /usr/local/bin to PATH for SLURM commands
export PATH="/usr/local/bin:$PATH"

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"

if [ ! -f "${CONDA_PATH}" ]; then
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    echo "Please install miniconda or update CONDA_PATH in this script"
    exit 1
fi

echo "Initializing conda..."
eval "$(${CONDA_PATH} shell.bash hook)"

# Activate the jax__kkt environment
echo "Activating conda environment: jax__kkt"
conda activate jax__kkt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment jax__kkt"
    echo ""
    echo "To create the environment, run:"
    echo "  conda create -n jax__kkt python=3.10"
    echo "  conda activate jax__kkt"
    echo "  pip install -e .[dev]"
    exit 1
fi

echo ""
echo "Environment ready:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo ""

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "WARNING: nvidia-smi not available"
fi
