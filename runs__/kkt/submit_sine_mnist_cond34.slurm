#!/bin/bash
### KKT Server - SINE & MNIST Conditions 3&4 (4 jobs on 4 GPUs)
### Runs Condition 3 and 4 for both sine and mnist in parallel
### USAGE: Submit from ContLearn directory: sbatch kkt_run/kkt/submit_sine_mnist_cond34.slurm

#SBATCH --job-name=sine_mnist_c34
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=kkt_run/kkt/logs/sine_mnist_cond34_%j.out
#SBATCH --error=kkt_run/kkt/logs/sine_mnist_cond34_%j.err

echo ""
echo "=========================================================================================================="
echo "SINE & MNIST - Conditions 3&4 (Arch Search + AWB Full)"
echo "=========================================================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Change to submission directory (should be ContLearn root)
cd ${SLURM_SUBMIT_DIR}

# Ensure we're in the ContLearn directory
if [ ! -f "run.py" ]; then
    echo "ERROR: Must be run from ContLearn directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Add /usr/local/bin to PATH for SLURM commands
export PATH="/usr/local/bin:$PATH"

# Initialize conda
CONDA_PATH="/home/kraghavan/miniconda3/condabin/conda"
if [ -f "${CONDA_PATH}" ]; then
    eval "$(${CONDA_PATH} shell.bash hook)"
    conda activate jax__kkt
else
    echo "ERROR: Conda not found at ${CONDA_PATH}"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
    echo ""
fi

# Create directories in kkt_run/kkt
mkdir -p kkt_run/kkt/logs
mkdir -p kkt_run/kkt/results

# GPU assignment (4 experiments on 4 GPUs)
JOBS=(
    "sine:condition3_arch_no_transfer:0"
    "sine:condition4_awb_full:1"
    "mnist:condition3_arch_no_transfer:2"
    "mnist:condition4_awb_full:3"
)

echo "=========================================="
echo "Running 4 experiments in parallel:"
echo "  GPU 0: sine - condition3_arch_no_transfer"
echo "  GPU 1: sine - condition4_awb_full"
echo "  GPU 2: mnist - condition3_arch_no_transfer"
echo "  GPU 3: mnist - condition4_awb_full"
echo "=========================================="
echo ""

# Track background processes
declare -A PIDS
declare -A STATUS

# Launch all 4 jobs in parallel
for JOB in "${JOBS[@]}"; do
    IFS=':' read -r DATASET CONDITION GPU_ID <<< "$JOB"

    CONFIG_FILE="${DATASET}_${CONDITION}.json"
    CONFIG_PATH="kkt_run/configs/${CONFIG_FILE}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="kkt_run/kkt/logs/${DATASET}_${CONDITION}_${TIMESTAMP}.log"

    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "ERROR: Config not found: ${CONFIG_PATH}"
        continue
    fi

    echo "[$(date)] Starting: ${DATASET} - ${CONDITION} on GPU ${GPU_ID}"
    echo "  Config: ${CONFIG_PATH}"
    echo "  Log: ${LOG_FILE}"
    echo "  Note: Results directory specified in config file"

    # Run in background with dedicated GPU
    (
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        # GPU optimization flags
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_PREALLOCATE=true
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform

        python run.py "${CONFIG_PATH}" > "${LOG_FILE}" 2>&1
    ) &

    JOB_KEY="${DATASET}_${CONDITION}"
    PIDS[${JOB_KEY}]=$!
    echo "  PID: ${PIDS[${JOB_KEY}]}"
    echo ""
done

# Wait for all jobs to complete
echo "Waiting for all 4 jobs to complete..."
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

for JOB in "${JOBS[@]}"; do
    IFS=':' read -r DATASET CONDITION GPU_ID <<< "$JOB"
    JOB_KEY="${DATASET}_${CONDITION}"

    if [ -n "${PIDS[${JOB_KEY}]}" ]; then
        PID=${PIDS[${JOB_KEY}]}

        if wait ${PID}; then
            STATUS[${JOB_KEY}]="SUCCESS"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "[$(date)] ✓ SUCCESS: ${DATASET} - ${CONDITION}"
        else
            STATUS[${JOB_KEY}]="FAILED"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "[$(date)] ✗ FAILED: ${DATASET} - ${CONDITION}"
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "All jobs completed!"
echo "Successful: ${SUCCESS_COUNT}/4"
echo "Failed: ${FAILED_COUNT}/4"
echo "Completed at: $(date)"
echo "=========================================="
echo ""

# Print status for each job
echo "Job Status Summary:"
for JOB in "${JOBS[@]}"; do
    IFS=':' read -r DATASET CONDITION GPU_ID <<< "$JOB"
    JOB_KEY="${DATASET}_${CONDITION}"
    echo "  ${JOB_KEY}: ${STATUS[${JOB_KEY}]}"
done
echo ""

# Return non-zero if any failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi

exit 0
