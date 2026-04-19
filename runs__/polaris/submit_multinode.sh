#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -A datascience
#PBS -N contlearn_multinode

# Multi-node continual learning on Polaris
# Usage: qsub submit_multinode.sh
#   or:  qsub -v CONFIG=path/to/config.json submit_multinode.sh

set -e

NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
NTOTAL=$((NNODES * NGPUS_PER_NODE))

PROJECT_ROOT="/lus/eagle/projects/datascience/vsastry/projects/KRaghavan/ContLearn"
CONFIG="${CONFIG:-runs__/configs/mnist_transformer_baseline.json}"

cd "${PROJECT_ROOT}"

# Activate environment
module load conda/2024-08-08
conda activate /lus/eagle/projects/datascience/vsastry/venv/sophia/ct_new

# GPU settings
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "================================================"
echo "Multi-node training"
echo "  Nodes:     ${NNODES}"
echo "  GPUs/node: ${NGPUS_PER_NODE}"
echo "  Total GPUs:${NTOTAL}"
echo "  Config:    ${CONFIG}"
echo "  Job ID:    ${PBS_JOBID}"
echo "================================================"

# 1 rank per GPU, 4 ranks per node
mpiexec -n ${NTOTAL} -ppn ${NGPUS_PER_NODE} \
    --hostfile ${PBS_NODEFILE} \
    python run_multinode.py "${CONFIG}" --runs 0
