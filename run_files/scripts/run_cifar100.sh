#!/bin/bash
# Test script for CIFAR-100 classification (CNN3D)

#-------------------------------------------------------
# Change to project root directory
cd "$(dirname "$0")/.."

#-------------------------------------------------------
# Activate conda environment (adjust for your local setup)
source ~/miniconda3/etc/profile.d/conda.sh

#-------------------------------------------------------
# for JLSE
conda activate jaxss

#-------------------------------------------------------
# Run experiment
python scripts/run.py config/cifar100.json "$@"
