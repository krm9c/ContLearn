#!/bin/bash
# Test script for CIFAR-10 classification with AWB (CNN3D)

#-------------------------------------------------------
# Change to project root directory
cd "$(dirname "$0")/.."

#-------------------------------------------------------
# Activate conda environment (adjust for your local setup)
source ~/miniconda3/etc/profile.d/conda.sh


conda activate jaxss

#-------------------------------------------------------
# Run experiment
python scripts/run.py config/cifar10_awb.json "$@"
