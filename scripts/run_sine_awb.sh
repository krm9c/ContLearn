#!/bin/bash
# Test script for sine regression with AWB (MLP)

#-------------------------------------------------------
# Change to project root directory
cd "$(dirname "$0")/.."

#-------------------------------------------------------
# Activate conda environment (adjust for your local setup)
source ~/miniconda3/etc/profile.d/conda.sh

#-------------------------------------------------------
# for JLSE
# conda activate jax__

#-------------------------------------------------------
# for local
conda activate jaxss

#-------------------------------------------------------
# Run experiment
python scripts/run.py config/sine_awb.json "$@"
