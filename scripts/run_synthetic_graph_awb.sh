#!/bin/bash
# Test script for synthetic graph classification with AWB (GCN)

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
python scripts/run.py config/synthetic_graph_awb.json "$@"
