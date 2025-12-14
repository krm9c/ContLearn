#!/bin/bash
# Test script to verify sine regression dataset works without errors
# Tests both standard CL and AWB-enabled training

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
echo "=========================================="
echo "Testing CL Framework - Sine Regression"
echo "=========================================="
echo "Working directory: $(pwd)"

# #-------------------------------------------------------
# # Sine dataset test (standard CL, no AWB) - Quick test config
# echo ""
# echo "=========================================="
# echo "Test 1: Sine Regression (Standard CL) - Quick"
# echo "=========================================="
# echo "Config: config/test_sine.json"
# python scripts/run.py config/test_sine.json --runs 1

# #-------------------------------------------------------
# # AWB-enabled sine dataset test - Quick test config
# echo ""
# echo "=========================================="
# echo "Test 2: Sine Regression (AWB Pipeline) - Quick"
# echo "=========================================="
# echo "Config: config/test_sine_awb.json"
# python scripts/run.py config/test_sine_awb.json --runs 1

# #-------------------------------------------------------
# # Sine dataset test (standard CL, no AWB) - Full config
# echo ""
# echo "=========================================="
# echo "Test 3: Sine Regression (Standard CL) - Full"
# echo "=========================================="
# echo "Config: config/sine.json"
# python scripts/run.py config/sine.json --runs 1

#-------------------------------------------------------
# AWB-enabled sine dataset test - Full config
echo ""
echo "=========================================="
echo "Test 4: Sine Regression (AWB Pipeline) - Full"
echo "=========================================="
echo "Config: config/sine_awb.json"
python scripts/run.py config/sine_awb.json --runs 1

#-------------------------------------------------------
echo ""
echo "=========================================="
echo "All dataset tests completed!"
echo "=========================================="
