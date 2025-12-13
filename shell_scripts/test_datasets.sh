#!/bin/bash
# Test script to verify all datasets work without errors
# Uses minimal epochs (1 epoch, 2 tasks) to quickly check for import/runtime errors

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
echo "Testing ContLearn with all datasets"
echo "=========================================="
echo "Working directory: $(pwd)"

#-------------------------------------------------------
# # sine dataset test
echo ""
echo "=========================================="
echo "Testing Sine"
echo "=========================================="
python scripts/run.py train 1 "test_param_sine.json"
#-------------------------------------------------------
# AWB-enabled sine dataset test
echo ""
echo "=========================================="
echo "Testing AWB Pipeline (Sine Regression)"
echo "=========================================="
python scripts/run.py train 1 "test_awb_sine.json"

#-------------------------------------------------------
# Graph Synthetic
echo ""
echo "=========================================="
echo "Testing Graph Synthetic..."
echo "=========================================="
python scripts/run.py train 1 "test_paramgraph_synthetic.json"


#-------------------------------------------------------
# Test Permuted MNIST
echo ""
echo "=========================================="
echo "Testing Permuted MNIST..."
echo "=========================================="
python scripts/run.py train 1 "test_permuted_mnist.json"


#-------------------------------------------------------
# Test CIFAR-10
echo ""
echo "=========================================="
echo "Testing CIFAR-10..."
echo "=========================================="
python scripts/run.py train 1 "test_cifar10.json"

#-------------------------------------------------------
# Test CIFAR-100
echo ""
echo "=========================================="
echo "Testing CIFAR-100..."
echo "=========================================="
python scripts/run.py train 1 "test_cifar100.json"


echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="



# #-------------------------------------------------------
# # Graph mutag
# echo ""
# echo "=========================================="
# echo "Testing Graph Synthetic..."
# echo "=========================================="
# python scripts/run.py train 1 "paramgraph_mutag.json"


# #-------------------------------------------------------
# # Test Permuted MNIST
# echo ""
# echo "=========================================="
# echo "Testing Permuted OMNIGLOT..."
# echo "=========================================="
# python scripts/run.py train 1 "paramomni.json"

