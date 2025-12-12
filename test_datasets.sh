#!/bin/bash
# Test script to verify all datasets work without errors
# Uses minimal epochs (1 epoch, 2 tasks) to quickly check for import/runtime errors


export https_proxy="http://proxy.ftm.alcf.anl.gov:3128"
export http_proxy="http://proxy.ftm.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.ftm.alcf.anl.gov:3128"


#-------------------------------------------------------
## The following is for running on JLSE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax__
#-------------------------------------------------------
echo "=========================================="
echo "Testing ContLearn with all datasets"
echo "=========================================="


#-------------------------------------------------------
# # sine dataset test
echo ""
echo "=========================================="
echo "Testing Sine"
echo "=========================================="
python run.py train 1 "param_sine.json"

#-------------------------------------------------------
# Graph Synthetic
echo ""
echo "=========================================="
echo "Testing Graph Synthetic..."
echo "=========================================="
python run.py train 1 "paramgraph_synthetic.json"


# Test Permuted MNIST
echo ""
echo "=========================================="
echo "Testing Permuted MNIST..."
echo "=========================================="
python run.py train 1 "test_permuted_mnist.json"


# Test CIFAR-10
echo ""
echo "=========================================="
echo "Testing CIFAR-10..."
echo "=========================================="
python run.py train 1 "test_cifar10.json"

# Test CIFAR-100
echo ""
echo "=========================================="
echo "Testing CIFAR-100..."
echo "=========================================="
python run.py train 1 "test_cifar100.json"



echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
