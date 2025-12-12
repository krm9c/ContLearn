#!/bin/bash
# Test script to verify all datasets work without errors
# Uses minimal epochs (1 epoch, 2 tasks) to quickly check for import/runtime errors

echo "=========================================="
echo "Testing ContLearn with all datasets"
echo "=========================================="

cd /Users/kraghavan/Desktop/JMLR_paper/ContLearn

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

# # Test Permuted MNIST
# echo ""
# echo "=========================================="
# echo "Testing Permuted MNIST..."
# echo "=========================================="
# python run.py train 1 "test_permuted_mnist.json"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
