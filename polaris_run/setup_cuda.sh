#!/bin/bash
# CUDA 12.1 Installation Script for Polaris
# Run this after activating your virtual environment

set -e

echo "=========================================="
echo "Installing PyTorch and JAX with CUDA 12.1"
echo "=========================================="
echo ""

# Upgrade pip first
pip install --upgrade pip

# Install base package with extras (no torch/jax yet)
echo "Installing base package..."
pip install -e ".[graph,plotting,dev]"

# Install PyTorch with CUDA 12.1
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric
echo ""
echo "Installing torch-geometric..."
pip install torch-geometric

# Install JAX with CUDA 12
echo ""
echo "Installing JAX with CUDA 12..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

python -c "
import torch
import jax
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA available:', torch.cuda.is_available())
print('PyTorch CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print()
print('JAX version:', jax.__version__)
print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
