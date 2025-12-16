#!/bin/bash
# CUDA 12.1 Installation Script for Polaris
# Run this after activating your virtual environment

set -e

echo "=========================================="
echo "Installing JAX with CUDA 12 + PyTorch CPU"
echo "=========================================="
echo "Note: JAX handles GPU computation, PyTorch used for data loading only"
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    echo ""
else
    echo "WARNING: nvidia-smi not found. Proceeding with installation anyway..."
    echo ""
fi

# Upgrade pip first
pip install --upgrade pip

# Install base package with extras (no torch/jax yet)
echo "Installing base package..."
pip install -e ".[graph,plotting,dev]"

# Install JAX with CUDA 12 FIRST (priority for GPU computation)
echo ""
echo "Installing JAX with CUDA 12..."
# First, try uninstalling any existing JAX
pip uninstall -y jax jaxlib 2>/dev/null || true

# Install JAX with CUDA 12 support
# Using cuda12_pip for systems with CUDA 12.x installed
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch CPU version (datasets/data loading only)
echo ""
echo "Installing PyTorch CPU version (for data loading)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
echo ""
echo "Installing torch-geometric..."
pip install torch-geometric

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

python << 'PYEOF'
import sys

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch build: {'CPU-only' if not torch.cuda.is_available() else 'CUDA'}")
    print()
except Exception as e:
    print(f"PyTorch import failed: {e}")
    sys.exit(1)

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # Try to use GPU
    if jax.default_backend() == 'cpu':
        print("\nWARNING: JAX is using CPU backend!")
        print("This might be due to:")
        print("  1. CUDA libraries not in LD_LIBRARY_PATH")
        print("  2. Incompatible CUDA version")
        print("  3. jaxlib not properly installed with CUDA support")
        print("\nTrying alternative JAX installation...")
        sys.exit(1)
    else:
        print("\n✓ JAX GPU support confirmed!")
except Exception as e:
    print(f"JAX import/check failed: {e}")
    sys.exit(1)
PYEOF

JAX_EXIT=$?

if [ $JAX_EXIT -ne 0 ]; then
    echo ""
    echo "Attempting alternative JAX installation with cuda12_local..."
    pip uninstall -y jax jaxlib
    pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    echo ""
    echo "Re-verifying JAX installation..."
    python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
