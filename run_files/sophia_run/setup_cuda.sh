#!/bin/bash
# CUDA 12.1 Installation Script for Polaris
# Run this after activating your virtual environment

echo "=========================================="
echo "Installing JAX with CUDA + PyTorch CPU"
echo "=========================================="
echo "Note: JAX handles GPU computation, PyTorch used for data loading only"
echo ""

# Check proxy (should be set in bash_profile)
if [ -z "$HTTP_PROXY" ]; then
    echo "ERROR: HTTP_PROXY not set! Check your bash_profile."
    exit 1
fi
echo "Proxy: $HTTP_PROXY"

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
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install base package with extras (no torch/jax yet)
echo "Installing base package..."
if pip install -e ".[graph,plotting,dev]"; then
    echo "✓ Base package installed successfully"
else
    echo "Warning: Base package installation failed, continuing with core dependencies..."
    # Install core dependencies manually
    pip install equinox optax numpy tqdm scikit-learn matplotlib seaborn pytest pytest-cov black isort mypy
fi

# Install JAX with CUDA (let pip figure out dependencies)
echo ""
echo "Installing JAX with CUDA support..."
# First, try uninstalling any existing JAX
pip uninstall -y jax jaxlib 2>/dev/null || true

# Install JAX with CUDA support (let pip resolve versions)
pip install "jax[cuda]"

# Install PyTorch CPU version (datasets/data loading only)
echo ""
echo "Installing PyTorch CPU version (for data loading)..."
pip install torch torchvision torchaudio

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
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"  PyTorch build: {'CPU-only (expected)' if not torch.cuda.is_available() else 'CUDA'}")
    print()
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    print()

try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  JAX devices: {jax.devices()}")

    # Try to use GPU
    if jax.default_backend() == 'cpu':
        print("\n⚠ WARNING: JAX is using CPU backend!")
        print("  This might be due to:")
        print("    1. CUDA libraries not in LD_LIBRARY_PATH")
        print("    2. Incompatible CUDA version")
        print("    3. jaxlib not properly installed with CUDA support")
        print("  Trying alternative JAX installation...")
        sys.exit(1)
    else:
        print("\n✓ JAX GPU support confirmed!")
        sys.exit(0)
except Exception as e:
    print(f"✗ JAX import/check failed: {e}")
    sys.exit(1)
PYEOF

JAX_EXIT=$?

if [ $JAX_EXIT -ne 0 ]; then
    echo ""
    echo "JAX GPU support failed. Keeping JAX as-is (may be CPU-only)."
    echo "If you need GPU support, check CUDA drivers and LD_LIBRARY_PATH."
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
