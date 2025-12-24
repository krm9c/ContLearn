# Installation Instructions

## For Polaris (CUDA 12.x)

**Automated Installation:**
```bash
cd ~/Projects/ContLearn
source venv/bin/activate  # or conda activate base
bash polaris_run/setup_cuda.sh
```

This installs:
- **JAX with CUDA 12 GPU support** (primary compute engine)
- **PyTorch CPU version** (data loading only)
- All other dependencies

**Manual Installation:**
```bash
# Install base package
pip install -e ".[graph,plotting,dev]"

# Install JAX with CUDA 12
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch CPU (data loading)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
pip install torch-geometric
```

## For Local Development (CPU)

```bash
pip install -e ".[all]"

# Install JAX CPU
pip install jax jaxlib

# Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## For Local Development (CUDA)

**For CUDA 11.8:**
```bash
pip install -e ".[all]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch-geometric
```

**For CUDA 12.x:**
```bash
pip install -e ".[all]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch-geometric
```

## Verification

```bash
python -c "
import jax
import torch
print('JAX Backend:', jax.default_backend())
print('JAX Devices:', jax.devices())
print('PyTorch CUDA:', torch.cuda.is_available())
"
```

**Expected output on Polaris:**
```
JAX Backend: gpu
JAX Devices: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]
PyTorch CUDA: False
```

Note: PyTorch CUDA is False because we install CPU version (data loading only).

## Why This Installation Strategy?

1. **JAX handles all GPU computation** - Faster, better GPU memory management
2. **PyTorch for data loading only** - CPU version is sufficient and smaller
3. **Avoids CUDA library conflicts** - Only JAX needs GPU support
4. **Saves disk space** - No need for duplicate CUDA libraries

## Troubleshooting

**JAX shows `cpu` backend despite CUDA being available:**
```bash
# Try alternative installation
pip uninstall jax jaxlib -y
pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Missing CUDA libraries:**
```bash
# On Polaris, ensure CUDA module is loaded
module load cudatoolkit-standalone
```

**Import errors:**
```bash
# Reinstall everything
bash polaris_run/setup_cuda.sh
```
