"""
Continual Learning Framework.

A modular JAX/Equinox-based framework for continual learning with:
- Shared training, validation, and recording infrastructure
- Separate model files (MLP, CNN, GNN)
- Separate dataset files (Sine, MNIST, CIFAR, Graph)
- AWB (Adaptive Weight Basis) support for architecture morphing
"""

from .core.trainer import Trainer

__version__ = "0.1.0"
__all__ = ["Trainer"]
