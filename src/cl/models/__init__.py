"""
Model architectures for continual learning.

Each model file contains:
- Model class with AWB (A/B matrices) support
- Standard forward pass: model(x)
- AWB forward pass: model.getAWB(x) or model.get_AWBT(x)
"""

from .layers import Linear, LinearGCN, Linear2, Dropout
from .mlp import MLP, create_mlp
from .cnn import CNN, CNN3D, CNNorig
from .gcn import GCN, GCNLayer, Linear3, Pool, GraphPooling

__all__ = [
    "Linear",
    "LinearGCN",
    "Linear2",
    "Dropout",
    "MLP",
    "create_mlp",
    "CNN",
    "CNN3D",
    "CNNorig",
    # Added by Claude: Graph models
    "GCN",
    "GCNLayer",
    "Linear3",
    "Pool",
    "GraphPooling",
]
