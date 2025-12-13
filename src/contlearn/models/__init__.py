"""Neural network models for continual learning."""

# MLP models
from .mlp import MLP, MLPorig

# CNN models
from .cnn import CNN, CNN3D, CNNorig

# Graph models
from .graph import (
    GCN,
    GCNorig,
    myNN,
    myNNorig,
    SingleHeadGAT,
    MultiHeadGAT,
    Pool,
    GraphPooling,
    sp_matmul,
)

# Layer building blocks
from .layers import Linear, Linear1, Linear2, Linear3, Dropout

__all__ = [
    # MLP
    "MLP",
    "MLPorig",
    # CNN
    "CNN",
    "CNN3D",
    "CNNorig",
    # Graph
    "GCN",
    "GCNorig",
    "myNN",
    "myNNorig",
    "SingleHeadGAT",
    "MultiHeadGAT",
    "Pool",
    "GraphPooling",
    "sp_matmul",
    # Layers
    "Linear",
    "Linear1",
    "Linear2",
    "Linear3",
    "Dropout",
]
