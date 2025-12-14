"""
Training runners for different problem types.

Each runner orchestrates:
- Model initialization
- Dataset loading
- Training loop execution
- Optional AWB pipeline
- Model/record saving
"""

from .regression import train_model_reg, load_regression_checkpoint
from .classification import train_model_class, load_classification_checkpoint
from .graph_classification import train_model_graph, load_graph_checkpoint

__all__ = [
    "train_model_reg",
    "load_regression_checkpoint",
    "train_model_class",
    "load_classification_checkpoint",
    # Added by Claude: Graph classification runner
    "train_model_graph",
    "load_graph_checkpoint",
]
