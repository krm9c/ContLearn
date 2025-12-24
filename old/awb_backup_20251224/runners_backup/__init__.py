"""
Training runners for different problem types.

Each runner orchestrates:
- Model initialization
- Dataset loading
- Training loop execution
- Optional AWB pipeline
- Model/record saving

Added by Claude: generic_runner consolidates all runners into one unified implementation.
"""

from .regression import train_model_reg, load_regression_checkpoint
from .classification import train_model_class, load_classification_checkpoint
from .graph_classification import train_model_graph, load_graph_checkpoint
from .generic_runner import train_model, load_checkpoint, create_optimizer

__all__ = [
    # Legacy runners (backward compatibility)
    "train_model_reg",
    "load_regression_checkpoint",
    "train_model_class",
    "load_classification_checkpoint",
    "train_model_graph",
    "load_graph_checkpoint",
    # Generic unified runner (preferred)
    "train_model",
    "load_checkpoint",
    "create_optimizer",
]
