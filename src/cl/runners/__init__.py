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

from .generic_runner import train_model, load_checkpoint, create_optimizer

__all__ = [
    "train_model",
    "load_checkpoint",
    "create_optimizer",
]
