"""
Core training infrastructure for continual learning.

This module contains shared components used across all models and datasets:
- LossMixin: Loss and metric computation
- HamiltonianMixin: Gradient computation with CL regularization
- TrainingLoopsMixin: Unified training loop
- RecordingMixin: Metric recording and eigenvalue tracking
- AWB utilities: Architecture morphing support
- Architecture search utilities: Generic model-agnostic search
"""

from .trainer import Trainer
from .losses import LossMixin
from .hamiltonian import HamiltonianMixin
from .loops import TrainingLoopsMixin
from .recording import RecordingMixin

# Added by Claude: Export architecture search utilities
from .arch_search import (
    search_architecture,
    load_search_config,
    compute_search_loss,
    partition_for_search,
    reinitialize_weights,
)

__all__ = [
    "Trainer",
    "LossMixin",
    "HamiltonianMixin",
    "TrainingLoopsMixin",
    "RecordingMixin",
    # Architecture search
    "search_architecture",
    "load_search_config",
    "compute_search_loss",
    "partition_for_search",
    "reinitialize_weights",
]
