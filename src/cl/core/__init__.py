"""
Core training infrastructure for continual learning.

This module contains shared components used across all models and datasets:
- LossMixin: Loss and metric computation
- HamiltonianMixin: Gradient computation with CL regularization
- TrainingLoopsMixin: Unified training loop
- RecordingMixin: Metric recording and eigenvalue tracking
- AWB utilities: Architecture morphing support
"""

from .trainer import Trainer
from .losses import LossMixin
from .hamiltonian import HamiltonianMixin
from .loops import TrainingLoopsMixin
from .recording import RecordingMixin

__all__ = [
    "Trainer",
    "LossMixin",
    "HamiltonianMixin",
    "TrainingLoopsMixin",
    "RecordingMixin",
]
