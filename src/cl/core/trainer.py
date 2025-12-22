"""
Main Trainer class for continual learning.

This module combines all mixin classes to create the unified Trainer
that works across all problem types (regression, classification, graph).
"""

import equinox as eqx
import jax

from .losses import LossMixin
from .hamiltonian import HamiltonianMixin
from .loops import TrainingLoopsMixin
from .recording import RecordingMixin

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

class Trainer(LossMixin, HamiltonianMixin, TrainingLoopsMixin, RecordingMixin, eqx.Module):
    """Main Trainer class for continual learning.

    This class combines functionality from four mixins:
    - LossMixin: Loss and metric computation methods
    - HamiltonianMixin: Hamiltonian computation methods for continual learning
    - TrainingLoopsMixin: Training loop methods for all problem types
    - RecordingMixin: Unified metric recording with eigenvalue tracking

    Attributes:
        loss: Loss function type ('mse' or 'class')
        problem: Problem type ('vectors' or 'graph')
        metric: Metric function type ('mse' or 'class')

    Example:
        >>> trainer = Trainer(loss='mse', metric='mse', problem='vectors')
        >>> params, static, opt_state, records = trainer.train__CL(
        ...     train__=(train_loader, exp_loader, val_loader, test_loader),
        ...     params=params, static=static, opt_state=opt_state, optim=optim,
        ...     n_iter=100, problem_type='vectors', loss_type='regression'
        ... )
    """
    loss: str
    problem: str
    metric: str

    def __init__(self, loss='mse', metric='mse', problem='vectors'):
        """Initialize the Trainer.

        Args:
            loss: Loss function type ('mse' for regression, 'class' for classification)
            metric: Metric function type ('mse' for MSE, 'class' for accuracy)
            problem: Problem type ('vectors' for MLP/CNN, 'graph' for GNN)
        """
        self.loss = loss
        self.problem = problem
        self.metric = metric
