import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np_
import pickle
import optax
import torch_geometric as pyg

from utils.trainer_losses import LossMixin
from utils.trainer_hamiltonian import HamiltonianMixin
from utils.trainer_loops import TrainingLoopsMixin
from utils.trainer_recording import RecordingMixin

jax.config.update("jax_enable_x64", True)


class Trainer(LossMixin, HamiltonianMixin, TrainingLoopsMixin, RecordingMixin, eqx.Module):
    """Main Trainer class for continual learning.

    This class combines functionality from four mixins:
    - LossMixin: Loss and metric computation methods
    - HamiltonianMixin: Hamiltonian computation methods for continual learning
    - TrainingLoopsMixin: Training loop methods for graphs, regression, and classification
    - RecordingMixin: Unified metric recording with eigenvalue tracking
    """
    loss: str
    problem: str
    metric: str
    dict: dict()

    def __init__(self, Loss='mse', metric='mse', problem='vectors'):
        self.loss = Loss
        self.problem = problem
        self.metric = metric
        self.dict = {}

    def writer(self, dict, epoch, string_scalers=['train'], metric_scaler=['training_loss', 'validation_loss', 'loss', 'acc']):
        for (string, metric) in zip(string_scalers, metric_scaler):
            self.writer.add_scalar(str(string), dict[metric], epoch)
        pickle.dump(dict['params'], open("best_ckpt.pkl"), "wb")
