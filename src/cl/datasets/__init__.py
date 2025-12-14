"""
Dataset implementations for continual learning.

Each dataset file contains:
- Dataset class with experience replay support
- generate_dataset(task_id, batch_size, phase) method
- append_to_experience(task_id) method
"""

from .base import BaseDataset, ContinualDataset
from .sine import SineDataset, generate_sine_data
from .mnist import MNISTDataset, PermutedMNISTDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .synthetic_graph import (
    BaseGraphDataset,
    SyntheticGraphDataset,
    TUGraphDataset,
    load_graph_dataset,
)

__all__ = [
    "BaseDataset",
    "ContinualDataset",
    "SineDataset",
    "generate_sine_data",
    "MNISTDataset",
    "PermutedMNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    # Added by Claude: Graph datasets
    "BaseGraphDataset",
    "SyntheticGraphDataset",
    "TUGraphDataset",
    "load_graph_dataset",
]
