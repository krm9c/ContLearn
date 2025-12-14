"""
CIFAR-10/100 datasets for continual learning classification.

Provides CIFAR-10 and CIFAR-100 datasets with experience replay support
for testing continual learning algorithms on image classification with CNN3D.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from typing import Dict, Any

from .base import BaseDataset
from ..config.constants import (
    DEFAULT_INPUT_SIZE_CIFAR,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_ROTATION_RANGE,
    DEFAULT_SCALING_RANGE,
)


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset for continual learning classification.

    Features (X): 32x32 RGB images (3, 32, 32)
    Target (y): Class labels 0-9

    Task transitions use rotation and scaling transforms to create
    distribution shift between tasks.

    Args:
        config: Configuration dictionary with:
            - batch_size: Batch size for DataLoaders
            - len_exp_replay: Max experience replay buffer size
            - debug_mode: Enable debug mode with limited data
            - debug_limit: Number of samples in debug mode
            - n_task: Number of tasks to use
            - rotation_range: Max rotation angle (default: 180)
            - scaling_range: Scaling range tuple (default: (1, 2))

    Example:
        >>> config = {'batch_size': 64, 'n_task': 5}
        >>> dataset = CIFAR10Dataset(config)
        >>> train_loader, exp_loader = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the CIFAR-10 dataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self._n_tasks = config.get('n_task', 5)
        self.rotation_range = config.get('rotation_range', DEFAULT_ROTATION_RANGE)
        self.scaling_range = config.get('scaling_range', DEFAULT_SCALING_RANGE)
        self.train_split = config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)

        # Load CIFAR-10 dataset
        print("Loading CIFAR-10 dataset")
        my_transforms = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True, transform=my_transforms
        )

        # Extract images and labels
        [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
        self.images = torch.stack(self.images, dim=0)
        self.labels = np.array(self.labels)
        print(f"CIFAR-10 data loaded: {len(self.images)} samples")

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting data from {len(self.images)} to {self.debug_limit} samples")
            self.images = self.images[:self.debug_limit]
            self.labels = self.labels[:self.debug_limit]

    def _load_task_data(self, task_id: int) -> None:
        """Load data for a specific CIFAR-10 task with transforms.

        Applies rotation and scaling transforms based on task_id to create
        distribution shift between tasks.

        Args:
            task_id: Task identifier (0-indexed)
        """
        X = self.images.clone()
        y = self.labels.copy()

        # Apply task-specific transformations
        rot_angle = np.random.random() * self.rotation_range
        scaling_min, scaling_max = self.scaling_range
        scaling = np.random.random() * (scaling_max - scaling_min) + scaling_min

        X = torchvision.transforms.functional.affine(
            X, rot_angle,
            translate=(scaling, scaling),
            scale=1, shear=rot_angle
        )

        # Split the data
        n_samples = X.shape[0]
        n_train = int(self.train_split * n_samples)

        train_idx = np.random.randint(0, n_samples, n_train)
        test_idx = np.random.randint(0, n_samples, n_samples - n_train)

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

    @property
    def input_size(self) -> int:
        """Input dimension: (3, 32, 32) -> 3072 for FCNN."""
        return 3 * DEFAULT_INPUT_SIZE_CIFAR * DEFAULT_INPUT_SIZE_CIFAR

    @property
    def output_size(self) -> int:
        """Output dimension: 10 classes."""
        return 10

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        return self._n_tasks


class CIFAR100Dataset(BaseDataset):
    """CIFAR-100 dataset for continual learning classification.

    Features (X): 32x32 RGB images (3, 32, 32)
    Target (y): Class labels 0-99

    Task transitions use rotation and scaling transforms to create
    distribution shift between tasks.

    Args:
        config: Configuration dictionary with:
            - batch_size: Batch size for DataLoaders
            - len_exp_replay: Max experience replay buffer size
            - debug_mode: Enable debug mode with limited data
            - debug_limit: Number of samples in debug mode
            - n_task: Number of tasks to use
            - rotation_range: Max rotation angle (default: 180)
            - scaling_range: Scaling range tuple (default: (1, 2))
            - n_class: Number of classes (default: 100)

    Example:
        >>> config = {'batch_size': 64, 'n_task': 5, 'n_class': 100}
        >>> dataset = CIFAR100Dataset(config)
        >>> train_loader, exp_loader = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the CIFAR-100 dataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self._n_tasks = config.get('n_task', 5)
        self._n_classes = config.get('n_class', 100)
        self.rotation_range = config.get('rotation_range', DEFAULT_ROTATION_RANGE)
        self.scaling_range = config.get('scaling_range', DEFAULT_SCALING_RANGE)
        self.train_split = config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)

        # Load CIFAR-100 dataset
        print("Loading CIFAR-100 dataset")
        my_transforms = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.CIFAR100(
            './data', train=True, download=True, transform=my_transforms
        )

        # Extract images and labels
        [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
        self.images = torch.stack(self.images, dim=0)
        self.labels = np.array(self.labels)
        print(f"CIFAR-100 data loaded: {len(self.images)} samples")

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting data from {len(self.images)} to {self.debug_limit} samples")
            self.images = self.images[:self.debug_limit]
            self.labels = self.labels[:self.debug_limit]

    def _load_task_data(self, task_id: int) -> None:
        """Load data for a specific CIFAR-100 task with transforms.

        Applies rotation and scaling transforms based on task_id to create
        distribution shift between tasks.

        Args:
            task_id: Task identifier (0-indexed)
        """
        X = self.images.clone()
        y = self.labels.copy()

        # Apply task-specific transformations
        rot_angle = np.random.random() * self.rotation_range
        scaling_min, scaling_max = self.scaling_range
        scaling = np.random.random() * (scaling_max - scaling_min) + scaling_min

        X = torchvision.transforms.functional.affine(
            X, rot_angle,
            translate=(scaling, scaling),
            scale=1, shear=rot_angle
        )

        # Split the data
        n_samples = X.shape[0]
        n_train = int(self.train_split * n_samples)

        train_idx = np.random.randint(0, n_samples, n_train)
        test_idx = np.random.randint(0, n_samples, n_samples - n_train)

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

    @property
    def input_size(self) -> int:
        """Input dimension: (3, 32, 32) -> 3072 for FCNN."""
        return 3 * DEFAULT_INPUT_SIZE_CIFAR * DEFAULT_INPUT_SIZE_CIFAR

    @property
    def output_size(self) -> int:
        """Output dimension: n_class classes."""
        return self._n_classes

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        return self._n_tasks
