"""
MNIST dataset for continual learning classification.

Provides MNIST and Permuted MNIST datasets with experience replay support
for testing continual learning algorithms on image classification.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from typing import Dict, Any

from .base import BaseDataset
from ..config.constants import (
    DEFAULT_INPUT_SIZE_MNIST,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_ROTATION_RANGE,
    DEFAULT_SCALING_RANGE,
    DEFAULT_PERMUTATION_SEED_MULTIPLIER,
)


class MNISTDataset(BaseDataset):
    """MNIST dataset for continual learning classification.

    Features (X): 28x28 grayscale images (1, 28, 28)
    Target (y): Digit labels 0-9

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
        >>> dataset = MNISTDataset(config)
        >>> train_loader, exp_loader = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the MNIST dataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self._n_tasks = config.get('n_task', 5)
        self.rotation_range = config.get('rotation_range', DEFAULT_ROTATION_RANGE)
        self.scaling_range = config.get('scaling_range', DEFAULT_SCALING_RANGE)
        self.train_split = config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)

        # Load MNIST dataset
        print("Loading MNIST dataset")
        my_transforms = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=my_transforms
        )

        # Extract images and labels
        [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
        self.images = torch.stack(self.images, dim=0)
        self.labels = np.array(self.labels)

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting data from {len(self.images)} to {self.debug_limit} samples")
            self.images = self.images[:self.debug_limit]
            self.labels = self.labels[:self.debug_limit]

    def _load_task_data(self, task_id: int) -> None:
        """Load data for a specific MNIST task with transforms.

        Applies rotation and scaling transforms based on task_id to create
        distribution shift between tasks.

        Args:
            task_id: Task identifier (0-indexed)
        """
        # Fixed by Claude: Set deterministic seed for reproducible task generation
        # Critical for accurate CL metrics - ensures task data is identical during training and evaluation
        np.random.seed(task_id * DEFAULT_PERMUTATION_SEED_MULTIPLIER)

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

        # Fixed by Claude: Use deterministic train/test split for reproducibility
        # The seed was already set above, so np.random calls will be deterministic
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
        """Input dimension: (1, 28, 28) -> 784 for FCNN."""
        return DEFAULT_INPUT_SIZE_MNIST * DEFAULT_INPUT_SIZE_MNIST

    @property
    def output_size(self) -> int:
        """Output dimension: 10 classes (digits 0-9)."""
        return 10

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        return self._n_tasks


class PermutedMNISTDataset(BaseDataset):
    """Permuted MNIST dataset for continual learning.

    Each task applies a different random permutation to the pixels,
    creating distinct but related classification problems.

    Args:
        config: Configuration dictionary with:
            - batch_size: Batch size for DataLoaders
            - len_exp_replay: Max experience replay buffer size
            - debug_mode: Enable debug mode with limited data
            - debug_limit: Number of samples in debug mode
            - n_task: Number of tasks (each with unique permutation)
            - permutation_seed_multiplier: Seed multiplier for permutations

    Example:
        >>> config = {'batch_size': 64, 'n_task': 10}
        >>> dataset = PermutedMNISTDataset(config)
        >>> train_loader, exp_loader = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Permuted MNIST dataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self._n_tasks = config.get('n_task', 10)
        self.train_split = config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)
        self.seed_multiplier = config.get('permutation_seed_multiplier', DEFAULT_PERMUTATION_SEED_MULTIPLIER)
        self.image_size = config.get('image_size', DEFAULT_INPUT_SIZE_MNIST)

        # Load MNIST dataset
        print("Loading Permuted MNIST dataset")
        my_transforms = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=my_transforms
        )

        # Extract images and labels
        [self.images, self.labels] = [list(t) for t in zip(*self.dataset)]
        self.images = torch.stack(self.images, dim=0)
        self.labels = np.array(self.labels)

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting data from {len(self.images)} to {self.debug_limit} samples")
            self.images = self.images[:self.debug_limit]
            self.labels = self.labels[:self.debug_limit]

    def _load_task_data(self, task_id: int) -> None:
        """Load data for a specific permutation task.

        Applies a task-specific permutation to all pixels.

        Args:
            task_id: Task identifier (0-indexed)
        """
        X = self.images.clone()
        y = self.labels.copy()

        # Generate task-specific permutation
        rng = np.random.RandomState(seed=task_id * self.seed_multiplier)
        perm = rng.permutation(self.image_size * self.image_size)

        # Flatten, permute, reshape
        # X shape: (N, 1, 28, 28) -> flatten to (N, 784) -> permute -> reshape back
        X = X.view(X.shape[0], -1)[:, perm].view(X.shape[0], 1, self.image_size, self.image_size)

        # Fixed by Claude: Use deterministic train/test split for reproducibility
        # Use the same RNG state for consistent splits across training and evaluation
        n_samples = X.shape[0]
        n_train = int(self.train_split * n_samples)

        train_idx = rng.randint(0, n_samples, n_train)
        test_idx = rng.randint(0, n_samples, n_samples - n_train)

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

    @property
    def input_size(self) -> int:
        """Input dimension: (1, 28, 28) -> 784 for FCNN."""
        return self.image_size * self.image_size

    @property
    def output_size(self) -> int:
        """Output dimension: 10 classes (digits 0-9)."""
        return 10

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        return self._n_tasks
