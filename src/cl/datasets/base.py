"""
Base dataset interface for continual learning.

This module defines the common interface that all dataset implementations must follow.
The interface ensures consistent task sampling and experience replay across all datasets.

Key concepts:
- Task: A learning episode with specific data distribution
- Experience Replay: Buffer storing samples from previous tasks
- Phase: 'training' or 'testing' data split
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class ContinualDataset(Dataset):
    """PyTorch Dataset wrapper for continual learning data.

    Handles reshaping based on network type (fcnn vs cnn).

    Args:
        config: Configuration dictionary with 'problem' and 'network' keys
        data_x: Input features (numpy array or torch tensor)
        data_y: Target labels/values (numpy array or torch tensor)
    """

    def __init__(self, config: Dict[str, Any], data_x, data_y):
        self.config = config
        self.x = data_x if torch.is_tensor(data_x) else torch.from_numpy(data_x.astype(np.float32))
        self.y = data_y if torch.is_tensor(data_y) else torch.from_numpy(data_y.astype(np.float32))

        # Reshape for fully connected networks (flatten images)
        if self.config.get('problem') == 'classification':
            if self.config.get('network') == 'fcnn':
                self.x = self.x.reshape([-1, 784])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


class BaseDataset(ABC):
    """Abstract base class for continual learning datasets.

    All dataset implementations must inherit from this class and implement
    the required methods. This ensures a consistent interface for:
    - Task generation/sampling
    - Experience replay management
    - Train/test data retrieval

    Attributes:
        config: Configuration dictionary
        exp_x_train: Experience replay buffer for training inputs
        exp_y_train: Experience replay buffer for training targets
        exp_x_test: Experience replay buffer for test inputs
        exp_y_test: Experience replay buffer for test targets
        X_train: Current task training inputs
        y_train: Current task training targets
        X_test: Current task test inputs
        y_test: Current task test targets
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset.

        Args:
            config: Configuration dictionary containing:
                - batch_size: Batch size for DataLoaders
                - len_exp_replay: Maximum experience replay buffer size
                - debug_mode: (optional) Enable debug mode with limited data
                - debug_limit: (optional) Number of samples in debug mode
        """
        self.config = config
        self.batch_size = config.get('batch_size', 64)
        self.len_exp_replay = config.get('len_exp_replay', 20000)
        self.debug_mode = config.get('debug_mode', False)
        self.debug_limit = config.get('debug_limit', 100)

        # Current task data
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[torch.Tensor] = None
        self.y_test: Optional[np.ndarray] = None

        # Experience replay buffers
        self.exp_x_train: List = []
        self.exp_y_train: List = []
        self.exp_x_test: List = []
        self.exp_y_test: List = []

        # Track if experience buffer has been initialized
        self._exp_initialized = False

    @abstractmethod
    def _load_task_data(self, task_id: int) -> None:
        """Internal method to load data for a specific task.

        Subclasses must implement this to populate:
        - self.X_train, self.y_train (training data)
        - self.X_test, self.y_test (test data)

        Args:
            task_id: Task identifier (0-indexed)
        """
        pass

    def load_task(self, task_id: int) -> None:
        """Load data for a specific task with debug limit applied.

        This wrapper calls _load_task_data and then applies debug limits.

        Args:
            task_id: Task identifier (0-indexed)
        """
        # Call the subclass implementation
        self._load_task_data(task_id)

        # Apply debug limit if enabled
        if self.debug_mode:
            self._apply_debug_limit()

    def _apply_debug_limit(self) -> None:
        """Apply debug limit to current task data.

        Limits X_train, y_train, X_test, y_test to debug_limit samples.
        """
        limit = self.debug_limit

        if self.X_train is not None and len(self.X_train) > limit:
            print(f"DEBUG MODE: Limiting training data from {len(self.X_train)} to {limit} samples")
            self.X_train = self.X_train[:limit]
            self.y_train = self.y_train[:limit]

        if self.X_test is not None and len(self.X_test) > limit:
            print(f"DEBUG MODE: Limiting test data from {len(self.X_test)} to {limit} samples")
            self.X_test = self.X_test[:limit]
            self.y_test = self.y_test[:limit]

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Return the input dimension for the model."""
        pass

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return the output dimension for the model."""
        pass

    @property
    @abstractmethod
    def n_tasks(self) -> int:
        """Return the total number of available tasks."""
        pass

    def append_to_experience(self, task_id: int) -> None:
        """Add current task data to the experience replay buffer.

        Manages buffer size by randomly sampling if it exceeds len_exp_replay.

        Args:
            task_id: Current task identifier
        """
        # Convert to tensor if needed
        X_train = self.X_train if torch.is_tensor(self.X_train) else torch.from_numpy(
            np.array(self.X_train, dtype=np.float32))
        X_test = self.X_test if torch.is_tensor(self.X_test) else torch.from_numpy(
            np.array(self.X_test, dtype=np.float32))
        y_train = self.y_train if isinstance(self.y_train, np.ndarray) else np.array(self.y_train)
        y_test = self.y_test if isinstance(self.y_test, np.ndarray) else np.array(self.y_test)

        if not self._exp_initialized:
            # First task: initialize buffers
            self.exp_x_train = X_train.clone()
            self.exp_y_train = y_train.copy()
            self.exp_x_test = X_test.clone()
            self.exp_y_test = y_test.copy()
            self._exp_initialized = True
        else:
            # Subsequent tasks: concatenate
            self.exp_x_train = torch.cat((self.exp_x_train, X_train), dim=0)
            self.exp_y_train = np.concatenate([self.exp_y_train, y_train], axis=0)
            self.exp_x_test = torch.cat((self.exp_x_test, X_test), dim=0)
            self.exp_y_test = np.concatenate([self.exp_y_test, y_test], axis=0)

        # Limit buffer size by random sampling
        if len(self.exp_x_train) > self.len_exp_replay:
            indices = np.random.choice(len(self.exp_x_train), self.len_exp_replay, replace=False)
            self.exp_x_train = self.exp_x_train[indices]
            self.exp_y_train = self.exp_y_train[indices]

        if len(self.exp_x_test) > self.len_exp_replay:
            indices = np.random.choice(len(self.exp_x_test), self.len_exp_replay, replace=False)
            self.exp_x_test = self.exp_x_test[indices]
            self.exp_y_test = self.exp_y_test[indices]

    def get_task_data(self, task_id: int, phase: str) -> Tuple[Tuple, Tuple]:
        """Retrieve current and experience data for a task.

        Args:
            task_id: Task identifier
            phase: 'training' or 'testing'

        Returns:
            Tuple of ((current_x, current_y), (experience_x, experience_y))
        """
        if phase == 'training':
            current = (self.X_train, self.y_train)
            if task_id > 0 and self._exp_initialized:
                experience = (self.exp_x_train, self.exp_y_train)
            else:
                experience = (self.X_train, self.y_train)  # Use current as experience for task 0
        else:  # testing
            current = (self.X_test, self.y_test)
            if task_id > 0 and self._exp_initialized:
                experience = (self.exp_x_test, self.exp_y_test)
            else:
                experience = (self.X_test, self.y_test)

        return current, experience

    def generate_dataset(self, task_id: int, batch_size: int, phase: str) -> Tuple[DataLoader, DataLoader]:
        """Generate DataLoaders for a specific task.

        This is the main interface method called by runners.

        Args:
            task_id: Task identifier (0-indexed)
            batch_size: Batch size for DataLoaders
            phase: 'training' or 'testing'

        Returns:
            Tuple of (current_task_loader, experience_replay_loader)
        """
        # Load task data if in training phase (test uses same task data)
        if phase == 'training':
            self.load_task(task_id)

        # Get current and experience data
        (x_curr, y_curr), (x_exp, y_exp) = self.get_task_data(task_id, phase)

        # Create datasets
        dataset_curr = ContinualDataset(self.config, x_curr, y_curr)
        dataset_exp = ContinualDataset(self.config, x_exp, y_exp)

        # Create DataLoaders
        loader_curr = DataLoader(dataset_curr, batch_size=batch_size, shuffle=True)
        loader_exp = DataLoader(dataset_exp, batch_size=batch_size, shuffle=True)

        return loader_curr, loader_exp

    def get_model_config(self) -> Dict[str, Any]:
        """Return configuration for model initialization.

        Returns:
            Dictionary with input_size, output_size, and any dataset-specific config
        """
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'n_tasks': self.n_tasks,
        }
