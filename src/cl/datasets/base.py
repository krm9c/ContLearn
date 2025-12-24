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

        # Added by Claude: Task ID tracking for balanced experience replay
        # Each element stores the task_id of the sample at the same index
        self.exp_task_ids_train: Optional[np.ndarray] = None
        self.exp_task_ids_test: Optional[np.ndarray] = None

        # Balanced replay settings
        self.balanced_replay_enabled = config.get('balanced_replay_enabled', True)
        self.recent_task_weight = config.get('recent_task_weight', 0.1)  # 10% for recent
        self.older_tasks_weight = config.get('older_tasks_weight', 0.8)  # 80% for older

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

    def _rebalance_buffer(self, task_id: int, is_train: bool = True) -> None:
        """Rebalance experience buffer with task-weighted sampling.

        Added by Claude: Implements balanced replay with recency weighting.
        Allocation: 10% recent task, 80% older tasks (equal split), 10% random.

        Args:
            task_id: Current task identifier (most recent task)
            is_train: True for training buffer, False for test buffer
        """
        if is_train:
            exp_x = self.exp_x_train
            exp_y = self.exp_y_train
            exp_task_ids = self.exp_task_ids_train
        else:
            exp_x = self.exp_x_test
            exp_y = self.exp_y_test
            exp_task_ids = self.exp_task_ids_test

        if len(exp_x) <= self.len_exp_replay:
            return  # No rebalancing needed

        n_tasks = task_id + 1

        if not self.balanced_replay_enabled or n_tasks == 1:
            # Disabled or first task: simple random sampling
            indices = np.random.choice(len(exp_x), self.len_exp_replay, replace=False)
        else:
            # Weighted allocation: 10% recent, 80% older, 10% random
            recent_quota = int(self.recent_task_weight * self.len_exp_replay)
            older_total = int(self.older_tasks_weight * self.len_exp_replay)
            older_quota_per_task = older_total // (n_tasks - 1) if n_tasks > 1 else 0

            selected_indices = []

            for task in range(n_tasks):
                task_mask = exp_task_ids == task
                task_indices = np.where(task_mask)[0]

                if task == task_id:
                    quota = recent_quota
                else:
                    quota = older_quota_per_task

                # Don't exceed available samples
                quota = min(quota, len(task_indices))

                if quota > 0 and len(task_indices) > 0:
                    selected = np.random.choice(task_indices, quota, replace=False)
                    selected_indices.extend(selected.tolist())

            # Add random samples for remaining quota (10% diversity)
            random_quota = self.len_exp_replay - len(selected_indices)
            if random_quota > 0:
                remaining = set(range(len(exp_x))) - set(selected_indices)
                if remaining:
                    random_samples = np.random.choice(
                        list(remaining),
                        min(random_quota, len(remaining)),
                        replace=False
                    )
                    selected_indices.extend(random_samples.tolist())

            indices = np.array(selected_indices)

        # Apply selection
        if is_train:
            self.exp_x_train = self.exp_x_train[indices]
            self.exp_y_train = self.exp_y_train[indices]
            self.exp_task_ids_train = self.exp_task_ids_train[indices]
        else:
            self.exp_x_test = self.exp_x_test[indices]
            self.exp_y_test = self.exp_y_test[indices]
            self.exp_task_ids_test = self.exp_task_ids_test[indices]

    def append_to_experience(self, task_id: int) -> None:
        """Add current task data to the experience replay buffer.

        Manages buffer size using task-balanced sampling when enabled.

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

        # Create task ID arrays for new samples
        new_task_ids_train = np.full(len(X_train), task_id, dtype=np.int8)
        new_task_ids_test = np.full(len(X_test), task_id, dtype=np.int8)

        if not self._exp_initialized:
            # First task: initialize buffers
            self.exp_x_train = X_train.clone()
            self.exp_y_train = y_train.copy()
            self.exp_x_test = X_test.clone()
            self.exp_y_test = y_test.copy()
            self.exp_task_ids_train = new_task_ids_train
            self.exp_task_ids_test = new_task_ids_test
            self._exp_initialized = True
        else:
            # Subsequent tasks: concatenate
            self.exp_x_train = torch.cat((self.exp_x_train, X_train), dim=0)
            self.exp_y_train = np.concatenate([self.exp_y_train, y_train], axis=0)
            self.exp_x_test = torch.cat((self.exp_x_test, X_test), dim=0)
            self.exp_y_test = np.concatenate([self.exp_y_test, y_test], axis=0)
            self.exp_task_ids_train = np.concatenate([self.exp_task_ids_train, new_task_ids_train])
            self.exp_task_ids_test = np.concatenate([self.exp_task_ids_test, new_task_ids_test])

        # Rebalance buffers with task-weighted sampling
        self._rebalance_buffer(task_id, is_train=True)
        self._rebalance_buffer(task_id, is_train=False)

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
        # Note: num_workers=0 required when using JAX because os.fork() is incompatible
        # with JAX's multithreaded runtime. pin_memory=False since PyTorch is CPU-only.
        # drop_last=True ensures all batches have same size, preventing JIT recompilation.
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0,  # Must be 0 with JAX (fork incompatibility)
            'pin_memory': False,  # PyTorch is CPU-only, JAX handles GPU
            'drop_last': True,  # Prevent JIT recompilation from variable batch sizes
        }

        loader_curr = DataLoader(dataset_curr, **loader_kwargs)
        loader_exp = DataLoader(dataset_exp, **loader_kwargs)

        return loader_curr, loader_exp

    def generate_test_loader(self, task_id: int, batch_size: int = None) -> DataLoader:
        """Generate test loader for a specific task (for CL metrics evaluation).

        Added by Claude: This method enables per-task evaluation needed for computing
        the performance matrix A[j][i] = accuracy on task i after training task j.

        Args:
            task_id: Task ID to generate test loader for
            batch_size: Batch size for DataLoader (uses self.batch_size if None)

        Returns:
            DataLoader for task-specific test data
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Load task data (populates self.X_test, self.y_test)
        self.load_task(task_id)

        # Create dataset and loader for test data
        dataset_test = ContinualDataset(self.config, self.X_test, self.y_test)

        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,  # No shuffle for evaluation
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,  # Use all test samples
        }

        return DataLoader(dataset_test, **loader_kwargs)

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
