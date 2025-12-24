"""
Graph dataset classes for continual learning.

Adapted from ContLearn for cl_framework with BaseDataset-like interface.
Uses torch_geometric for graph data handling.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import FakeDataset, TUDataset

from ..config.constants import (
    DEFAULT_GRAPH_SEED,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_SYNTHETIC_NUM_GRAPHS,
    DEFAULT_SYNTHETIC_NUM_CHANNELS,
    DEFAULT_SYNTHETIC_AVG_NUM_NODES,
    DEFAULT_SYNTHETIC_NUM_CLASSES,
    DEFAULT_BATCH_SIZE_GRAPH,
)


def _transform_graph(data):
    """Transform function to add n_nodes attribute to graph data."""
    data.n_nodes = data.num_nodes
    return data


class BaseGraphDataset(ABC):
    """Abstract base class for graph continual learning datasets.

    Similar interface to BaseDataset but adapted for graph data structures.
    Uses torch_geometric DataLoader instead of standard PyTorch DataLoader.

    Attributes:
        config: Configuration dictionary
        train_data: Training graph dataset
        test_data: Test graph dataset
        memory_train: Experience replay buffer (list of graph objects)
        _task_class_mapping: Persistent mapping of task_id -> class_list for reproducibility
        _task_train_data: Cache of task-specific training data
        _task_test_data: Cache of task-specific test data
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the graph dataset.

        Args:
            config: Configuration dictionary containing:
                - batch_size: Batch size for DataLoaders
                - n_class: Number of classes for task sampling
                - class_per_task: Number of classes per task
                - debug_mode: (optional) Enable debug mode
                - debug_limit: (optional) Number of samples in debug mode
        """
        self.config = config
        self.batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE_GRAPH)
        self.debug_mode = config.get('debug_mode', False)
        self.debug_limit = config.get('debug_limit', 100)

        # Graph datasets (to be set by subclass)
        self.dataset = None
        self.train_data = None
        self.test_data = None

        # Experience replay buffer (list of graph data objects)
        self.memory_train: List = []

        # Added by Claude: Task persistence for CL metrics
        # Maps task_id -> class_list to ensure deterministic task definition
        self._task_class_mapping: Dict[int, np.ndarray] = {}
        # Cache of filtered data per task for fast reload
        self._task_train_data: Dict[int, List] = {}
        self._task_test_data: Dict[int, List] = {}

    @abstractmethod
    def _load_dataset(self) -> None:
        """Load the graph dataset.

        Subclasses must implement this to populate:
        - self.dataset: Full dataset
        - self.train_data: Training split
        - self.test_data: Test split
        """
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Return number of node features."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of graph classes."""
        pass

    @property
    def input_size(self) -> int:
        """Alias for num_features (consistent with BaseDataset)."""
        return self.num_features

    @property
    def output_size(self) -> int:
        """Alias for num_classes (consistent with BaseDataset)."""
        return self.num_classes

    def generate_dataset(self, task_id: int, batch_size: int = None,
                         phase: str = 'training') -> Tuple[DataLoader, DataLoader]:
        """Generate train/memory dataloaders for a task.

        Implements continuum_Graph_classification logic with deterministic task selection.
        Uses persistent task-to-class mapping to ensure reproducibility for CL metrics.

        Args:
            task_id: Current task ID (0-indexed)
            batch_size: Batch size for DataLoader
            phase: 'training' or 'testing'

        Returns:
            Tuple of (current_loader, memory_loader)
        """
        if batch_size is None:
            batch_size = self.batch_size

        n_class = self.config.get('n_class', self.num_classes)
        select = self.config.get('class_per_task', 2)

        # Added by Claude: Use deterministic task-to-class mapping
        # If task already defined, use cached classes; otherwise create deterministically
        if task_id in self._task_class_mapping:
            tasks = self._task_class_mapping[task_id]
        else:
            # Use task_id as seed for reproducible class selection
            rng = np.random.RandomState(seed=task_id * 1000)
            tasks = rng.randint(0, n_class, select)
            self._task_class_mapping[task_id] = tasks

        # Choose data source based on phase
        source_data = self.train_data if phase == 'training' else self.test_data
        cache_dict = self._task_train_data if phase == 'training' else self._task_test_data

        # Check cache first for fast reload
        if task_id in cache_dict:
            datas = cache_dict[task_id]
        else:
            # Filter data for selected classes
            stack = [(source_data[j].y.numpy() in tasks) for j in range(len(source_data))]
            datas = [source_data[k] for k, val in enumerate(stack) if val == True]

            # Ensure n_nodes attribute is set
            for k in range(len(datas)):
                datas[k].n_nodes = datas[k].num_nodes

            # Fixed by Claude: Update memory BEFORE caching to avoid logic bug
            # Update memory buffer (only for training, only first time)
            if phase == 'training':
                self.memory_train += datas

            # Cache for future use
            cache_dict[task_id] = datas

        # Create DataLoaders
        train_loader = DataLoader(datas, batch_size=batch_size, shuffle=False)
        mem_train_loader = DataLoader(self.memory_train, batch_size=batch_size, shuffle=False)

        print(f"Task {task_id}: Selected classes {tasks}, "
              f"current={len(datas)}, memory={len(self.memory_train)}")

        return train_loader, mem_train_loader

    def get_test_loader(self, batch_size: int = None) -> DataLoader:
        """Get test data loader.

        Args:
            batch_size: Batch size for DataLoader

        Returns:
            DataLoader for test data
        """
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

    def generate_test_loader(self, task_id: int, batch_size: int = None) -> DataLoader:
        """Generate test loader for a specific task (for CL metrics evaluation).

        Added by Claude: This method enables per-task evaluation needed for computing
        the performance matrix A[j][i] = accuracy on task i after training task j.

        Args:
            task_id: Task ID to generate test loader for
            batch_size: Batch size for DataLoader

        Returns:
            DataLoader for task-specific test data
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Use generate_dataset in test phase to get task-specific data
        test_loader, _ = self.generate_dataset(task_id, batch_size, phase='testing')
        return test_loader

    def append_to_experience(self, task_id: int) -> None:
        """No-op for graph datasets.

        Experience replay is handled directly in generate_dataset by appending
        to memory_train. This method exists for interface compatibility.
        """
        pass

    def get_model_config(self) -> Dict[str, Any]:
        """Return configuration for model initialization.

        Returns:
            Dictionary with input_size, output_size, and graph-specific config
        """
        return {
            'input_size': self.num_features,
            'output_size': self.num_classes,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
        }


class SyntheticGraphDataset(BaseGraphDataset):
    """Synthetic graph dataset using torch_geometric FakeDataset.

    Args:
        config: Configuration dictionary with optional keys:
            - num_graphs: Number of graphs to generate (default: 1000)
            - num_channels: Number of node features (default: 5)
            - avg_num_nodes: Average number of nodes per graph (default: 2)
            - num_classes: Number of graph classes (default: 10)
            - batch_size: Batch size for DataLoader (default: 20)
            - debug_mode: If True, limit data size
            - debug_limit: Max samples in debug mode
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load synthetic graph dataset."""
        # Get dataset parameters from config
        num_graphs = self.config.get('num_graphs', DEFAULT_SYNTHETIC_NUM_GRAPHS)
        num_channels = self.config.get('num_channels', DEFAULT_SYNTHETIC_NUM_CHANNELS)
        avg_num_nodes = self.config.get('avg_num_nodes', DEFAULT_SYNTHETIC_AVG_NUM_NODES)
        num_classes = self.config.get('num_classes', DEFAULT_SYNTHETIC_NUM_CLASSES)

        # Set seed for reproducibility
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)

        # Create synthetic dataset
        self.dataset = FakeDataset(
            num_graphs=num_graphs,
            num_channels=num_channels,
            avg_num_nodes=avg_num_nodes,
            num_classes=num_classes,
            transform=_transform_graph
        )

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting synthetic graph data from {len(self.dataset)} to {self.debug_limit} samples")
            self.dataset = self.dataset[:self.debug_limit]

        # Split into train/test
        length = len(self.dataset)
        train_split = self.config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)
        self.train_data = self.dataset[:int(train_split * length)]
        self.test_data = self.dataset[int(train_split * length):]

        print(f'Synthetic Graph Dataset:')
        print('======================')
        print(f'Number of training graphs: {len(self.train_data)}')
        print(f'Number of test graphs: {len(self.test_data)}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

    @property
    def num_features(self) -> int:
        """Number of node features."""
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        """Number of graph classes."""
        return self.dataset.num_classes


class TUGraphDataset(BaseGraphDataset):
    """TU Dataset wrapper for graph classification (MUTAG, ENZYMES, PROTEINS).

    Args:
        config: Configuration dictionary with keys:
            - data: Dataset name ('MUTAG', 'ENZYMES', 'PROTEINS')
            - batch_size: Batch size for DataLoader
            - debug_mode: If True, limit data size
            - debug_limit: Max samples in debug mode
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load TU graph dataset."""
        data_name = self.config.get('data', 'MUTAG')

        # Set seed for reproducibility
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)

        # Load TU Dataset
        self.dataset = TUDataset(
            root='data/TUDataset',
            name=data_name,
            transform=_transform_graph
        ).shuffle()

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting {data_name} data from {len(self.dataset)} to {self.debug_limit} samples")
            self.dataset = self.dataset[:self.debug_limit]

        # Split into train/test
        length = len(self.dataset)
        train_split = self.config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)
        self.train_data = self.dataset[:int(train_split * length)]
        self.test_data = self.dataset[int(train_split * length):]

        print(f'Dataset: {data_name}')
        print('======================')
        print(f'Number of training graphs: {len(self.train_data)}')
        print(f'Number of test graphs: {len(self.test_data)}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

    @property
    def num_features(self) -> int:
        """Number of node features."""
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        """Number of graph classes."""
        return self.dataset.num_classes


def load_graph_dataset(config: Dict[str, Any]) -> BaseGraphDataset:
    """Factory function to load appropriate graph dataset.

    Args:
        config: Configuration dictionary with 'data' key

    Returns:
        Graph dataset instance (SyntheticGraphDataset or TUGraphDataset)
    """
    data_name = config.get('data', 'synthetic')

    if data_name == 'synthetic':
        return SyntheticGraphDataset(config)
    elif data_name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        return TUGraphDataset(config)
    else:
        raise ValueError(f"Unknown graph dataset: {data_name}. "
                         f"Supported: 'synthetic', 'MUTAG', 'ENZYMES', 'PROTEINS'")
