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
    # Added by Claude: Task-shift dataset constants
    DEFAULT_SYNTHETIC_TASK_SHIFT_ENABLED,
    DEFAULT_SYNTHETIC_NUM_CLASSES_PER_TASK,
    DEFAULT_SYNTHETIC_FEATURE_NOISE_BASE,
    DEFAULT_SYNTHETIC_EDGE_DROPOUT_BASE,
    DEFAULT_SYNTHETIC_FEATURE_SHIFT_BASE,
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

        # DEBUG: All tasks see all classes (no forgetting test) - REMOVE AFTER DEBUG
        tasks = np.arange(n_class)
        print(f"[DEBUG] Task {task_id}: Using ALL {n_class} classes")

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

        # Create synthetic dataset and shuffle (matching old code behavior)
        self.dataset = FakeDataset(
            num_graphs=num_graphs,
            num_channels=num_channels,
            avg_num_nodes=avg_num_nodes,
            num_classes=num_classes,
            transform=_transform_graph
        ).shuffle()

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


# Added by Claude: Task-shift synthetic graph dataset for continual learning
class TaskShiftGraphDataset(BaseGraphDataset):
    """Synthetic graph dataset with domain shift across tasks.

    Instead of class-incremental learning (which causes mode collapse when classes
    are added/removed), this dataset uses domain shift continual learning:

    - All tasks have the same N classes (default: 5)
    - Each task introduces progressive perturbations:
        1. Feature noise: Gaussian noise N(0, σ²) where σ = task_id * noise_base
        2. Edge dropout: Random edge removal with probability task_id * dropout_base
        3. Feature shift: Constant additive shift task_id * shift_base

    This tests continual learning under distribution shift (real-world scenario)
    and avoids mode collapse issues from class-incremental approaches.

    Args:
        config: Configuration dictionary with optional keys:
            - num_graphs: Number of graphs to generate (default: 1000)
            - num_channels: Number of node features (default: 5)
            - avg_num_nodes: Average number of nodes per graph (default: 20)
            - num_classes: Total classes in dataset (default: 5)
            - task_shift_enabled: Enable domain shift (default: True)
            - feature_noise_base: Base noise σ, scaled by task_id (default: 0.1)
            - edge_dropout_base: Base edge dropout rate (default: 0.05)
            - feature_shift_base: Base feature shift (default: 0.05)
            - n_tasks: Number of tasks (default: 5)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Task-shift parameters
        self.task_shift_enabled = config.get('task_shift_enabled', DEFAULT_SYNTHETIC_TASK_SHIFT_ENABLED)
        self.feature_noise_base = config.get('feature_noise_base', DEFAULT_SYNTHETIC_FEATURE_NOISE_BASE)
        self.edge_dropout_base = config.get('edge_dropout_base', DEFAULT_SYNTHETIC_EDGE_DROPOUT_BASE)
        self.feature_shift_base = config.get('feature_shift_base', DEFAULT_SYNTHETIC_FEATURE_SHIFT_BASE)
        self.n_tasks = config.get('n_task', 5)  # Fixed: was 'n_tasks', now matches rest of codebase

        # Store base (unperturbed) full dataset for generating task-shifted versions
        self._base_full_data = None
        self._train_split_idx = None  # Index to split train/test

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load synthetic graph dataset."""
        # Get dataset parameters from config
        num_graphs = self.config.get('num_graphs', DEFAULT_SYNTHETIC_NUM_GRAPHS)
        num_channels = self.config.get('num_channels', DEFAULT_SYNTHETIC_NUM_CHANNELS)
        avg_num_nodes = self.config.get('avg_num_nodes', DEFAULT_SYNTHETIC_AVG_NUM_NODES)
        # Use smaller number of classes for task-shift (all tasks see same classes)
        num_classes = self.config.get('num_classes', DEFAULT_SYNTHETIC_NUM_CLASSES_PER_TASK)

        # Set seed for reproducibility
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)

        # Create synthetic dataset
        self.dataset = FakeDataset(
            num_graphs=num_graphs,
            num_channels=num_channels,
            avg_num_nodes=avg_num_nodes,
            num_classes=num_classes,
            transform=_transform_graph
        ).shuffle()

        # Apply debug limit if enabled
        if self.debug_mode:
            print(f"DEBUG MODE: Limiting synthetic graph data from {len(self.dataset)} to {self.debug_limit} samples")
            self.dataset = self.dataset[:self.debug_limit]

        # Store full dataset and train/test split ratio
        self._base_full_data = list(self.dataset)
        self._train_split_ratio = self.config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)

        # Initialize train/test for compatibility (will be overwritten per task)
        split_idx = int(self._train_split_ratio * len(self._base_full_data))
        self.train_data = self._base_full_data[:split_idx]
        self.test_data = self._base_full_data[split_idx:]

        print(f'Task-Shift Synthetic Graph Dataset:')
        print('====================================')
        print(f'Total graphs: {len(self._base_full_data)}')
        print(f'Train/Test split: {self._train_split_ratio:.0%}/{1-self._train_split_ratio:.0%}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')
        print(f'Task-shift enabled: {self.task_shift_enabled}')
        print(f'  Feature noise base: {self.feature_noise_base}')
        print(f'  Edge dropout base: {self.edge_dropout_base}')
        print(f'  Feature shift base: {self.feature_shift_base}')

    def _apply_task_perturbation(self, data_list: List, task_id: int, seed: int = None) -> List:
        """Apply task-specific perturbations to graph data.

        Perturbations scale linearly with task_id:
        - Task 0: No perturbation (baseline)
        - Task k: feature_noise = k * noise_base, edge_dropout = k * dropout_base, etc.

        Args:
            data_list: List of graph data objects
            task_id: Current task ID (0 = no perturbation)
            seed: Random seed for reproducibility

        Returns:
            List of perturbed graph data objects
        """
        import copy
        import torch

        if seed is not None:
            np.random.seed(seed + task_id)
            torch.manual_seed(seed + task_id)

        # Task 0 has no perturbation
        if task_id == 0:
            return [copy.deepcopy(d) for d in data_list]

        # Compute perturbation magnitudes
        feature_noise_std = task_id * self.feature_noise_base
        edge_dropout_prob = min(task_id * self.edge_dropout_base, 0.5)  # Cap at 50%
        feature_shift = task_id * self.feature_shift_base

        perturbed_list = []
        for data in data_list:
            # Deep copy to avoid modifying original
            new_data = copy.deepcopy(data)

            # 1. Add feature noise: x' = x + N(0, σ²)
            if feature_noise_std > 0:
                noise = torch.randn_like(new_data.x) * feature_noise_std
                new_data.x = new_data.x + noise

            # 2. Add feature shift: x' = x + shift
            if feature_shift > 0:
                new_data.x = new_data.x + feature_shift

            # 3. Edge dropout: randomly remove edges
            if edge_dropout_prob > 0 and new_data.edge_index.shape[1] > 0:
                num_edges = new_data.edge_index.shape[1]
                keep_mask = torch.rand(num_edges) > edge_dropout_prob
                # Ensure at least some edges remain
                if keep_mask.sum() < 2:
                    keep_mask[:2] = True
                new_data.edge_index = new_data.edge_index[:, keep_mask]

            # Ensure n_nodes is set
            new_data.n_nodes = new_data.num_nodes

            perturbed_list.append(new_data)

        return perturbed_list

    def _get_or_generate_task_data(self, task_id: int) -> Tuple[List, List]:
        """Generate or retrieve cached train/test data for a task.

        For each task:
        1. Apply perturbations to full base dataset
        2. Shuffle the perturbed dataset (with fixed seed for reproducibility)
        3. Split into train/test

        This ensures train and test come from the same perturbed distribution.

        Args:
            task_id: Task ID

        Returns:
            Tuple of (train_data, test_data) for this task
        """
        # Check if already generated
        if task_id in self._task_train_data and task_id in self._task_test_data:
            return self._task_train_data[task_id], self._task_test_data[task_id]

        # Apply task-specific perturbations to FULL dataset
        perturbed_full = self._apply_task_perturbation(
            self._base_full_data, task_id, seed=DEFAULT_GRAPH_SEED
        )

        # Shuffle with fixed seed (based on task_id for reproducibility)
        import random
        rng = random.Random(DEFAULT_GRAPH_SEED + task_id)
        rng.shuffle(perturbed_full)

        # Split into train/test
        split_idx = int(self._train_split_ratio * len(perturbed_full))
        train_data = perturbed_full[:split_idx]
        test_data = perturbed_full[split_idx:]

        # Cache both
        self._task_train_data[task_id] = train_data
        self._task_test_data[task_id] = test_data

        return train_data, test_data

    def generate_dataset(self, task_id: int, batch_size: int = None,
                         phase: str = 'training') -> Tuple[DataLoader, DataLoader]:
        """Generate train/memory dataloaders for a task with domain shift.

        Each task uses the same classes but with task-specific perturbations.
        Memory buffer accumulates perturbed data from all seen tasks.

        Args:
            task_id: Current task ID (0-indexed)
            batch_size: Batch size for DataLoader
            phase: 'training' or 'testing'

        Returns:
            Tuple of (current_loader, memory_loader)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Track if this is first time generating this task's data
        is_new_task = task_id not in self._task_train_data

        # Get or generate train/test data for this task
        train_data, test_data = self._get_or_generate_task_data(task_id)

        # Select appropriate data based on phase
        task_data = train_data if phase == 'training' else test_data

        # Update memory buffer (only for training, only first time task is generated)
        if phase == 'training' and is_new_task:
            self.memory_train.extend(train_data)

        # Create DataLoaders
        current_loader = DataLoader(task_data, batch_size=batch_size, shuffle=False)
        mem_train_loader = DataLoader(self.memory_train, batch_size=batch_size, shuffle=False)

        # Compute perturbation info for logging
        noise_std = task_id * self.feature_noise_base
        dropout_prob = min(task_id * self.edge_dropout_base, 0.5)
        shift = task_id * self.feature_shift_base

        print(f"Task {task_id} ({phase}): All {self.num_classes} classes with perturbation "
              f"(noise_σ={noise_std:.2f}, edge_drop={dropout_prob:.2f}, shift={shift:.2f}), "
              f"current={len(task_data)}, memory={len(self.memory_train)}")

        return current_loader, mem_train_loader

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
        Graph dataset instance (SyntheticGraphDataset, TaskShiftGraphDataset, or TUGraphDataset)
    """
    data_name = config.get('data', 'synthetic')

    if data_name == 'synthetic':
        return SyntheticGraphDataset(config)
    # Added by Claude: Task-shift synthetic dataset for domain-shift CL
    elif data_name == 'synthetic_taskshift':
        return TaskShiftGraphDataset(config)
    elif data_name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        return TUGraphDataset(config)
    else:
        raise ValueError(f"Unknown graph dataset: {data_name}. "
                         f"Supported: 'synthetic', 'synthetic_taskshift', 'MUTAG', 'ENZYMES', 'PROTEINS'")
