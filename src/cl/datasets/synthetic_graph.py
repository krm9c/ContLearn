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
        # Added by Claude: Separate test buffer for proper experience evaluation
        self.memory_test: List = []

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

            # Cache for future use
            cache_dict[task_id] = datas

        # Update memory buffers (only first time task data is generated)
        if task_id not in self._task_train_data or task_id not in self._task_test_data:
            if phase == 'training':
                self.memory_train += datas
            else:
                self.memory_test += datas

        # Create DataLoaders
        train_loader = DataLoader(datas, batch_size=batch_size, shuffle=False)
        # Fixed by Claude: Use memory_test for evaluation to avoid testing on training data
        memory_data = self.memory_train if phase == 'training' else self.memory_test
        mem_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False)

        print(f"Task {task_id}: Selected classes {tasks}, "
              f"current={len(datas)}, memory_train={len(self.memory_train)}, memory_test={len(self.memory_test)}")

        return train_loader, mem_loader

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
# Fixed by Claude: Train/test split now happens ONCE at initialization to prevent data leakage
class TaskShiftGraphDataset(BaseGraphDataset):
    """Synthetic graph dataset with domain shift across tasks.

    Instead of class-incremental learning (which causes mode collapse when classes
    are added/removed), this dataset uses domain shift continual learning:

    - All tasks have the same N classes (default: 5)
    - Each task introduces progressive perturbations:
        1. Feature noise: Gaussian noise N(0, σ²)
        2. Edge dropout: Random edge removal with probability p
        3. Feature shift: Constant additive shift

    IMPORTANT: Train/test split happens ONCE at initialization. The same underlying
    graphs are always in train or test across all tasks. Only perturbations differ.
    This prevents data leakage between train and test sets.

    Perturbation Modes:
        - 'linear': perturbation = task_id * base_value (default)
        - 'exponential': perturbation = base_value * (growth_rate ^ task_id)
        - 'step': perturbation increases every N tasks
        - 'custom': user provides per-task perturbation values via config

    Args:
        config: Configuration dictionary with optional keys:
            - num_graphs: Number of graphs to generate (default: 1000)
            - num_channels: Number of node features (default: 5)
            - avg_num_nodes: Average number of nodes per graph (default: 20)
            - num_classes: Total classes in dataset (default: 5)
            - task_shift_enabled: Enable domain shift (default: True)
            - perturbation_mode: 'linear', 'exponential', 'step', or 'custom' (default: 'linear')
            - feature_noise_base: Base noise σ (default: 0.1)
            - edge_dropout_base: Base edge dropout rate (default: 0.05)
            - feature_shift_base: Base feature shift (default: 0.05)
            - perturbation_growth_rate: Growth rate for exponential mode (default: 1.5)
            - perturbation_step_size: Tasks per step for step mode (default: 2)
            - custom_perturbations: Dict mapping task_id -> {noise, dropout, shift} for custom mode
            - n_task: Number of tasks (default: 5)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Task-shift parameters
        self.task_shift_enabled = config.get('task_shift_enabled', DEFAULT_SYNTHETIC_TASK_SHIFT_ENABLED)
        self.feature_noise_base = config.get('feature_noise_base', DEFAULT_SYNTHETIC_FEATURE_NOISE_BASE)
        self.edge_dropout_base = config.get('edge_dropout_base', DEFAULT_SYNTHETIC_EDGE_DROPOUT_BASE)
        self.feature_shift_base = config.get('feature_shift_base', DEFAULT_SYNTHETIC_FEATURE_SHIFT_BASE)
        self.n_tasks = config.get('n_task', 5)

        # Added by Claude: Perturbation control modes
        self.perturbation_mode = config.get('perturbation_mode', 'linear')
        self.perturbation_growth_rate = config.get('perturbation_growth_rate', 1.5)
        self.perturbation_step_size = config.get('perturbation_step_size', 2)
        self.custom_perturbations = config.get('custom_perturbations', {})

        # Fixed by Claude: Store base train/test splits (indices) - split ONCE
        self._base_train_data = None  # List of unperturbed train graphs
        self._base_test_data = None   # List of unperturbed test graphs

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load synthetic graph dataset with FIXED train/test split."""
        # Get dataset parameters from config
        num_graphs = self.config.get('num_graphs', DEFAULT_SYNTHETIC_NUM_GRAPHS)
        num_channels = self.config.get('num_channels', DEFAULT_SYNTHETIC_NUM_CHANNELS)
        avg_num_nodes = self.config.get('avg_num_nodes', DEFAULT_SYNTHETIC_AVG_NUM_NODES)
        num_classes = self.config.get('num_classes', DEFAULT_SYNTHETIC_NUM_CLASSES_PER_TASK)

        # Set seed for reproducibility
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)

        # Create synthetic dataset and shuffle ONCE
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

        # Fixed by Claude: Split ONCE at initialization - no per-task shuffling
        full_data = list(self.dataset)
        self._train_split_ratio = self.config.get('train_test_split', DEFAULT_TRAIN_TEST_SPLIT)
        split_idx = int(self._train_split_ratio * len(full_data))

        # Store base (unperturbed) train and test sets SEPARATELY
        self._base_train_data = full_data[:split_idx]
        self._base_test_data = full_data[split_idx:]

        # Set train_data/test_data for compatibility (task 0, no perturbation)
        self.train_data = self._base_train_data
        self.test_data = self._base_test_data

        print(f'Task-Shift Synthetic Graph Dataset:')
        print('====================================')
        print(f'Total graphs: {len(full_data)}')
        print(f'Training graphs: {len(self._base_train_data)} (FIXED split)')
        print(f'Test graphs: {len(self._base_test_data)} (FIXED split)')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')
        print(f'Task-shift enabled: {self.task_shift_enabled}')
        print(f'Perturbation mode: {self.perturbation_mode}')
        print(f'  Feature noise base: {self.feature_noise_base}')
        print(f'  Edge dropout base: {self.edge_dropout_base}')
        print(f'  Feature shift base: {self.feature_shift_base}')
        if self.perturbation_mode == 'exponential':
            print(f'  Growth rate: {self.perturbation_growth_rate}')
        elif self.perturbation_mode == 'step':
            print(f'  Step size: {self.perturbation_step_size} tasks')
        elif self.perturbation_mode == 'custom':
            print(f'  Custom perturbations: {len(self.custom_perturbations)} tasks defined')

    def _get_perturbation_params(self, task_id: int) -> Tuple[float, float, float]:
        """Get perturbation parameters for a specific task based on mode.

        Added by Claude: Flexible perturbation control for different experimental designs.

        Args:
            task_id: Current task ID

        Returns:
            Tuple of (feature_noise_std, edge_dropout_prob, feature_shift)
        """
        # Task 0 always has no perturbation (baseline)
        if task_id == 0:
            return 0.0, 0.0, 0.0

        if self.perturbation_mode == 'linear':
            # Linear scaling: perturbation = task_id * base
            noise_std = task_id * self.feature_noise_base
            dropout_prob = task_id * self.edge_dropout_base
            shift = task_id * self.feature_shift_base

        elif self.perturbation_mode == 'exponential':
            # Exponential scaling: perturbation = base * (rate ^ task_id)
            rate = self.perturbation_growth_rate
            noise_std = self.feature_noise_base * (rate ** task_id)
            dropout_prob = self.edge_dropout_base * (rate ** task_id)
            shift = self.feature_shift_base * (rate ** task_id)

        elif self.perturbation_mode == 'step':
            # Step function: perturbation increases every N tasks
            step = task_id // self.perturbation_step_size
            noise_std = (step + 1) * self.feature_noise_base
            dropout_prob = (step + 1) * self.edge_dropout_base
            shift = (step + 1) * self.feature_shift_base

        elif self.perturbation_mode == 'custom':
            # User-defined per-task perturbations
            if task_id in self.custom_perturbations:
                params = self.custom_perturbations[task_id]
                noise_std = params.get('noise', 0.0)
                dropout_prob = params.get('dropout', 0.0)
                shift = params.get('shift', 0.0)
            else:
                # Fall back to linear if task not specified
                noise_std = task_id * self.feature_noise_base
                dropout_prob = task_id * self.edge_dropout_base
                shift = task_id * self.feature_shift_base

        else:
            raise ValueError(f"Unknown perturbation_mode: {self.perturbation_mode}. "
                           f"Supported: 'linear', 'exponential', 'step', 'custom'")

        # Cap edge dropout at 50% to maintain graph connectivity
        dropout_prob = min(dropout_prob, 0.5)

        return noise_std, dropout_prob, shift

    def _apply_task_perturbation(self, data_list: List, task_id: int, seed: int = None) -> List:
        """Apply task-specific perturbations to graph data.

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

        # Get perturbation parameters based on mode
        feature_noise_std, edge_dropout_prob, feature_shift = self._get_perturbation_params(task_id)

        # Task 0 has no perturbation - return copies
        if task_id == 0 or (feature_noise_std == 0 and edge_dropout_prob == 0 and feature_shift == 0):
            return [copy.deepcopy(d) for d in data_list]

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

        Fixed by Claude: Now applies perturbations to FIXED train/test splits
        instead of shuffling and re-splitting per task. This prevents data leakage.

        Args:
            task_id: Task ID

        Returns:
            Tuple of (train_data, test_data) for this task
        """
        # Check if already generated
        if task_id in self._task_train_data and task_id in self._task_test_data:
            return self._task_train_data[task_id], self._task_test_data[task_id]

        # Fixed by Claude: Apply perturbations to FIXED train and test sets SEPARATELY
        # This ensures the same underlying graphs are always in train or test
        train_data = self._apply_task_perturbation(
            self._base_train_data, task_id, seed=DEFAULT_GRAPH_SEED
        )
        test_data = self._apply_task_perturbation(
            self._base_test_data, task_id, seed=DEFAULT_GRAPH_SEED + 1000  # Different seed for test
        )

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

        # Update memory buffers (only first time task is generated)
        if is_new_task:
            self.memory_train.extend(train_data)
            # Added by Claude: Also track test data for proper experience evaluation
            self.memory_test.extend(test_data)

        # Create DataLoaders
        current_loader = DataLoader(task_data, batch_size=batch_size, shuffle=False)
        # Fixed by Claude: Use memory_test for evaluation to avoid testing on training data
        memory_data = self.memory_train if phase == 'training' else self.memory_test
        mem_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False)

        # Get perturbation info for logging
        noise_std, dropout_prob, shift = self._get_perturbation_params(task_id)

        print(f"Task {task_id} ({phase}): {self.num_classes} classes, mode={self.perturbation_mode}, "
              f"perturbation(noise={noise_std:.3f}, drop={dropout_prob:.3f}, shift={shift:.3f}), "
              f"current={len(task_data)}, memory_train={len(self.memory_train)}, memory_test={len(self.memory_test)}")

        return current_loader, mem_loader

    def get_perturbation_schedule(self) -> Dict[int, Dict[str, float]]:
        """Return the perturbation schedule for all tasks.

        Added by Claude: Useful for logging and visualization.

        Returns:
            Dict mapping task_id -> {noise, dropout, shift}
        """
        schedule = {}
        for task_id in range(self.n_tasks):
            noise, dropout, shift = self._get_perturbation_params(task_id)
            schedule[task_id] = {
                'noise': noise,
                'dropout': dropout,
                'shift': shift
            }
        return schedule

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
