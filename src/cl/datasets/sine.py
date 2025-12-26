"""
Sine wave dataset for continual learning regression.

Generates synthetic sine wave data with gradual task drift for
testing continual learning algorithms on regression problems.

Each task has slightly different frequency and amplitude, creating
a sequence of related but distinct learning problems.
"""

import numpy as np
import pickle
import os
import sys
from typing import Dict, Any, Optional
import sklearn.model_selection as model_selection

# Added by Claude: NumPy 2.0 compatibility fix for pickle
# NumPy 2.0 renamed numpy.core to numpy._core, but older pickle files may reference the old name
# This ensures compatibility when unpickling data across NumPy versions
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
if not hasattr(np, 'core'):
    sys.modules['numpy.core'] = np._core

from .base import BaseDataset
from ..config.constants import DEFAULT_SINE_TIME_STEP


def generate_sine_data(delta: float, n_tasks: int = 40, output_path: str = 'Incremental_Sine1e^4.p',
                       seed: int = 1) -> str:
    """Generate sine data for continual learning tasks.

    Creates a pickle file containing sine wave data for multiple tasks.
    Each task has gradually increasing frequency and amplitude.

    Args:
        delta: Perturbation value for gradual task drift
        n_tasks: Number of tasks to generate (default: 40)
        output_path: Path to save the pickle file
        seed: Random seed for reproducibility

    Returns:
        Path to the generated pickle file

    Data format per task:
        (y, time, phase, amplitude, frequency) where:
        - y: Sine wave values, shape (n_samples, n_time_points)
        - time: Time points array
        - phase: Phase values, shape (n_samples, 1)
        - amplitude: Amplitude values, shape (n_samples, 1)
        - frequency: Frequency values, shape (n_samples, 1)
    """
    # Added by Claude: use constant for time step
    time = np.arange(0, 1, DEFAULT_SINE_TIME_STEP)  # Time points based on step size
    data = {}
    # Conservative: 40,000 samples (10x original, enough for batch_size=2048-4096)
    total_samples = 40000
    np.random.seed(seed)
    frequency = (np.random.random([total_samples, 1]) * 60) * np.ones([total_samples, 1])
    amplitude = (np.random.random() * 1) * np.ones([total_samples, 1])
    phase = (np.random.random() * 90) * np.ones([total_samples, 1])

    for i in range(n_tasks):
        y = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # Gradual drift: increase frequency and amplitude
        frequency = frequency + delta
        amplitude = amplitude + delta
        data['task' + str(i)] = (y, time, phase, amplitude, frequency)

    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Sine data generated: {output_path} ({n_tasks} tasks)")
    return output_path


class SineDataset(BaseDataset):
    """Sine wave dataset for continual learning regression.

    Features (X): [phase, amplitude, frequency] - 3 input features
    Target (y): Sine wave values - shape depends on time points

    Args:
        config: Configuration dictionary with:
            - delta: Task drift perturbation value
            - data_path: Path to sine data pickle file (optional)
            - batch_size: Batch size for DataLoaders
            - len_exp_replay: Max experience replay buffer size
            - debug_mode: Enable debug mode with limited data
            - debug_limit: Number of samples in debug mode
            - n_task: Number of tasks to use (default: all available)

    Example:
        >>> config = {'delta': 0.001, 'batch_size': 64}
        >>> dataset = SineDataset(config)
        >>> train_loader, exp_loader = dataset.generate_dataset(task_id=0, batch_size=64, phase='training')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the sine dataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.delta = config.get('delta', 0.00001)
        self.data_path = config.get('data_path', 'data/Incremental_Sine1e^4.p')
        self._n_tasks = config.get('n_task', 40)
        self.test_size = config.get('test_size', 0.2)

        # Generate data if file doesn't exist
        if not os.path.exists(self.data_path):
            # Added by Claude: ensure parent directory exists before generating data
            data_dir = os.path.dirname(self.data_path)
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            generate_sine_data(self.delta, n_tasks=40, output_path=self.data_path)

        # Load the data
        # Added by Claude: Handle NumPy version compatibility when unpickling
        try:
            with open(self.data_path, 'rb') as fp:
                self.raw_data = pickle.load(fp)
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                # NumPy version mismatch - regenerate the data file
                print(f"Warning: Pickle file incompatible with current NumPy version. Regenerating data...")
                os.remove(self.data_path)
                generate_sine_data(self.delta, n_tasks=40, output_path=self.data_path)
                with open(self.data_path, 'rb') as fp:
                    self.raw_data = pickle.load(fp)
            else:
                raise

        # Determine actual number of tasks available
        self._available_tasks = len(self.raw_data)
        if self._n_tasks > self._available_tasks:
            print(f"Warning: Requested {self._n_tasks} tasks but only {self._available_tasks} available")
            self._n_tasks = self._available_tasks

    def _load_task_data(self, task_id: int) -> None:
        """Load data for a specific sine task.

        Extracts sine data for the given task and creates train/test splits.
        Features: [phase, amplitude, frequency]
        Target: sine wave values (flattened)

        Args:
            task_id: Task identifier (0-indexed)
        """
        if task_id >= self._available_tasks:
            raise ValueError(f"Task {task_id} not available. Max task: {self._available_tasks - 1}")

        # Extract task data
        y, time, phase, amplitude, frequency = self.raw_data['task' + str(task_id)]

        # Create feature matrix: [phase, amplitude, frequency]
        X = np.concatenate([phase, amplitude.reshape([-1, 1]), frequency.reshape([-1, 1])], axis=1)

        # Flatten y if needed (shape: n_samples x n_time_points -> n_samples)
        # For regression, we typically predict all time points
        # But original code treats y as target directly
        y = y.astype(np.float32)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            X.astype(np.float32), y, test_size=self.test_size, random_state=42 + task_id
        )

    @property
    def input_size(self) -> int:
        """Input dimension: 3 (phase, amplitude, frequency)."""
        return 3

    @property
    def output_size(self) -> int:
        """Output dimension: number of time points in sine wave."""
        # Get output size from actual loaded data
        # Data format: (y, time, phase, amplitude, frequency) where y.shape = (n_samples, n_time_points)
        if hasattr(self, 'raw_data') and self.raw_data:
            # Get the time array from the first task
            _, time, _, _, _ = self.raw_data['task0']
            return len(time)
        # Fallback: compute from constant if data not yet loaded
        return len(np.arange(0, 1, DEFAULT_SINE_TIME_STEP))

    @property
    def n_tasks(self) -> int:
        """Number of available tasks."""
        return self._n_tasks
