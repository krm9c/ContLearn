"""
Unit tests for datasets in datasets/base.py and datasets/sine.py.
Tests BaseDataset, SineDataset, and experience replay functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax.numpy as jnp
import numpy as np
import torch
import pytest

from cl.datasets import BaseDataset, SineDataset, generate_sine_data
from cl.config.constants import DEFAULT_SINE_TIME_STEP


class TestSineDataset:
    """Tests for SineDataset class."""

    def test_sine_dataset_initialization(self, test_sine_config):
        """Test SineDataset initializes correctly from config."""
        dataset = SineDataset(test_sine_config)

        assert dataset.batch_size == test_sine_config['batch_size']
        assert dataset.len_exp_replay == test_sine_config['len_exp_replay']
        assert dataset.debug_mode == test_sine_config['debug_mode']
        assert dataset.debug_limit == test_sine_config['debug_limit']

    def test_sine_dataset_properties(self, test_sine_config):
        """Test SineDataset property methods."""
        dataset = SineDataset(test_sine_config)

        assert dataset.input_size == 3  # phase, amplitude, frequency
        # Added by Claude: output_size is computed dynamically from time step constant
        expected_output_size = len(np.arange(0, 1, DEFAULT_SINE_TIME_STEP))
        assert dataset.output_size == expected_output_size
        assert dataset.n_tasks == test_sine_config['n_task']

    def test_sine_dataset_load_task(self, test_sine_config):
        """Test loading a specific task."""
        dataset = SineDataset(test_sine_config)
        dataset.load_task(0)

        # Check data is loaded
        assert dataset.X_train is not None
        assert dataset.y_train is not None
        assert dataset.X_test is not None
        assert dataset.y_test is not None

        # Check input dimension
        assert dataset.X_train.shape[1] == 3  # 3 features

    def test_sine_dataset_debug_limit(self, test_sine_config):
        """Test debug limit is applied correctly."""
        config = test_sine_config.copy()
        config['debug_mode'] = True
        config['debug_limit'] = 20

        dataset = SineDataset(config)
        dataset.load_task(0)

        # Data should be limited to debug_limit
        assert len(dataset.X_train) <= 20
        assert len(dataset.y_train) <= 20

    def test_sine_dataset_multiple_tasks(self, test_sine_config):
        """Test loading multiple tasks sequentially."""
        dataset = SineDataset(test_sine_config)

        for task_id in range(test_sine_config['n_task']):
            dataset.load_task(task_id)

            assert dataset.X_train is not None
            assert dataset.y_train is not None

    def test_sine_dataset_generate_dataset(self, test_sine_config):
        """Test generate_dataset returns DataLoaders."""
        dataset = SineDataset(test_sine_config)

        train_loader, exp_loader = dataset.generate_dataset(
            task_id=0,
            batch_size=test_sine_config['batch_size'],
            phase='training'
        )

        assert train_loader is not None
        assert exp_loader is not None

        # Get a batch
        batch = next(iter(train_loader))
        x, y = batch
        assert x.shape[1] == 3  # 3 input features


class TestExperienceReplay:
    """Tests for experience replay buffer functionality."""

    def test_append_to_experience_first_task(self, test_sine_config):
        """Test appending first task to experience buffer."""
        dataset = SineDataset(test_sine_config)
        dataset.load_task(0)
        dataset.append_to_experience(0)

        # Buffer should be initialized
        assert dataset._exp_initialized == True
        assert len(dataset.exp_x_train) > 0
        assert len(dataset.exp_y_train) > 0

    def test_append_to_experience_multiple_tasks(self, test_sine_config):
        """Test appending multiple tasks to experience buffer."""
        dataset = SineDataset(test_sine_config)

        sizes = []
        for task_id in range(2):
            dataset.load_task(task_id)
            dataset.append_to_experience(task_id)
            sizes.append(len(dataset.exp_x_train))

        # Buffer should grow with each task
        assert sizes[1] > sizes[0]

    def test_experience_replay_buffer_limit(self, test_exp_replay_config):
        """Test experience buffer is limited to len_exp_replay."""
        dataset = SineDataset(test_exp_replay_config)
        max_size = test_exp_replay_config['len_exp_replay']

        # Load and append multiple tasks
        for task_id in range(test_exp_replay_config['n_task']):
            dataset.load_task(task_id)
            dataset.append_to_experience(task_id)

        # Buffer should not exceed limit
        assert len(dataset.exp_x_train) <= max_size
        assert len(dataset.exp_y_train) <= max_size

    def test_get_task_data_training(self, test_sine_config):
        """Test get_task_data returns correct data for training phase."""
        dataset = SineDataset(test_sine_config)

        # Load and append first task
        dataset.load_task(0)
        dataset.append_to_experience(0)

        # Get data for task 0
        (current_x, current_y), (exp_x, exp_y) = dataset.get_task_data(0, 'training')

        assert current_x is not None
        assert current_y is not None
        # For task 0, experience should equal current
        assert len(exp_x) == len(current_x)

    def test_get_task_data_with_experience(self, test_sine_config):
        """Test get_task_data includes experience data for task > 0."""
        dataset = SineDataset(test_sine_config)

        # Load and append task 0
        dataset.load_task(0)
        dataset.append_to_experience(0)
        task0_size = len(dataset.X_train)

        # Load task 1
        dataset.load_task(1)

        # Get data for task 1
        (current_x, current_y), (exp_x, exp_y) = dataset.get_task_data(1, 'training')

        # Experience buffer should have task 0 data
        assert len(exp_x) >= task0_size


class TestGenerateSineData:
    """Tests for sine data generation function."""

    def test_generate_sine_data_creates_file(self, tmp_path):
        """Test generate_sine_data creates pickle file."""
        output_path = str(tmp_path / "test_sine.p")
        result_path = generate_sine_data(delta=0.001, n_tasks=5, output_path=output_path)

        assert os.path.exists(result_path)
        assert result_path == output_path

    def test_generate_sine_data_content(self, tmp_path):
        """Test generated sine data has correct structure."""
        import pickle

        output_path = str(tmp_path / "test_sine.p")
        generate_sine_data(delta=0.001, n_tasks=5, output_path=output_path)

        with open(output_path, 'rb') as f:
            data = pickle.load(f)

        # Should have 5 tasks
        assert len(data) == 5

        # Each task should have (y, time, phase, amplitude, frequency)
        task_data = data['task0']
        assert len(task_data) == 5

        y, time, phase, amplitude, frequency = task_data
        # Added by Claude: calculate expected time points from constant
        expected_time_points = len(np.arange(0, 1, DEFAULT_SINE_TIME_STEP))
        assert y.shape[1] == expected_time_points

    def test_generate_sine_data_delta_drift(self, tmp_path):
        """Test sine data has gradual drift with delta."""
        import pickle

        output_path = str(tmp_path / "test_sine.p")
        delta = 0.1  # Large delta for visible drift
        generate_sine_data(delta=delta, n_tasks=3, output_path=output_path)

        with open(output_path, 'rb') as f:
            data = pickle.load(f)

        # Frequency should increase between tasks
        freq_0 = data['task0'][4][0, 0]  # frequency of task 0
        freq_1 = data['task1'][4][0, 0]  # frequency of task 1
        freq_2 = data['task2'][4][0, 0]  # frequency of task 2

        assert freq_1 > freq_0
        assert freq_2 > freq_1


class TestDataLoaderIntegration:
    """Integration tests for DataLoader creation."""

    def test_dataloaders_batch_size(self, test_sine_config):
        """Test DataLoaders return correct batch sizes."""
        dataset = SineDataset(test_sine_config)
        batch_size = test_sine_config['batch_size']

        train_loader, exp_loader = dataset.generate_dataset(
            task_id=0, batch_size=batch_size, phase='training'
        )

        # Get a batch
        for batch in train_loader:
            x, y = batch
            # Last batch may be smaller
            assert x.shape[0] <= batch_size
            break

    def test_dataloaders_training_vs_testing(self, test_sine_config):
        """Test DataLoaders work for both training and testing phases."""
        dataset = SineDataset(test_sine_config)

        # Training phase
        train_loader, _ = dataset.generate_dataset(
            task_id=0, batch_size=32, phase='training'
        )
        train_batch = next(iter(train_loader))
        assert train_batch is not None

        # Testing phase (uses same loaded task data)
        test_loader, _ = dataset.generate_dataset(
            task_id=0, batch_size=32, phase='testing'
        )
        test_batch = next(iter(test_loader))
        assert test_batch is not None

    def test_get_model_config(self, test_sine_config):
        """Test get_model_config returns correct configuration."""
        dataset = SineDataset(test_sine_config)

        model_config = dataset.get_model_config()

        assert model_config['input_size'] == 3
        # Added by Claude: output_size is computed dynamically from time step constant
        expected_output_size = len(np.arange(0, 1, DEFAULT_SINE_TIME_STEP))
        assert model_config['output_size'] == expected_output_size
        assert model_config['n_tasks'] == test_sine_config['n_task']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
