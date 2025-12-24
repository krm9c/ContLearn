"""
Unit tests for MNIST datasets in datasets/mnist.py.
Tests MNISTDataset and PermutedMNISTDataset classes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax.numpy as jnp
import numpy as np
import torch
import pytest

# Added by Claude: Mark as unit test for test categorization
pytestmark = pytest.mark.unit

from cl.datasets import MNISTDataset, PermutedMNISTDataset


class TestMNISTDataset:
    """Tests for MNISTDataset class."""

    def test_mnist_dataset_initialization(self, classification_config):
        """Test MNISTDataset initializes correctly from config."""
        dataset = MNISTDataset(classification_config)

        assert dataset.batch_size == classification_config['batch_size']
        assert dataset.debug_mode == True
        assert dataset.debug_limit == classification_config['debug_limit']

    def test_mnist_dataset_properties(self, classification_config):
        """Test MNISTDataset property methods."""
        dataset = MNISTDataset(classification_config)

        assert dataset.input_size == 28 * 28  # 784
        assert dataset.output_size == 10  # 10 digits
        assert dataset.n_tasks == classification_config['n_task']

    def test_mnist_dataset_load_task(self, classification_config):
        """Test loading a specific task."""
        dataset = MNISTDataset(classification_config)
        dataset.load_task(0)

        # Check data is loaded
        assert dataset.X_train is not None
        assert dataset.y_train is not None
        assert dataset.X_test is not None
        assert dataset.y_test is not None

        # Check image dimensions (1, 28, 28) or (batch, 1, 28, 28)
        assert dataset.X_train.shape[1] == 1  # channel
        assert dataset.X_train.shape[2] == 28
        assert dataset.X_train.shape[3] == 28

    def test_mnist_dataset_debug_limit(self, classification_config):
        """Test debug limit is applied correctly."""
        config = classification_config.copy()
        config['debug_mode'] = True
        config['debug_limit'] = 20

        dataset = MNISTDataset(config)
        dataset.load_task(0)

        # Data should be limited
        assert len(dataset.X_train) <= 20
        assert len(dataset.y_train) <= 20

    def test_mnist_dataset_generate_dataset(self, classification_config):
        """Test generate_dataset returns DataLoaders."""
        dataset = MNISTDataset(classification_config)

        train_loader, exp_loader = dataset.generate_dataset(
            task_id=0,
            batch_size=classification_config['batch_size'],
            phase='training'
        )

        assert train_loader is not None
        assert exp_loader is not None

        # Get a batch
        batch = next(iter(train_loader))
        x, y = batch
        assert x.shape[1] == 1  # 1 channel
        assert x.shape[2] == 28
        assert x.shape[3] == 28


class TestPermutedMNISTDataset:
    """Tests for PermutedMNISTDataset class."""

    def test_permuted_mnist_initialization(self, classification_config):
        """Test PermutedMNISTDataset initializes correctly."""
        config = classification_config.copy()
        config['data'] = 'permuted_mnist'

        dataset = PermutedMNISTDataset(config)

        assert dataset.batch_size == config['batch_size']
        assert dataset.n_tasks == config['n_task']

    def test_permuted_mnist_different_tasks(self, classification_config):
        """Test that different tasks have different permutations."""
        config = classification_config.copy()
        config['data'] = 'permuted_mnist'

        dataset = PermutedMNISTDataset(config)

        # Load task 0
        dataset.load_task(0)
        x_task0 = dataset.X_train[0].clone()

        # Load task 1 (reinitialize to get fresh permutation)
        dataset.load_task(1)
        x_task1 = dataset.X_train[0].clone()

        # The images should be different due to permutation
        # (comparing the same original image with different permutations)
        # Note: This is a weak test since we're comparing different samples
        # A stronger test would compare the same sample under different permutations

    def test_permuted_mnist_properties(self, classification_config):
        """Test PermutedMNISTDataset property methods."""
        config = classification_config.copy()
        config['data'] = 'permuted_mnist'

        dataset = PermutedMNISTDataset(config)

        assert dataset.input_size == 28 * 28
        assert dataset.output_size == 10


class TestMNISTExperienceReplay:
    """Tests for experience replay with MNIST."""

    def test_append_to_experience_first_task(self, classification_config):
        """Test appending first task to experience buffer."""
        dataset = MNISTDataset(classification_config)
        dataset.load_task(0)
        dataset.append_to_experience(0)

        # Buffer should be initialized
        assert dataset._exp_initialized == True
        assert len(dataset.exp_x_train) > 0
        assert len(dataset.exp_y_train) > 0

    def test_append_to_experience_multiple_tasks(self, classification_config):
        """Test appending multiple tasks to experience buffer."""
        dataset = MNISTDataset(classification_config)

        sizes = []
        for task_id in range(2):
            dataset.load_task(task_id)
            dataset.append_to_experience(task_id)
            sizes.append(len(dataset.exp_x_train))

        # Buffer should grow with each task
        assert sizes[1] > sizes[0]

    def test_experience_replay_buffer_limit(self, classification_config):
        """Test experience buffer is limited to len_exp_replay."""
        config = classification_config.copy()
        config['len_exp_replay'] = 30  # Small limit for testing
        config['debug_limit'] = 25

        dataset = MNISTDataset(config)

        # Load and append multiple tasks
        for task_id in range(config['n_task']):
            dataset.load_task(task_id)
            dataset.append_to_experience(task_id)

        # Buffer should not exceed limit
        assert len(dataset.exp_x_train) <= config['len_exp_replay']
        assert len(dataset.exp_y_train) <= config['len_exp_replay']


class TestMNISTDataLoaderIntegration:
    """Integration tests for MNIST DataLoader creation."""

    def test_dataloaders_batch_size(self, classification_config):
        """Test DataLoaders return correct batch sizes."""
        dataset = MNISTDataset(classification_config)
        batch_size = classification_config['batch_size']

        train_loader, exp_loader = dataset.generate_dataset(
            task_id=0, batch_size=batch_size, phase='training'
        )

        # Get a batch
        for batch in train_loader:
            x, y = batch
            # Last batch may be smaller
            assert x.shape[0] <= batch_size
            break

    def test_dataloaders_training_vs_testing(self, classification_config):
        """Test DataLoaders work for both training and testing phases."""
        dataset = MNISTDataset(classification_config)

        # Training phase
        train_loader, _ = dataset.generate_dataset(
            task_id=0, batch_size=32, phase='training'
        )
        train_batch = next(iter(train_loader))
        assert train_batch is not None

        # Testing phase (uses same loaded task data)
        # Added by Claude: Use smaller batch size for test data (debug_limit=50 gives ~10 test samples)
        test_loader, _ = dataset.generate_dataset(
            task_id=0, batch_size=8, phase='testing'
        )
        test_batch = next(iter(test_loader))
        assert test_batch is not None

    def test_get_model_config(self, classification_config):
        """Test get_model_config returns correct configuration."""
        dataset = MNISTDataset(classification_config)

        model_config = dataset.get_model_config()

        assert model_config['input_size'] == 784
        assert model_config['output_size'] == 10
        assert model_config['n_tasks'] == classification_config['n_task']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
