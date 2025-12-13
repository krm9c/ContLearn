"""
Unit tests for data handling in utils/data.py.
Tests data_return class, experience replay, and dataset loaders.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytest

from contlearn.data.datasets import data_return, Continual_Dataset


class TestDataReturn:
    """Tests for data_return class."""

    def test_data_return_sine_initialization(self):
        """Test data_return initializes correctly for sine regression."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)

        # After initialization, X_train/y_train are None until generate_dataset is called
        # But dataset should be loaded
        assert data.dataset is not None
        # Experience replay starts as empty lists
        assert len(data.exp_x_train) == 0
        assert len(data.exp_y_train) == 0

    def test_data_return_mnist_initialization(self):
        """Test data_return initializes correctly for MNIST."""
        config = {
            'data_id': 'mnist',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)

        # MNIST images and labels should be loaded
        assert data.images is not None
        assert data.labels is not None
        # MNIST has shape [N, 1, 28, 28]
        assert data.images.shape[1:] == (1, 28, 28)

    def test_data_return_cifar10_initialization(self):
        """Test data_return initializes correctly for CIFAR-10."""
        config = {
            'data_id': 'cifar10',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)

        # CIFAR-10 images and labels should be loaded
        assert data.images is not None
        assert data.labels is not None
        # CIFAR-10: 3 channels, 32x32
        assert data.images.shape[1:] == (3, 32, 32)

    def test_generate_dataset_sine(self):
        """Test dataset generation for sine regression."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)
        train_loader, exp_loader = data.generate_dataset(
            task_id=0, batch_size=32, phase='training'
        )

        assert train_loader is not None
        # First task has no experience replay
        assert exp_loader is not None

        # Check batch shapes
        for x, y in train_loader:
            assert x.shape[0] <= 32  # Batch size
            assert y.shape[0] <= 32
            break

    def test_append_to_experience_regression(self):
        """Test appending data to experience replay buffer (regression)."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)
        data.generate_dataset(task_id=0, batch_size=32, phase='training')

        # Initially exp_x_train is an empty list
        initial_exp_size = len(data.exp_x_train)
        data.append_to_experience(task_id=0)

        # Experience replay should have grown (now a tensor)
        assert data.exp_x_train.shape[0] > initial_exp_size
        assert data.exp_y_train.shape[0] > initial_exp_size

    def test_append_to_experience_classification(self):
        """Test appending data to experience replay buffer (classification)."""
        config = {
            'data_id': 'mnist',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)
        data.generate_dataset(task_id=0, batch_size=32, phase='training')

        # Initially exp_x_train is an empty list
        initial_exp_size = len(data.exp_x_train)
        data.append_to_experience(task_id=0)

        # Experience replay should have grown (now a tensor)
        assert data.exp_x_train.shape[0] > initial_exp_size
        assert data.exp_y_train.shape[0] > initial_exp_size

        # Check that channel dimension is correct for MNIST (1 channel)
        assert len(data.exp_x_train.shape) == 4
        assert data.exp_x_train.shape[1] == 1  # 1-channel

    def test_append_to_experience_cifar(self):
        """Test appending CIFAR data preserves 3 channels."""
        config = {
            'data_id': 'cifar10',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)
        data.generate_dataset(task_id=0, batch_size=32, phase='training')
        data.append_to_experience(task_id=0)

        # Check that 3-channel images are preserved correctly
        assert len(data.exp_x_train.shape) == 4
        assert data.exp_x_train.shape[1] == 3  # 3-channel

    def test_experience_replay_buffer_limit(self):
        """Test that experience replay buffer respects size limit."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 100,  # Small limit
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)

        # Add multiple tasks
        for task_id in range(3):
            data.generate_dataset(task_id=task_id, batch_size=32, phase='training')
            data.append_to_experience(task_id=task_id)

        # Buffer should not exceed limit (or be close to it)
        assert data.exp_x_train.shape[0] <= config['len_exp_replay'] * 1.5

    def test_continual_learning_multiple_tasks(self):
        """Test generating datasets for multiple continual learning tasks."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)

        for task_id in range(3):
            train_loader, exp_loader = data.generate_dataset(
                task_id=task_id, batch_size=32, phase='training'
            )

            assert train_loader is not None
            data.append_to_experience(task_id=task_id)

        # After 3 tasks, experience replay should have data
        assert data.exp_x_train.shape[0] > 0


class TestContinualDataset:
    """Tests for Continual_Dataset class."""

    def test_continual_dataset_regression(self):
        """Test Continual_Dataset for regression data."""
        config = {
            'problem': 'regression',
            'network': 'fcnn'
        }

        x = torch.randn(100, 10)
        y = torch.randn(100, 5)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)

        assert len(dataset) == 100
        assert dataset[0][0].shape == (10,)
        assert dataset[0][1].shape == (5,)

    def test_continual_dataset_classification(self):
        """Test Continual_Dataset for classification data."""
        config = {
            'problem': 'classification',
            'network': 'cnn'
        }

        x = torch.randn(100, 1, 28, 28)
        y = np.random.randint(0, 10, 100)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)

        assert len(dataset) == 100
        assert dataset[0][0].shape == (1, 28, 28)
        assert isinstance(dataset[0][1], (int, np.integer))

    def test_continual_dataset_graph(self):
        """Test Continual_Dataset for graph data."""
        config = {
            'problem': 'graph',
            'network': 'gnn'
        }

        # Graph data structure is different
        x = torch.randn(100, 10)
        y = np.random.randint(0, 5, 100)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)

        assert len(dataset) == 100

    def test_continual_dataset_with_dataloader(self):
        """Test Continual_Dataset works with PyTorch DataLoader."""
        config = {
            'problem': 'regression',
            'network': 'fcnn'
        }

        x = torch.randn(100, 10)
        y = torch.randn(100, 5)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        batch_count = 0
        for batch_x, batch_y in dataloader:
            assert batch_x.shape[0] <= 32
            assert batch_y.shape[0] <= 32
            batch_count += 1

        assert batch_count >= 3  # 100 samples / 32 batch size

    def test_continual_dataset_empty(self):
        """Test Continual_Dataset handles empty data correctly."""
        config = {
            'problem': 'regression',
            'network': 'fcnn'
        }

        x = torch.randn(0, 10)
        y = torch.randn(0, 5)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)
        assert len(dataset) == 0

    def test_continual_dataset_single_sample(self):
        """Test Continual_Dataset with single sample."""
        config = {
            'problem': 'regression',
            'network': 'fcnn'
        }

        x = torch.randn(1, 10)
        y = torch.randn(1, 5)

        dataset = Continual_Dataset(config, data_x=x, data_y=y)

        assert len(dataset) == 1
        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (10,)
        assert sample_y.shape == (5,)


class TestDataTransformations:
    """Tests for data transformations and preprocessing."""

    def test_mnist_channel_dimension(self):
        """Test MNIST data gets proper channel dimension."""
        config = {
            'data_id': 'mnist',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)
        data.generate_dataset(task_id=0, batch_size=32, phase='training')
        data.append_to_experience(task_id=0)

        # Should have shape [N, 1, 28, 28]
        assert data.exp_x_train.shape[1] == 1

    def test_cifar_channel_dimension(self):
        """Test CIFAR data preserves 3-channel dimension."""
        config = {
            'data_id': 'cifar10',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'classification',
            'network': 'cnn'
        }

        data = data_return(config)
        data.generate_dataset(task_id=0, batch_size=32, phase='training')
        data.append_to_experience(task_id=0)

        # Should have shape [N, 3, 32, 32]
        assert data.exp_x_train.shape[1] == 3
        assert data.exp_x_train.shape[2:] == (32, 32)

    def test_sine_data_shape(self):
        """Test sine regression data has correct shapes."""
        config = {
            'data_id': 'sine',
            'len_exp_replay': 1000,
            'batch_size': 32,
            'problem': 'regression',
            'network': 'fcnn'
        }

        data = data_return(config)
        train_loader, _ = data.generate_dataset(
            task_id=0, batch_size=32, phase='training'
        )

        for x, y in train_loader:
            # Sine: x is typically 1D or 2D input, y is output
            assert len(x.shape) == 2  # [batch, features]
            assert len(y.shape) == 2  # [batch, outputs]
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
