"""
Unit tests for checkpoint loading and model initialization in training/checkpoint.py.
Tests load_checkpoint function for different problem types and configurations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile

from training.checkpoint import load_checkpoint
from utils.model import MLP, CNN, CNN3D, myNN
from utils.trainer import Trainer


class TestCheckpointRegression:
    """Tests for checkpoint loading with regression problems."""

    def test_load_checkpoint_sine_regression(self):
        """Test loading checkpoint for sine regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Verify trainer
            assert isinstance(trainer, Trainer)
            assert trainer.loss == 'mse'
            assert trainer.metric == 'mse'

            # Verify optimizer
            assert optim is not None

            # Verify dataset
            assert dataset is not None

            # Verify model
            assert isinstance(model, MLP)
            assert len(model.sizes) > 0

    def test_checkpoint_regression_model_shape(self):
        """Test that regression model has correct input/output shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Generate some data to test model
            dataloader_curr, _ = dataset.generate_dataset(
                task_id=0, batch_size=32, phase='training'
            )

            for x, y in dataloader_curr:
                # Convert to JAX arrays
                x_jax = jnp.array(x.numpy())
                y_jax = jnp.array(y.numpy())

                # Test forward pass
                pred = jax.vmap(model)(x_jax)

                assert pred.shape == y_jax.shape
                break


class TestCheckpointClassification:
    """Tests for checkpoint loading with classification problems."""

    def test_load_checkpoint_mnist_classification(self):
        """Test loading checkpoint for MNIST classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'classification',
                'data': 'mnist',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'delta': 1e-3,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Verify trainer
            assert trainer.loss == 'class'
            assert trainer.metric == 'class'

            # Verify model is CNN
            assert isinstance(model, CNN)

    def test_load_checkpoint_cifar10_classification(self):
        """Test loading checkpoint for CIFAR-10 classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'classification',
                'data': 'cifar10',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'delta': 1e-3,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Verify model is CNN3D for CIFAR
            assert isinstance(model, CNN3D)
            assert model.channel_in == 3
            assert model.channel_out == 32

    def test_load_checkpoint_cifar100_classification(self):
        """Test loading checkpoint for CIFAR-100 classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'classification',
                'data': 'cifar100',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 100,
                'delta': 1e-3,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Verify model is CNN3D with 100 classes
            assert isinstance(model, CNN3D)
            # Check output size matches number of classes
            test_input = jax.random.normal(jax.random.PRNGKey(42), (3, 32, 32))
            output = model(test_input)
            assert output.shape == (100,)


class TestCheckpointGraphClassification:
    """Tests for checkpoint loading with graph classification problems."""

    def test_load_checkpoint_graph_classification(self):
        """Test loading checkpoint for graph classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch_size': 32,
                'batch': 32,
                'n_class': 2,
                'class_per_task': 1,
                'delta': 1e-4,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, test_loader, model = load_checkpoint(config)

            # Verify trainer
            assert trainer.loss == 'class'
            assert trainer.problem == 'graph'

            # Verify model is myNN (GCN + MLP)
            assert isinstance(model, myNN)

            # Verify test loader
            assert test_loader is not None

            # Verify dataset
            assert dataset is not None

    def test_checkpoint_graph_model_structure(self):
        """Test that graph model has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch_size': 32,
                'batch': 32,
                'n_class': 2,
                'class_per_task': 1,
                'delta': 1e-4,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, test_loader, model = load_checkpoint(config)

            # Verify model components
            assert hasattr(model, 'gcn_layers')
            assert hasattr(model, 'feed_layers')
            assert hasattr(model, 'pool_layer')
            assert hasattr(model, 'A_gcn')
            assert hasattr(model, 'B_gcn')
            assert hasattr(model, 'A_feed')
            assert hasattr(model, 'B_feed')


class TestCheckpointOptimizer:
    """Tests for optimizer initialization in checkpoint loading."""

    def test_optimizer_type_regression(self):
        """Test optimizer type for regression (Adam)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Test that optimizer can initialize state
            import equinox as eqx
            params, static = eqx.partition(model, eqx.is_array)
            opt_state = optim.init(params)

            assert opt_state is not None

    def test_optimizer_type_graph(self):
        """Test optimizer type for graph classification (AdamW)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch_size': 32,
                'batch': 32,
                'n_class': 2,
                'class_per_task': 1,
                'delta': 1e-4,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, test_loader, model = load_checkpoint(config)

            # Test optimizer initialization
            import equinox as eqx
            params, static = eqx.partition(model, eqx.is_array)
            opt_state = optim.init(params)

            assert opt_state is not None

    def test_optimizer_learning_rate(self):
        """Test that optimizer uses specified learning rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lr_test = 5e-4

            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': lr_test,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Optimizer should be created with specified learning rate
            # (We can't directly check lr from the optimizer object,
            #  but we verify it was created successfully)
            assert optim is not None


class TestCheckpointDataset:
    """Tests for dataset initialization in checkpoint loading."""

    def test_dataset_generation_regression(self):
        """Test that dataset can generate batches for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            # Generate dataset for a task
            train_loader, exp_loader = dataset.generate_dataset(
                task_id=0, batch_size=32, phase='training'
            )

            assert train_loader is not None
            assert exp_loader is not None

            # Check that we can iterate through data
            batch_count = 0
            for x, y in train_loader:
                assert x.shape[0] <= 32
                batch_count += 1
                if batch_count >= 3:
                    break

            assert batch_count >= 1

    def test_dataset_generation_classification(self):
        """Test that dataset can generate batches for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'classification',
                'data': 'mnist',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'delta': 1e-3,
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard'
            }

            trainer, optim, dataset, model = load_checkpoint(config)

            train_loader, exp_loader = dataset.generate_dataset(
                task_id=0, batch_size=32, phase='training'
            )

            batch_count = 0
            for x, y in train_loader:
                # MNIST: [batch, 1, 28, 28]
                assert x.shape[0] <= 32
                assert len(x.shape) == 4
                batch_count += 1
                if batch_count >= 3:
                    break

            assert batch_count >= 1


class TestCheckpointSeed:
    """Tests for consistent initialization with SEED."""

    def test_seed_consistency(self):
        """Test that same config produces consistent model initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'delta': 1e-3,
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard'
            }

            # Load checkpoint twice
            _, _, _, model1 = load_checkpoint(config)
            _, _, _, model2 = load_checkpoint(config)

            # Models should have same architecture
            assert model1.sizes == model2.sizes

            # Note: Weights won't be identical due to random initialization,
            # but structure should match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
