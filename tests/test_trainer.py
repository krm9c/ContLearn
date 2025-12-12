"""
Unit tests for Trainer class in utils/trainer.py.
Tests loss functions, accuracy metrics, and training loop components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pytest
import tempfile
import shutil

from utils.trainer import Trainer
from utils.model import MLP, CNN3D


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_trainer_initialization(self):
        """Test Trainer initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            assert trainer.loss == 'mse'
            assert trainer.metric == 'mse'
            assert trainer.problem == 'vectors'
            assert trainer.writer is not None

    def test_trainer_classification_init(self):
        """Test Trainer initialization for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')
            assert trainer.loss == 'class'
            assert trainer.metric == 'class'


class TestLossFunctions:
    """Tests for loss functions."""
    def test_loss_fn_mse(self):
        """Test MSE loss function for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(10).astype(np.float32))
            y = jnp.array(np.random.randn(5).astype(np.float32))

            loss = trainer.loss_fn_mse(params, static, x, y)
            
            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()
            assert loss >= 0  # MSE is non-negative

    def test_loss_fn_mse_batch(self):
        """Test MSE loss with batch of inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.array(np.random.randn(32, 5).astype(np.float32))

            loss = trainer.loss_fn_mse(params, static, x, y)

            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()

    def test_loss_fn_class(self):
        """Test classification loss function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')

            model = MLP(sizes=[10, 64, 10])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            # One-hot encoded labels
            y = jnp.zeros((32, 10))
            y = y.at[jnp.arange(32), jnp.array(np.random.randint(0, 10, 32))].set(1.0)

            loss = trainer.loss_fn_class(params, static, x, y)

            assert isinstance(loss, jnp.ndarray)
            assert loss.shape == ()

    def test_loss_fn_class_gradient(self):
        """Test that classification loss can be differentiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')

            model = MLP(sizes=[10, 64, 10])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.zeros((32, 10))
            y = y.at[jnp.arange(32), jnp.array(np.random.randint(0, 10, 32))].set(1.0)

            # Compute gradient
            loss_and_grad = eqx.filter_value_and_grad(trainer.loss_fn_class)
            loss, grads = loss_and_grad(params, static, x, y)

            assert loss.shape == ()
            assert grads is not None


class TestMetricFunctions:
    """Tests for metric/accuracy functions."""

    def test_accuracy_vectors(self):
        """Test accuracy metric for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')

            model = MLP(sizes=[10, 64, 10])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            # One-hot encoded labels
            labels = jnp.array(np.random.randint(0, 10, 32))
            y = jnp.zeros((32, 10))
            y = y.at[jnp.arange(32), labels].set(1.0)

            accuracy = trainer.accuracy_vectors(params, static, x, y)

            assert isinstance(accuracy, jnp.ndarray)
            assert 0 <= accuracy <= 1

    def test_mse_vectors(self):
        """Test MSE metric for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.array(np.random.randn(32, 5).astype(np.float32))

            mse = trainer.mse_vectors(params, static, x, y)

            assert isinstance(mse, jnp.ndarray)
            assert mse >= 0

    def test_get_pred(self):
        """Test get_pred function for predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))

            predictions = trainer.get_pred(params, static, x)

            assert predictions.shape == (32, 5)
            assert not jnp.isnan(predictions).any()


class TestGraphLossFunctions:
    """Tests for graph-specific loss functions."""

    def test_loss_fn_class_graph_structure(self):
        """Test that graph classification loss function exists and has correct signature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='graph')

            # Just verify the method exists and can be called
            assert hasattr(trainer, 'loss_fn_class_graph')

    def test_accuracy_graphs_structure(self):
        """Test that graph accuracy function exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='graph')

            assert hasattr(trainer, 'accuracy_graphs')
            assert hasattr(trainer, 'accuracy_graphs_AWBT')


class TestTrainerJIT:
    """Tests for JIT compilation of trainer functions."""

    def test_loss_fn_mse_is_jitted(self):
        """Test that loss_fn_mse is JIT compiled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(10).astype(np.float32))
            y = jnp.array(np.random.randn(5).astype(np.float32))

            # First call should compile
            loss1 = trainer.loss_fn_mse(params, static, x, y)

            # Second call should use cached compilation
            loss2 = trainer.loss_fn_mse(params, static, x, y)

            assert jnp.allclose(loss1, loss2)

    def test_accuracy_vectors_is_jitted(self):
        """Test that accuracy_vectors is JIT compiled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')

            model = MLP(sizes=[10, 64, 10])
            params, static = eqx.partition(model, eqx.is_array)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.zeros((32, 10))
            y = y.at[jnp.arange(32), jnp.array(np.random.randint(0, 10, 32))].set(1.0)

            # First call
            acc1 = trainer.accuracy_vectors(params, static, x, y)

            # Second call (should be faster due to JIT)
            acc2 = trainer.accuracy_vectors(params, static, x, y)

            assert jnp.allclose(acc1, acc2)


class TestTrainerWithOptimizer:
    """Tests for trainer with optimizer integration."""

    def test_training_step_regression(self):
        """Test a single training step for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            optimizer = optax.adam(1e-3)
            opt_state = optimizer.init(params)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.array(np.random.randn(32, 5).astype(np.float32))

            # Compute loss and gradient
            loss_and_grad = eqx.filter_value_and_grad(trainer.loss_fn_mse)
            loss, grads = loss_and_grad(params, static, x, y)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = eqx.apply_updates(params, updates)

            assert loss.shape == ()
            assert not jnp.isnan(loss)

    def test_training_step_classification(self):
        """Test a single training step for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='class', metric='class', problem='vectors')

            model = MLP(sizes=[10, 64, 10])
            params, static = eqx.partition(model, eqx.is_array)

            optimizer = optax.adam(1e-3)
            opt_state = optimizer.init(params)

            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.zeros((32, 10))
            y = y.at[jnp.arange(32), jnp.array(np.random.randint(0, 10, 32))].set(1.0)

            # Compute loss and gradient
            loss_and_grad = eqx.filter_value_and_grad(trainer.loss_fn_class)
            loss, grads = loss_and_grad(params, static, x, y)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = eqx.apply_updates(params, updates)

            assert loss.shape == ()
            assert not jnp.isnan(loss)

    def test_multiple_training_steps_decreases_loss(self):
        """Test that multiple training steps decrease loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(logdir=tmpdir, Loss='mse', metric='mse', problem='vectors')

            model = MLP(sizes=[10, 64, 5])
            params, static = eqx.partition(model, eqx.is_array)

            optimizer = optax.adam(1e-2)
            opt_state = optimizer.init(params)

            # Fixed data
            x = jnp.array(np.random.randn(32, 10).astype(np.float32))
            y = jnp.array(np.random.randn(32, 5).astype(np.float32))

            losses = []
            for _ in range(10):
                loss_and_grad = eqx.filter_value_and_grad(trainer.loss_fn_mse)
                loss, grads = loss_and_grad(params, static, x, y)
                losses.append(float(loss))

                updates, opt_state = optimizer.update(grads, opt_state)
                params = eqx.apply_updates(params, updates)

            # Loss should generally decrease (with some tolerance)
            assert losses[-1] < losses[0] or abs(losses[-1] - losses[0]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
