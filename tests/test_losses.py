"""
Unit tests for loss and metric functions in core/losses.py.
Tests LossMixin methods for regression and classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from cl.core.losses import LossMixin
from cl.models import MLP


class MockTrainer(LossMixin):
    """Mock trainer class that inherits LossMixin for testing."""

    def __init__(self, loss='mse', problem='vectors', metric='mse'):
        self.loss = loss
        self.problem = problem
        self.metric = metric


class TestMSELoss:
    """Tests for MSE loss function."""

    def test_loss_fn_mse_basic(self, jax_key):
        """Test MSE loss computation."""
        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        y = jnp.array(np.random.randn(16, 10).astype(np.float32))

        loss = trainer.loss_fn_mse(params, static, x, y)

        assert loss.shape == ()  # Scalar
        assert not jnp.isnan(loss)
        assert loss >= 0  # MSE is non-negative

    def test_loss_fn_mse_zero_when_perfect(self, jax_key):
        """Test MSE loss is zero when predictions match targets."""
        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        # Get model predictions as target
        y = jax.vmap(model)(x)
        y = y.squeeze(1)  # Remove extra dimension

        loss = trainer.loss_fn_mse(params, static, x, y)

        assert jnp.allclose(loss, 0.0, atol=1e-5)

    def test_mse_vectors_metric(self, jax_key):
        """Test MSE metric computation for vectors."""
        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        y = jnp.array(np.random.randn(16, 10).astype(np.float32))

        metric = trainer.mse_vectors(params, static, x, y)

        assert metric.shape == ()
        assert not jnp.isnan(metric)
        assert metric >= 0


class TestClassificationLoss:
    """Tests for classification loss function."""

    def test_loss_fn_class_basic(self, jax_key):
        """Test cross-entropy loss computation."""
        trainer = MockTrainer(loss='class', problem='vectors', metric='class')
        model = MLP(sizes=[784, 64, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 784).astype(np.float32))
        # One-hot encoded labels
        y = jnp.zeros((16, 10))
        y = y.at[jnp.arange(16), np.random.randint(0, 10, 16)].set(1.0)

        loss = trainer.loss_fn_class(params, static, x, y)

        assert loss.shape == ()
        assert not jnp.isnan(loss)

    def test_accuracy_vectors(self, jax_key):
        """Test accuracy computation for classification."""
        trainer = MockTrainer(loss='class', problem='vectors', metric='class')
        model = MLP(sizes=[784, 64, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 784).astype(np.float32))
        # One-hot encoded labels - standard 2D format (batch, classes)
        y = jnp.zeros((16, 10))
        labels = np.random.randint(0, 10, 16)
        y = y.at[jnp.arange(16), labels].set(1.0)

        accuracy = trainer.accuracy_vectors(params, static, x, y)

        assert accuracy.shape == ()
        assert 0.0 <= accuracy <= 1.0


class TestReturnLossGrad:
    """Tests for return_loss_grad unified interface."""

    def test_return_loss_grad_mse(self, jax_key):
        """Test return_loss_grad for MSE regression."""
        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        y = jnp.array(np.random.randn(16, 10).astype(np.float32))

        loss, grads = trainer.return_loss_grad(params, (x, y), static)

        assert loss.shape == ()
        assert not jnp.isnan(loss)
        # Check gradients exist
        for layer in grads.layers:
            assert layer.weight is not None
            assert not jnp.isnan(layer.weight).any()

    def test_return_loss_grad_class(self, jax_key):
        """Test return_loss_grad for classification."""
        trainer = MockTrainer(loss='class', problem='vectors', metric='class')
        model = MLP(sizes=[784, 64, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 784).astype(np.float32))
        y = jnp.zeros((16, 10))
        y = y.at[jnp.arange(16), np.random.randint(0, 10, 16)].set(1.0)

        loss, grads = trainer.return_loss_grad(params, (x, y), static)

        assert loss.shape == ()
        assert not jnp.isnan(loss)


class TestReturnMetric:
    """Tests for return_metric unified interface."""

    def test_return_metric_mse(self, jax_key):
        """Test return_metric for MSE regression."""
        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        y = jnp.array(np.random.randn(16, 10).astype(np.float32))

        metric = trainer.return_metric(params, static, (x, y), notABTrain=True)

        assert metric.shape == ()
        assert not jnp.isnan(metric)
        assert metric >= 0

    def test_return_metric_classification(self, jax_key):
        """Test return_metric for classification."""
        trainer = MockTrainer(loss='class', problem='vectors', metric='class')
        model = MLP(sizes=[784, 64, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 784).astype(np.float32))
        # Added by Claude: return_metric for classification expects integer labels
        # Model output is (16, 1, 10), return_metric does argmax on predictions
        # and compares with y (which should be integer class indices)
        y = jnp.array(np.random.randint(0, 10, 16).astype(np.int64))

        metric = trainer.return_metric(params, static, (x, y), notABTrain=True)

        assert metric.shape == ()
        assert 0.0 <= metric <= 1.0


class TestGetPred:
    """Tests for prediction methods."""

    def test_get_pred_vectors(self, jax_key):
        """Test get_pred for vector inputs."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))

        pred = trainer.get_pred(params, static, x)

        assert pred.shape == (16, 1, 10)  # Batch, 1 (from Linear), output_dim
        assert not jnp.isnan(pred).any()


class TestGradientComputation:
    """Tests for gradient computation through loss functions."""

    def test_loss_gradient_decreases_loss(self, jax_key):
        """Test that applying gradient decreases loss."""
        import optax

        trainer = MockTrainer(loss='mse', problem='vectors', metric='mse')
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        x = jnp.array(np.random.randn(16, 3).astype(np.float32))
        y = jnp.array(np.random.randn(16, 10).astype(np.float32))

        # Initial loss
        loss_before, grads = trainer.return_loss_grad(params, (x, y), static)

        # Apply gradient update
        optim = optax.sgd(learning_rate=0.01)
        opt_state = optim.init(params)
        updates, opt_state = optim.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)

        # Loss after update
        loss_after = trainer.loss_fn_mse(params_new, static, x, y)

        # Loss should decrease (for small learning rate)
        assert loss_after < loss_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
