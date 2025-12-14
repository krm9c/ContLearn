"""
Unit tests for neural network layers in models/layers.py.
Tests Linear, LinearGCN, and Dropout layers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cl.models.layers import Linear, LinearGCN, Dropout


class TestLinear:
    """Tests for Linear layer."""

    def test_linear_initialization(self, jax_key):
        """Test Linear layer initializes with correct shapes."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        # Weight shape: (out_size, in_size)
        assert layer.weight.shape == (5, 10)
        # Bias shape: (1, out_size) for broadcasting
        assert layer.bias.shape == (1, 5)

    def test_linear_forward_pass(self, jax_key):
        """Test Linear layer forward pass."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = layer(x)

        # Output shape: (1, out_size) due to bias broadcasting
        assert output.shape == (1, 5)
        assert not jnp.isnan(output).any()

    def test_linear_batch_forward(self, jax_key):
        """Test Linear layer with batched inputs using vmap."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        x = jnp.array(np.random.randn(32, 10).astype(np.float32))
        # Use vmap for batch processing
        output = jax.vmap(layer)(x)

        assert output.shape == (32, 1, 5)
        assert not jnp.isnan(output).any()

    def test_linear_weight_initialization_glorot(self, jax_key):
        """Test that weights are initialized with Glorot uniform."""
        layer = Linear(in_size=100, out_size=50, key=jax_key)

        # Glorot uniform: values should be within reasonable bounds
        # Bound = sqrt(6 / (fan_in + fan_out)) = sqrt(6/150) ~ 0.2
        assert jnp.abs(layer.weight).max() < 1.0
        assert jnp.abs(layer.bias).max() < 1.0

    def test_linear_forward_computation(self, jax_key):
        """Test that forward pass computes x @ W.T + bias correctly."""
        layer = Linear(in_size=3, out_size=2, key=jax_key)

        x = jnp.array([1.0, 2.0, 3.0])
        output = layer(x)

        # Manual computation
        expected = x @ layer.weight.T + layer.bias
        assert jnp.allclose(output, expected)


class TestLinearGCN:
    """Tests for LinearGCN layer (GCN variant)."""

    def test_linear_gcn_initialization(self, jax_key):
        """Test LinearGCN layer initializes with correct shapes."""
        layer = LinearGCN(in_size=10, out_size=5, key=jax_key)

        # Weight shape: (out_size, in_size)
        assert layer.weight.shape == (5, 10)
        # Bias shape: (out_size, 1) for GCN
        assert layer.bias.shape == (5, 1)

    def test_linear_gcn_forward_pass(self, jax_key):
        """Test LinearGCN layer forward pass."""
        layer = LinearGCN(in_size=10, out_size=5, key=jax_key)

        # GCN layer expects column vector or matrix
        x = jnp.array(np.random.randn(10, 1).astype(np.float32))
        output = layer(x)

        # Output shape: W @ x + bias = (5, 1)
        assert output.shape == (5, 1)
        assert not jnp.isnan(output).any()

    def test_linear_gcn_forward_computation(self, jax_key):
        """Test that GCN forward pass computes W @ x + bias correctly."""
        layer = LinearGCN(in_size=3, out_size=2, key=jax_key)

        x = jnp.array([[1.0], [2.0], [3.0]])  # Column vector
        output = layer(x)

        # Manual computation: W @ x + bias
        expected = layer.weight @ x + layer.bias
        assert jnp.allclose(output, expected)


class TestDropout:
    """Tests for Dropout layer."""

    def test_dropout_initialization(self):
        """Test Dropout layer initializes with correct rate."""
        dropout = Dropout(rate=0.5)
        assert dropout.rate == 0.5

        dropout_high = Dropout(rate=0.8)
        assert dropout_high.rate == 0.8

    def test_dropout_training_mode(self, jax_key):
        """Test Dropout applies dropout during training."""
        dropout = Dropout(rate=0.5)

        x = jnp.ones((100,))
        output = dropout(x, jax_key, is_training=True)

        # Some values should be zeroed out
        zero_count = jnp.sum(output == 0)
        assert zero_count > 0  # Some dropout occurred
        assert zero_count < 100  # Not all dropped

    def test_dropout_inference_mode(self, jax_key):
        """Test Dropout does nothing during inference."""
        dropout = Dropout(rate=0.5)

        x = jnp.ones((100,))
        output = dropout(x, jax_key, is_training=False)

        # During inference, output should equal input
        assert jnp.allclose(output, x)

    def test_dropout_requires_rng(self):
        """Test Dropout raises error without RNG key."""
        dropout = Dropout(rate=0.5)

        x = jnp.ones((10,))
        with pytest.raises(ValueError, match="requires a PRNG key"):
            dropout(x, None, is_training=True)

    def test_dropout_scaling(self, jax_key):
        """Test Dropout scales non-zeroed values by 1/rate."""
        dropout = Dropout(rate=0.5)

        x = jnp.ones((1000,))
        output = dropout(x, jax_key, is_training=True)

        # Non-zero values should be scaled by 1/rate = 2.0
        non_zero_values = output[output != 0]
        assert jnp.allclose(non_zero_values, 2.0)

    def test_dropout_different_rates(self, jax_key):
        """Test Dropout with different keep rates."""
        for rate in [0.3, 0.5, 0.7, 0.9]:
            dropout = Dropout(rate=rate)
            x = jnp.ones((1000,))
            output = dropout(x, jax_key, is_training=True)

            # Approximate check: kept fraction should be close to rate
            kept_fraction = jnp.mean(output != 0)
            assert abs(kept_fraction - rate) < 0.1  # Allow 10% deviation


class TestLayerGradients:
    """Tests for gradient computation through layers."""

    def test_linear_gradient_flow(self, jax_key):
        """Test gradients flow through Linear layer."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        def loss_fn(layer, x):
            return jnp.sum(layer(x) ** 2)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        grads = jax.grad(loss_fn)(layer, x)

        # Check gradients exist and are finite
        assert grads.weight is not None
        assert grads.bias is not None
        assert not jnp.isnan(grads.weight).any()
        assert not jnp.isnan(grads.bias).any()

    def test_linear_gcn_gradient_flow(self, jax_key):
        """Test gradients flow through LinearGCN layer."""
        layer = LinearGCN(in_size=10, out_size=5, key=jax_key)

        def loss_fn(layer, x):
            return jnp.sum(layer(x) ** 2)

        x = jnp.array(np.random.randn(10, 1).astype(np.float32))
        grads = jax.grad(loss_fn)(layer, x)

        assert grads.weight is not None
        assert grads.bias is not None
        assert not jnp.isnan(grads.weight).any()
        assert not jnp.isnan(grads.bias).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
