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

# Added by Claude: Mark as unit test for test categorization
pytestmark = pytest.mark.unit

from cl.models.layers import (
    Linear, LinearGCN, Linear2, Dropout,
    AWBLayerSpec, AWBShapeError,
    compute_V_conv2d_single_channel,
    compute_V_conv2d_multi_channel
)


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


# Added by Claude: AWB tests for layer transformations
class TestLinearAWB:
    """Tests for Linear layer AWB methods."""

    def test_linear_compute_V_weight_identity(self, jax_key):
        """Test V computation with identity A and B matrices."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        # Identity matrices (no transformation)
        A = jnp.eye(5, 5)
        B = jnp.eye(10, 10)

        V = layer.compute_V_weight(A, layer.weight, B)

        # With identity matrices, V should equal W
        assert jnp.allclose(V, layer.weight)

    def test_linear_compute_V_weight_scaling(self, jax_key):
        """Test V computation with scaling transformations."""
        layer = Linear(in_size=4, out_size=3, key=jax_key)

        # Scaling matrices
        A = 2.0 * jnp.eye(3, 3)
        B = 0.5 * jnp.eye(4, 4)

        V = layer.compute_V_weight(A, layer.weight, B)

        # V = A @ W @ B.T = 2 * W * 0.5 = W
        assert jnp.allclose(V, layer.weight)

    def test_linear_compute_V_weight_shape_error(self, jax_key):
        """Test that shape mismatches raise AWBShapeError."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        # Wrong A shape
        A_wrong = jnp.eye(5, 4)  # Should be (5, 5) to match weight.shape[0]
        B = jnp.eye(10, 10)

        with pytest.raises(AWBShapeError, match="A.shape.*incompatible"):
            layer.compute_V_weight(A_wrong, layer.weight, B)

    def test_linear_compute_V_bias(self, jax_key):
        """Test bias transformation for Linear layer."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)

        # A matrix changes output dimension from 5 to 7
        A = jnp.ones((7, 5))
        B = jnp.eye(10, 10)

        V_bias = layer.compute_V_bias(A, B, layer.bias)

        # Linear: bias @ A.T, bias shape (1, 5) -> (1, 7)
        assert V_bias.shape == (1, 7)
        expected = layer.bias @ A.T
        assert jnp.allclose(V_bias, expected)


class TestLinear2AWB:
    """Tests for Linear2 layer AWB methods."""

    def test_linear2_compute_V_weight_identity(self, jax_key):
        """Test V computation with identity matrices for Linear2."""
        layer = Linear2(in_size=8, out_size=4, key=jax_key)

        A = jnp.eye(4, 4)
        B = jnp.eye(8, 8)

        V = layer.compute_V_weight(A, layer.weight, B)

        assert jnp.allclose(V, layer.weight)

    def test_linear2_compute_V_bias(self, jax_key):
        """Test bias transformation for Linear2 layer."""
        layer = Linear2(in_size=8, out_size=4, key=jax_key)

        # A changes output from 4 to 6
        A = jnp.ones((6, 4))
        B = jnp.eye(8, 8)

        V_bias = layer.compute_V_bias(A, B, layer.bias)

        # Linear2: A @ bias, bias shape (4, 1) -> (6, 1)
        assert V_bias.shape == (6, 1)
        expected = A @ layer.bias
        assert jnp.allclose(V_bias, expected)


class TestLinearGCNAWB:
    """Tests for LinearGCN layer AWB methods."""

    def test_linear_gcn_compute_V_weight_identity(self, jax_key):
        """Test V computation with identity matrices for LinearGCN."""
        layer = LinearGCN(in_size=6, out_size=3, key=jax_key)

        A = jnp.eye(3, 3)
        B = jnp.eye(6, 6)

        V = layer.compute_V_weight(A, layer.weight, B)

        assert jnp.allclose(V, layer.weight)

    def test_linear_gcn_compute_V_bias(self, jax_key):
        """Test bias transformation for LinearGCN layer."""
        layer = LinearGCN(in_size=6, out_size=3, key=jax_key)

        # For GCN, B should match output dimension for bias transformation
        # bias shape is (3, 1), so B.T should allow (3, 1) @ B.T
        # This means B.T should have first dim = 1
        # So B should be (new_dim, 1) where new_dim is the target output
        A = jnp.eye(3, 3)
        B = jnp.ones((5, 1))  # B.T will be (1, 5), so (3, 1) @ (1, 5) = (3, 5)

        V_bias = layer.compute_V_bias(A, B, layer.bias)

        # Check computation is correct
        expected = layer.bias @ B.T
        assert jnp.allclose(V_bias, expected)
        assert V_bias.shape == (3, 5)


class TestAWBLayerSpec:
    """Tests for AWBLayerSpec dataclass and validation."""

    def test_awb_layer_spec_creation(self, jax_key):
        """Test creating AWBLayerSpec."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)
        A = jnp.eye(5, 5)
        B = jnp.eye(10, 10)

        spec = AWBLayerSpec(
            layer=layer,
            A=A,
            B=B,
            layer_type='linear',
            layer_index=0
        )

        assert spec.layer is layer
        assert spec.A is A
        assert spec.B is B
        assert spec.layer_type == 'linear'
        assert spec.layer_index == 0

    def test_awb_layer_spec_validate_correct(self, jax_key):
        """Test validation passes for correct shapes."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)
        A = jnp.eye(7, 5)  # (new_out, old_out)
        B = jnp.eye(12, 10)  # (new_in, old_in)

        spec = AWBLayerSpec(layer=layer, A=A, B=B, layer_type='linear', layer_index=0)

        errors = spec.validate()
        assert len(errors) == 0

    def test_awb_layer_spec_validate_wrong_A(self, jax_key):
        """Test validation catches wrong A shape."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)
        A = jnp.eye(7, 4)  # Wrong: should be (7, 5) to match weight.shape[0]=5
        B = jnp.eye(12, 10)

        spec = AWBLayerSpec(layer=layer, A=A, B=B, layer_type='linear', layer_index=0)

        errors = spec.validate()
        assert len(errors) == 1
        assert 'A.shape' in errors[0]

    def test_awb_layer_spec_validate_wrong_B(self, jax_key):
        """Test validation catches wrong B shape."""
        layer = Linear(in_size=10, out_size=5, key=jax_key)
        A = jnp.eye(7, 5)
        B = jnp.eye(12, 8)  # Wrong: should be (12, 10) to match weight.shape[1]=10

        spec = AWBLayerSpec(layer=layer, A=A, B=B, layer_type='linear', layer_index=0)

        errors = spec.validate()
        assert len(errors) == 1
        assert 'B.shape' in errors[0]


class TestConvAWBUtilities:
    """Tests for Conv2d AWB utility functions."""

    def test_compute_V_conv2d_single_channel(self, jax_key):
        """Test single-channel conv AWB transformation."""
        # Setup: 2 output channels, 1 input channel, 3x3 filters
        channel_out = 2
        channel_in = 1
        filter_size = 3

        # Create dummy conv weights [channel_out, channel_in, H, W]
        W = jax.random.normal(jax_key, (channel_out, channel_in, filter_size, filter_size))

        # Identity A and B (no transformation)
        A_list = [jnp.eye(filter_size, filter_size) for _ in range(channel_out)]
        B_list = [jnp.eye(filter_size, filter_size) for _ in range(channel_out)]

        result = compute_V_conv2d_single_channel(A_list, W, B_list, channel_out)

        # Should return list of lists [channel_out][1]
        assert len(result) == channel_out
        assert len(result[0]) == 1

        # With identity matrices, transformed weights should equal original
        for i in range(channel_out):
            assert jnp.allclose(result[i][0], W[i][0])

    def test_compute_V_conv2d_multi_channel(self, jax_key):
        """Test multi-channel conv AWB transformation."""
        # Setup: 2 output channels, 3 input channels, 3x3 filters
        channel_out = 2
        channel_in = 3
        filter_size = 3

        # Create dummy conv weights [channel_out, channel_in, H, W]
        W = jax.random.normal(jax_key, (channel_out, channel_in, filter_size, filter_size))

        # Identity A and B (no transformation)
        A_list = [[jnp.eye(filter_size, filter_size) for _ in range(channel_in)]
                  for _ in range(channel_out)]
        B_list = [[jnp.eye(filter_size, filter_size) for _ in range(channel_in)]
                  for _ in range(channel_out)]

        result = compute_V_conv2d_multi_channel(A_list, W, B_list, channel_out, channel_in)

        # Should return nested list [channel_out][channel_in]
        assert len(result) == channel_out
        assert len(result[0]) == channel_in

        # With identity matrices, transformed weights should equal original
        for i in range(channel_out):
            for c in range(channel_in):
                assert jnp.allclose(result[i][c], W[i][c])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
