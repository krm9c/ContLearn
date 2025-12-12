"""
Unit tests for neural network models in utils/model.py.
Tests MLP, CNN, CNN3D architectures with and without AWB transformations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from utils.model import MLP, MLPorig, CNN, CNN3D, Linear, Linear2, Linear3


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_initialization(self):
        """Test MLP initializes with correct layer sizes."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes)

        assert model.sizes == sizes
        assert len(model.layers) == len(sizes) - 1
        assert len(model.A) == len(sizes) - 1
        assert len(model.B) == len(sizes) - 1

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = model(x)

        assert output.shape == (5,)
        assert not jnp.isnan(output).any()

    def test_mlp_batch_forward(self):
        """Test MLP forward pass with batch of inputs."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes)

        x = jnp.array(np.random.randn(32, 10).astype(np.float32))
        output = jax.vmap(model)(x)

        assert output.shape == (32, 5)
        assert not jnp.isnan(output).any()

    def test_mlp_awb_forward(self):
        """Test MLP forward pass with AWB transformation."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = model.getAWB(x)

        # AWB output shape depends on A matrix dimensions
        assert output.shape[0] == model.A[-1].shape[0]
        assert not jnp.isnan(output).any()

    def test_mlp_awb_matrices_shape(self):
        """Test that AWB matrices have correct shapes."""
        sizes = [10, 64, 32, 5]
        model = MLP(sizes=sizes)

        # A matrices: output dimension for each layer
        assert model.A[0].shape[0] == sizes[1]  # First hidden layer
        assert model.A[-1].shape[0] == sizes[-1]  # Output layer

        # B matrices: transformation between layers
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            assert model.B[i].shape[0] == out_size
            assert model.B[i].shape[1] == in_size


class TestMLPorig:
    """Tests for MLPorig (original MLP architecture)."""

    def test_mlporig_initialization(self):
        """Test MLPorig initializes correctly."""
        model = MLPorig(key=42, input_dim=10, out_dim=5, n_layers=4, hln=64)

        assert model.input_layer is not None
        assert model.output_layers is not None
        assert len(model.feed_layers) == 4 - 2  # n_layers - 2

    def test_mlporig_forward_pass(self):
        """Test MLPorig forward pass."""
        model = MLPorig(key=42, input_dim=10, out_dim=5, n_layers=4, hln=64)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = model(x)

        assert output.shape == (5,)
        assert not jnp.isnan(output).any()

    def test_mlporig_with_output_activation(self):
        """Test MLPorig with custom output activation."""
        model = MLPorig(key=42, input_dim=10, out_dim=5, n_layers=4, hln=64)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = model(x, outfunc=jax.nn.softmax)

        assert output.shape == (5,)
        assert jnp.allclose(jnp.sum(output), 1.0, atol=1e-5)  # Softmax sums to 1


class TestCNN:
    """Tests for CNN model (1-channel images)."""

    def test_cnn_initialization(self):
        """Test CNN initializes correctly."""
        key = jax.random.PRNGKey(42)
        filter_size = 3
        feed_sizes = [1875, 512, 64, 10]

        model = CNN(key, filter_size=filter_size, feed_sizes=feed_sizes)

        assert model.filter_size == filter_size
        assert model.feed_sizes == feed_sizes
        assert len(model.feed_layers) == len(feed_sizes) - 1

    def test_cnn_forward_pass(self):
        """Test CNN forward pass with 1x28x28 input (MNIST-style)."""
        key = jax.random.PRNGKey(42)
        model = CNN(key, filter_size=3, feed_sizes=[1875, 512, 64, 10])

        # Create 1-channel 28x28 input
        x = jax.random.normal(key, (1, 28, 28))
        output = model(x)

        assert output.shape == (10,)
        assert not jnp.isnan(output).any()

    def test_cnn_awbt_forward(self):
        """Test CNN AWB transformation forward pass."""
        key = jax.random.PRNGKey(42)
        model = CNN(key, filter_size=3, feed_sizes=[1875, 512, 64, 10])

        x = jax.random.normal(key, (1, 28, 28))
        output = model.get_AWBT(x)

        assert not jnp.isnan(output).any()


class TestCNN3D:
    """Tests for CNN3D model (3-channel images)."""

    def test_cnn3d_initialization(self):
        """Test CNN3D initializes correctly."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        assert model.filter_size == 3
        assert model.channel_in == 3
        assert model.channel_out == 32
        assert model.num_classes == 10

    def test_cnn3d_forward_pass(self):
        """Test CNN3D forward pass with 3x32x32 input (CIFAR-style)."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        x = jax.random.normal(key, (3, 32, 32))
        output = model(x)

        assert output.shape == (10,)
        assert not jnp.isnan(output).any()

    def test_cnn3d_batch_forward(self):
        """Test CNN3D with batch of inputs."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        x = jax.random.normal(key, (16, 3, 32, 32))
        output = jax.vmap(model)(x)

        assert output.shape == (16, 10)
        assert not jnp.isnan(output).any()

    def test_cnn3d_calc_output_size(self):
        """Test CNN3D output size calculation."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        # 32x32 -> conv(3) -> 30 -> pool -> 15
        result = model.calc_output_size(32, 3)
        assert result == 15

        # 15 -> conv(3) -> 13 -> pool -> 6
        result = model.calc_output_size(15, 3)
        assert result == 6

    def test_cnn3d_awbt_forward(self):
        """Test CNN3D AWB transformation."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        x = jax.random.normal(key, (3, 32, 32))
        output = model.get_AWBT(x)

        assert not jnp.isnan(output).any()


class TestLinearLayers:
    """Tests for custom Linear layer variants."""

    def test_linear_initialization(self):
        """Test Linear layer initialization."""
        key = jax.random.PRNGKey(42)
        layer = Linear(in_size=10, out_size=5, key=key)

        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (5,)

    def test_linear_forward(self):
        """Test Linear layer forward pass."""
        key = jax.random.PRNGKey(42)
        layer = Linear(in_size=10, out_size=5, key=key)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = layer(x)

        assert output.shape == (5,)
        assert not jnp.isnan(output).any()

    def test_linear2_initialization(self):
        """Test Linear2 layer initialization (bias shape (1, out_size))."""
        key = jax.random.PRNGKey(42)
        layer = Linear2(in_size=10, out_size=5, key=key)

        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (1, 5)

    def test_linear2_forward(self):
        """Test Linear2 layer forward pass."""
        key = jax.random.PRNGKey(42)
        layer = Linear2(in_size=10, out_size=5, key=key)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = layer(x)

        assert output.shape == (5,)
        assert not jnp.isnan(output).any()

    def test_linear3_initialization(self):
        """Test Linear3 layer initialization (bias shape (out_size, 1))."""
        key = jax.random.PRNGKey(42)
        layer = Linear3(in_size=10, out_size=5, key=key)

        assert layer.weight.shape == (5, 10)
        assert layer.bias.shape == (5, 1)

    def test_linear3_forward(self):
        """Test Linear3 layer forward pass."""
        key = jax.random.PRNGKey(42)
        layer = Linear3(in_size=10, out_size=5, key=key)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = layer(x)

        assert output.shape == (5,)
        assert not jnp.isnan(output).any()


class TestModelSerialization:
    """Tests for model serialization with Equinox."""

    def test_mlp_serialization(self, tmp_path):
        """Test MLP can be serialized and deserialized."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes)

        # Test input
        x = jnp.array(np.random.randn(10).astype(np.float32))
        output_before = model(x)

        # Serialize
        filepath = tmp_path / "model.eqx"
        eqx.tree_serialise_leaves(str(filepath), model)

        # Deserialize
        model_loaded = eqx.tree_deserialise_leaves(str(filepath), model)
        output_after = model_loaded(x)

        # Check outputs match
        assert jnp.allclose(output_before, output_after)

    def test_cnn3d_serialization(self, tmp_path):
        """Test CNN3D can be serialized and deserialized."""
        key = jax.random.PRNGKey(42)
        model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                      channel_in=3, channel_out=32, num_classes=10)

        x = jax.random.normal(key, (3, 32, 32))
        output_before = model(x)

        # Serialize
        filepath = tmp_path / "cnn3d.eqx"
        eqx.tree_serialise_leaves(str(filepath), model)

        # Deserialize
        model_loaded = eqx.tree_deserialise_leaves(str(filepath), model)
        output_after = model_loaded(x)

        # Check outputs match
        assert jnp.allclose(output_before, output_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
