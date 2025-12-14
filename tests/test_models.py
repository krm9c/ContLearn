"""
Unit tests for neural network models in models/mlp.py.
Tests MLP model with and without AWB transformations.
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

from cl.models import MLP, create_mlp, Linear
from cl.models.layers import AWBLayerSpec


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_initialization(self, jax_key):
        """Test MLP initializes with correct layer sizes."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        assert model.sizes == sizes
        assert len(model.layers) == len(sizes) - 1  # 3 layers
        assert model.awb_enabled == False
        assert model.A is None
        assert model.B is None

    def test_mlp_initialization_with_awb(self, jax_key):
        """Test MLP initializes with AWB matrices when enabled."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        assert model.sizes == sizes
        assert model.awb_enabled == True
        assert model.A is not None
        assert model.B is not None
        assert len(model.A) == len(sizes) - 1
        assert len(model.B) == len(sizes) - 1

    def test_mlp_forward_pass(self, jax_key):
        """Test MLP forward pass."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output = model(x)

        # Linear layer outputs shape (1, out_size) due to bias broadcasting
        assert output.shape == (1, 5)
        assert not jnp.isnan(output).any()

    def test_mlp_batch_forward(self, jax_key):
        """Test MLP forward pass with batch of inputs."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        x = jnp.array(np.random.randn(32, 10).astype(np.float32))
        output = jax.vmap(model)(x)

        # Batch output includes extra dimension from Linear layer
        assert output.shape == (32, 1, 5)
        assert not jnp.isnan(output).any()

    def test_mlp_awb_matrices_shape(self, jax_key):
        """Test AWB transformation matrices have correct shapes."""
        sizes = [10, 64, 32, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        # A matrices: (out_size, 1) column vectors
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            assert model.A[i].shape == (out_size, 1)
            assert model.B[i].shape == (out_size, in_size)


class TestCreateMLP:
    """Tests for create_mlp factory function."""

    def test_create_mlp_basic(self, test_sine_config, jax_key):
        """Test create_mlp creates model from config."""
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        config = test_sine_config.copy()
        config['input_size'] = 3
        config['output_size'] = 10
        model = create_mlp(config)

        assert model is not None
        assert model.sizes[0] == 3   # input
        assert model.sizes[-1] == 10  # output

    def test_create_mlp_with_awb(self, test_sine_awb_config, jax_key):
        """Test create_mlp creates AWB-enabled model."""
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        config = test_sine_awb_config.copy()
        config['input_size'] = 3
        config['output_size'] = 10
        model = create_mlp(config)

        assert model.awb_enabled == True
        assert model.A is not None
        assert model.B is not None

    def test_create_mlp_architecture(self, test_sine_config, jax_key):
        """Test create_mlp creates correct architecture from config."""
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        config = test_sine_config.copy()
        config['n_layers'] = 3
        config['hln'] = 64
        config['input_size'] = 5
        config['output_size'] = 2

        model = create_mlp(config)

        # Sizes: [input, hln, hln, ..., output]
        assert model.sizes[0] == 5
        assert model.sizes[-1] == 2
        assert len(model.layers) == config['n_layers'] - 1


class TestMLPSerialization:
    """Tests for model serialization with Equinox."""

    def test_mlp_serialization(self, jax_key, tmp_path):
        """Test MLP can be serialized and deserialized."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

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

    def test_mlp_awb_serialization(self, jax_key, tmp_path):
        """Test MLP with AWB can be serialized and deserialized."""
        sizes = [10, 64, 64, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output_before = model(x)

        # Serialize
        filepath = tmp_path / "model_awb.eqx"
        eqx.tree_serialise_leaves(str(filepath), model)

        # Deserialize
        model_loaded = eqx.tree_deserialise_leaves(str(filepath), model)
        output_after = model_loaded(x)

        # Check outputs match
        assert jnp.allclose(output_before, output_after)

        # Check AWB matrices preserved
        for i in range(len(sizes) - 1):
            assert jnp.allclose(model.A[i], model_loaded.A[i])
            assert jnp.allclose(model.B[i], model_loaded.B[i])


class TestMLPGradients:
    """Tests for gradient computation through MLP."""

    def test_mlp_gradient_flow(self, jax_key):
        """Test gradients flow through MLP."""
        sizes = [10, 32, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        # Added by Claude: partition model to separate arrays from non-arrays (like sizes)
        # This prevents JAX from trying to differentiate integer values
        params, static = eqx.partition(model, eqx.is_array)

        def loss_fn(params, static, x):
            model = eqx.combine(params, static)
            return jnp.sum(model(x) ** 2)

        x = jnp.array(np.random.randn(10).astype(np.float64))
        grads = jax.grad(loss_fn)(params, static, x)

        # Check gradients exist for all layers
        for i, layer in enumerate(grads.layers):
            assert layer.weight is not None
            assert layer.bias is not None
            assert not jnp.isnan(layer.weight).any()
            assert not jnp.isnan(layer.bias).any()

    def test_mlp_awb_gradient_flow(self, jax_key):
        """Test gradients flow to AWB matrices when enabled."""
        sizes = [10, 32, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        # Added by Claude: partition model to separate arrays from non-arrays
        params, static = eqx.partition(model, eqx.is_array)

        def loss_fn(params, static, x):
            model = eqx.combine(params, static)
            return jnp.sum(model(x) ** 2)

        x = jnp.array(np.random.randn(10).astype(np.float64))
        grads = jax.grad(loss_fn)(params, static, x)

        # Check gradients exist for A and B matrices
        for i in range(len(sizes) - 1):
            assert grads.A[i] is not None
            assert grads.B[i] is not None


class TestMLPPartitioning:
    """Tests for Equinox model partitioning."""

    def test_mlp_partition_is_array(self, jax_key):
        """Test MLP can be partitioned by is_array filter."""
        sizes = [10, 32, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        params, static = eqx.partition(model, eqx.is_array)

        # Added by Claude: sizes is a Python list (not array), so it appears in static
        # params has arrays (weights/biases), static has non-arrays (sizes, awb_enabled)
        assert static.sizes == sizes
        for layer in params.layers:
            assert layer.weight is not None
            assert layer.bias is not None

    def test_mlp_combine_after_partition(self, jax_key):
        """Test MLP can be combined after partitioning."""
        sizes = [10, 32, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        x = jnp.array(np.random.randn(10).astype(np.float32))
        output_before = model(x)

        # Partition and recombine
        params, static = eqx.partition(model, eqx.is_array)
        model_combined = eqx.combine(params, static)
        output_after = model_combined(x)

        assert jnp.allclose(output_before, output_after)


# Added by Claude: Tests for MLP AWBModel interface
class TestMLPAWBInterface:
    """Tests for MLP AWBModel interface methods."""

    def test_get_awb_layer_specs_disabled(self, jax_key):
        """Test get_awb_layer_specs when AWB is disabled."""
        sizes = [3, 10, 10, 2]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=False)

        specs = model.get_awb_layer_specs()

        assert len(specs) == 3  # 3 layers
        for i, spec in enumerate(specs):
            assert isinstance(spec, AWBLayerSpec)
            assert spec.layer_type == 'linear'
            assert spec.layer_index == i
            assert spec.A is None
            assert spec.B is None

    def test_get_awb_layer_specs_enabled(self, jax_key):
        """Test get_awb_layer_specs when AWB is enabled."""
        sizes = [3, 10, 10, 2]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        specs = model.get_awb_layer_specs()

        assert len(specs) == 3
        for i, spec in enumerate(specs):
            assert spec.layer_type == 'linear'
            assert spec.layer_index == i
            assert spec.A is not None
            assert spec.B is not None

        # Note: Initial A/B matrices are placeholders with shapes (out, 1) and (out, in),
        # not the correct (new_out, old_out) and (new_in, old_in).
        # Proper A/B initialization happens in with_new_AB_matrices().

    def test_apply_V_transformation(self, jax_key):
        """Test apply_V_transformation computes V = A @ W @ B.T."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        # Set identity A and B matrices
        model = eqx.tree_at(lambda x: x.A, model, [jnp.eye(3, 3), jnp.eye(2, 2)])
        model = eqx.tree_at(lambda x: x.B, model, [jnp.eye(2, 2), jnp.eye(3, 3)])

        # Store original weights
        original_weights = [layer.weight.copy() for layer in model.layers]

        # Apply V transformation
        model_transformed = model.apply_V_transformation()

        # With identity matrices, weights should remain unchanged
        for i, layer in enumerate(model_transformed.layers):
            assert jnp.allclose(layer.weight, original_weights[i])

    def test_partition_for_AB_training(self, jax_key):
        """Test partition_for_AB_training freezes W, trains A/B."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        diff_model, static_model = model.partition_for_AB_training()

        # A and B should be in diff_model (trainable)
        assert diff_model.A is not None
        assert diff_model.B is not None

        # Layer weights should be in static_model (frozen)
        for layer in diff_model.layers:
            assert layer.weight is None
            assert layer.bias is None

    def test_partition_for_standard_training(self, jax_key):
        """Test partition_for_standard_training freezes A/B, trains W."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        params, static = model.partition_for_standard_training()

        # A and B should be in static (frozen)
        assert static.A is not None
        assert static.B is not None

        # A and B should be None in params
        assert params.A is None
        assert params.B is None

        # Layer weights should be in params (trainable)
        for layer in params.layers:
            assert layer.weight is not None
            assert layer.bias is not None

    def test_with_new_AB_matrices(self, jax_key):
        """Test with_new_AB_matrices initializes A/B for arch transition."""
        original_arch = [3, 10, 10, 2]
        new_arch = [3, 15, 20, 2]

        model = MLP(sizes=original_arch, key=jax_key, awb_enabled=True)
        model_new = model.with_new_AB_matrices(original_arch, new_arch, seed=5)

        # Check sizes updated
        assert model_new.sizes == new_arch

        # Check A matrix shapes: A[i] should be (new_out, old_out)
        assert model_new.A[0].shape == (15, 10)  # Layer 0: 10 -> 15
        assert model_new.A[1].shape == (20, 10)  # Layer 1: 10 -> 20
        assert model_new.A[2].shape == (2, 2)    # Layer 2: 2 -> 2

        # Check B matrix shapes: B[i] should be (new_in, old_in)
        assert model_new.B[0].shape == (3, 3)    # Layer 0: 3 -> 3
        assert model_new.B[1].shape == (15, 10)  # Layer 1: 10 -> 15
        assert model_new.B[2].shape == (20, 10)  # Layer 2: 10 -> 20

    def test_awb_workflow_integration(self, jax_key):
        """Test complete AWB workflow using new interface."""
        # Step 1: Create model with AWB
        original_arch = [3, 10, 10, 2]
        model = MLP(sizes=original_arch, key=jax_key, awb_enabled=True)

        # Step 2: Simulate architecture change
        new_arch = [3, 15, 15, 2]
        model = model.with_new_AB_matrices(original_arch, new_arch)

        # Step 3: Partition for AB training
        diff_model, static_model = model.partition_for_AB_training()
        model = eqx.combine(diff_model, static_model)

        # Step 4: Apply V transformation
        model = model.apply_V_transformation()

        # Step 5: Partition for standard training
        params, static = model.partition_for_standard_training()
        model = eqx.combine(params, static)

        # Verify model still works
        x = jnp.ones((3,))
        output = model(x)
        assert output.shape == (1, 2)  # Output size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
