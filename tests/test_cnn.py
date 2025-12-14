"""
Unit tests for CNN models in models/cnn.py.
Tests CNN and CNN3D classes with AWB support.
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

from cl.models import CNN, CNN3D, CNNorig, Linear2
from cl.core.awb import (
    partition_for_AB_training_cnn,
    partition_for_standard_training_cnn,
    partition_for_AB_training_cnn3d,
    partition_for_standard_training_cnn3d,
)
from cl.arch_search.cnn_search import prepABs, prepABs_CNN3D


class TestLinear2:
    """Tests for Linear2 layer used by CNN."""

    def test_linear2_initialization(self, jax_key):
        """Test Linear2 initializes correctly."""
        layer = Linear2(10, 20, key=jax_key)

        assert layer.weight.shape == (20, 10)
        assert layer.bias.shape == (20, 1)

    def test_linear2_forward(self, jax_key):
        """Test Linear2 forward pass."""
        layer = Linear2(10, 20, key=jax_key)
        x = jnp.ones(10)

        output = layer(x)

        assert output.shape == (20,)


class TestCNN:
    """Tests for CNN class."""

    def test_cnn_initialization(self, jax_key, cnn_config):
        """Test CNN initializes correctly."""
        filter_size = cnn_config['filter_size']
        feed_sizes = cnn_config['feed_sizes']

        model = CNN(
            key=jax_key,
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        assert len(model.conv_layers) == 1
        assert len(model.feed_layers) == len(feed_sizes) - 1
        assert model.filter_size == filter_size
        assert model.channel_out == 3

    def test_cnn_forward_pass(self, jax_key, cnn_config):
        """Test CNN forward pass produces correct output shape."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        # MNIST input: (1, 28, 28)
        x = jnp.ones((1, 28, 28))
        output = model(x)

        assert output.shape == (10,)  # 10 classes

    def test_cnn_batched_forward(self, jax_key, cnn_config):
        """Test CNN with batched input via vmap."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        # Batch of MNIST inputs: (batch, 1, 28, 28)
        x = jnp.ones((8, 1, 28, 28))
        output = jax.vmap(model)(x)

        assert output.shape == (8, 10)  # (batch, classes)

    def test_cnn_awb_matrices(self, jax_key, cnn_config):
        """Test CNN has AWB matrices initialized."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        # AWB matrices should exist
        assert model.A_conv is not None
        assert model.B_conv is not None
        assert model.A_feed is not None
        assert model.B_feed is not None

        # Check shapes
        assert len(model.A_conv) == model.channel_out
        assert len(model.B_conv) == model.channel_out

    def test_cnn_calc_output_size(self, jax_key, cnn_config):
        """Test CNN output size calculation."""
        model = CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        # 28 - 4 + 1 = 25 (with padding=0, stride=1)
        conv_out = model.calc_output_size(4)
        assert conv_out == 25

        # After pool(2): 25 // 2 = 12
        pool_out = model.pool_output_size(2, conv_out)
        assert pool_out == 12


class TestCNN3D:
    """Tests for CNN3D class (3-channel images like CIFAR)."""

    def test_cnn3d_initialization(self, jax_key):
        """Test CNN3D initializes correctly."""
        # CIFAR config: 3-channel 32x32 images
        feed_sizes = [2304, 256, 10]

        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32
        )

        assert len(model.conv_layers) == 2  # CNN3D has 2 conv layers
        assert model.channel_in == 3
        assert model.channel_out == 32

    def test_cnn3d_forward_pass(self, jax_key):
        """Test CNN3D forward pass produces correct output shape."""
        feed_sizes = [2304, 256, 10]

        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32
        )

        # CIFAR input: (3, 32, 32)
        x = jnp.ones((3, 32, 32))
        output = model(x)

        assert output.shape == (10,)  # 10 classes

    def test_cnn3d_awb_matrices(self, jax_key):
        """Test CNN3D has multi-layer AWB matrices."""
        feed_sizes = [2304, 256, 10]

        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32
        )

        # CNN3D has separate AWB matrices for each conv layer
        assert model.A_conv1 is not None
        assert model.B_conv1 is not None
        assert model.A_conv2 is not None
        assert model.B_conv2 is not None


class TestCNNPartitioning:
    """Tests for CNN parameter partitioning for AWB training."""

    def test_cnn_partition(self, jax_key, cnn_config):
        """Test CNN can be partitioned for training."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        params, static = eqx.partition(model, eqx.is_array)

        # After partitioning, params should have arrays, static should have non-arrays
        assert params is not None
        assert static is not None

    def test_cnn_recombine(self, jax_key, cnn_config):
        """Test CNN can be recombined after partitioning."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        params, static = eqx.partition(model, eqx.is_array)
        model_recombined = eqx.combine(params, static)

        # Test that recombined model produces same output
        x = jnp.ones((1, 28, 28))
        out1 = model(x)
        out2 = model_recombined(x)

        assert jnp.allclose(out1, out2)

    def test_cnn_freeze_awb(self, jax_key, cnn_config):
        """Test freezing AWB matrices in CNN."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        params, static = eqx.partition(model, eqx.is_array)

        # Move AWB to static (frozen)
        static = eqx.tree_at(
            lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
            static,
            replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed)
        )
        params = eqx.tree_at(
            lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
            params,
            replace=(None, None, None, None)
        )

        # AWB should be in static, not params
        assert params.A_conv is None
        assert static.A_conv is not None


class TestCNN3DExtended:
    """Extended tests for CNN3D model."""

    def test_cnn3d_batched_forward(self, jax_key, cnn3d_config):
        """Test CNN3D with batched input via vmap."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        # Batch of CIFAR inputs: (batch, 3, 32, 32)
        x = jnp.ones((8, 3, 32, 32))
        output = jax.vmap(model)(x)

        assert output.shape == (8, 10)  # (batch, classes)

    def test_cnn3d_get_awbt(self, jax_key, cnn3d_config):
        """Test CNN3D get_AWBT forward pass."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        # Single CIFAR input: (3, 32, 32)
        x = jnp.ones((3, 32, 32))
        output = model.get_AWBT(x)

        assert output.shape == (10,)  # 10 classes

    def test_cnn3d_awb_matrices_shapes(self, jax_key, cnn3d_config):
        """Test CNN3D AWB matrices have correct shapes."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        channel_in = cnn3d_config['channel_in']
        channel_out = cnn3d_config['channel_out']

        # Conv1 AWB matrices: [channel_out][channel_in] each
        assert len(model.A_conv1) == channel_out
        assert len(model.A_conv1[0]) == channel_in
        assert len(model.B_conv1) == channel_out
        assert len(model.B_conv1[0]) == channel_in

        # Conv2 AWB matrices: [channel_out * 2][channel_out] each
        assert len(model.A_conv2) == channel_out * 2
        assert len(model.A_conv2[0]) == channel_out
        assert len(model.B_conv2) == channel_out * 2
        assert len(model.B_conv2[0]) == channel_out

        # Feed AWB matrices
        assert len(model.A_feed) == len(cnn3d_config['feed_sizes']) - 1
        assert len(model.B_feed) == len(cnn3d_config['feed_sizes']) - 1


class TestCNNAWBPartitioning:
    """Tests for CNN AWB partition functions."""

    def test_partition_for_AB_training_cnn(self, jax_key, cnn_config):
        """Test partitioning CNN for A/B training."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=cnn_config['input_size'],
            channel_in=cnn_config['channel_in'],
            channel_out=cnn_config['channel_out']
        )

        diff_model, static_model = partition_for_AB_training_cnn(model)

        # A/B should be trainable (in diff_model)
        assert diff_model.A_conv is not None
        assert diff_model.B_conv is not None
        assert diff_model.A_feed is not None
        assert diff_model.B_feed is not None

        # Conv/feed layers should be frozen (in static_model)
        for layer in diff_model.feed_layers:
            assert layer.weight is None
            assert layer.bias is None

    def test_partition_for_standard_training_cnn(self, jax_key, cnn_config):
        """Test partitioning CNN for standard training."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=cnn_config['input_size'],
            channel_in=cnn_config['channel_in'],
            channel_out=cnn_config['channel_out']
        )

        params, static = partition_for_standard_training_cnn(model)

        # A/B should be frozen (None in params, values in static)
        assert params.A_conv is None
        assert params.B_conv is None
        assert static.A_conv is not None
        assert static.B_conv is not None

    def test_partition_for_AB_training_cnn3d(self, jax_key, cnn3d_config):
        """Test partitioning CNN3D for A/B training."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        diff_model, static_model = partition_for_AB_training_cnn3d(model)

        # All A/B should be trainable
        assert diff_model.A_conv1 is not None
        assert diff_model.B_conv1 is not None
        assert diff_model.A_conv2 is not None
        assert diff_model.B_conv2 is not None
        assert diff_model.A_feed is not None
        assert diff_model.B_feed is not None

    def test_partition_for_standard_training_cnn3d(self, jax_key, cnn3d_config):
        """Test partitioning CNN3D for standard training."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        params, static = partition_for_standard_training_cnn3d(model)

        # All A/B should be frozen
        assert params.A_conv1 is None
        assert params.B_conv1 is None
        assert params.A_conv2 is None
        assert params.B_conv2 is None
        assert static.A_conv1 is not None
        assert static.B_conv1 is not None

    def test_cnn_partition_recombine(self, jax_key, cnn_config):
        """Test CNN can be recombined after partitioning for standard training."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=cnn_config['input_size'],
            channel_in=cnn_config['channel_in'],
            channel_out=cnn_config['channel_out']
        )

        params, static = partition_for_standard_training_cnn(model)
        recombined = eqx.combine(params, static)

        # Test forward pass works
        x = jnp.ones((1, 28, 28))
        out1 = model(x)
        out2 = recombined(x)

        assert jnp.allclose(out1, out2)

    def test_cnn3d_partition_recombine(self, jax_key, cnn3d_config):
        """Test CNN3D can be recombined after partitioning for standard training."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        params, static = partition_for_standard_training_cnn3d(model)
        recombined = eqx.combine(params, static)

        # Test forward pass works
        x = jnp.ones((3, 32, 32))
        out1 = model(x)
        out2 = recombined(x)

        assert jnp.allclose(out1, out2)


class TestPrepABs:
    """Tests for prepABs functions that prepare A/B matrices for architecture transitions."""

    def test_prepABs_new_filter_only(self, jax_key, cnn_config):
        """Test prepABs for CNN when only filter size changes."""
        model = CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=[1728, 64, 10],
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        # Same hidden layers, but filter will change
        prev_feed_sizes = [1728, 64, 10]
        prev_filter_size = 4

        # Simulate model with new filter size (which changes feed_sizes[0])
        new_filter = 5
        model = eqx.tree_at(lambda x: x.filter_size, model, new_filter)

        A_feed, B_feed, A_conv, B_conv = prepABs(model, prev_feed_sizes, prev_filter_size)

        # Conv A/B should be transformation matrices
        assert len(A_conv) == model.channel_out
        assert A_conv[0].shape == (new_filter, prev_filter_size)

        # B_feed[0] should be transformation matrix (not identity) due to flattened size change
        # The rest should be identity
        assert len(B_feed) == len(prev_feed_sizes) - 1

    def test_prepABs_new_feed_only(self, jax_key, cnn_config):
        """Test prepABs for CNN when only feed sizes change."""
        model = CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=[1728, 128, 10],  # Changed hidden layer
            input_size=28,
            channel_in=1,
            channel_out=3
        )

        prev_feed_sizes = [1728, 64, 10]
        prev_filter_size = 4

        A_feed, B_feed, A_conv, B_conv = prepABs(model, prev_feed_sizes, prev_filter_size)

        # Conv A/B should be identity (filter didn't change)
        assert jnp.allclose(A_conv[0], jnp.eye(4, 4))
        assert jnp.allclose(B_conv[0], jnp.eye(4, 4))

        # Feed A/B should be transformation matrices
        assert A_feed[0].shape == (128, 64)  # (new_out, old_out)

    def test_prepABs_CNN3D_new_filter_only(self, jax_key, cnn3d_config):
        """Test prepABs_CNN3D when only filter size changes."""
        model = CNN3D(
            key=jax_key,
            filter_size=4,  # Changed from 3
            feed_sizes=[2304, 256, 10],  # Note: should be recalculated for new filter
            input_size=32,
            channel_in=3,
            channel_out=32
        )

        prev_feed_sizes = [2304, 256, 10]
        prev_filter_size = 3

        A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2 = prepABs_CNN3D(
            model, prev_feed_sizes, prev_filter_size
        )

        # Conv matrices should be transformation matrices
        assert len(A_conv1) == model.channel_out
        assert len(A_conv1[0]) == model.channel_in
        assert A_conv1[0][0].shape == (4, 3)  # (new_filter, old_filter)

        assert len(A_conv2) == model.channel_out * 2
        assert len(A_conv2[0]) == model.channel_out


class TestCNNAWBIntegration:
    """Integration tests for CNN AWB workflow."""

    def test_cnn_awb_workflow(self, jax_key, cnn_config):
        """Test complete CNN AWB workflow: partition, train A/B, compute V, partition standard."""
        model = CNN(
            key=jax_key,
            filter_size=cnn_config['filter_size'],
            feed_sizes=cnn_config['feed_sizes'],
            input_size=cnn_config['input_size'],
            channel_in=cnn_config['channel_in'],
            channel_out=cnn_config['channel_out']
        )

        # 1. Partition for AB training
        diff_model, static_model = partition_for_AB_training_cnn(model)

        # 2. Simulate training (just modify A/B slightly)
        # In real training, optimizer would update diff_model

        # 3. Recombine
        model = eqx.combine(diff_model, static_model)

        # 4. Partition for standard training
        params, static = partition_for_standard_training_cnn(model)

        # 5. Verify model still works
        final_model = eqx.combine(params, static)
        x = jnp.ones((1, 28, 28))
        output = final_model(x)

        assert output.shape == (10,)

    def test_cnn3d_awb_workflow(self, jax_key, cnn3d_config):
        """Test complete CNN3D AWB workflow."""
        model = CNN3D(
            key=jax_key,
            filter_size=cnn3d_config['filter_size'],
            feed_sizes=cnn3d_config['feed_sizes'],
            input_size=cnn3d_config['input_size'],
            channel_in=cnn3d_config['channel_in'],
            channel_out=cnn3d_config['channel_out']
        )

        # 1. Partition for AB training
        diff_model, static_model = partition_for_AB_training_cnn3d(model)

        # 2. Recombine (simulating after AB training)
        model = eqx.combine(diff_model, static_model)

        # 3. Partition for standard training
        params, static = partition_for_standard_training_cnn3d(model)

        # 4. Verify model still works
        final_model = eqx.combine(params, static)
        x = jnp.ones((3, 32, 32))
        output = final_model(x)

        assert output.shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
