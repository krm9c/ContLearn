"""
AWB Baseline Correctness Tests for JIT Optimization.

These tests verify that the AWB forward pass (get_AWBT/getAWB) produces correct
outputs and gradients for all model types. They MUST pass before and after
any optimization changes.

Run with: pytest tests/test_awb_baseline.py -v

This file tests:
1. AWB forward pass produces valid (non-NaN, correct shape) output
2. Gradients flow correctly through get_AWBT/getAWB
3. Full training step with AWB works correctly
4. Output consistency between standard forward and AWB forward (when A=I, B=I)

Models tested: MLP (Sine), CNN (MNIST), CNN3D (CIFAR), GCN (Synthetic Graph)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pytest

from cl.models import MLP, CNN, CNN3D, GCN
from cl.models.layers import Linear2

# Mark all tests as unit tests for fast CI runs
pytestmark = pytest.mark.unit


# =============================================================================
# MLP (Sine Regression) AWB Baseline Tests
# =============================================================================

class TestMLPAWBBaseline:
    """Baseline tests for MLP AWB - MUST PASS before and after optimization."""

    @pytest.fixture
    def mlp_model(self, jax_key):
        """Create MLP with properly initialized AWB matrices.

        Note: MLP AWB matrices are placeholders by default. We must call
        with_new_AB_matrices() to properly initialize them before using getAWB().
        """
        original_sizes = [3, 32, 32, 10]
        new_sizes = [3, 40, 40, 10]  # Expanded architecture for AWB
        model = MLP(sizes=original_sizes, key=jax_key, awb_enabled=True)
        # Properly initialize A/B matrices for architecture transition
        model = model.with_new_AB_matrices(original_sizes, new_sizes, seed=42)
        return model

    @pytest.fixture
    def mlp_input(self):
        """Sample input for MLP (sine regression input)."""
        return jnp.array(np.random.randn(3).astype(np.float32))

    def test_mlp_awb_forward_valid_output(self, mlp_model, mlp_input):
        """Verify MLP getAWB produces valid (non-NaN, finite) output."""
        output = mlp_model.getAWB(mlp_input)

        assert output is not None, "Output should not be None"
        assert not jnp.isnan(output).any(), "Output contains NaN"
        assert jnp.isfinite(output).all(), "Output contains Inf"

    def test_mlp_awb_forward_shape(self, mlp_model, mlp_input):
        """Verify MLP getAWB produces correct output shape."""
        output = mlp_model.getAWB(mlp_input)

        # MLP AWB output: new_sizes[-1] (expanded architecture)
        # After with_new_AB_matrices, sizes is updated to new_sizes
        expected_shape = (mlp_model.sizes[-1],)  # Should be 10
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_mlp_awb_gradient_flow(self, mlp_model, mlp_input, jax_key):
        """Verify gradients flow through MLP getAWB correctly."""
        target = jax.random.normal(jax_key, (mlp_model.sizes[-1],))

        def loss_fn(model):
            pred = model.getAWB(mlp_input)
            return jnp.mean((pred - target) ** 2)

        # Use eqx.filter_grad to handle non-array fields (like sizes which is int)
        grad = eqx.filter_grad(loss_fn)(mlp_model)

        # Check A and B gradients exist and are finite
        assert grad.A is not None, "A gradient should not be None"
        assert grad.B is not None, "B gradient should not be None"

        for i, (grad_a, grad_b) in enumerate(zip(grad.A, grad.B)):
            assert jnp.isfinite(grad_a).all(), f"A[{i}] gradient contains Inf/NaN"
            assert jnp.isfinite(grad_b).all(), f"B[{i}] gradient contains Inf/NaN"

    def test_mlp_awb_training_step(self, mlp_model, mlp_input, jax_key):
        """Verify full training step with MLP AWB works."""
        target = jax.random.normal(jax_key, (mlp_model.sizes[-1],))

        # Partition for AB training (freeze W, train A/B)
        diff_model, static_model = mlp_model.partition_for_AB_training()

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(diff_model)

        def loss_fn(diff, static, x, y):
            model = eqx.combine(diff, static)
            pred = model.getAWB(x)
            return jnp.mean((pred - y) ** 2)

        # Training step
        loss_before = loss_fn(diff_model, static_model, mlp_input, target)
        grad = jax.grad(loss_fn)(diff_model, static_model, mlp_input, target)
        updates, opt_state = optimizer.update(grad, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        loss_after = loss_fn(diff_model, static_model, mlp_input, target)

        # Verify training made progress (loss changed)
        assert jnp.isfinite(loss_before), "Loss before is not finite"
        assert jnp.isfinite(loss_after), "Loss after is not finite"
        assert loss_before != loss_after, "Loss should change after training step"


# =============================================================================
# CNN (MNIST) AWB Baseline Tests
# =============================================================================

class TestCNNAWBBaseline:
    """Baseline tests for CNN AWB - MUST PASS before and after optimization.

    Note: CNN has a complex AWB architecture where the conv filter dimensions
    change via A @ W @ B.T transformation. This test uses the default CNN
    architecture from the original codebase which was designed to work together.
    """

    @pytest.fixture
    def cnn_model(self, jax_key):
        """Create CNN with original MNIST config matching DEFAULT_AWB_CNN_ARCH.

        Original architecture designed for MNIST with filter_size=4, channel_out=3:
        - Conv output: (28 - 4 + 1) = 25
        - After pool(2): 25 // 2 = 12
        - Flattened: 3 * 12 * 12 = 432

        AWB conv transforms 4x4 -> 5x5 filter:
        - AWB conv output: (28 - 5 + 1) = 24
        - After pool(2): 24 // 2 = 12
        - Flattened: 3 * 12 * 12 = 432 (same!)

        But when filter_size=4 and default awb_filter_size=6 (4+2):
        - AWB conv output: (28 - 6 + 1) = 23
        - After pool(2): (23 - 2) // 2 + 1 = 11
        - Flattened: 3 * 11 * 11 = 363 (DIFFERENT from 432!)

        To avoid this, we set awb_filter_size=filter_size so AWB conv uses same filter size.
        """
        feed_sizes = [432, 64, 10]
        awb_arch = [432, 64, 10]  # Same as feed_sizes for identity transform
        return CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=feed_sizes,
            input_size=28,
            channel_in=1,
            channel_out=3,
            awb_arch=awb_arch,
            awb_filter_size=4  # CRITICAL: Same as filter_size to avoid dimension mismatch
        )

    @pytest.fixture
    def cnn_input(self, jax_key):
        """Sample MNIST-like input."""
        # Use JAX random to ensure consistent dtype with model
        return jax.random.normal(jax_key, (1, 28, 28))

    def test_cnn_awb_forward_valid_output(self, cnn_model, cnn_input):
        """Verify CNN get_AWBT produces valid output."""
        output = cnn_model.get_AWBT(cnn_input)

        assert output is not None, "Output should not be None"
        assert not jnp.isnan(output).any(), "Output contains NaN"
        assert jnp.isfinite(output).all(), "Output contains Inf"

    def test_cnn_awb_forward_shape(self, cnn_model, cnn_input):
        """Verify CNN get_AWBT produces correct output shape."""
        output = cnn_model.get_AWBT(cnn_input)

        # CNN output: num_classes (10 for MNIST)
        expected_shape = (10,)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_cnn_awb_gradient_flow(self, cnn_model, cnn_input, jax_key):
        """Verify gradients flow through CNN get_AWBT correctly."""
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(model):
            logits = model.get_AWBT(cnn_input)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target)

        # Use eqx.filter_grad to handle non-array fields
        grad = eqx.filter_grad(loss_fn)(cnn_model)

        # Check conv AWB gradients
        assert grad.A_conv is not None, "A_conv gradient should not be None"
        assert grad.B_conv is not None, "B_conv gradient should not be None"
        for i, (grad_a, grad_b) in enumerate(zip(grad.A_conv, grad.B_conv)):
            assert jnp.isfinite(grad_a).all(), f"A_conv[{i}] gradient contains Inf/NaN"
            assert jnp.isfinite(grad_b).all(), f"B_conv[{i}] gradient contains Inf/NaN"

        # Check feed AWB gradients
        assert grad.A_feed is not None, "A_feed gradient should not be None"
        assert grad.B_feed is not None, "B_feed gradient should not be None"
        for i, (grad_a, grad_b) in enumerate(zip(grad.A_feed, grad.B_feed)):
            assert jnp.isfinite(grad_a).all(), f"A_feed[{i}] gradient contains Inf/NaN"
            assert jnp.isfinite(grad_b).all(), f"B_feed[{i}] gradient contains Inf/NaN"

    def test_cnn_awb_training_step(self, cnn_model, cnn_input, jax_key):
        """Verify full training step with CNN AWB works."""
        target = jax.random.randint(jax_key, (), 0, 10)

        diff_model, static_model = cnn_model.partition_for_AB_training()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(diff_model)

        def loss_fn(diff, static, x, y):
            model = eqx.combine(diff, static)
            logits = model.get_AWBT(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y)

        loss_before = loss_fn(diff_model, static_model, cnn_input, target)
        grad = jax.grad(loss_fn)(diff_model, static_model, cnn_input, target)
        updates, opt_state = optimizer.update(grad, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        loss_after = loss_fn(diff_model, static_model, cnn_input, target)

        assert jnp.isfinite(loss_before), "Loss before is not finite"
        assert jnp.isfinite(loss_after), "Loss after is not finite"
        assert loss_before != loss_after, "Loss should change after training step"

    def test_cnn_awb_batched_gradient(self, cnn_model, jax_key):
        """Verify batched gradient computation for CNN AWB."""
        batch_size = 8
        x_batch = jax.random.normal(jax_key, (batch_size, 1, 28, 28))
        y_batch = jax.random.randint(jax_key, (batch_size,), 0, 10)

        def batch_loss_fn(model):
            logits = jax.vmap(model.get_AWBT)(x_batch)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_batch))

        # Use eqx.filter_grad to handle non-array fields
        grad = eqx.filter_grad(batch_loss_fn)(cnn_model)

        # Verify all gradients are finite
        for i in range(len(cnn_model.A_conv)):
            assert jnp.isfinite(grad.A_conv[i]).all(), f"Batched A_conv[{i}] grad has Inf/NaN"
            assert jnp.isfinite(grad.B_conv[i]).all(), f"Batched B_conv[{i}] grad has Inf/NaN"


# =============================================================================
# CNN3D (CIFAR-10/100) AWB Baseline Tests
# =============================================================================

class TestCNN3DAWBBaseline:
    """Baseline tests for CNN3D AWB - MUST PASS before and after optimization."""

    @pytest.fixture
    def cnn3d_model(self, jax_key):
        """Create CNN3D with CIFAR-10 config."""
        return CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=[2304, 256, 10],  # 32 * 2 * 6 * 6 = 2304
            input_size=32,
            channel_in=3,
            channel_out=32
        )

    @pytest.fixture
    def cnn3d_input(self, jax_key):
        """Sample CIFAR-like input."""
        return jax.random.normal(jax_key, (3, 32, 32))

    def test_cnn3d_awb_forward_valid_output(self, cnn3d_model, cnn3d_input):
        """Verify CNN3D get_AWBT produces valid output."""
        output = cnn3d_model.get_AWBT(cnn3d_input)

        assert output is not None, "Output should not be None"
        assert not jnp.isnan(output).any(), "Output contains NaN"
        assert jnp.isfinite(output).all(), "Output contains Inf"

    def test_cnn3d_awb_forward_shape(self, cnn3d_model, cnn3d_input):
        """Verify CNN3D get_AWBT produces correct output shape."""
        output = cnn3d_model.get_AWBT(cnn3d_input)

        expected_shape = (10,)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_cnn3d_awb_gradient_flow(self, cnn3d_model, cnn3d_input, jax_key):
        """Verify gradients flow through CNN3D get_AWBT correctly."""
        target = jax.random.randint(jax_key, (), 0, 10)

        def loss_fn(model):
            logits = model.get_AWBT(cnn3d_input)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target)

        # Use eqx.filter_grad to handle non-array fields
        grad = eqx.filter_grad(loss_fn)(cnn3d_model)

        # Check conv1 AWB gradients (nested structure)
        assert grad.A_conv1 is not None, "A_conv1 gradient should not be None"
        for i in range(len(grad.A_conv1)):
            for c in range(len(grad.A_conv1[i])):
                assert jnp.isfinite(grad.A_conv1[i][c]).all(), f"A_conv1[{i}][{c}] gradient has Inf/NaN"
                assert jnp.isfinite(grad.B_conv1[i][c]).all(), f"B_conv1[{i}][{c}] gradient has Inf/NaN"

        # Check conv2 AWB gradients
        assert grad.A_conv2 is not None, "A_conv2 gradient should not be None"
        for i in range(len(grad.A_conv2)):
            for c in range(len(grad.A_conv2[i])):
                assert jnp.isfinite(grad.A_conv2[i][c]).all(), f"A_conv2[{i}][{c}] gradient has Inf/NaN"
                assert jnp.isfinite(grad.B_conv2[i][c]).all(), f"B_conv2[{i}][{c}] gradient has Inf/NaN"

        # Check feed AWB gradients
        for i in range(len(grad.A_feed)):
            assert jnp.isfinite(grad.A_feed[i]).all(), f"A_feed[{i}] gradient has Inf/NaN"
            assert jnp.isfinite(grad.B_feed[i]).all(), f"B_feed[{i}] gradient has Inf/NaN"

    def test_cnn3d_awb_training_step(self, cnn3d_model, cnn3d_input, jax_key):
        """Verify full training step with CNN3D AWB works."""
        target = jax.random.randint(jax_key, (), 0, 10)

        diff_model, static_model = cnn3d_model.partition_for_AB_training()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(diff_model)

        def loss_fn(diff, static, x, y):
            model = eqx.combine(diff, static)
            logits = model.get_AWBT(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y)

        loss_before = loss_fn(diff_model, static_model, cnn3d_input, target)
        grad = jax.grad(loss_fn)(diff_model, static_model, cnn3d_input, target)
        updates, opt_state = optimizer.update(grad, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        loss_after = loss_fn(diff_model, static_model, cnn3d_input, target)

        assert jnp.isfinite(loss_before), "Loss before is not finite"
        assert jnp.isfinite(loss_after), "Loss after is not finite"
        assert loss_before != loss_after, "Loss should change after training step"

    def test_cnn3d_cifar100_output_shape(self, jax_key):
        """Verify CNN3D works with CIFAR-100 (100 classes)."""
        feed_sizes = [2304, 256, 100]
        # CNN3D AWB uses num_classes parameter for output layer dimension
        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=feed_sizes,
            input_size=32,
            channel_in=3,
            channel_out=32,
            num_classes=100  # Must specify this for AWB to use correct output dim
        )

        x = jax.random.normal(jax_key, (3, 32, 32))
        output = model.get_AWBT(x)

        assert output.shape == (100,), f"Expected (100,), got {output.shape}"
        assert jnp.isfinite(output).all(), "Output contains Inf/NaN"


# =============================================================================
# GCN (Synthetic Graph) AWB Baseline Tests
# =============================================================================

class TestGCNAWBBaseline:
    """Baseline tests for GCN AWB - MUST PASS before and after optimization."""

    @pytest.fixture
    def gcn_model(self):
        """Create GCN with synthetic graph config.

        Note: GCN doesn't use a key parameter directly - it uses SEED in config.
        """
        return GCN(
            in_size=5,
            gcn_sizes=[5, 64],
            feed_sizes=[64, 32, 10],
            node_num=10,
            out_size=10,
            SEED=42
        )

    @pytest.fixture
    def graph_inputs(self, jax_key):
        """Sample graph inputs."""
        num_nodes = 10
        num_features = 5

        # Node features
        x = jax.random.normal(jax_key, (num_nodes, num_features))

        # Adjacency matrix (sparse random graph, normalized)
        adj = jax.random.uniform(jax_key, (num_nodes, num_nodes))
        adj = (adj > 0.7).astype(jnp.float32)
        adj = adj + adj.T  # Make symmetric
        adj = jnp.clip(adj, 0, 1)  # Binary adjacency
        # Add self-loops and normalize (simplified GCN norm)
        adj = adj + jnp.eye(num_nodes)
        deg = jnp.sum(adj, axis=1, keepdims=True)
        adj = adj / jnp.sqrt(deg) / jnp.sqrt(deg.T)

        # Batch assignment (single graph)
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)

        # Number of nodes per graph
        n_nodes = jnp.array([num_nodes])

        return x, adj, batch, n_nodes

    def test_gcn_awb_forward_valid_output(self, gcn_model, graph_inputs):
        """Verify GCN get_AWBT produces valid output."""
        x, adj, batch, n_nodes = graph_inputs
        output = gcn_model.get_AWBT(x, adj, batch, n_nodes)

        assert output is not None, "Output should not be None"
        assert not jnp.isnan(output).any(), "Output contains NaN"
        assert jnp.isfinite(output).all(), "Output contains Inf"

    def test_gcn_awb_forward_shape(self, gcn_model, graph_inputs):
        """Verify GCN get_AWBT produces correct output shape."""
        x, adj, batch, n_nodes = graph_inputs
        output = gcn_model.get_AWBT(x, adj, batch, n_nodes)

        # GCN output: (batch_size, num_classes) -> (1, 10)
        expected_shape = (1, 10)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_gcn_awb_gradient_flow(self, gcn_model, graph_inputs, jax_key):
        """Verify gradients flow through GCN get_AWBT correctly."""
        x, adj, batch, n_nodes = graph_inputs
        target = jax.random.randint(jax_key, (1,), 0, 10)

        def loss_fn(model):
            logits = model.get_AWBT(x, adj, batch, n_nodes)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()

        # Use eqx.filter_grad to handle non-array fields
        grad = eqx.filter_grad(loss_fn)(gcn_model)

        # Check GCN AWB gradients
        assert grad.A_gcn is not None, "A_gcn gradient should not be None"
        assert grad.B_gcn is not None, "B_gcn gradient should not be None"
        for i in range(len(grad.A_gcn)):
            assert jnp.isfinite(grad.A_gcn[i]).all(), f"A_gcn[{i}] gradient has Inf/NaN"
            assert jnp.isfinite(grad.B_gcn[i]).all(), f"B_gcn[{i}] gradient has Inf/NaN"

        # Check feed AWB gradients
        assert grad.A_feed is not None, "A_feed gradient should not be None"
        assert grad.B_feed is not None, "B_feed gradient should not be None"
        for i in range(len(grad.A_feed)):
            assert jnp.isfinite(grad.A_feed[i]).all(), f"A_feed[{i}] gradient has Inf/NaN"
            assert jnp.isfinite(grad.B_feed[i]).all(), f"B_feed[{i}] gradient has Inf/NaN"

    def test_gcn_awb_training_step(self, gcn_model, graph_inputs, jax_key):
        """Verify full training step with GCN AWB works."""
        x, adj, batch, n_nodes = graph_inputs
        target = jax.random.randint(jax_key, (1,), 0, 10)

        diff_model, static_model = gcn_model.partition_for_AB_training()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(diff_model)

        def loss_fn(diff, static):
            model = eqx.combine(diff, static)
            logits = model.get_AWBT(x, adj, batch, n_nodes)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()

        loss_before = loss_fn(diff_model, static_model)
        grad = jax.grad(loss_fn)(diff_model, static_model)
        updates, opt_state = optimizer.update(grad, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        loss_after = loss_fn(diff_model, static_model)

        assert jnp.isfinite(loss_before), "Loss before is not finite"
        assert jnp.isfinite(loss_after), "Loss after is not finite"
        assert loss_before != loss_after, "Loss should change after training step"


# =============================================================================
# Cross-Model Consistency Tests
# =============================================================================

class TestAWBConsistency:
    """Tests to verify AWB output consistency with identity transformations."""

    def test_mlp_identity_awb_finite_output(self, jax_key):
        """MLP: With identity A, B, AWB forward should produce finite output.

        Note: MLP getAWB applies tanh to ALL layers (including output),
        while standard forward only applies tanh to hidden layers.
        So outputs won't match exactly, but both should be finite.
        """
        sizes = [3, 16, 16, 5]
        model = MLP(sizes=sizes, key=jax_key, awb_enabled=True)

        # Set A and B to identity matrices of correct shapes
        # For AWB: V = A @ W @ B.T, so if A=I, B=I, then V=W
        A_identity = [
            jnp.eye(sizes[i+1], sizes[i+1]) for i in range(len(sizes) - 1)
        ]
        B_identity = [
            jnp.eye(sizes[i], sizes[i]) for i in range(len(sizes) - 1)
        ]

        model = eqx.tree_at(lambda x: x.A, model, A_identity)
        model = eqx.tree_at(lambda x: x.B, model, B_identity)
        # Update sizes to reflect "identity" arch change (same dimensions)
        model = eqx.tree_at(lambda x: x.sizes, model, sizes)

        x = jax.random.normal(jax_key, (3,))

        # Standard forward
        standard_out = model(x)

        # AWB forward with identity
        awb_out = model.getAWB(x)

        assert jnp.isfinite(standard_out).all(), "Standard forward has Inf/NaN"
        assert jnp.isfinite(awb_out).all(), "AWB forward has Inf/NaN"
        # Outputs won't match exactly due to activation difference on last layer,
        # but shapes should match
        assert standard_out.flatten().shape == awb_out.shape, \
            f"Output shapes should match: {standard_out.flatten().shape} vs {awb_out.shape}"

    def test_cnn_both_forwards_finite(self, jax_key):
        """CNN: Both standard and AWB forward should produce finite outputs.

        Note: CNN AWB has a dimension constraint - awb_arch[0] must equal the
        flatten size produced by the AWB conv layer. When awb_filter_size=filter_size,
        the flatten size equals feed_sizes[0]. Using awb_arch != feed_sizes with
        mismatched flatten size causes dimension errors.
        """
        feed_sizes = [432, 64, 10]
        # Must use identity awb_arch when awb_filter_size=filter_size
        awb_arch = [432, 64, 10]
        model = CNN(
            key=jax_key,
            filter_size=4,
            feed_sizes=feed_sizes,
            input_size=28,
            channel_in=1,
            channel_out=3,
            awb_arch=awb_arch,
            awb_filter_size=4  # Same as filter_size, so flatten=432
        )

        x = jax.random.normal(jax_key, (1, 28, 28))

        standard_out = model(x)
        awb_out = model.get_AWBT(x)

        assert jnp.isfinite(standard_out).all(), "Standard forward has Inf/NaN"
        assert jnp.isfinite(awb_out).all(), "AWB forward has Inf/NaN"
        assert standard_out.shape == awb_out.shape, "Output shapes should match"

    def test_cnn3d_both_forwards_finite(self, jax_key):
        """CNN3D: Both standard and AWB forward should produce finite outputs."""
        model = CNN3D(
            key=jax_key,
            filter_size=3,
            feed_sizes=[2304, 256, 10],
            input_size=32,
            channel_in=3,
            channel_out=32
        )

        x = jax.random.normal(jax_key, (3, 32, 32))

        standard_out = model(x)
        awb_out = model.get_AWBT(x)

        assert jnp.isfinite(standard_out).all(), "Standard forward has Inf/NaN"
        assert jnp.isfinite(awb_out).all(), "AWB forward has Inf/NaN"
        assert standard_out.shape == awb_out.shape, "Output shapes should match"

    def test_gcn_both_forwards_finite(self, jax_key):
        """GCN: Both standard and AWB forward should produce finite outputs."""
        model = GCN(
            in_size=5,
            gcn_sizes=[5, 64],
            feed_sizes=[64, 32, 10],
            node_num=10,
            out_size=10,
            SEED=42
        )

        num_nodes = 10
        x = jax.random.normal(jax_key, (num_nodes, 5))
        adj = jnp.eye(num_nodes)  # Simple identity adjacency
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        n_nodes = jnp.array([num_nodes])

        standard_out = model(x, adj, batch, n_nodes)
        awb_out = model.get_AWBT(x, adj, batch, n_nodes)

        assert jnp.isfinite(standard_out).all(), "Standard forward has Inf/NaN"
        assert jnp.isfinite(awb_out).all(), "AWB forward has Inf/NaN"
        assert standard_out.shape == awb_out.shape, "Output shapes should match"


# =============================================================================
# Partition Function Tests
# =============================================================================

class TestPartitionFunctions:
    """Tests for model partition functions used in AWB training."""

    def test_mlp_partition_preserves_structure(self, jax_key):
        """MLP partition/combine preserves model structure."""
        original_sizes = [3, 16, 10]
        new_sizes = [3, 20, 10]
        model = MLP(sizes=original_sizes, key=jax_key, awb_enabled=True)
        # Properly initialize A/B matrices
        model = model.with_new_AB_matrices(original_sizes, new_sizes, seed=42)
        x = jax.random.normal(jax_key, (3,))

        # Test AB training partition
        diff, static = model.partition_for_AB_training()
        recombined = eqx.combine(diff, static)
        assert jnp.allclose(model.getAWB(x), recombined.getAWB(x))

        # Test standard training partition
        params, static = model.partition_for_standard_training()
        recombined = eqx.combine(params, static)
        assert jnp.allclose(model(x), recombined(x))

    def test_cnn_partition_preserves_structure(self, jax_key):
        """CNN partition/combine preserves model structure."""
        feed_sizes = [432, 64, 10]
        # Must use identity awb_arch when awb_filter_size=filter_size
        awb_arch = [432, 64, 10]
        model = CNN(key=jax_key, filter_size=4, feed_sizes=feed_sizes,
                   input_size=28, channel_in=1, channel_out=3, awb_arch=awb_arch,
                   awb_filter_size=4)  # Same as filter_size, so flatten=432
        x = jax.random.normal(jax_key, (1, 28, 28))

        diff, static = model.partition_for_AB_training()
        recombined = eqx.combine(diff, static)
        assert jnp.allclose(model.get_AWBT(x), recombined.get_AWBT(x))

        params, static = model.partition_for_standard_training()
        recombined = eqx.combine(params, static)
        assert jnp.allclose(model(x), recombined(x))

    def test_cnn3d_partition_preserves_structure(self, jax_key):
        """CNN3D partition/combine preserves model structure."""
        model = CNN3D(key=jax_key, filter_size=3, feed_sizes=[2304, 256, 10],
                     input_size=32, channel_in=3, channel_out=32)
        x = jax.random.normal(jax_key, (3, 32, 32))

        diff, static = model.partition_for_AB_training()
        recombined = eqx.combine(diff, static)
        assert jnp.allclose(model.get_AWBT(x), recombined.get_AWBT(x))

        params, static = model.partition_for_standard_training()
        recombined = eqx.combine(params, static)
        assert jnp.allclose(model(x), recombined(x))

    def test_gcn_partition_preserves_structure(self, jax_key):
        """GCN partition/combine preserves model structure."""
        model = GCN(in_size=5, gcn_sizes=[5, 64],
                   feed_sizes=[64, 32, 10], node_num=10, out_size=10, SEED=42)

        num_nodes = 10
        x = jax.random.normal(jax_key, (num_nodes, 5))
        adj = jnp.eye(num_nodes)
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        n_nodes = jnp.array([num_nodes])

        diff, static = model.partition_for_AB_training()
        recombined = eqx.combine(diff, static)
        assert jnp.allclose(model.get_AWBT(x, adj, batch, n_nodes),
                          recombined.get_AWBT(x, adj, batch, n_nodes))

        params, static = model.partition_for_standard_training()
        recombined = eqx.combine(params, static)
        assert jnp.allclose(model(x, adj, batch, n_nodes),
                          recombined(x, adj, batch, n_nodes))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
