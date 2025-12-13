"""
Unit tests for graph neural network models in utils/model.py.
Tests GCN, GAT, myNN, and graph pooling operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from contlearn.models import GCN, GCNorig, SingleHeadGAT, MultiHeadGAT, myNN, myNNorig
from contlearn.models import Pool, GraphPooling, Linear, Linear2, Linear3, sp_matmul


class TestGCNLayers:
    """Tests for GCN (Graph Convolutional Network) layers."""

    def test_gcn_initialization(self):
        """Test GCN layer initializes correctly."""
        key = jax.random.PRNGKey(42)
        layer = GCN(in_size=10, out_size=64, key=key, bias=True, sparse=False)

        assert layer.weight.shape == (10, 64)
        assert layer.bias.shape == (1, 64)
        assert layer.bias_flag == True
        assert layer.sparse == False

    def test_gcn_no_bias(self):
        """Test GCN layer without bias."""
        key = jax.random.PRNGKey(42)
        layer = GCN(in_size=10, out_size=64, key=key, bias=False)

        assert layer.bias is None
        assert layer.bias_flag == False

    def test_gcn_forward_pass(self):
        """Test GCN forward pass."""
        key = jax.random.PRNGKey(42)
        layer = GCN(in_size=10, out_size=64, key=key)

        # Create dummy node features and adjacency matrix
        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes).astype(np.float32))

        output = layer(x, adj)

        assert output.shape == (num_nodes, 64)
        assert not jnp.isnan(output).any()

    def test_gcn_sparse_mode(self):
        """Test GCN initialization with sparse mode."""
        key = jax.random.PRNGKey(42)
        layer = GCN(in_size=10, out_size=64, key=key, sparse=True)

        assert layer.sparse == True

    def test_gcnorig_forward_pass(self):
        """Test GCNorig forward pass."""
        key = jax.random.PRNGKey(42)
        layer = GCNorig(in_size=10, out_size=64, key=key)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes).astype(np.float32))

        output = layer(x, adj)

        assert output.shape == (num_nodes, 64)


class TestGATLayers:
    """Tests for GAT (Graph Attention Network) layers."""

    def test_single_head_gat_initialization(self):
        """Test SingleHeadGAT initializes correctly."""
        key = jax.random.PRNGKey(42)
        layer = SingleHeadGAT(in_size=10, out_size=64, key=key)

        assert layer.weight.shape == (10, 64)
        assert layer.a1.shape == (64, 1)
        assert layer.a2.shape == (64, 1)
        assert hasattr(layer, 'dropout')

    def test_single_head_gat_forward(self):
        """Test SingleHeadGAT forward pass."""
        key = jax.random.PRNGKey(42)
        layer = SingleHeadGAT(in_size=10, out_size=64, key=key)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes) > 0.5)  # Binary adjacency

        rng_key = jax.random.PRNGKey(123)
        output = layer(x, adj, rng=rng_key, is_training=False)

        assert output.shape == (num_nodes, 64)
        assert not jnp.isnan(output).any()

    def test_multi_head_gat_initialization(self):
        """Test MultiHeadGAT initializes correctly."""
        key = jax.random.PRNGKey(42)
        n_heads = 4
        layer = MultiHeadGAT(n_heads=n_heads, in_size=10, out_size=64, key=key)

        assert layer.n_heads == n_heads
        assert len(layer.layer) == n_heads

    def test_multi_head_gat_forward_concat(self):
        """Test MultiHeadGAT forward pass with concatenation (not last layer)."""
        key = jax.random.PRNGKey(42)
        layer = MultiHeadGAT(n_heads=4, in_size=10, out_size=64, key=key, last_layer=False)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes) > 0.5)

        rng_key = jax.random.PRNGKey(123)
        output = layer(x, adj, rng=rng_key, is_training=False)

        # Concatenated output: 4 heads * 64 = 256
        assert output.shape == (num_nodes, 4 * 64)

    def test_multi_head_gat_forward_mean(self):
        """Test MultiHeadGAT forward pass with averaging (last layer)."""
        key = jax.random.PRNGKey(42)
        layer = MultiHeadGAT(n_heads=4, in_size=10, out_size=64, key=key, last_layer=True)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes) > 0.5)

        rng_key = jax.random.PRNGKey(123)
        output = layer(x, adj, rng=rng_key, is_training=False)

        # Averaged output: still 64
        assert output.shape == (num_nodes, 64)


class TestGraphPooling:
    """Tests for graph pooling operations."""

    def test_pool_sum(self):
        """Test sum pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])  # 2 graphs with 2 nodes each
        num_nodes = jnp.array([2, 2])

        output = Pool.sum(x, batch, num_nodes)

        assert output.shape == (2, 2)
        # Graph 0: [1+3, 2+4] = [4, 6]
        # Graph 1: [5+7, 6+8] = [12, 14]
        assert jnp.allclose(output[0], jnp.array([4.0, 6.0]))
        assert jnp.allclose(output[1], jnp.array([12.0, 14.0]))

    def test_pool_mean(self):
        """Test mean pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])
        num_nodes = jnp.array([2, 2])

        output = Pool.mean(x, batch, num_nodes)

        assert output.shape == (2, 2)
        # Graph 0: [4/2, 6/2] = [2, 3]
        # Graph 1: [12/2, 14/2] = [6, 7]
        assert jnp.allclose(output[0], jnp.array([2.0, 3.0]))
        assert jnp.allclose(output[1], jnp.array([6.0, 7.0]))

    def test_pool_max(self):
        """Test max pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])
        num_nodes = jnp.array([2, 2])

        output = Pool.max(x, batch, num_nodes)

        assert output.shape == (2, 2)
        # Graph 0: max([1,3], [2,4]) = [3, 4]
        # Graph 1: max([5,7], [6,8]) = [7, 8]
        assert jnp.allclose(output[0], jnp.array([3.0, 4.0]))
        assert jnp.allclose(output[1], jnp.array([7.0, 8.0]))

    def test_pool_identity(self):
        """Test identity pooling (no pooling)."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        batch = jnp.array([0, 1])
        num_nodes = jnp.array([1, 1])

        output = Pool.identity(x, batch, num_nodes)

        assert jnp.allclose(output, x)

    def test_graph_pooling_wrapper(self):
        """Test GraphPooling wrapper class."""
        pool_layer = GraphPooling(Pool.sum)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        batch = jnp.array([0, 0])
        num_nodes = jnp.array([2])

        output = pool_layer(x, batch, num_nodes)

        assert output.shape == (1, 2)
        assert jnp.allclose(output[0], jnp.array([4.0, 6.0]))


class TestMyNN:
    """Tests for myNN (combined GCN + MLP for graph classification)."""

    def test_mynn_initialization(self):
        """Test myNN initializes correctly."""
        model = myNN(in_size=10, feed_sizes=[128, 128, 128, 5],
                     gcn_sizes=[10, 128], node_num=20, out_size=5)

        assert len(model.gcn_layers) > 0
        assert len(model.feed_layers) > 0
        assert model.pool_layer is not None
        assert hasattr(model, 'A_gcn')
        assert hasattr(model, 'B_gcn')
        assert hasattr(model, 'A_feed')
        assert hasattr(model, 'B_feed')

    def test_mynn_forward_pass(self):
        """Test myNN forward pass."""
        model = myNN(in_size=10, feed_sizes=[128, 128, 128, 5],
                     gcn_sizes=[10, 128], node_num=20, out_size=5)

        # Create dummy graph data
        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes).astype(np.float32))
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)  # Single graph
        n_nodes = jnp.array([num_nodes])

        output = model(x, adj, batch, n_nodes)

        # Output should be graph-level prediction: [1, num_classes]
        assert output.shape[0] == 1  # One graph
        assert output.shape[1] == 5  # 5 classes
        assert not jnp.isnan(output).any()

    def test_mynn_awbt_forward(self):
        """Test myNN AWB transformation forward pass."""
        model = myNN(in_size=10, feed_sizes=[128, 128, 128, 5],
                     gcn_sizes=[10, 128], node_num=20, out_size=5)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes).astype(np.float32))
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        n_nodes = jnp.array([num_nodes])

        output = model.get_AWBT(x, adj, batch, n_nodes)

        assert not jnp.isnan(output).any()

    def test_mynn_multiple_graphs(self):
        """Test myNN with multiple graphs in a batch."""
        model = myNN(in_size=10, feed_sizes=[128, 128, 128, 5],
                     gcn_sizes=[10, 128], node_num=20, out_size=5)

        # Two graphs: first has 15 nodes, second has 10 nodes
        num_nodes_total = 25
        x = jnp.array(np.random.randn(num_nodes_total, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes_total, num_nodes_total).astype(np.float32))
        batch = jnp.array([0]*15 + [1]*10, dtype=jnp.int32)
        n_nodes = jnp.array([15, 10])

        output = model(x, adj, batch, n_nodes)

        # Output should be [2, num_classes] for 2 graphs
        assert output.shape == (2, 5)
        assert not jnp.isnan(output).any()

    def test_mynnorig_forward(self):
        """Test myNNorig (original version) forward pass."""
        model = myNNorig(in_size=10, hid_size=128, node_num=20, out_size=5)

        num_nodes = 20
        x = jnp.array(np.random.randn(num_nodes, 10).astype(np.float32))
        adj = jnp.array(np.random.rand(num_nodes, num_nodes).astype(np.float32))
        batch = jnp.zeros(num_nodes, dtype=jnp.int32)
        n_nodes = jnp.array([num_nodes])

        output = model(x, adj, batch, n_nodes)

        assert output.shape[0] == 1  # One graph
        assert output.shape[1] == 5  # 5 classes


class TestSparseMatmul:
    """Tests for sparse matrix multiplication utility."""

    def test_sp_matmul_basic(self):
        """Test sparse matrix multiplication."""
        # Create a simple sparse matrix: 3x3 identity
        # indexes: (rows, cols)
        rows = jnp.array([0, 1, 2])
        cols = jnp.array([0, 1, 2])
        values = jnp.array([1.0, 1.0, 1.0])
        sparse_A = ((rows, cols), values)

        # Dense matrix B: 3x2
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Result should be same as B (identity matrix)
        result = sp_matmul(sparse_A, B, 3)

        assert result.shape == (3, 2)
        assert jnp.allclose(result, B)

    def test_sp_matmul_non_identity(self):
        """Test sparse matrix multiplication with non-identity matrix."""
        # Sparse matrix with some specific pattern
        rows = jnp.array([0, 0, 1, 2])
        cols = jnp.array([0, 1, 1, 2])
        values = jnp.array([2.0, 3.0, 1.0, 4.0])
        sparse_A = ((rows, cols), values)

        B = jnp.array([[1.0], [2.0], [3.0]])

        result = sp_matmul(sparse_A, B, 3)

        assert result.shape == (3, 1)
        # Row 0: 2*1 + 3*2 = 8
        # Row 1: 1*2 = 2
        # Row 2: 4*3 = 12
        expected = jnp.array([[8.0], [2.0], [12.0]])
        assert jnp.allclose(result, expected)


class TestLinearVariants:
    """Tests for Linear layer variants used in graph models."""

    def test_linear_graph_usage(self):
        """Test Linear layer as used in graph models."""
        key = jax.random.PRNGKey(42)
        layer = Linear(in_size=128, out_size=64, key=key)

        # After pooling, graph features are typically [num_graphs, features]
        x = jnp.array(np.random.randn(5, 128).astype(np.float32))
        output = layer(x)

        assert output.shape == (5, 64)

    def test_linear2_bias_shape(self):
        """Test Linear2 has correct bias shape for graph operations."""
        key = jax.random.PRNGKey(42)
        layer = Linear2(in_size=128, out_size=64, key=key)

        assert layer.bias.shape == (64, 1)

    def test_linear3_bias_shape(self):
        """Test Linear3 has correct bias shape."""
        key = jax.random.PRNGKey(42)
        layer = Linear3(in_size=128, out_size=64, key=key)

        assert layer.bias.shape == (1, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
