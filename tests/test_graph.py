"""
Unit tests for graph neural network models and graph classification pipeline.
Tests GCN model with AWB transformations, graph datasets, and graph classification runner.

Added by Claude: Comprehensive graph model testing.
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

from cl.models.gcn import GCN, GCNLayer, Linear3, Pool, GraphPooling
from cl.datasets.synthetic_graph import (
    SyntheticGraphDataset,
    TUGraphDataset,
    BaseGraphDataset,
    load_graph_dataset,
)
from cl.core.awb import (
    partition_for_AB_training_gnn,
    partition_for_standard_training_gnn,
    compute_V_from_AWB_gcn,
    save_gcn_layer_weights,
    restore_gcn_layer_weights,
)
from cl.arch_search.gcn_search import prepABs_GCN


@pytest.fixture
def gcn_config():
    """Configuration for creating test GCN models."""
    return {
        'in_size': 5,
        'gcn_sizes': [5, 64],
        'feed_sizes': [64, 32, 16, 10],
        'node_num': 10,
        'out_size': 10,
        'SEED': 42,
    }


@pytest.fixture
def graph_dataset_config():
    """Configuration for creating test graph datasets."""
    return {
        'data': 'synthetic',
        'batch_size': 4,
        'n_class': 10,
        'class_per_task': 2,
        'debug_mode': True,
        'debug_limit': 50,
        'num_graphs': 100,
        'num_channels': 5,
        'avg_num_nodes': 3,
        'num_classes': 10,
    }


class TestGCNLayer:
    """Tests for GCNLayer module."""

    def test_gcn_layer_initialization(self, jax_key):
        """Test GCNLayer initializes with correct dimensions."""
        layer = GCNLayer(in_size=10, out_size=32, key=jax_key)

        assert layer.weight.shape == (10, 32)
        assert layer.bias.shape == (1, 32)
        assert layer.bias_flag == True
        assert layer.sparse == False

    def test_gcn_layer_no_bias(self, jax_key):
        """Test GCNLayer without bias."""
        layer = GCNLayer(in_size=10, out_size=32, key=jax_key, bias=False)

        assert layer.weight.shape == (10, 32)
        assert layer.bias is None
        assert layer.bias_flag == False

    def test_gcn_layer_forward(self, jax_key):
        """Test GCNLayer forward pass."""
        layer = GCNLayer(in_size=5, out_size=16, key=jax_key)

        # Create node features and adjacency matrix
        x = jnp.ones((10, 5))  # 10 nodes, 5 features
        adj = jnp.eye(10)  # Identity adjacency (self-loops only)

        output = layer(x, adj)

        assert output.shape == (10, 16)
        assert not jnp.isnan(output).any()


class TestLinear3:
    """Tests for Linear3 layer (feed-forward in GCN)."""

    def test_linear3_initialization(self, jax_key):
        """Test Linear3 initializes with correct dimensions."""
        layer = Linear3(in_size=32, out_size=16, key=jax_key)

        # Note: weight shape is (out_size, in_size) for Linear3
        assert layer.weight.shape == (16, 32)
        assert layer.bias.shape == (1, 16)

    def test_linear3_forward(self, jax_key):
        """Test Linear3 forward pass: x @ W.T + bias."""
        layer = Linear3(in_size=32, out_size=16, key=jax_key)

        x = jnp.ones((8, 32))  # 8 samples, 32 features
        output = layer(x)

        assert output.shape == (8, 16)
        assert not jnp.isnan(output).any()


class TestPool:
    """Tests for graph pooling operations."""

    def test_pool_sum(self):
        """Test sum pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])
        num_nodes = jnp.array([2, 2])

        result = Pool.sum(x, batch, num_nodes)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [4.0, 6.0])
        np.testing.assert_allclose(result[1], [12.0, 14.0])

    def test_pool_mean(self):
        """Test mean pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])
        num_nodes = jnp.array([2, 2])

        result = Pool.mean(x, batch, num_nodes)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [6.0, 7.0])

    def test_pool_max(self):
        """Test max pooling."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        batch = jnp.array([0, 0, 1, 1])
        num_nodes = jnp.array([2, 2])

        result = Pool.max(x, batch, num_nodes)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [3.0, 4.0])
        np.testing.assert_allclose(result[1], [7.0, 8.0])


class TestGCN:
    """Tests for GCN model."""

    def test_gcn_initialization(self, gcn_config):
        """Test GCN initializes with correct architecture."""
        model = GCN(**gcn_config)

        assert model.gcn_sizes == gcn_config['gcn_sizes']
        assert model.feed_sizes == gcn_config['feed_sizes']
        assert len(model.gcn_layers) == len(gcn_config['gcn_sizes']) - 1
        assert len(model.feed_layers) == len(gcn_config['feed_sizes']) - 1

    def test_gcn_awb_matrices_initialization(self, gcn_config):
        """Test GCN initializes AWB matrices."""
        model = GCN(**gcn_config)

        # A_gcn, B_gcn for GCN layers
        assert len(model.A_gcn) == len(gcn_config['gcn_sizes']) - 1
        assert len(model.B_gcn) == len(gcn_config['gcn_sizes']) - 1

        # A_feed, B_feed for feed layers
        assert len(model.A_feed) == len(gcn_config['feed_sizes']) - 1
        assert len(model.B_feed) == len(gcn_config['feed_sizes']) - 1

    def test_gcn_forward_pass(self, gcn_config):
        """Test GCN standard forward pass."""
        model = GCN(**gcn_config)

        # Create test data (10 nodes, 5 features)
        x = jnp.ones((10, gcn_config['in_size']))
        adj = jnp.eye(10)  # Self-loops only
        batch = jnp.array([0] * 5 + [1] * 5)  # 2 graphs, 5 nodes each
        n_nodes = jnp.array([5, 5])

        output = model(x, adj, batch, n_nodes)

        # Output shape: (num_graphs, num_classes)
        assert output.shape == (2, gcn_config['out_size'])
        assert not jnp.isnan(output).any()

    def test_gcn_awb_forward_pass(self, gcn_config):
        """Test GCN AWB forward pass (get_AWBT)."""
        model = GCN(**gcn_config)

        # Create test data
        x = jnp.ones((10, gcn_config['in_size']))
        adj = jnp.eye(10)
        batch = jnp.array([0] * 5 + [1] * 5)
        n_nodes = jnp.array([5, 5])

        output = model.get_AWBT(x, adj, batch, n_nodes)

        # Output shape: (num_graphs, num_classes)
        assert output.shape == (2, gcn_config['out_size'])
        assert not jnp.isnan(output).any()


class TestGCNAWBPartitioning:
    """Tests for GCN AWB partitioning functions."""

    def test_partition_for_AB_training(self, gcn_config):
        """Test partitioning GCN for A/B training (freeze W, train A/B)."""
        model = GCN(**gcn_config)

        diff_model, static_model = partition_for_AB_training_gnn(model)

        # A and B should be in diff_model (trainable)
        assert diff_model.A_gcn is not None
        assert diff_model.B_gcn is not None
        assert diff_model.A_feed is not None
        assert diff_model.B_feed is not None

    def test_partition_for_standard_training(self, gcn_config):
        """Test partitioning GCN for standard training (freeze A/B, train W)."""
        model = GCN(**gcn_config)

        params, static = partition_for_standard_training_gnn(model)

        # A and B should be in static (frozen) - set to None in params
        assert params.A_gcn is None
        assert params.B_gcn is None
        assert params.A_feed is None
        assert params.B_feed is None

        # A and B should be preserved in static
        assert static.A_gcn is not None
        assert static.B_gcn is not None


class TestGCNAWBCompute:
    """Tests for GCN AWB computation functions."""

    def test_compute_V_from_AWB_gcn(self, gcn_config):
        """Test computing V = A @ W @ B.T for GCN."""
        model = GCN(**gcn_config)

        # Store original weights
        orig_gcn_weight = model.gcn_layers[0].weight.copy()
        orig_feed_weight = model.feed_layers[0].weight.copy()

        # Compute V = AWB
        model_v = compute_V_from_AWB_gcn(model)

        # After AWB: input/output dimensions preserved, hidden layers expanded
        # GCN has no hidden layers: gcn_sizes=[5,64] -> awb_gcn_arch=[5,64]
        assert model_v.gcn_layers[0].weight.shape == orig_gcn_weight.shape  # (5, 64) preserved

        # Feed has 2 hidden layers: feed_sizes=[64,32,16,10] -> awb_fnn_arch=[64,100,140,10]
        # feed_layers[0]: 64→32 expands to 64→100, so weight shape (32,64) -> (100,64)
        assert model_v.feed_layers[0].weight.shape == (100, 64)  # Hidden layer expanded from 32 to 100

    def test_save_and_restore_gcn_weights(self, gcn_config):
        """Test saving and restoring GCN layer weights."""
        model = GCN(**gcn_config)

        # Save original weights
        gcn_w, gcn_b, mlp_w, mlp_b = save_gcn_layer_weights(model)

        # Modify model weights (simulate architecture search)
        new_weight = jnp.zeros_like(model.gcn_layers[0].weight)
        model = eqx.tree_at(lambda m: m.gcn_layers[0].weight, model, new_weight)

        # Verify weights changed
        assert jnp.allclose(model.gcn_layers[0].weight, jnp.zeros_like(new_weight))

        # Restore weights
        model = restore_gcn_layer_weights(model, gcn_w, gcn_b, mlp_w, mlp_b)

        # Verify weights restored
        assert jnp.allclose(model.gcn_layers[0].weight, gcn_w[0])


class TestPrepABsGCN:
    """Tests for prepABs_GCN function."""

    def test_prepABs_no_change(self, gcn_config):
        """Test prepABs_GCN with no architecture change (identity matrices)."""
        model = GCN(**gcn_config)

        prev_feed_sizes = list(model.feed_sizes)
        prev_gcn_sizes = list(model.gcn_sizes)

        A_feed, B_feed, A_gcn, B_gcn = prepABs_GCN(model, prev_feed_sizes, prev_gcn_sizes)

        # Should return identity matrices when no change
        for i, a in enumerate(A_gcn):
            expected_size = prev_gcn_sizes[i]
            assert a.shape == (expected_size, expected_size)

    def test_prepABs_gcn_change(self, gcn_config):
        """Test prepABs_GCN with GCN architecture change."""
        model = GCN(**gcn_config)

        prev_gcn_sizes = [5, 32]  # Smaller previous size
        prev_feed_sizes = [32, 32, 16, 10]

        # Update model to new sizes
        new_gcn_sizes = [5, 64]
        new_feed_sizes = [64, 32, 16, 10]
        model = eqx.tree_at(lambda m: m.gcn_sizes, model, new_gcn_sizes)
        model = eqx.tree_at(lambda m: m.feed_sizes, model, new_feed_sizes)

        A_feed, B_feed, A_gcn, B_gcn = prepABs_GCN(model, prev_feed_sizes, prev_gcn_sizes)

        # Transformation matrices should map from old to new sizes
        assert len(A_gcn) == len(new_gcn_sizes) - 1
        assert len(B_gcn) == len(new_gcn_sizes) - 1


class TestSyntheticGraphDataset:
    """Tests for SyntheticGraphDataset."""

    def test_synthetic_dataset_initialization(self, graph_dataset_config):
        """Test SyntheticGraphDataset initializes correctly."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        assert dataset.train_data is not None
        assert dataset.test_data is not None
        assert dataset.num_features > 0
        assert dataset.num_classes > 0

    def test_synthetic_dataset_properties(self, graph_dataset_config):
        """Test SyntheticGraphDataset properties."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        assert dataset.input_size == dataset.num_features
        assert dataset.output_size == dataset.num_classes

    def test_synthetic_dataset_generate(self, graph_dataset_config):
        """Test generating dataloaders for a task."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        train_loader, mem_loader = dataset.generate_dataset(task_id=0, batch_size=4, phase='training')

        # Should return DataLoader objects
        assert train_loader is not None
        assert mem_loader is not None

    def test_synthetic_dataset_test_loader(self, graph_dataset_config):
        """Test getting test data loader."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        test_loader = dataset.get_test_loader(batch_size=4)

        assert test_loader is not None

    def test_synthetic_dataset_model_config(self, graph_dataset_config):
        """Test get_model_config returns correct values."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        model_config = dataset.get_model_config()

        assert 'input_size' in model_config
        assert 'output_size' in model_config
        assert 'num_features' in model_config
        assert 'num_classes' in model_config


class TestLoadGraphDataset:
    """Tests for load_graph_dataset factory function."""

    def test_load_synthetic(self, graph_dataset_config):
        """Test loading synthetic graph dataset."""
        graph_dataset_config['data'] = 'synthetic'
        dataset = load_graph_dataset(graph_dataset_config)

        assert isinstance(dataset, SyntheticGraphDataset)

    def test_load_unknown_raises(self, graph_dataset_config):
        """Test loading unknown dataset raises error."""
        graph_dataset_config['data'] = 'unknown_dataset'

        with pytest.raises(ValueError, match="Unknown graph dataset"):
            load_graph_dataset(graph_dataset_config)


class TestGCNIntegration:
    """Integration tests for GCN with synthetic data."""

    def test_gcn_with_synthetic_data(self, graph_dataset_config):
        """Test GCN forward pass with actual synthetic graph data."""
        dataset = SyntheticGraphDataset(graph_dataset_config)

        # Create GCN model matching dataset
        model = GCN(
            in_size=dataset.num_features,
            gcn_sizes=[dataset.num_features, 32],
            feed_sizes=[32, 16, dataset.num_classes],
            node_num=10,
            out_size=dataset.num_classes,
        )

        # Get a batch of data
        test_loader = dataset.get_test_loader(batch_size=2)
        batch = next(iter(test_loader))

        # Forward pass through model
        x = jnp.array(batch.x.numpy())
        edge_index = batch.edge_index.numpy()

        # Build adjacency matrix from edge_index
        num_nodes = x.shape[0]
        adj = jnp.zeros((num_nodes, num_nodes))
        adj = adj.at[edge_index[0], edge_index[1]].set(1.0)

        batch_idx = jnp.array(batch.batch.numpy())
        n_nodes = jnp.array([batch.n_nodes[i].item() for i in range(len(batch.n_nodes))])

        output = model(x, adj, batch_idx, n_nodes)

        assert output.shape[1] == dataset.num_classes
        assert not jnp.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
