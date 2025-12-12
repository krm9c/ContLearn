"""
Unit tests for utility functions in utils/utils.py.
Tests sparse matrix operations, graph preprocessing, and visualization utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import pytest
import torch_geometric as pyg
from torch_geometric.data import Data

from utils.utils import (
    sp_matmul, normalize, preprocess_features, normalize_adj,
    preprocess_adj, to_sparse, plot_dists, visualize_gradients
)


class TestSparseMatmul:
    """Tests for sparse matrix multiplication."""

    def test_sp_matmul_identity(self):
        """Test sparse matmul with identity matrix."""
        # Identity matrix: diagonal entries
        rows = jnp.array([0, 1, 2])
        cols = jnp.array([0, 1, 2])
        values = jnp.array([1.0, 1.0, 1.0])
        sparse_A = ((rows, cols), values)

        # Dense matrix B
        B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = sp_matmul(sparse_A, B, 3)

        # Result should be same as B (identity multiplication)
        assert result.shape == (3, 2)
        assert jnp.allclose(result, B)

    def test_sp_matmul_general(self):
        """Test sparse matmul with general sparse matrix."""
        # Sparse matrix with specific pattern
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

    def test_sp_matmul_multiple_columns(self):
        """Test sparse matmul with multiple columns in B."""
        rows = jnp.array([0, 1])
        cols = jnp.array([0, 1])
        values = jnp.array([2.0, 3.0])
        sparse_A = ((rows, cols), values)

        B = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = sp_matmul(sparse_A, B, 2)

        assert result.shape == (2, 2)
        # Row 0: [2*1, 2*2] = [2, 4]
        # Row 1: [3*3, 3*4] = [9, 12]
        expected = jnp.array([[2.0, 4.0], [9.0, 12.0]])
        assert jnp.allclose(result, expected)

    def test_sp_matmul_jit_compiled(self):
        """Test that sp_matmul is JIT compiled and works correctly."""
        rows = jnp.array([0, 1])
        cols = jnp.array([0, 1])
        values = jnp.array([1.0, 1.0])
        sparse_A = ((rows, cols), values)

        B = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # First call (compiles)
        result1 = sp_matmul(sparse_A, B, 2)

        # Second call (uses cached compilation)
        result2 = sp_matmul(sparse_A, B, 2)

        assert jnp.allclose(result1, result2)


class TestNormalization:
    """Tests for matrix normalization functions."""

    def test_normalize_basic(self):
        """Test row normalization of sparse matrix."""
        # Create a simple sparse matrix
        mx = sp.csr_matrix([[2.0, 0.0], [0.0, 4.0]])

        result = normalize(mx)

        # After normalization, each row should sum to 1
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_normalize_with_zeros(self):
        """Test normalization handles zero rows correctly."""
        # Matrix with a zero row
        mx = sp.csr_matrix([[2.0, 2.0], [0.0, 0.0], [3.0, 3.0]])

        result = normalize(mx)

        # Non-zero rows should sum to 1, zero row should remain 0
        row_sums = np.array(result.sum(axis=1)).flatten()
        assert np.allclose(row_sums[0], 1.0)
        assert np.allclose(row_sums[1], 0.0)
        assert np.allclose(row_sums[2], 1.0)

    def test_preprocess_features(self):
        """Test feature matrix preprocessing (row normalization)."""
        features = sp.csr_matrix([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])

        result = preprocess_features(features)

        # Each row should sum to 1
        row_sums = np.array(result.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0)

    def test_preprocess_features_with_zeros(self):
        """Test feature preprocessing handles zero rows."""
        features = sp.csr_matrix([[1.0, 1.0], [0.0, 0.0]])

        result = preprocess_features(features)

        row_sums = np.array(result.sum(axis=1)).flatten()
        assert np.allclose(row_sums[0], 1.0)
        assert np.allclose(row_sums[1], 0.0)


class TestGraphPreprocessing:
    """Tests for graph adjacency matrix preprocessing."""

    def test_to_sparse(self):
        """Test conversion to sparse representation."""
        # Create a sparse COO matrix
        adj = sp.coo_matrix([[1.0, 0.0], [0.0, 1.0]])

        result = to_sparse(adj)

        # Result should be (indices, values)
        assert isinstance(result, tuple)
        assert len(result) == 2
        indices, values = result
        assert len(indices) == 2  # Row and column indices
        assert len(values) == 2  # Two non-zero values

    def test_normalize_adj_basic(self):
        """Test symmetric normalization of adjacency matrix."""
        # Create simple graph edge index
        edge_index = [[0, 1], [1, 0]]  # Bidirectional edge

        result = normalize_adj(edge_index)

        # Result should be sparse representation
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preprocess_adj(self):
        """Test adjacency preprocessing (adds self-loops and normalizes)."""
        # Create adjacency matrix
        adj = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])

        result = preprocess_adj(adj)

        # Result should be normalized sparse representation
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestVisualization:
    """Tests for visualization utility functions."""

    def test_plot_dists_single_layer(self):
        """Test plot_dists with single layer."""
        val_dict = {
            'Layer 0': np.random.randn(100)
        }

        # Should create figure without errors
        fig = plot_dists(val_dict)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dists_multiple_layers(self):
        """Test plot_dists with multiple layers."""
        val_dict = {
            'Layer 0': np.random.randn(100),
            'Layer 2': np.random.randn(100),
            'Layer 4': np.random.randn(100)
        }

        fig = plot_dists(val_dict)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dists_2d_arrays(self):
        """Test plot_dists with 2D arrays (weights)."""
        val_dict = {
            'Layer 0': np.random.randn(64, 32),
            'Layer 2': np.random.randn(32, 16)
        }

        fig = plot_dists(val_dict)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dists_custom_color(self):
        """Test plot_dists with custom color."""
        val_dict = {'Layer 0': np.random.randn(100)}

        fig = plot_dists(val_dict, color="C1")

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_visualize_gradients_basic(self):
        """Test gradient visualization."""
        # Create dummy gradients
        grads = {
            'layer1': jnp.array(np.random.randn(64, 32).astype(np.float32)),
            'layer2': jnp.array(np.random.randn(32, 16).astype(np.float32))
        }
        params = grads  # Not used in function, but required parameter

        # Should run without errors (creates and closes plot)
        visualize_gradients(grads, params, print_variance=False)

    def test_visualize_gradients_with_variance(self):
        """Test gradient visualization with variance printing."""
        grads = {
            'layer1': jnp.array(np.random.randn(64, 32).astype(np.float32))
        }
        params = grads

        # Should run without errors
        visualize_gradients(grads, params, print_variance=True)

    def test_visualize_gradients_filters_bias(self):
        """Test that gradient visualization filters out 1D parameters (bias)."""
        grads = {
            'weight': jnp.array(np.random.randn(64, 32).astype(np.float32)),
            'bias': jnp.array(np.random.randn(64).astype(np.float32))  # 1D, should be filtered
        }
        params = grads

        # Should run without errors, only plotting weight gradients
        visualize_gradients(grads, params, print_variance=False)


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_graph_preprocessing_pipeline(self):
        """Test full graph preprocessing pipeline."""
        # Create simple graph
        adj = sp.csr_matrix([[0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [1.0, 1.0, 0.0]])

        # Preprocess adjacency
        adj_preprocessed = preprocess_adj(adj)

        assert isinstance(adj_preprocessed, tuple)
        assert len(adj_preprocessed) == 2

    def test_feature_and_adj_preprocessing(self):
        """Test preprocessing both features and adjacency."""
        features = sp.csr_matrix([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        adj = sp.csr_matrix([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])

        # Preprocess both
        features_norm = preprocess_features(features)
        adj_norm = preprocess_adj(adj)

        # Both should be normalized
        assert features_norm is not None
        assert adj_norm is not None

        # Features should be row-normalized
        row_sums = np.array(features_norm.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_normalize_empty_matrix(self):
        """Test normalization with empty matrix."""
        mx = sp.csr_matrix((0, 0))

        result = normalize(mx)

        assert result.shape == (0, 0)

    def test_preprocess_features_single_feature(self):
        """Test feature preprocessing with single feature dimension."""
        features = sp.csr_matrix([[5.0], [10.0]])

        result = preprocess_features(features)

        # Each row should sum to 1
        row_sums = np.array(result.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0)

    def test_sp_matmul_with_zeros(self):
        """Test sparse matmul with sparse matrix containing zeros."""
        rows = jnp.array([0, 1])
        cols = jnp.array([0, 1])
        values = jnp.array([0.0, 2.0])
        sparse_A = ((rows, cols), values)

        B = jnp.array([[1.0], [3.0]])

        result = sp_matmul(sparse_A, B, 2)

        # Row 0: 0*1 = 0
        # Row 1: 2*3 = 6
        expected = jnp.array([[0.0], [6.0]])
        assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
