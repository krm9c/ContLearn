"""
Graph Neural Network models for graph classification.

Adapted from ContLearn/src/contlearn/models/graph.py for cl_framework.
Contains GCN layers and myNN model with AWB (Adaptive Weight Basis) support.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Callable, List, Optional, Tuple, Dict, Any

from ..config.constants import (
    DEFAULT_AWB_FNN_ARCH,
    DEFAULT_AWB_GCN_ARCH,
    DEFAULT_GCN_SIZES,
    DEFAULT_GCN_MLP_SIZES,
)
from .layers import Linear, Dropout, AWBLayerSpec
import jax.tree_util as jtu


@partial(jax.jit, static_argnums=(2,))
def sp_matmul(A, B, shape):
    """Sparse matrix multiplication for graph operations.

    Args:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M, K) dense matrix
        shape: value of N

    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_ * values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res


class Pool:
    """Static graph pooling operations."""

    @staticmethod
    def sum(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        out_shape = num_nodes.shape[0]
        return jax.ops.segment_sum(x, batch, out_shape)

    @staticmethod
    def mean(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        added = Pool.sum(x, batch, num_nodes)
        return added / jnp.array(num_nodes).reshape([-1, 1])

    @staticmethod
    def max(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return jax.ops.segment_max(x, batch, num_nodes.shape[0])

    @staticmethod
    def identity(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return x


class GraphPooling:
    """Graph pooling wrapper class."""

    def __init__(self, pool: Callable) -> None:
        self.pool = pool

    def __call__(self, x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return self.pool(x, batch, num_nodes)


class GCNLayer(eqx.Module):
    """Graph Convolutional Network layer with AWB support.

    Performs: x = A_adj @ (x @ W) + bias
    where A_adj is the normalized adjacency matrix.

    Args:
        in_size: Input feature dimension
        out_size: Output feature dimension
        key: JAX PRNG key for initialization
        bias: Whether to use bias (default: True)
        sparse: Whether adjacency is sparse (default: False)
    """
    weight: jax.Array
    bias: jax.Array
    sparse: bool
    bias_flag: bool
    initializer: None

    def __init__(self, in_size: int, out_size: int, key: jax.Array,
                 bias: bool = True, sparse: bool = False):
        self.bias_flag = bias
        self.sparse = sparse
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (in_size, out_size))
        if self.bias_flag:
            self.bias = self.initializer(bkey, (1, out_size))
        else:
            self.bias = None

    def matmul(self, A, B, shape):
        """Matrix multiplication supporting sparse adjacency."""
        if self.sparse:
            return sp_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    # Added by Claude: Standard GCN normalization (Kipf & Welling, 2017)
    def normalize_adjacency(self, adj: jax.Array) -> jax.Array:
        """Apply symmetric normalization with self-loops.

        Implements the standard GCN normalization from Kipf & Welling (2017):
        Â = D̃^(-1/2) @ (A + I) @ D̃^(-1/2)

        where:
        - A is the adjacency matrix
        - I is the identity matrix (self-loops)
        - D̃ is the degree matrix of (A + I)

        Args:
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Normalized adjacency matrix (num_nodes, num_nodes)
        """
        # Add self-loops: Ã = A + I
        num_nodes = adj.shape[0]
        adj_with_loops = adj + jnp.eye(num_nodes)

        # Compute degree matrix: D̃_ii = sum_j Ã_ij
        degree = jnp.sum(adj_with_loops, axis=1)

        # Compute D̃^(-1/2) with numerical stability (handle isolated nodes)
        deg_inv_sqrt = jnp.power(degree, -0.5)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0., deg_inv_sqrt)

        # Symmetric normalization: D̃^(-1/2) @ Ã @ D̃^(-1/2)
        deg_mat_inv_sqrt = jnp.diag(deg_inv_sqrt)
        adj_normalized = deg_mat_inv_sqrt @ adj_with_loops @ deg_mat_inv_sqrt

        return adj_normalized

    def __call__(self, x: jax.Array, adj: jax.Array) -> jax.Array:
        """Forward pass with standard GCN normalization.

        Implements: Â @ X @ W + b
        where Â = D̃^(-1/2) @ (A + I) @ D̃^(-1/2)

        Args:
            x: Node features (num_nodes, in_size)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Updated node features (num_nodes, out_size)
        """
        # Added by Claude: Apply standard GCN normalization
        adj_normalized = self.normalize_adjacency(adj)

        # Graph convolution with normalized adjacency
        support = x @ self.weight
        x = self.matmul(adj_normalized, support, support.shape)
        if self.bias_flag:
            x += self.bias
        return x


class Linear3(eqx.Module):
    """Linear layer with bias shape (1, out_size) for GCN feed layers.

    Weight shape: (out_size, in_size)
    Bias shape: (1, out_size)

    Forward: x @ W.T + bias

    Args:
        in_size: Input dimension
        out_size: Output dimension
        key: JAX PRNG key for initialization
    """
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size))
        self.bias = initializer(bkey, (1, out_size))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: x @ W.T + bias"""
        return x @ self.weight.T + self.bias


class GCN(eqx.Module):
    """Graph Neural Network with GCN + MLP and AWB support.

    Architecture:
    - GCN layers for message passing on graph
    - Graph pooling (max by default)
    - Feed-forward MLP for classification

    AWB Attributes:
    - A_gcn, B_gcn: Transformation matrices for GCN layers
    - A_feed, B_feed: Transformation matrices for feed layers

    Args:
        in_size: Input feature dimension
        feed_sizes: List of feed-forward layer sizes [gcn_out, hidden1, hidden2, ..., n_class]
        gcn_sizes: List of GCN layer sizes [in_size, hidden, ...]
        node_num: Number of nodes (for reference)
        SEED: Random seed for initialization
        out_size: Number of output classes
        graph: Whether to use graph pooling (default: True)
        awb_fnn_arch: AWB architecture for feed layers
        awb_gcn_arch: AWB architecture for GCN layers
    """
    gcn_layers: list
    pool_layer: GraphPooling
    SEED: int
    graph: bool
    node_num: int
    feed_layers: list
    feed_sizes: list
    gcn_sizes: list
    A_gcn: list
    B_gcn: list
    A_feed: list
    B_feed: list
    sparse: bool

    def __init__(self, in_size: int, feed_sizes: List[int] = None,
                 gcn_sizes: List[int] = None, node_num: int = 0,
                 SEED: int = 1234, out_size: int = 2, graph: bool = True,
                 awb_fnn_arch: List[int] = None, awb_gcn_arch: List[int] = None):
        """Initialize GCN model."""
        self.SEED = SEED
        self.graph = graph
        self.node_num = node_num

        # Default architectures
        if gcn_sizes is None:
            gcn_sizes = [in_size, 128]
        if feed_sizes is None:
            feed_sizes = [128, 128, 128, out_size]

        # Ensure first GCN layer matches input size
        gcn_sizes[0] = in_size
        self.gcn_sizes = gcn_sizes

        # Ensure last feed layer matches output size
        feed_sizes[-1] = out_size
        self.feed_sizes = feed_sizes

        # AWB architectures - use provided or defaults
        # AWB should preserve input and output dimensions but can expand hidden layers
        if awb_fnn_arch is None:
            # Preserve input (feed_sizes[0]) and output (feed_sizes[-1])
            # Expand hidden layers using DEFAULT_AWB_FNN_ARCH
            num_hidden_feed = len(self.feed_sizes) - 2
            if num_hidden_feed > 0:
                awb_hidden_feed = DEFAULT_AWB_FNN_ARCH[:num_hidden_feed]
                awb_fnn_arch = [self.feed_sizes[0]] + awb_hidden_feed + [out_size]
            else:
                awb_fnn_arch = [self.feed_sizes[0], out_size]

        if awb_gcn_arch is None:
            # Preserve input (gcn_sizes[0]) and output (gcn_sizes[-1])
            # Expand hidden layers using DEFAULT_AWB_GCN_ARCH
            num_hidden_gcn = len(self.gcn_sizes) - 2
            if num_hidden_gcn > 0:
                awb_hidden_gcn = DEFAULT_AWB_GCN_ARCH[:num_hidden_gcn]
                awb_gcn_arch = [self.gcn_sizes[0]] + awb_hidden_gcn + [self.gcn_sizes[-1]]
            else:
                awb_gcn_arch = [self.gcn_sizes[0], self.gcn_sizes[-1]]

        # Initialize AWB matrices
        initializer = jax.nn.initializers.glorot_uniform()
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                       for x, y in zip(self.feed_sizes[1:], awb_fnn_arch[1:])]
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                       for x, y in zip(self.feed_sizes[:-1], awb_fnn_arch[:-1])]
        self.B_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                      for x, y in zip(self.gcn_sizes[1:], awb_gcn_arch[1:])]
        self.A_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                      for x, y in zip(self.gcn_sizes[:-1], awb_gcn_arch[:-1])]

        # Build GCN layers
        self.gcn_layers = []
        for i in range(len(self.gcn_sizes) - 1):
            self.gcn_layers.append(
                GCNLayer(in_size=self.gcn_sizes[i], out_size=self.gcn_sizes[i + 1],
                         key=jax.random.PRNGKey(SEED))
            )

        self.sparse = self.gcn_layers[0].sparse if self.gcn_layers else False

        # Build feed-forward layers
        self.feed_layers = []
        if self.graph:
            for (in_layer, out_layer) in zip(self.feed_sizes[:-1], self.feed_sizes[1:]):
                self.feed_layers.append(
                    Linear3(in_size=in_layer, out_size=out_layer,
                            key=jax.random.PRNGKey(self.SEED))
                )

        self.pool_layer = GraphPooling(Pool.max)

    def matmul(self, A, B, shape):
        """Matrix multiplication supporting sparse adjacency."""
        if self.sparse:
            return sp_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    def __call__(self, x: jax.Array, adj: jax.Array, batch: jax.Array,
                 n_nodes: jax.Array) -> jax.Array:
        """Standard forward pass.

        Args:
            x: Node features (total_nodes, in_size)
            adj: Adjacency matrix (total_nodes, total_nodes)
            batch: Batch assignment for each node (total_nodes,)
            n_nodes: Number of nodes per graph (batch_size,)

        Returns:
            Class logits (batch_size, n_class)
        """
        # GCN layers with LeakyReLU
        for layer in self.gcn_layers:
            x = jax.nn.leaky_relu(layer(x, adj))

        # Graph pooling
        x = self.pool_layer(x, batch, n_nodes)

        # Feed-forward layers (all but last with LeakyReLU)
        for i in range(len(self.feed_sizes) - 2):
            x = jax.nn.leaky_relu(self.feed_layers[i](x))

        # Final layer (no activation)
        x = self.feed_layers[-1](x)

        return x

    def get_AWBT(self, x: jax.Array, adj: jax.Array, batch: jax.Array,
                 n_nodes: jax.Array) -> jax.Array:
        """Forward pass using AWB transformation with standard GCN normalization.

        Uses V = A @ W @ B.T for both GCN and feed layers.
        Applies standard GCN adjacency normalization: Â = D̃^(-1/2) @ (A + I) @ D̃^(-1/2)

        Args:
            x: Node features (total_nodes, in_size)
            adj: Adjacency matrix (total_nodes, total_nodes)
            batch: Batch assignment for each node (total_nodes,)
            n_nodes: Number of nodes per graph (batch_size,)

        Returns:
            Class logits (batch_size, n_class)
        """
        # GCN layers with AWB transformation
        for i in range(len(self.gcn_layers)):
            # Added by Claude: Apply standard GCN normalization
            adj_normalized = self.gcn_layers[i].normalize_adjacency(adj)

            # Compute AWB transformed weight: V = A_gcn @ W @ B_gcn^T
            transformed_weight = (self.A_gcn[i] @ self.gcn_layers[i].weight
                                  @ jnp.transpose(self.B_gcn[i]))
            support = x @ transformed_weight
            # Use normalized adjacency for graph convolution
            x = self.matmul(adj_normalized, support, support.shape)

            if self.gcn_layers[i].bias_flag:
                # Transform bias: bias @ B.T
                x += (self.gcn_layers[i].bias @ self.B_gcn[i].T)

            x = jax.nn.leaky_relu(x)

        # Graph pooling
        x = self.pool_layer(x, batch, n_nodes)

        # Feed-forward layers with AWB transformation (all but last with LeakyReLU)
        for i in range(len(self.feed_sizes) - 2):
            # Compute AWB transformed: x @ (A @ W.T @ B.T) + bias @ B.T
            transformed_weight = (self.A_feed[i] @ self.feed_layers[i].weight.T
                                  @ jnp.transpose(self.B_feed[i]))
            x = x @ transformed_weight + (self.feed_layers[i].bias @ self.B_feed[i].T)
            x = jax.nn.leaky_relu(x)

        # Final layer with AWB transformation (no activation)
        transformed_weight = (self.A_feed[-1] @ self.feed_layers[-1].weight.T
                              @ jnp.transpose(self.B_feed[-1]))
        x = x @ transformed_weight + (self.feed_layers[-1].bias @ self.B_feed[-1].T)

        return x

    # Added by Claude: AWBModel interface for GCN
    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB specs for feed layers only (GCN layers handled separately)."""
        return [
            AWBLayerSpec(layer=self.feed_layers[i], A=self.A_feed[i], B=self.B_feed[i],
                        layer_type='linear', layer_index=i)
            for i in range(len(self.feed_layers))
        ]

    def partition_for_AB_training(self):
        """Partition for A/B training (freeze W, train A/B)."""
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
                                  filter_spec, replace=(True, True, True, True))
        return eqx.partition(self, filter_spec)

    def partition_for_standard_training(self):
        """Partition for standard training (freeze A/B, train W)."""
        params, static = eqx.partition(self, eqx.is_array)
        static = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), static,
                            replace=(self.A_gcn, self.B_gcn, self.A_feed, self.B_feed))
        params = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), params,
                            replace=(None, None, None, None))
        return params, static

    # Added by Claude: Architecture search interface methods
    def generate_search_candidates(self, iteration: int, current_best: Tuple[List[int], List[int]],
                                   config: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
        """Generate candidate architectures for GCN using neighborhood search.

        GCN architecture has two components:
        - gcn_sizes: [in_size, gcn_hidden, ...] for graph convolution layers
        - feed_sizes: [gcn_out, mlp_h1, mlp_h2, n_class] for feedforward layers

        Search strategy:
        - Uses neighborhood search with expanding radius (n)
        - For each iteration, explores 3x3x3 = 27 candidates
        - GCN hidden: z2 + n*(j+1)*step_gcn for j in [0,1,2]
        - MLP hidden1: x1 + n*(k+1)*step_mlp for k in [0,1,2]
        - MLP hidden2: x2 + n*(r+1)*step_mlp for r in [0,1,2]

        Args:
            iteration: Current search iteration (controls neighborhood size n)
            current_best: Tuple of (gcn_sizes, mlp_sizes) lists
            config: Configuration dict with step_gcn, step_mlp parameters

        Returns:
            List of (gcn_sizes, mlp_sizes) tuples to evaluate
        """
        from ..config.constants import (
            DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN,
            DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP,
        )

        # Unpack current best architecture
        current_gcn, current_mlp = current_best

        # Get search parameters from config
        step_gcn = config.get('arch_search_step_size_gcn', DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN)
        step_mlp = config.get('arch_search_step_size_mlp', DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP)

        # Extract base dimensions from current architecture
        # gcn_sizes = [in_size, z2]
        z2 = current_gcn[1] if len(current_gcn) > 1 else current_gcn[0]

        # mlp_sizes = [gcn_out, x1, x2, n_class]
        x1 = current_mlp[1] if len(current_mlp) > 1 else current_mlp[0]
        x2 = current_mlp[2] if len(current_mlp) > 2 else x1

        # Neighborhood radius grows with iteration
        n = iteration + 1

        # Generate candidate architectures
        candidates = []

        # Search over GCN architecture (3 candidates)
        for j in range(3):
            new_gcn_hidden = z2 + n * (j + 1) * step_gcn
            curr_gcn = [current_gcn[0], new_gcn_hidden]  # Preserve input size

            # Search over MLP architecture (3x3 = 9 candidates per GCN size)
            for k in range(3):
                for r in range(3):
                    new_mlp_h1 = x1 + n * (k + 1) * step_mlp
                    new_mlp_h2 = x2 + n * (r + 1) * step_mlp

                    # MLP connects from GCN output to final class count
                    curr_mlp = [
                        new_gcn_hidden,  # First MLP layer takes GCN output
                        new_mlp_h1,
                        new_mlp_h2,
                        current_mlp[-1]  # Preserve output size (n_class)
                    ]

                    candidates.append((curr_gcn, curr_mlp))

        return candidates

    @classmethod
    def create_with_architecture(cls, arch_spec: Tuple[List[int], List[int]],
                                 seed: int = 0, awb_enabled: bool = True) -> 'GCN':
        """Create GCN model with specified architecture.

        Args:
            arch_spec: Tuple of (gcn_sizes, feed_sizes) lists
            seed: Random seed for weight initialization
            awb_enabled: Whether to enable AWB (always True for GCN)

        Returns:
            New GCN instance with specified architecture
        """
        gcn_sizes, feed_sizes = arch_spec

        # Infer parameters from architecture
        in_size = gcn_sizes[0]
        out_size = feed_sizes[-1]

        # Create new GCN with specified architecture
        return cls(
            in_size=in_size,
            feed_sizes=feed_sizes,
            gcn_sizes=gcn_sizes,
            node_num=0,  # Will be set from actual data
            SEED=seed,
            out_size=out_size,
            graph=True,
            awb_fnn_arch=None,  # Use defaults
            awb_gcn_arch=None   # Use defaults
        )

    def reinitialize_weights(self, seed: int = 0) -> 'GCN':
        """Reinitialize GCN weights for fair architecture comparison.

        For GCN models, we create a fresh instance because the architecture
        is fixed at initialization (GCN and feed layers).

        Args:
            seed: Random seed for initialization

        Returns:
            New GCN instance with reinitialized weights
        """
        # Create fresh GCN with same architecture
        return self.create_with_architecture(
            arch_spec=(self.gcn_sizes, self.feed_sizes),
            seed=seed,
            awb_enabled=True
        )


# Alias for backward compatibility
myNN = GCN