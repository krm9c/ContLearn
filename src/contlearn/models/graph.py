"""Graph Neural Network models for graph classification."""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Any, Callable

from contlearn.config.constants import (
    DEFAULT_AWB_FNN_ARCH,
    DEFAULT_AWB_GCN_ARCH,
)

from .layers import Dropout, Linear, Linear3


@partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """Sparse matrix multiplication for graph operations.

    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    rows, cols = indexes
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res


class SingleHeadGAT(eqx.Module):
    """Single-head Graph Attention Network layer."""
    weight: jax.Array
    a1: jax.Array
    a2: jax.Array
    sparse:bool
    dropout: callable

    def __init__(self, in_size, out_size,\
                 key, sparse=False):
        ## Something to be handled later on
        # output_shape = input_shape[:-1] + (out_dim,)
        ## Still need the different parameters required
        self.sparse=sparse
        wkey,  a1key, a2key, drop_key\
              = jax.random.split(key, 4)
        ## This needs to be taken care of    output_shape = in_size + (out_size,)
        self.dropout = Dropout(rate=0.5)
        self.weight = jax.random.normal(wkey,   (in_size, out_size))
        self.a1     = jax.random.normal(a1key,  ( out_size, 1))
        self.a2     = jax.random.normal(a2key,  (out_size, 1))

    def __call__(self, x, adj, rng, is_training=True):
        x =self.dropout(x, rng, is_training=is_training)
        # print("weights", self.weight.shape, x.shape)
        x   = jnp.dot(x,self.weight)
        f_1 = jnp.dot(x, self.a1)
        f_2 = jnp.dot(x, self.a2)
        logits = f_1 + f_2.T
        # print("logits", logits.shape)
        coefs = jax.nn.softmax(
            jax.nn.leaky_relu(logits, negative_slope=0.2) + jnp.where(adj, 0., -1e9))
        x = self.dropout(x, rng, is_training=is_training)
        # print("coefs", coefs.shape)
        # print(jnp.dot(coefs, x).shape)
        return jnp.dot(coefs, x)


class MultiHeadGAT(eqx.Module):
    """Multi-head Graph Attention Network layer."""
    layer:list
    n_heads: int
    last_layer: bool

    def __init__(self, n_heads, in_size,  out_size,\
                 key, dropout=0.5,last_layer=True):
        self.n_heads=n_heads
        self.last_layer = last_layer
        self.layer =   [ SingleHeadGAT(in_size, out_size, key)\
                         for _ in range(self.n_heads) ]

    def __call__(self, x, adj, rng, is_training=False):
        layer_out=[]
        for head_i in self.layer:
            layer_out.append(head_i(x, adj, rng, is_training=is_training))
        if not self.last_layer:
            x = jnp.concatenate(layer_out, axis=1)
        else:
            x = jnp.mean(jnp.stack(layer_out), axis=0)
        # print("The thing coming out of  multi head", x)
        return x


class Pool:
    """Static graph pooling operations."""

    def sum(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        out_shape = num_nodes.shape[0]
        return jax.ops.segment_sum(x, batch, out_shape)

    def mean(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        added = Pool.sum(x, batch, num_nodes)
        return added / jnp.array(num_nodes).reshape([-1,1])

    def max(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return jax.ops.segment_max(x, batch, num_nodes.shape[0])

    def identity(x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return x


class GraphPooling:
    """Graph pooling wrapper class."""

    def __init__(self, pool: Callable) -> None:
        self.pool = pool

    def __call__(self, x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return self.pool(x, batch, num_nodes)


class GCNorig(eqx.Module):
    """Original GCN layer implementation."""
    weight: jax.Array
    bias: jax.Array
    sparse:bool
    bias_flag:bool
    initializer: None

    def __init__(self, in_size, out_size, key, bias=True, sparse=False):
        self.bias_flag=bias
        self.sparse=sparse
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        ## This needs to be taken care of    output_shape = in_size + (out_size,)
        self.weight = self.initializer(wkey, (in_size, out_size))
        if self.bias_flag:
            self.bias =self.initializer(bkey, (1, out_size))
        else:
            self.bias = None

    def matmul(self, A, B, shape):
        # print("adjacency", A.shape, "node", B.shape)
        if self.sparse:
            return sp_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    def __call__(self, x, adj):
        print("adj", adj.shape, "x shape", x.shape, "weight", self.weight.shape)
        support = x @ self.weight
        print("support", support.shape)
        x = self.matmul(adj, support, support.shape)
        print("x after adj* support", x.shape)
        if self.bias_flag:
            x += self.bias
            print("x after bias", x.shape)
        return x


class GCN(eqx.Module):
    """Graph Convolutional Network layer with AWB support."""
    weight: jax.Array
    bias: jax.Array
    sparse:bool
    bias_flag:bool
    initializer: None

    def __init__(self, in_size, out_size, key, bias=True, sparse=False):
        self.bias_flag=bias
        self.sparse=sparse
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        ## This needs to be taken care of    output_shape = in_size + (out_size,)
        self.weight = self.initializer(wkey, (in_size, out_size))
        if self.bias_flag:
            self.bias =self.initializer(bkey, (1, out_size))
        else:
            self.bias = None

    def matmul(self, A, B, shape):
        # print("adjacency", A.shape, "node", B.shape)
        if self.sparse:
            return sp_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    def __call__(self, x, adj):
        #print("IN GCN: adj ", adj.shape, "x shape ", x.shape, "weight ", self.weight.shape)
        support = x @ self.weight
        #print("support", support.shape)
        x = self.matmul(adj, support, support.shape)
        if self.bias_flag:
            x += self.bias
            #print("x after bias", x.shape)
        return x


class myNNorig(eqx.Module):
    """Original myNN implementation for graph classification."""
    gcn_layers: list
    linear_layer:list
    output_layer:None
    pool_layer:None
    SEED:None
    graph:bool
    node_num:None

    def __init__(self, in_size, hid_size, node_num, SEED=1234, out_size=2, graph = True):
        self.SEED=SEED
        self.graph=graph
        self.node_num=node_num
        self.gcn_layers = [
                            GCNorig(in_size=in_size, out_size=hid_size, key=jax.random.PRNGKey(self.SEED)),
                            # GCN(in_size=hid_size, out_size=hid_size, key=jax.random.PRNGKey(self.SEED))
                            # MultiHeadGAT(n_heads = 5, in_size=in_size, out_size=hid_size, key = jax.random.PRNGKey(self.SEED))
                            #MultiHeadGAT(n_heads = 5, in_size=hid_size, out_size=hid_size, key = jax.random.PRNGKey(self.SEED))
                          ]
        self.graph = graph
        if self.graph:
            self.linear_layer = [# Linear(hid_size, hid_size, key=jax.random.PRNGKey(self.SEED)),
                               Linear(hid_size, hid_size, key=jax.random.PRNGKey(self.SEED)),
                               Linear(hid_size, hid_size, key=jax.random.PRNGKey(self.SEED))
            ]
            self.output_layer = Linear(hid_size, out_size, key=jax.random.PRNGKey(self.SEED))
        else:
            self.linear_layer = []

        self.pool_layer = GraphPooling(Pool.max)

    def __call__(self, x, adj, batch, n_nodes):
        for layer in self.gcn_layers:
            x = jax.nn.leaky_relu(layer(x, adj) )
        # ------------------------------------
        #  pooling here
        x = self.pool_layer(x, batch, n_nodes)
        # x = jnp.mean(x, axis = 0).reshape([-1, 1
        # print(x.shape)
        # print("linear")
        for layer in self.linear_layer:
            x = jax.nn.leaky_relu(layer(x))
        # print("outputs")
        x = self.output_layer(x)
        # print("out", x.shape)
        return x


class myNN(eqx.Module):
    """Graph Neural Network with GCN + MLP and AWB support."""
    gcn_layers: list
    #linear_layer:list
    #output_layer:None
    pool_layer:None
    SEED:None
    graph:bool
    node_num:None
    feed_layers: list
    feed_sizes: list
    gcn_sizes: list
    A_gcn: jax.Array
    B_gcn: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
    sparse: bool

    def __init__(self, in_size, feed_sizes, gcn_sizes, node_num, SEED=1234, out_size=2, graph=True,
                 awb_fnn_arch=None, awb_gcn_arch=None):
        """
        Args:
            in_size: Input feature size
            feed_sizes: List of feed-forward layer sizes
            gcn_sizes: List of GCN layer sizes
            node_num: Number of nodes
            SEED: Random seed
            out_size: Output size
            graph: Whether to use graph pooling
            awb_fnn_arch: AWB FNN architecture (default: [100, 140, 140, out_size])
            awb_gcn_arch: AWB GCN architecture (default: [in_size, 100])
        """
        self.SEED = SEED
        self.graph = graph
        self.node_num = node_num
        gcn_sizes[0] = in_size  # make sure the first layer is the input size
        self.gcn_sizes = gcn_sizes
        self.gcn_layers = []
        self.feed_layers = []
        feed_sizes[-1] = out_size  # make sure the last layer is the output size
        self.feed_sizes = feed_sizes

        # AWB architectures - use provided or defaults
        if awb_fnn_arch is None:
            awb_fnn_arch = DEFAULT_AWB_FNN_ARCH.copy() + [out_size]
        if awb_gcn_arch is None:
            awb_gcn_arch = [in_size] + DEFAULT_AWB_GCN_ARCH.copy()

        initializer = jax.nn.initializers.glorot_uniform()
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(self.feed_sizes[1:], awb_fnn_arch[1:])]
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(self.feed_sizes[:-1], awb_fnn_arch[:-1])]
        self.B_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(self.gcn_sizes[1:], awb_gcn_arch[1:])]
        self.A_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(self.gcn_sizes[:-1], awb_gcn_arch[:-1])]

        for i in range(0, len(self.gcn_sizes) - 1):
            self.gcn_layers.append(GCN(in_size=self.gcn_sizes[i], out_size=self.gcn_sizes[i + 1], key=jax.random.PRNGKey(SEED)))

        self.graph = graph
        self.sparse = self.gcn_layers[0].sparse
        if self.graph:
            for (in_layer, out_layer) in zip(self.feed_sizes[:-1], self.feed_sizes[1:]):
                self.feed_layers.append(Linear3(in_size=in_layer, out_size=out_layer, key=jax.random.PRNGKey(self.SEED)))
        else:
            self.feed_layers = []

        self.pool_layer = GraphPooling(Pool.max)

    def matmul(self, A, B, shape):
        # print("adjacency", A.shape, "node", B.shape)
        if self.sparse:
            return sp_matmul(A, B, shape)
        else:
            return jnp.matmul(A, B)

    def __call__(self, x, adj, batch, n_nodes):
        for layer in self.gcn_layers:
            x = jax.nn.leaky_relu(layer(x, adj) )
        # ------------------------------------
        #  pooling here
        x = self.pool_layer(x, batch, n_nodes)
        # x = jnp.mean(x, axis = 0).reshape([-1, 1
        # print(x.shape)
        # print("linear")
        for i in range(0,len(self.feed_sizes)-2):
            x = jax.nn.leaky_relu(self.feed_layers[i](x))
        #for layer in self.linear_layer:
            #x = jax.nn.leaky_relu(layer(x))
        # print("outputs")
        #x = self.output_layer(x)
        x = self.feed_layers[-1](x)
        # print("out", x.shape)
        return x

    def get_AWBT(self, x, adj, batch, n_nodes):
        """Forward pass using AWB transformation."""
        for i in range(len(self.gcn_layers)):
            #print("A_gch[i] shape: ", self.A_gcn[i].shape)
            #print("weights shape: ", self.gcn_layers[i].weight.shape)
            #print("B_gcn[i].T shape: ", self.B_gcn[i].T.shape)
            #print("shape of AW: ", (self.A_gcn[i] @ self.gcn_layers[i].weight).shape)
            #print('shape AWB: ', (self.A_gcn[i] @ self.gcn_layers[i].weight @ jnp.transpose(self.B_gcn[i])).shape)
            support = x@ (self.A_gcn[i] @ self.gcn_layers[i].weight @ jnp.transpose(self.B_gcn[i]))
            #print('support shape: ', support.shape)
            x = self.matmul(adj, support, support.shape)
            #print("x after adj* support", x.shape)
            if self.gcn_layers[i].bias_flag:
                new_bias = (self.gcn_layers[i].bias@ self.B_gcn[i].T) #.squeeze(1)
                #print("new bias shape: ", new_bias.shape)
                x += (self.gcn_layers[i].bias @ self.B_gcn[i].T) #squeeze(1)
            x = jax.nn.leaky_relu(x)
        # ------------------------------------
        #  pooling here
        x = self.pool_layer(x, batch, n_nodes)

        for i in range(0,len(self.feed_sizes)-2):
            #print("i: ", i)
            #print("after pooling: ", x.shape)
            #print("A: ", self.A_feed[i].shape)
            #print("W: ", self.feed_layers[i].weight.shape)
            #print("B.T: ", jnp.transpose(self.B_feed[i]).shape)
            #print("AWBT: ", (self.A_feed[i] @ self.feed_layers[i].weight.T @ jnp.transpose(self.B_feed[i])).shape)
            #print("AWBTx: ", (x@(self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]))).shape)
            #print("bias part: ", (self.feed_layers[i].bias@self.B_feed[i].T).shape)
            x = x@(self.A_feed[i] @ self.feed_layers[i].weight.T @ jnp.transpose(self.B_feed[i])) + (self.feed_layers[i].bias@ self.B_feed[i].T) #.squeeze(1)
            x = jax.nn.leaky_relu(x)
            #print("after activation function: ", x.shape)
        #print("last row AWBT shape: ", (self.A_feed[-1] @ self.feed_layers[-1].weight.T @ jnp.transpose(self.B_feed[-1])).shape)
        #print("last row bias: ", (self.feed_layers[-1].bias@self.B_feed[-1].T).shape)
        x = x@(self.A_feed[-1] @ self.feed_layers[-1].weight.T @ jnp.transpose(self.B_feed[-1])) + (self.feed_layers[-1].bias@self.B_feed[-1].T) #.squeeze(1)
        #print(x.shape)
        return x
