#------------------------------------------------------------------------------------------------------------------------------------------------
# imports
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from functools import partial

import numpy as np_
from jax import lax
import diffrax
import equinox as eqx
## Train now a CNN and test the trainer and then, the older model

from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
############################################################################################################################
@partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
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



# #---------------------------------------------------------------------------------------------------------
## MLP layer
# #---------------------------------------------------------------------------------------------------------
class MLP(eqx.Module):
    input_layer: None
    feed_layers: list
    output_layers: None
    
    # outfunc__: jax.nn.relu
    def __init__(self, key, input_dim=100, out_dim=100, n_layers=4, hln=200):
        self.input_layer = eqx.nn.Linear(
            input_dim, hln, key=jax.random.PRNGKey(key))
        self.feed_layers = [eqx.nn.Linear(
            hln, hln, key=jax.random.PRNGKey(i)) for i in range(1, n_layers-2)]
        self.output_layers =  eqx.nn.Linear( hln, out_dim, key=jax.random.PRNGKey(n_layers-1))

    def __call__(self, x, actfunc__=jax.nn.tanh, outfunc=None):
        x = actfunc__(self.input_layer(x))
        for element in self.feed_layers:
            x = actfunc__(element(x))
        if outfunc is None:
            return self.output_layers(x)
        else:
            return outfunc(self.output_layers(x))
    


# # class CNN(eqx.Module):
# #     layers: list

# #     def __init__(self, key):
# #         key1, key2, key3, key4 = jax.random.split(key, 4)
# #         # Standard CNN setup: convolutional layer, followed by flattening,
# #         # with a small MLP on top.
# #         self.layers = [
# #             eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
# #             eqx.nn.MaxPool2d(kernel_size=2),
# #             jax.nn.relu,
# #             jnp.ravel,
# #             eqx.nn.Linear(1728, 512, key=key2),
# #             jax.nn.sigmoid,
# #             eqx.nn.Linear(512, 64, key=key3),
# #             jax.nn.relu,
# #             eqx.nn.Linear(64, 10, key=key4),
# #             jax.nn.log_softmax,
# #         ]

# #     def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
# #         print(x.shape)
# #         for layer in self.layers:
# #             x = layer(x)
#         return x


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Dropout Layer
class Dropout(eqx.Module):
    rate: float 
    def __init__(self, rate=0.5):
        self.rate=rate       
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.

    Arguments:
        rate (float): Probability of keeping and element.
    """
    def __call__ (self, inputs, rng, is_training=True):
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        # print(self.rate)
        keep = jax.random.bernoulli(rng, self.rate, shape = inputs.shape)
        # print(keep)
        outs = jnp.where(keep, inputs / self.rate, 0)
        # if not training, just return inputs and discard any computation done
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out
    
#------------------------------------------------------------------------------------------------------------------------------------------------
# Single Head GAT Layer 
class SingleHeadGAT(eqx.Module):
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
    
#---------------------------------------------------------------------------------------------------------
# Multi Head GAT Layer
class MultiHeadGAT(eqx.Module):
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


# #---------------------------------------------------------------------------------------------------------
## GCN Layers  
class CNN(eqx.Module):
    conv_layers: list
    feed_layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.conv_layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            ]
        self.feed_layers =[
            eqx.nn.Linear(1728, 512, key=key2),
            eqx.nn.Linear(512, 64, key=key3),
            eqx.nn.Linear(64, 10, key=key4),
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:     
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2)(self.conv_layers[0](x))))
        x = jax.nn.relu(self.feed_layers[0](x))
        x = jax.nn.relu(self.feed_layers[1](x))
        x = self.feed_layers[2](x)
        return x
    
    
class Pool:
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

#---------------------------------------------------------------------------------------------------------
# graph pooling wrapper
from typing import Any, Callable
class GraphPooling:
    def __init__(self, pool: Callable) -> None:
        self.pool = pool

    def __call__(self, x: jnp.ndarray, batch: jnp.ndarray, num_nodes: jnp.ndarray) -> jnp.ndarray:
        return self.pool(x, batch, num_nodes)
    

# ---------------------------------------------------------------------------------------------------------
## GCN Layers
class GCN(eqx.Module):
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
        # print("adj", adj.shape, "x shape", x.shape, "weight", self.weight.shape)
        support = x @ self.weight
        # print("support", support.shape)
        x = self.matmul(adj, support, support.shape)
        if self.bias_flag:
            x += self.bias
        return x

# ---------------------------------------------------------------------------------------------------------
## Simple feedforward NN
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (1, out_size))
    def __call__(self, x):
        # print(self.weight.shape, x.shape)
        x = x @ self.weight.T
        # print(x.shape)
        x = x+ self.bias
        # print(x.shape)
        return x
    


# ---------------------------------------------------------------------------------------------------------
class myNN(eqx.Module):
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
                        GCN(in_size=in_size, out_size=hid_size, key=jax.random.PRNGKey(self.SEED)),
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