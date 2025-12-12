#------------------------------------------------------------------------------------------------------------------------------------------------
# imports
import matplotlib.pyplot as plt
import jax
import numpy as np
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
## MLP orig layer
# #---------------------------------------------------------------------------------------------------------
class MLPorig(eqx.Module):
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

 #------------------------NEW---------------------------------------------------
 class MLP(eqx.Module):
    layers: list
    sizes: list
    #act_fn: Callable
    #act_options: list
    A: jax.Array
    B: jax.Array
    
    def __init__(self, sizes):
        self.layers = []
        #self.act_fn = act_fn
        #self.act_options = act_options
        self.A = [jax.random.normal(jax.random.PRNGKey(0),shape = (y,1)) for y in sizes[1:]]
        self.B = [jax.random.normal(jax.random.PRNGKey(0),shape = (y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        i = 0
        self.sizes  = sizes
        for (in_layer,out_layer) in zip(sizes[:-1],sizes[1:]):
            self.layers.append(Linear(in_layer,out_layer, key = jax.random.PRNGKey(i)))
            i+=1

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.tanh(lin(x))
        x = self.layers[-1](x)
        return x

    def getAWB(self,x):
        for i in range(0,len(self.sizes)-1):
            #x=(model.A[i] @ model.layers[i].weight @ jnp.transpose(model.B[i]))
            #x = (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x) + (self.A[i] @ self.layers[i].bias)
            #print("this is the shape of AWB.T each time: ", (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i])).shape)
            #print("this is the shape of AWB.Tx each time: ", (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x).shape)
            #print("this is the shape of new bias each time: ", ((self.layers[i].bias @ self.A[i].T).T.squeeze(1)).shape)
            x = (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x) + (self.layers[i].bias @ self.A[i].T).T.squeeze(1)
            x = jax.nn.tanh(x)
            #print("this is the shape of x each time: ", x.shape)
        return x
        
class Linear1(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_normal()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (out_size, 1))
        
    def __call__(self, x):
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x


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
class CNNorig(eqx.Module):
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
    

#---------My New CNN---------------#
class CNN(eqx.Module):
    conv_layers: list
    feed_layers: list
    A_conv: jax.Array
    B_conv: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
    filter_size: int
    channel_out: int

    def __init__(self, key, filter_size, feed_sizes):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        i=0
        self.feed_sizes = feed_sizes
        self.filter_size = filter_size
        self.feed_layers = []
        self.channel_out = 3
        self.conv_layers = [
            eqx.nn.Conv2d(1,self.channel_out, kernel_size=filter_size, key=key1),
            ]
        for (in_layer,out_layer) in zip(feed_sizes[:-1],feed_sizes[1:]):
            self.feed_layers.append(Linear2(in_layer,out_layer, key = jax.random.PRNGKey(i)))
            i+=1
        new_arch = [1875,700,100,10]
        initializer = jax.nn.initializers.glorot_uniform()
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[1:],new_arch[1:])]
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[:-1],new_arch[:-1])]
        #below should produce the three weights filters
        new_filter_size = 5
        self.B_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]
        self.A_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]

    def calc_output_size(self,fil_size):
        omni_input = 28
        padding = 0 #this is apparently the default for Conv2d
        stride = 1
        output = ((omni_input-fil_size+2*padding)/stride) + 1
        return int(output)
    
    def pool_output_size(self,pool_size,conv_inputsize):
        stride = 1
        output = ((conv_inputsize-pool_size)/stride) + 1
        return int(output)

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:  
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2)(self.conv_layers[0](x))))
        for lin in self.feed_layers[:-1]:
            #print("x in model:", x.shape)
            x = jax.nn.relu(lin(x))
        x = self.feed_layers[-1](x)
        #for lin in self.feed_layers:
            #x = jax.nn.relu(lin(x))
        #x = self.feed_layers[0](x)
        return x
        #x = jax.nn.relu(self.feed_layers[0](x))
        #x = jax.nn.relu(self.feed_layers[1](x))
        #x = self.feed_layers[2](x)
        #return x


    def get_AWBT(self,x):
        #print("x before: ", x.shape)
        #print("shape of A_conv: ", self.A_conv[0].shape)
        #print("shape of B_conv: ", self.B_conv[0].shape)
        #print("shape of weights: ", self.conv_layers[0].weight)
        #print("before weights: ", self.conv_layers[0].weight)
        #rint("shape of A_conv@weights: ", (self.A_conv[0]@self.conv_layers[0].weight).shape)
        #rint("weights after: ", self.A_conv[0]@self.conv_layers[0].weight)
        weights_list = [[(self.A_conv[i]@(self.conv_layers[0].weight[i][0])@jnp.transpose(self.B_conv[i]))] for i in range(0,self.channel_out)]
        #print("weights list shape: ", jnp.array(weights_list).shape)
        x = jnp.expand_dims(x, axis=0)
        #print("x: ", x.shape)
        x = jax.lax.conv_general_dilated(lhs = x, rhs = jnp.array(weights_list), window_strides= (1,1), padding="VALID")
        #print("shape of x after:", x.shape)
        x = x.squeeze(0)
        #print("x: ", x.shape)
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2)(x)))
        #print("x after: ", x.shape)
        for i in range(0,len(self.feed_sizes)-1):
            #print(x.shape)
            #print("AWBTx: ", (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x).shape)
            #print("bias part: ", (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1).shape)
            x = (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x) + (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1)
            #print("after: ", x.shape)
            x = jax.nn.relu(x)
            #print(x.shape)
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
class GCNorig(eqx.Module):
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
    
class Linear2(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (out_size,1))
    def __call__(self, x):
        #print("w,x: ", self.weight.shape, x.shape)
        x = self.weight @ x        
        #print("x", x.shape)
        x = x+ self.bias.squeeze(1)
        #print("after bias: ", x.shape)
        return x

class Linear3(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (1,out_size))
    def __call__(self, x):
        #print("x@wT: ", x.shape, self.weight.T.shape)
        x = x@ self.weight.T     
        #print("x", x.shape)
        x = x+ self.bias
        #print("after bias: ", x.shape)
        return x
    
# ------------------------------------------------
class myNNorig(eqx.Module):
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



#================= GCN NEW ========================#
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
        #print("IN GCN: adj ", adj.shape, "x shape ", x.shape, "weight ", self.weight.shape)
        support = x @ self.weight
        #print("support", support.shape)
        x = self.matmul(adj, support, support.shape)
        if self.bias_flag:
            x += self.bias
            #print("x after bias", x.shape)
        return x
    

#-------------------myNN New-------------------#
class myNN(eqx.Module):
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
    def __init__(self, in_size, feed_sizes, gcn_sizes, node_num, SEED=1234, out_size=2, graph = True):
        self.SEED=SEED
        self.graph=graph
        self.node_num=node_num
        gcn_sizes[0] = in_size # make sure the first layer is the input size
        self.gcn_sizes = gcn_sizes
        self.gcn_layers = []
        self.feed_layers = []
        feed_sizes[-1] = out_size # make sure the last layer is the output size
        self.feed_sizes = feed_sizes
        #self.in_size = in_size

        new_FNNarch = [100,140,140,out_size] #in other file set new_arch[0] = gcn_arch[-1]
        initializer = jax.nn.initializers.glorot_uniform()
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(self.feed_sizes[1:],new_FNNarch[1:])]
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(self.feed_sizes[:-1],new_FNNarch[:-1])]
        gcn_arch = [in_size,100]
        self.B_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(self.gcn_sizes[1:],gcn_arch[1:])]
        self.A_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(self.gcn_sizes[:-1],gcn_arch[:-1])]

        for i in range(0,len(self.gcn_sizes)-1):
            self.gcn_layers.append(GCN(in_size=self.gcn_sizes[i], out_size=self.gcn_sizes[i+1], key=jax.random.PRNGKey(SEED)))
        #self.gcn_layers = [GCN(in_size=in_size, out_size=hid_size, key=jax.random.PRNGKey(self.SEED))]
        self.graph = graph
        self.sparse = self.gcn_layers[0].sparse
        if self.graph:
            for (in_layer, out_layer) in zip(self.feed_sizes[:-1], self.feed_sizes[1:]):
                self.feed_layers.append(Linear3(in_size=in_layer, out_size=out_layer, key=jax.random.PRNGKey(self.SEED)))
            #self.linear_layer = [Linear(hid_size, hid_size, key=jax.random.PRNGKey(self.SEED)),\
                                #Linear(hid_size, hid_size, key=jax.random.PRNGKey(self.SEED))]
            #self.output_layer = Linear(hid_size, out_size, key=jax.random.PRNGKey(self.SEED))
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

