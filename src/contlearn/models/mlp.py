"""Multi-Layer Perceptron models for continual learning."""

import jax
import jax.numpy as jnp
import equinox as eqx

from .layers import Linear


class MLPorig(eqx.Module):
    """Original MLP implementation using Equinox linear layers."""
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


class MLP(eqx.Module):
    """Multi-Layer Perceptron with AWB (Adaptive Weight Basis) support."""
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
        """Forward pass using AWB transformation: A @ W @ B.T"""
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
