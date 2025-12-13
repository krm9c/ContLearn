"""Basic neural network layers for ContLearn models."""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax


class Linear(eqx.Module):
    """Linear layer with shape (out_size, in_size) and bias shape (1, out_size)."""
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


class Linear1(eqx.Module):
    """Linear layer variant with bias shape (out_size, 1)."""
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


class Linear2(eqx.Module):
    """Linear layer variant with bias shape (out_size, 1) and different computation."""
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
    """Linear layer variant with bias shape (1, out_size)."""
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


class Dropout(eqx.Module):
    """Dropout layer with configurable rate.

    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.
    """
    rate: float

    def __init__(self, rate=0.5):
        self.rate=rate

    def __call__ (self, inputs, rng, is_training=True):
        """
        Arguments:
            inputs: Input array
            rng: PRNG key (required)
            is_training: Whether to apply dropout (default: True)
        """
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
