"""Convolutional Neural Network models for image classification."""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from ..config.constants import (
    DEFAULT_CHANNEL_OUT_CNN,
    DEFAULT_CHANNEL_OUT_CNN3D,
    DEFAULT_CHANNEL_IN_MNIST,
    DEFAULT_CHANNEL_IN_CIFAR,
    DEFAULT_INPUT_SIZE_MNIST,
    DEFAULT_INPUT_SIZE_CIFAR,
    DEFAULT_AWB_FILTER_INCREMENT,
    DEFAULT_AWB_CNN_ARCH,
    DEFAULT_AWB_CNN3D_HIDDEN,
    DEFAULT_PADDING,
    DEFAULT_STRIDE,
    DEFAULT_POOL_SIZE,
    DEFAULT_POOL_STRIDE,
)

from .layers import Linear2, AWBLayerSpec, compute_V_conv2d_single_channel, compute_V_conv2d_multi_channel
import jax.tree_util as jtu
from typing import List


class CNNorig(eqx.Module):
    """Original CNN implementation for MNIST."""
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
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2, stride=2)(self.conv_layers[0](x))))
        x = jax.nn.relu(self.feed_layers[0](x))
        x = jax.nn.relu(self.feed_layers[1](x))
        x = self.feed_layers[2](x)
        return x


class CNN(eqx.Module):
    """CNN with AWB (Adaptive Weight Basis) support for MNIST/Omniglot."""
    conv_layers: list
    feed_layers: list
    A_conv: jax.Array
    B_conv: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
    filter_size: int
    channel_out: int
    channel_in: int
    input_size: int
    padding: int
    stride: int

    def __init__(self, key, filter_size, feed_sizes,
                 input_size=None,
                 channel_in=1,
                 channel_out=None,
                 awb_arch=None,
                 awb_filter_size=None,
                 padding=None,
                 stride=None):
        """
        Args:
            key: PRNG key
            filter_size: Convolutional filter size
            feed_sizes: List of feed-forward layer sizes
            input_size: Input image size (default: 28 for MNIST/Omniglot)
            channel_in: Number of input channels (default: 1)
            channel_out: Number of output channels (default: 3)
            awb_arch: AWB architecture (default: [1875, 700, 100, 10])
            awb_filter_size: AWB filter size (default: 5)
            padding: Convolution padding (default: 0)
            stride: Convolution stride (default: 1)
        """
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Set defaults from constants
        self.input_size = input_size if input_size is not None else DEFAULT_INPUT_SIZE_MNIST
        self.channel_in = channel_in
        self.channel_out = channel_out if channel_out is not None else DEFAULT_CHANNEL_OUT_CNN
        self.padding = padding if padding is not None else DEFAULT_PADDING
        self.stride = stride if stride is not None else DEFAULT_STRIDE

        i=0
        self.feed_sizes = feed_sizes
        self.filter_size = filter_size
        self.feed_layers = []
        self.conv_layers = [
            eqx.nn.Conv2d(self.channel_in, self.channel_out, kernel_size=filter_size, key=key1),
        ]
        for (in_layer,out_layer) in zip(feed_sizes[:-1],feed_sizes[1:]):
            self.feed_layers.append(Linear2(in_layer,out_layer, key = jax.random.PRNGKey(i)))
            i+=1

        # AWB architecture - use provided or default
        new_arch = awb_arch if awb_arch is not None else DEFAULT_AWB_CNN_ARCH.copy()
        # Ensure new_arch has the same number of layers as feed_sizes
        # If new_arch has more layers, truncate the middle hidden layers to match
        if len(new_arch) != len(feed_sizes):
            # Keep input size, output size, and adjust hidden layers
            num_hidden = len(feed_sizes) - 2
            if num_hidden > 0:
                # Take first num_hidden elements from the new_arch hidden layers
                new_arch = [new_arch[0]] + new_arch[1:1+num_hidden] + [new_arch[-1]]
            else:
                # No hidden layers, just input and output
                new_arch = [new_arch[0], new_arch[-1]]

        initializer = jax.nn.initializers.glorot_uniform()
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[1:],new_arch[1:])]
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[:-1],new_arch[:-1])]

        # AWB convolution filters
        new_filter_size = awb_filter_size if awb_filter_size is not None else (filter_size + DEFAULT_AWB_FILTER_INCREMENT)
        self.B_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]
        self.A_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]

    def calc_output_size(self, fil_size, input_size=None):
        """Calculate output size after convolution"""
        if input_size is None:
            input_size = self.input_size
        output = ((input_size - fil_size + 2 * self.padding) / self.stride) + 1
        return int(output)

    def pool_output_size(self, pool_size, conv_inputsize, pool_stride=None):
        """Calculate output size after pooling"""
        if pool_stride is None:
            pool_stride = DEFAULT_POOL_STRIDE
        output = ((conv_inputsize - pool_size) / pool_stride) + 1
        return int(output)

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2, stride=2)(self.conv_layers[0](x))))
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
        """Forward pass using AWB transformation."""
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
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)))
        #print("x after: ", x.shape)
        for i in range(0,len(self.feed_sizes)-1):
            #print(x.shape)
            #print("AWBTx: ", (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x).shape)
            #print("bias part: ", (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1).shape)
            x = (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x) + (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1)
            #print("after: ", x.shape)
            # Apply relu to all layers except the final output layer
            if i < len(self.feed_sizes) - 2:
                x = jax.nn.relu(x)
            #print(x.shape)
        return x

    # Added by Claude: AWBModel interface for CNN
    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB specs for feed layers only (conv handled separately)."""
        return [
            AWBLayerSpec(layer=self.feed_layers[i], A=self.A_feed[i], B=self.B_feed[i],
                        layer_type='linear2', layer_index=i)
            for i in range(len(self.feed_layers))
        ]

    def partition_for_AB_training(self):
        """Partition for A/B training (freeze W, train A/B)."""
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
                                  filter_spec, replace=(True, True, True, True))
        return eqx.partition(self, filter_spec)

    def partition_for_standard_training(self):
        """Partition for standard training (freeze A/B, train W)."""
        params, static = eqx.partition(self, eqx.is_array)
        static = eqx.tree_at(lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed), static,
                            replace=(self.A_conv, self.B_conv, self.A_feed, self.B_feed))
        params = eqx.tree_at(lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed), params,
                            replace=(None, None, None, None))
        return params, static


class CNN3D(eqx.Module):
    """CNN3D with AWB support for CIFAR-10/100 (3-channel 32x32 images)."""
    conv_layers: list
    feed_layers: list
    A_conv: jax.Array
    B_conv: jax.Array
    A_conv1: jax.Array
    B_conv1: jax.Array
    A_conv2: jax.Array
    B_conv2: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
    filter_size: int
    channel_in: int
    channel_out: int
    input_size: int

    def __init__(self, key, filter_size, feed_sizes,
                 input_size=None,
                 channel_in=None,
                 channel_out=None,
                 num_classes=10,
                 awb_filter_increment=None,
                 awb_hidden_layers=None):
        """
        Args:
            key: PRNG key
            filter_size: Convolutional filter size
            feed_sizes: List of feed-forward layer sizes
            input_size: Input image size (default: 32 for CIFAR)
            channel_in: Number of input channels (default: 3)
            channel_out: Number of output channels (default: 32)
            num_classes: Number of output classes (default: 10)
            awb_filter_increment: Increment for AWB filter size (default: 2)
            awb_hidden_layers: AWB hidden layer sizes (default: [512, 256])
        """
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        # Set defaults from constants
        if input_size is None:
            input_size = DEFAULT_INPUT_SIZE_CIFAR
        if channel_in is None:
            channel_in = DEFAULT_CHANNEL_IN_CIFAR
        if channel_out is None:
            channel_out = DEFAULT_CHANNEL_OUT_CNN3D
        if awb_filter_increment is None:
            awb_filter_increment = DEFAULT_AWB_FILTER_INCREMENT
        if awb_hidden_layers is None:
            awb_hidden_layers = DEFAULT_AWB_CNN3D_HIDDEN.copy()

        i = 0
        self.feed_sizes = feed_sizes
        self.filter_size = filter_size
        self.feed_layers = []
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.input_size = input_size

        # Two conv layers for better feature extraction
        self.conv_layers = [
            eqx.nn.Conv2d(channel_in, channel_out, kernel_size=filter_size, key=key1),
            eqx.nn.Conv2d(channel_out, channel_out * 2, kernel_size=filter_size, key=key2),
        ]

        for (in_layer, out_layer) in zip(feed_sizes[:-1], feed_sizes[1:]):
            self.feed_layers.append(Linear2(in_layer, out_layer, key=jax.random.PRNGKey(i)))
            i += 1

        # AWB transformation matrices for architecture search
        # Calculate the flattened size after AWB convolutions
        new_filter_size = filter_size + awb_filter_increment
        after_awb_conv1 = (input_size - new_filter_size + 1) // 2  # conv1 + pool
        after_awb_conv2 = (after_awb_conv1 - new_filter_size + 1) // 2  # conv2 + pool
        awb_flatten_size = after_awb_conv2 * after_awb_conv2 * channel_out * 2

        # Match the number of hidden layers in the original architecture
        # feed_sizes has (len - 2) hidden layers, new_arch should have the same
        num_hidden_layers = len(feed_sizes) - 2
        awb_hidden_subset = awb_hidden_layers[:num_hidden_layers]
        new_arch = [awb_flatten_size] + awb_hidden_subset + [num_classes]
        initializer = jax.nn.initializers.glorot_uniform()
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(feed_sizes[1:], new_arch[1:])]
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(feed_sizes[:-1], new_arch[:-1])]

        # Conv AWB matrices for first conv layer (channel_in -> channel_out)
        # For 3-channel input: each output filter has channel_in input channels
        # A_conv1[i][c] transforms the (i,c) filter: shape (new_filter_size, filter_size)
        self.A_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * channel_in + c), shape=(new_filter_size, filter_size))
                         for c in range(channel_in)] for j in range(channel_out)]
        self.B_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * channel_in + c + 100), shape=(new_filter_size, filter_size))
                         for c in range(channel_in)] for j in range(channel_out)]

        # Conv AWB matrices for second conv layer (channel_out -> channel_out * 2)
        self.A_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * channel_out + c + 200), shape=(new_filter_size, filter_size))
                         for c in range(channel_out)] for j in range(channel_out * 2)]
        self.B_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * channel_out + c + 300), shape=(new_filter_size, filter_size))
                         for c in range(channel_out)] for j in range(channel_out * 2)]

        # Keep old attributes for backward compatibility
        self.A_conv = self.A_conv1
        self.B_conv = self.B_conv1

    def calc_output_size(self, input_size, fil_size, pool_size=2):
        """Calculate output size after convolution and pooling."""
        # After conv: (input_size - fil_size + 1)
        # After pool: floor((conv_out) / pool_size)
        conv_out = input_size - fil_size + 1
        pool_out = conv_out // pool_size
        return pool_out

    def __call__(self, x: Float[Array, "3 32 32"]) -> Float[Array, "num_classes"]:
        # First conv + pool
        x = jax.nn.relu(self.conv_layers[0](x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # Second conv + pool
        x = jax.nn.relu(self.conv_layers[1](x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # Flatten
        x = jnp.ravel(x)
        # Feed forward layers
        for lin in self.feed_layers[:-1]:
            x = jax.nn.relu(lin(x))
        x = self.feed_layers[-1](x)
        return x

    def get_AWBT(self, x):
        """Forward pass using AWB transformation."""
        # AWB transformation on first conv layer (channel_in -> channel_out)
        # For each output filter i, apply AWB to each input channel c: A[i][c] @ W[i][c] @ B[i][c].T
        weights_list1 = [[(self.A_conv1[i][c] @ self.conv_layers[0].weight[i][c] @ jnp.transpose(self.B_conv1[i][c]))
                          for c in range(self.channel_in)] for i in range(self.channel_out)]
        x = jnp.expand_dims(x, axis=0)
        x = jax.lax.conv_general_dilated(lhs=x, rhs=jnp.array(weights_list1), window_strides=(1, 1), padding="VALID")
        x = x.squeeze(0)
        x = jax.nn.relu(x)
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # AWB transformation on second conv layer (channel_out -> channel_out * 2)
        weights_list2 = [[(self.A_conv2[i][c] @ self.conv_layers[1].weight[i][c] @ jnp.transpose(self.B_conv2[i][c]))
                          for c in range(self.channel_out)] for i in range(self.channel_out * 2)]
        x = jnp.expand_dims(x, axis=0)
        x = jax.lax.conv_general_dilated(lhs=x, rhs=jnp.array(weights_list2), window_strides=(1, 1), padding="VALID")
        x = x.squeeze(0)
        x = jax.nn.relu(x)
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = jnp.ravel(x)

        # AWB transformation on feed layers
        for i in range(0, len(self.feed_sizes) - 1):
            x = (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x) + (self.A_feed[i] @ self.feed_layers[i].bias).squeeze(1)
            # Apply relu to all layers except the final output layer
            if i < len(self.feed_sizes) - 2:
                x = jax.nn.relu(x)
        return x

    # Added by Claude: AWBModel interface for CNN3D
    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB specs for feed layers only (conv handled separately)."""
        return [
            AWBLayerSpec(layer=self.feed_layers[i], A=self.A_feed[i], B=self.B_feed[i],
                        layer_type='linear2', layer_index=i)
            for i in range(len(self.feed_layers))
        ]

    def partition_for_AB_training(self):
        """Partition for A/B training (freeze W, train A/B)."""
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                                  filter_spec, replace=(True, True, True, True, True, True))
        return eqx.partition(self, filter_spec)

    def partition_for_standard_training(self):
        """Partition for standard training (freeze A/B, train W)."""
        params, static = eqx.partition(self, eqx.is_array)
        static = eqx.tree_at(lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed), static,
                            replace=(self.A_conv1, self.B_conv1, self.A_conv2, self.B_conv2, self.A_feed, self.B_feed))
        params = eqx.tree_at(lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed), params,
                            replace=(None, None, None, None, None, None))
        return params, static
