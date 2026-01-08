"""
Basic neural network layers for continual learning models.

Provides standard building blocks used across different model architectures.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# Unified AWB Transform (Added by Claude)
# =============================================================================

def awb_transform(A: jax.Array, W: jax.Array, B: jax.Array) -> jax.Array:
    """Unified AWB weight transformation: V = A @ W @ B.T

    Works for any number of batch dimensions using einsum with ellipsis.
    This enables efficient batched computation for:
    - MLP: 2D matrices (no batch dims)
    - CNN: 3D matrices (batch over output channels)
    - CNN3D: 4D matrices (batch over output and input channels)
    - Transformers: arbitrary batch dims (layers, heads, etc.)

    Args:
        A: Transformation matrix with shape (..., new_out, old_out)
        W: Weight matrix with shape (..., old_out, old_in)
        B: Transformation matrix with shape (..., new_in, old_in)

    Returns:
        Transformed weight V with shape (..., new_out, new_in)

    Example:
        # Single layer (MLP-style)
        V = awb_transform(A, W, B)  # A: (m,k), W: (k,n), B: (p,n) -> V: (m,p)

        # Batched over channels (CNN3D-style)
        V = awb_transform(A, W, B)  # A: (o,i,m,k), W: (o,i,k,n), B: (o,i,p,n) -> V: (o,i,m,p)
    """
    return jnp.einsum('...ij,...jk,...lk->...il', A, W, B)


# Added by Claude: AWBLayerSpec for tracking AWB transformations
@dataclass
class AWBLayerSpec:
    """Specification for AWB transformation of a layer.

    Args:
        layer: The layer module to be transformed
        A: A matrix for output dimension transformation (new_out, old_out)
        B: B matrix for input dimension transformation (new_in, old_in)
        layer_type: Type identifier ('linear', 'linear2', 'linear_gcn', 'conv2d', etc.)
        layer_index: Index of this layer in the model
    """
    layer: eqx.Module
    A: Optional[jax.Array]
    B: Optional[jax.Array]
    layer_type: str
    layer_index: int

    def validate(self) -> List[str]:
        """Validate AWB matrix shapes against layer dimensions.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if self.A is not None and hasattr(self.layer, 'weight'):
            # For most layers: A transforms output dimension
            expected_A_cols = self.layer.weight.shape[0]  # old_out
            if self.A.shape[1] != expected_A_cols:
                errors.append(
                    f"Layer {self.layer_index} ({self.layer_type}): "
                    f"A.shape[1]={self.A.shape[1]} != weight.shape[0]={expected_A_cols}"
                )

        if self.B is not None and hasattr(self.layer, 'weight'):
            # For most layers: B transforms input dimension
            expected_B_cols = self.layer.weight.shape[1]  # old_in
            if self.B.shape[1] != expected_B_cols:
                errors.append(
                    f"Layer {self.layer_index} ({self.layer_type}): "
                    f"B.shape[1]={self.B.shape[1]} != weight.shape[1]={expected_B_cols}"
                )

        return errors


# Added by Claude: Custom exception for AWB shape errors
class AWBShapeError(ValueError):
    """Raised when AWB matrix shapes are incompatible with layer dimensions."""
    pass


class Linear(eqx.Module):
    """Linear layer with Glorot uniform initialization.

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

    # Added by Claude: AWB methods for Linear layer
    def compute_V_weight(self, A: jax.Array, W: jax.Array, B: jax.Array) -> jax.Array:
        """Compute transformed weight V = A @ W @ B.T for Linear layer.

        Args:
            A: Output transformation matrix (new_out, old_out)
            W: Original weight matrix (old_out, old_in)
            B: Input transformation matrix (new_in, old_in)

        Returns:
            Transformed weight V with shape (new_out, new_in)
        """
        # Validate shapes
        if A.shape[1] != W.shape[0]:
            raise AWBShapeError(
                f"Linear layer: A.shape={A.shape} incompatible with W.shape={W.shape}. "
                f"Expected A.shape[1]={A.shape[1]} == W.shape[0]={W.shape[0]}"
            )
        if B.shape[1] != W.shape[1]:
            raise AWBShapeError(
                f"Linear layer: B.shape={B.shape} incompatible with W.shape={W.shape}. "
                f"Expected B.shape[1]={B.shape[1]} == W.shape[1]={W.shape[1]}"
            )

        return A @ W @ jnp.transpose(B)

    def compute_V_bias(self, A: jax.Array, B: jax.Array, bias: jax.Array) -> jax.Array:
        """Compute transformed bias for Linear layer.

        For Linear layers with bias shape (1, out_size), transformation is: bias @ A.T

        Args:
            A: Output transformation matrix (new_out, old_out)
            B: Input transformation matrix (not used for bias in Linear)
            bias: Original bias vector (1, old_out)

        Returns:
            Transformed bias with shape (1, new_out)
        """
        # Linear layer: bias @ A.T (because bias shape is (1, out))
        return bias @ A.T


class LinearGCN(eqx.Module):
    """Linear layer variant for GCN with different bias shape.

    Weight shape: (out_size, in_size)
    Bias shape: (out_size, 1)

    Forward: W @ x + bias

    Args:
        in_size: Input dimension
        out_size: Output dimension
        key: JAX PRNG key for initialization
    """
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        initializer = jax.nn.initializers.glorot_normal()
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size))
        self.bias = initializer(bkey, (out_size, 1))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: W @ x + bias"""
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x

    # Added by Claude: AWB methods for LinearGCN layer
    def compute_V_weight(self, A: jax.Array, W: jax.Array, B: jax.Array) -> jax.Array:
        """Compute transformed weight V = A @ W @ B.T for LinearGCN layer.

        Args:
            A: Output transformation matrix (new_out, old_out)
            W: Original weight matrix (old_out, old_in)
            B: Input transformation matrix (new_in, old_in)

        Returns:
            Transformed weight V with shape (new_out, new_in)
        """
        # Validate shapes
        if A.shape[1] != W.shape[0]:
            raise AWBShapeError(
                f"LinearGCN layer: A.shape={A.shape} incompatible with W.shape={W.shape}. "
                f"Expected A.shape[1]={A.shape[1]} == W.shape[0]={W.shape[0]}"
            )
        if B.shape[1] != W.shape[1]:
            raise AWBShapeError(
                f"LinearGCN layer: B.shape={B.shape} incompatible with W.shape={W.shape}. "
                f"Expected B.shape[1]={B.shape[1]} == W.shape[1]={W.shape[1]}"
            )

        return A @ W @ jnp.transpose(B)

    def compute_V_bias(self, A: jax.Array, B: jax.Array, bias: jax.Array) -> jax.Array:
        """Compute transformed bias for LinearGCN layer.

        For LinearGCN with bias shape (out_size, 1), transformation is: bias @ B.T

        Args:
            A: Output transformation matrix (not used for bias in LinearGCN)
            B: Input transformation matrix (new_in, old_in)
            bias: Original bias vector (old_out, 1)

        Returns:
            Transformed bias with shape (new_out, 1)
        """
        # LinearGCN layer: bias @ B.T (because bias shape is (out, 1))
        return bias @ B.T


class Linear2(eqx.Module):
    """Linear layer variant with bias shape (out_size, 1) for CNN feed layers.

    Weight shape: (out_size, in_size)
    Bias shape: (out_size, 1)

    Forward: W @ x + bias.squeeze(1)

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
        self.bias = initializer(bkey, (out_size, 1))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: W @ x + bias.squeeze(1)"""
        x = self.weight @ x
        x = x + self.bias.squeeze(1)
        return x

    # Added by Claude: AWB methods for Linear2 layer
    def compute_V_weight(self, A: jax.Array, W: jax.Array, B: jax.Array) -> jax.Array:
        """Compute transformed weight V = A @ W @ B.T for Linear2 layer.

        Args:
            A: Output transformation matrix (new_out, old_out)
            W: Original weight matrix (old_out, old_in)
            B: Input transformation matrix (new_in, old_in)

        Returns:
            Transformed weight V with shape (new_out, new_in)
        """
        # Validate shapes
        if A.shape[1] != W.shape[0]:
            raise AWBShapeError(
                f"Linear2 layer: A.shape={A.shape} incompatible with W.shape={W.shape}. "
                f"Expected A.shape[1]={A.shape[1]} == W.shape[0]={W.shape[0]}"
            )
        if B.shape[1] != W.shape[1]:
            raise AWBShapeError(
                f"Linear2 layer: B.shape={B.shape} incompatible with W.shape={W.shape}. "
                f"Expected B.shape[1]={B.shape[1]} == W.shape[1]={W.shape[1]}"
            )

        return A @ W @ jnp.transpose(B)

    def compute_V_bias(self, A: jax.Array, B: jax.Array, bias: jax.Array) -> jax.Array:
        """Compute transformed bias for Linear2 layer.

        For Linear2 with bias shape (out_size, 1), transformation is: A @ bias

        Args:
            A: Output transformation matrix (new_out, old_out)
            B: Input transformation matrix (not used for bias in Linear2)
            bias: Original bias vector (old_out, 1)

        Returns:
            Transformed bias with shape (new_out, 1)
        """
        # Linear2 layer: A @ bias (because bias shape is (out, 1) and forward is W @ x)
        return A @ bias


class Dropout(eqx.Module):
    """Dropout layer with configurable rate.

    During training, randomly zeroes elements with probability (1 - rate)
    and scales remaining elements by 1/rate.

    Args:
        rate: Keep probability (default 0.5)
    """
    rate: float

    def __init__(self, rate: float = 0.5):
        self.rate = rate

    def __call__(self, inputs: jax.Array, rng: jax.Array, is_training: bool = True) -> jax.Array:
        """Apply dropout.

        Args:
            inputs: Input array
            rng: JAX PRNG key (required)
            is_training: Whether to apply dropout (default: True)

        Returns:
            Dropout-applied array if training, inputs otherwise
        """
        if rng is None:
            raise ValueError(
                "Dropout layer requires a PRNG key argument. "
                "Call with `apply_fun(params, inputs, rng)` where rng is a jax.random.PRNGKey."
            )

        keep = jax.random.bernoulli(rng, self.rate, shape=inputs.shape)
        outs = jnp.where(keep, inputs / self.rate, 0)
        # Return inputs unchanged if not training
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out


# Added by Claude: Conv AWB utility functions for per-filter transformations
def compute_V_conv2d_single_channel(
    A_list: List[jax.Array],
    W: jax.Array,
    B_list: List[jax.Array],
    channel_out: int
) -> List[List[jax.Array]]:
    """Compute V = A @ W @ B.T for Conv2d with single input channel (MNIST).

    For single-channel Conv2d (e.g., MNIST), each output filter is transformed
    independently. This is used in CNN models with 1-channel input.

    Args:
        A_list: List of A matrices, one per output filter [channel_out]
                Each A has shape (new_filter_size, old_filter_size)
        W: Conv weights with shape [channel_out, channel_in, H, W]
        B_list: List of B matrices, one per output filter [channel_out]
                Each B has shape (new_filter_size, old_filter_size)
        channel_out: Number of output channels

    Returns:
        Transformed weights as list of lists: [channel_out][1]
        Each element has shape (new_filter_size, new_filter_size)
    """
    new_conv_weights = []
    for i in range(channel_out):
        # For single input channel: W[i][0] is the 2D filter
        # Transform: A[i] @ W[i][0] @ B[i].T
        transformed = A_list[i] @ W[i][0] @ jnp.transpose(B_list[i])
        new_conv_weights.append([transformed])

    return new_conv_weights


def compute_V_conv2d_multi_channel(
    A_list: List[List[jax.Array]],
    W: jax.Array,
    B_list: List[List[jax.Array]],
    channel_out: int,
    channel_in: int
) -> List[List[jax.Array]]:
    """Compute V = A @ W @ B.T for Conv2d with multiple input channels (CIFAR).

    For multi-channel Conv2d (e.g., CIFAR with 3 channels), each filter is
    transformed per input-output channel pair. This enables fine-grained
    control over channel-wise transformations.

    Args:
        A_list: Nested list of A matrices [channel_out][channel_in]
                Each A has shape (new_filter_size, old_filter_size)
        W: Conv weights with shape [channel_out, channel_in, H, W]
        B_list: Nested list of B matrices [channel_out][channel_in]
                Each B has shape (new_filter_size, old_filter_size)
        channel_out: Number of output channels
        channel_in: Number of input channels

    Returns:
        Transformed weights as nested list: [channel_out][channel_in]
        Each element has shape (new_filter_size, new_filter_size)
    """
    new_conv_weights = []
    for i in range(channel_out):
        channel_weights = []
        for c in range(channel_in):
            # Transform each input-output channel pair independently
            # A[i][c] @ W[i][c] @ B[i][c].T
            transformed = A_list[i][c] @ W[i][c] @ jnp.transpose(B_list[i][c])
            channel_weights.append(transformed)
        new_conv_weights.append(channel_weights)

    return new_conv_weights


# =============================================================================
# AWBMixin - Unified AWB Interface for All Models (Added by Claude)
# =============================================================================

class AWBMixin:
    """Mixin providing unified AWB (Adaptive Weight Basis) interface for all models.

    This mixin standardizes AWB operations across different model architectures:
    - MLP: Linear layers with 2D weight matrices
    - CNN/CNN3D: Convolutional layers with 3D/4D weight tensors
    - GCN: Graph layers with linear transformations
    - Transformers: Multi-head attention with batched projections

    The core AWB transformation is: V = A @ W @ B.T
    This mixin uses einsum with ellipsis notation to handle arbitrary batch dimensions.

    Models using this mixin should:
    1. Define A/B matrices as class attributes (lists or stacked arrays)
    2. Implement get_awb_layer_specs() for layer-level tracking
    3. Use awb_transform_linear() for linear layers
    4. Use awb_transform_conv() for convolutional layers

    Example:
        class MyModel(AWBMixin, eqx.Module):
            A: jax.Array  # Stacked A matrices
            B: jax.Array  # Stacked B matrices
            layers: List[Linear]

            def get_AWBT(self, x):
                # Transform all linear layers at once
                W_stacked = jnp.stack([l.weight for l in self.layers])
                V = self.awb_transform_linear(self.A, W_stacked, self.B)
                ...
    """

    @staticmethod
    def awb_transform_linear(
        A: jax.Array,
        W: jax.Array,
        B: jax.Array
    ) -> jax.Array:
        """Transform linear layer weights: V = A @ W @ B.T

        Uses einsum with ellipsis for arbitrary batch dimensions.
        Works for single layers (2D) or batched layers (3D+).

        Args:
            A: Transformation matrix (..., new_out, old_out)
            W: Weight matrix (..., old_out, old_in)
            B: Transformation matrix (..., new_in, old_in)

        Returns:
            Transformed weight V (..., new_out, new_in)

        Example:
            # Single layer
            V = AWBMixin.awb_transform_linear(A, W, B)
            # A: (m,k), W: (k,n), B: (p,n) -> V: (m,p)

            # Batched layers
            V = AWBMixin.awb_transform_linear(A_stacked, W_stacked, B_stacked)
            # A: (L,m,k), W: (L,k,n), B: (L,p,n) -> V: (L,m,p)
        """
        return awb_transform(A, W, B)

    @staticmethod
    def awb_transform_conv(
        A: jax.Array,
        W: jax.Array,
        B: jax.Array
    ) -> jax.Array:
        """Transform convolutional layer weights: V = A @ W @ B.T

        Uses einsum with ellipsis for arbitrary channel dimensions.
        Works for single-channel (3D) or multi-channel (4D) conv weights.

        Args:
            A: Transformation matrix (out_ch, [in_ch], new_f, old_f)
            W: Conv weight (out_ch, [in_ch], old_f, old_f)
            B: Transformation matrix (out_ch, [in_ch], new_f, old_f)

        Returns:
            Transformed weight V (out_ch, [in_ch], new_f, new_f)

        Example:
            # CNN single-channel (MNIST)
            V = AWBMixin.awb_transform_conv(A, W, B)
            # A: (o,m,k), W: (o,k,n), B: (o,p,n) -> V: (o,m,p)

            # CNN3D multi-channel (CIFAR)
            V = AWBMixin.awb_transform_conv(A, W, B)
            # A: (o,i,m,k), W: (o,i,k,n), B: (o,i,p,n) -> V: (o,i,m,p)
        """
        return awb_transform(A, W, B)

    @staticmethod
    def awb_transform_bias_linear(
        A: jax.Array,
        bias: jax.Array,
        bias_shape: str = 'row'
    ) -> jax.Array:
        """Transform linear layer bias.

        Args:
            A: Transformation matrix (new_out, old_out)
            bias: Original bias
            bias_shape: 'row' for (1, out) or 'col' for (out, 1)

        Returns:
            Transformed bias with same shape convention
        """
        if bias_shape == 'row':
            # Linear layer: bias @ A.T -> (1, new_out)
            return bias @ A.T
        else:
            # Linear2/LinearGCN: A @ bias -> (new_out, 1)
            return A @ bias

    def get_awb_matrices_spec(self) -> dict:
        """Get specification of AWB matrices for this model.

        Override in subclasses to specify which attributes hold A/B matrices.

        Returns:
            Dict with keys:
                - 'linear_A': attribute name(s) for linear layer A matrices
                - 'linear_B': attribute name(s) for linear layer B matrices
                - 'conv_A': attribute name(s) for conv layer A matrices (optional)
                - 'conv_B': attribute name(s) for conv layer B matrices (optional)
        """
        raise NotImplementedError("Subclasses must implement get_awb_matrices_spec()")

    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB layer specifications for all transformable layers.

        Override in subclasses to provide layer-specific tracking.

        Returns:
            List of AWBLayerSpec for each layer
        """
        raise NotImplementedError("Subclasses must implement get_awb_layer_specs()")

    def get_AWBT(self, *args, **kwargs):
        """Forward pass using AWB transformation.

        Override in subclasses with model-specific implementation.
        """
        raise NotImplementedError("Subclasses must implement get_AWBT()")

    def partition_for_AB_training(self):
        """Partition model for A/B training (freeze W, train A/B).

        Override in subclasses with model-specific partition logic.
        """
        raise NotImplementedError("Subclasses must implement partition_for_AB_training()")

    def partition_for_standard_training(self):
        """Partition model for standard training (freeze A/B, train W).

        Override in subclasses with model-specific partition logic.
        """
        raise NotImplementedError("Subclasses must implement partition_for_standard_training()")
