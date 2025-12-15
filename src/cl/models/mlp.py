"""
Multi-Layer Perceptron model for continual learning.

Provides MLP with optional AWB (Adaptive Weight Basis) support for
architecture morphing during lifelong learning.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional

from .layers import Linear, AWBLayerSpec
import jax.tree_util as jtu


class MLP(eqx.Module):
    """Multi-Layer Perceptron with optional AWB (Adaptive Weight Basis) support.

    The MLP supports two forward pass modes:
    1. Standard: x -> tanh(W1 @ x + b1) -> ... -> Wn @ x + bn
    2. AWB: Uses transformed weights V = A @ W @ B.T for architecture morphing

    Attributes:
        layers: List of Linear layers
        sizes: List of layer dimensions [input, h1, h2, ..., output]
        A: List of output transformation matrices for AWB (None if AWB disabled)
        B: List of input transformation matrices for AWB (None if AWB disabled)
        awb_enabled: Whether AWB matrices are initialized

    Example:
        >>> model = MLP(sizes=[3, 128, 128, 1])  # Standard MLP, no AWB
        >>> model = MLP(sizes=[3, 128, 128, 1], awb_enabled=True)  # With AWB support
        >>> y = model(x)  # Standard forward
        >>> y_awb = model.getAWB(x)  # AWB forward (only if awb_enabled=True)
    """
    layers: List[Linear]
    sizes: List[int]
    A: Optional[List[jax.Array]]
    B: Optional[List[jax.Array]]
    awb_enabled: bool

    def __init__(self, sizes: List[int], key: Optional[jax.Array] = None, awb_enabled: bool = False):
        """Initialize the MLP.

        Args:
            sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            key: Optional JAX PRNG key (uses default if not provided)
            awb_enabled: Whether to initialize A/B matrices for AWB (default: False)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        self.sizes = sizes
        self.layers = []
        self.awb_enabled = awb_enabled

        # Only initialize A/B matrices if AWB is enabled
        if awb_enabled:
            # A transforms output dimension: shape (out_size, 1) initially
            # B transforms input dimension: shape (out_size, in_size) initially
            self.A = [
                jax.random.normal(jax.random.PRNGKey(0), shape=(y, 1))
                for y in sizes[1:]
            ]
            self.B = [
                jax.random.normal(jax.random.PRNGKey(0), shape=(y, x))
                for x, y in zip(sizes[:-1], sizes[1:])
            ]
        else:
            self.A = None
            self.B = None

        # Initialize layers
        keys = jax.random.split(key, len(sizes) - 1)
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(in_size, out_size, key=keys[i]))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Standard forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim) or (input_dim,)

        Returns:
            Output tensor of shape (batch, output_dim) or (output_dim,)
        """
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def getAWB(self, x: jax.Array) -> jax.Array:
        """Forward pass using AWB transformation: A @ W @ B.T.

        Used during AWB training (Step 3b) when A/B matrices are being optimized
        while W is frozen. The effective weight becomes V = A @ W @ B.T.

        Args:
            x: Input tensor of shape (input_dim,) - note: expects unbatched input

        Returns:
            Output tensor after AWB transformation

        Raises:
            ValueError: If AWB is not enabled (A/B matrices are None)
        """
        if not self.awb_enabled or self.A is None or self.B is None:
            raise ValueError(
                "AWB forward pass requires awb_enabled=True. "
                "Initialize MLP with awb_enabled=True to use getAWB()."
            )

        for i in range(len(self.sizes) - 1):
            # Compute transformed weight and bias
            # V = A @ W @ B.T
            # bias_transformed = bias @ A.T
            weight_transformed = self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i])
            bias_transformed = (self.layers[i].bias @ self.A[i].T).T.squeeze(1)

            x = weight_transformed @ x + bias_transformed
            x = jax.nn.tanh(x)

        return x

    # Added by Claude: AWBModel interface methods for layer-level abstraction
    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        """Get AWB layer specifications for each transformable layer.

        Returns:
            List of AWBLayerSpec, one per layer, containing layer, A, B matrices

        Example:
            >>> specs = model.get_awb_layer_specs()
            >>> for spec in specs:
            ...     errors = spec.validate()  # Check shape compatibility
        """
        if not self.awb_enabled or self.A is None or self.B is None:
            # If AWB not enabled, return specs with None A/B
            return [
                AWBLayerSpec(
                    layer=self.layers[i],
                    A=None,
                    B=None,
                    layer_type='linear',
                    layer_index=i
                )
                for i in range(len(self.layers))
            ]

        return [
            AWBLayerSpec(
                layer=self.layers[i],
                A=self.A[i] if self.A else None,
                B=self.B[i] if self.B else None,
                layer_type='linear',
                layer_index=i
            )
            for i in range(len(self.layers))
        ]

    def apply_V_transformation(self) -> 'MLP':
        """Apply V = A @ W @ B.T transformation to all layers.

        This is STEP 4 of the AWB algorithm. After training A/B matrices,
        compute the effective weights V and update the model.

        Returns:
            New model with transformed weights V = A @ W @ B.T

        Raises:
            ValueError: If AWB is not enabled

        Example:
            >>> # After Step 3b (A/B training)
            >>> model = model.apply_V_transformation()
            >>> # Now model has V weights, ready for Step 5 training
        """
        if not self.awb_enabled or self.A is None or self.B is None:
            raise ValueError(
                "apply_V_transformation requires awb_enabled=True and A/B matrices. "
                "Train A/B matrices first (Step 3b)."
            )

        model = self
        for i, spec in enumerate(self.get_awb_layer_specs()):
            layer = spec.layer
            # Use layer's AWB methods to compute V
            Vw = layer.compute_V_weight(spec.A, layer.weight, spec.B)
            Vb = layer.compute_V_bias(spec.A, spec.B, layer.bias)

            # Update model with transformed weights
            model = eqx.tree_at(lambda x, idx=i: x.layers[idx].weight, model, Vw)
            model = eqx.tree_at(lambda x, idx=i: x.layers[idx].bias, model, Vb)

        return model

    def partition_for_AB_training(self):
        """Partition model for A/B training (freeze W, train A/B).

        This is used in STEP 3b of AWB algorithm where we train A/B matrices
        while keeping layer weights W frozen.

        Returns:
            Tuple of (diff_model, static_model) where:
                - diff_model: Contains only A and B (trainable)
                - static_model: Contains layer weights (frozen)

        Example:
            >>> diff_model, static_model = model.partition_for_AB_training()
            >>> # Train diff_model (only A/B parameters)
            >>> model = eqx.combine(diff_model, static_model)
        """
        # Create filter spec: True for A/B, False for everything else
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda x: (x.A, x.B),
            filter_spec,
            replace=(True, True)
        )
        diff_model, static_model = eqx.partition(self, filter_spec)
        return diff_model, static_model

    def partition_for_standard_training(self):
        """Partition model for standard training (freeze A/B, train W).

        This is used in standard CL training and STEP 5 of AWB (train V with A/B frozen).

        Returns:
            Tuple of (params, static) where:
                - params: Contains trainable arrays (layer weights, biases)
                - static: Contains A, B matrices (frozen)

        Example:
            >>> params, static = model.partition_for_standard_training()
            >>> # Train params (only layer weights/biases)
            >>> model = eqx.combine(params, static)
        """
        params, static = eqx.partition(self, eqx.is_array)

        if self.awb_enabled and self.A is not None and self.B is not None:
            # Move A and B to static (frozen)
            static = eqx.tree_at(
                lambda x: (x.A, x.B),
                static,
                replace=(self.A, self.B)
            )

            # Remove A and B from params (set to None)
            params = eqx.tree_at(
                lambda x: (x.A, x.B),
                params,
                replace=(None, None)
            )

        return params, static

    def with_new_AB_matrices(self, original_arch: List[int], new_arch: List[int], seed: int = 5) -> 'MLP':
        """Initialize A/B matrices for architecture transition.

        This is used in STEP 3a of AWB algorithm when changing from original
        architecture to new architecture.

        Args:
            original_arch: Original sizes [in, h1, h2, ..., out]
            new_arch: New sizes [in, h1', h2', ..., out]
            seed: Random seed for initialization

        Returns:
            Model with new A/B matrices initialized

        Example:
            >>> # After architecture search finds optimal size
            >>> model = model.with_new_AB_matrices([3, 10, 10, 2], [3, 15, 20, 2])
            >>> # Now A/B matrices are ready for Step 3b training
        """
        initializer = jax.nn.initializers.glorot_uniform()

        # A matrices: transform output dimensions [new_out, old_out]
        A_list = [
            initializer(jax.random.PRNGKey(seed + i), (y_new, y_old))
            for i, (y_old, y_new) in enumerate(zip(original_arch[1:], new_arch[1:]))
        ]

        # B matrices: transform input dimensions [new_in, old_in]
        B_list = [
            initializer(jax.random.PRNGKey(seed + 100 + i), (x_new, x_old))
            for i, (x_old, x_new) in enumerate(zip(original_arch[:-1], new_arch[:-1]))
        ]

        # Update model with new sizes and A/B matrices
        model = eqx.tree_at(lambda x: x.sizes, self, new_arch)
        model = eqx.tree_at(lambda x: x.A, model, A_list)
        model = eqx.tree_at(lambda x: x.B, model, B_list)

        return model


def create_mlp(config: dict) -> MLP:
    """Factory function to create MLP from configuration.

    Args:
        config: Configuration dict with:
            - input_size: Input dimension
            - output_size: Output dimension
            - n_layers: Total number of layers (including input/output)
            - hln: Hidden layer size
            - awb_enabled: Whether to enable AWB (default: False)

    Returns:
        Initialized MLP model
    """
    input_size = config['input_size']
    output_size = config['output_size']
    n_layers = config.get('n_layers', 4)
    hidden_size = config.get('hln', 128)
    awb_enabled = config.get('awb_enabled', False)

    # Build sizes list: [input, hidden, hidden, ..., output]
    sizes = [input_size] + [hidden_size] * (n_layers - 2) + [output_size]

    return MLP(sizes=sizes, awb_enabled=awb_enabled)
