"""
Multi-Layer Perceptron model for continual learning.

Provides MLP with optional AWB (Adaptive Weight Basis) support for
architecture morphing during lifelong learning.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Dict, Any

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
            # Fixed by Claude: Keep bias shape as (1, new_out) for proper broadcasting
            # Don't squeeze - bias should match the shape used in compute_V_from_AWB
            bias_transformed = self.layers[i].bias @ self.A[i].T

            x = weight_transformed @ x + bias_transformed.squeeze(0)
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

    # Added by Claude: Architecture search interface methods
    def generate_search_candidates(
        self,
        iteration: int,
        current_best: List[int],
        config: Dict[str, Any]
    ) -> List[List[int]]:
        """Generate candidate architectures for MLP search.

        # Added by Claude: Model-specific search strategy
        MLP uses additive grid search over hidden layer dimensions.
        This implements the logic from arch_search_MLP() lines 278-328.

        Args:
            iteration: Current search iteration (0-indexed)
            current_best: Current best architecture [in, h1, h2, ..., out]
            config: Configuration dict with search hyperparameters

        Returns:
            List of candidate architecture lists to evaluate
        """
        from ..config.constants import (
            DEFAULT_ARCH_SEARCH_MLP_INCREMENT,
            DEFAULT_ARCH_SEARCH_LARGE_INCREMENT,
            DEFAULT_ARCH_SEARCH_RANGE,
        )

        # Get search hyperparameters
        mlp_increment = config.get('arch_search_mlp_increment', DEFAULT_ARCH_SEARCH_MLP_INCREMENT)
        large_increment = config.get('arch_search_large_increment', DEFAULT_ARCH_SEARCH_LARGE_INCREMENT)
        search_range = config.get('arch_search_range', DEFAULT_ARCH_SEARCH_RANGE)

        # For MLP, we search over first two hidden layer dimensions
        if len(current_best) < 4:
            return []  # Need at least [input, h1, h2, output]

        input_size = current_best[0]
        output_size = current_best[-1]
        h1_base = current_best[1]
        h2_base = current_best[2]

        candidates = []

        # Grid search over hidden dimensions
        for n in range(search_range):
            for m in range(search_range):
                h1_candidate = h1_base + mlp_increment * n
                h2_candidate = h2_base + mlp_increment * m

                # Build candidate architecture
                if len(current_best) == 4:
                    candidate = [input_size, h1_candidate, h2_candidate, output_size]
                else:
                    # Preserve additional layers
                    candidate = [input_size, h1_candidate, h2_candidate] + current_best[3:]
                    
                candidates.append(candidate)

        return candidates

    @classmethod
    def create_with_architecture(cls, arch_spec: List[int], seed: int = 0, awb_enabled: bool = False) -> 'MLP':
        """Create MLP instance with specified architecture.

        # Added by Claude: Factory method for search
        Creates a fresh MLP with given architecture specification.

        Args:
            arch_spec: Architecture sizes [input, h1, h2, ..., output]
            seed: Random seed for weight initialization
            awb_enabled: Whether to enable AWB matrices

        Returns:
            New MLP instance with specified architecture
        """
        key = jax.random.PRNGKey(seed)
        return cls(sizes=arch_spec, key=key, awb_enabled=awb_enabled)

    def reinitialize_weights(self, seed: int = 0) -> 'MLP':
        """Reinitialize model weights for fair architecture comparison.

        # Added by Claude: Weight reinitialization for search
        Uses glorot uniform initialization. Ensures fair comparison between
        candidate architectures by starting from fresh random weights.

        This implements the logic from _reinitialize_weights() in mlp_search.py
        (lines 50-76).

        Args:
            seed: Random seed for reproducible initialization

        Returns:
            Model with freshly initialized weights
        """
        initializer = jax.nn.initializers.glorot_uniform()

        model = self
        # Reinitialize each layer's weights and biases
        for j in range(len(self.sizes) - 1):
            in_size = self.sizes[j]
            out_size = self.sizes[j + 1]

            weight = initializer(jax.random.PRNGKey(seed + j), (out_size, in_size))
            bias = initializer(jax.random.PRNGKey(seed + j + 100), (1, out_size))

            model = eqx.tree_at(lambda x, idx=j: x.layers[idx].weight, model, weight)
            model = eqx.tree_at(lambda x, idx=j: x.layers[idx].bias, model, bias)

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


# Added by Claude: AWB Operations implementation for MLP
class MLPAWBOps:
    """MLP-specific AWB operations implementation.

    Implements the AWBOperations interface for MLP models.
    Delegates to existing functions in awb.py and arch_search/mlp_search.py.
    """

    def search_architecture(self, model, task_id, baseline_loss, dataloader_curr,
                           dataloader_exp, test_loader_curr, test_loader_exp, config, trainer=None):
        """Search for optimal MLP architecture."""
        from ..core.arch_search import search_architecture
        return search_architecture(
            model=model, baseline_arch=list(model.sizes), task_id=task_id,
            baseline_loss=baseline_loss, dataloader_curr=dataloader_curr,
            dataloader_exp=dataloader_exp, test_loader_curr=test_loader_curr,
            test_loader_exp=test_loader_exp, config=config, trainer=trainer, model_type='mlp'
        )

    def set_AB_matrices(self, model, original_arch, new_arch):
        """Initialize A/B matrices for architecture transition."""
        from ..core.awb import set_new_AB_matrices
        return set_new_AB_matrices(model, original_arch, new_arch)

    def partition_for_AB_training(self, model):
        """Partition model for AB training (freeze W, train A/B)."""
        from ..core.awb import partition_for_AB_training
        return partition_for_AB_training(model)

    def compute_V(self, model):
        """Compute V = A @ W @ B^T."""
        from ..core.awb import compute_V_from_AWB
        return compute_V_from_AWB(model)

    def partition_for_standard_training(self, model):
        """Partition model for standard training (train V, freeze A/B)."""
        from ..core.awb import partition_for_standard_training
        return partition_for_standard_training(model)

    def get_model_architecture(self, model):
        """Extract architecture specification from model."""
        return list(model.sizes)

    def save_weights(self, model):
        """Save current model weights."""
        from ..core.awb import save_layer_weights
        return save_layer_weights(model)

    def restore_weights(self, model, saved_weights):
        """Restore model weights."""
        from ..core.awb import restore_layer_weights
        weight_list, bias_list = saved_weights
        return restore_layer_weights(model, weight_list, bias_list)
