"""
AWB (Adaptive Weight Basis) pipeline utilities.

This module contains helper functions for the 5-step AWB continual learning
algorithm that enables architecture morphing during lifelong learning.

The 5-step algorithm:
    Task 0: Standard CL training
    Tasks 1+:
        STEP 1: Train for preliminary epochs on new task
        STEP 2: Decide if architecture change needed (loss ratio thresholds)
        If change_arch == True:
            STEP 3a: Search for new architecture, set new A/B matrices
            STEP 3b: Train A/B with W frozen (notABTrain=False)
            STEP 4: Set new weights V = A @ W @ B.T
            STEP 5: Train V with A/B frozen (notABTrain=True)
        Else:
            Continue normal training
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax

from ..config.constants import (
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_CHANGE_THRESHOLD_HIGH,
    DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA,
    DEFAULT_AWB_AB_THRESHOLD_BASE,
)


def compute_avg_loss(record_dict, task_id, epochs, window=None):
    """Compute average loss over last `window` epochs from training record.

    Args:
        record_dict: Dictionary containing training records (iterations dict)
        task_id: Current task ID
        epochs: Total epochs per task
        window: Number of epochs to average (default: DEFAULT_AWB_AVERAGING_WINDOW)

    Returns:
        Average loss value over the specified window
    """
    if window is None:
        window = DEFAULT_AWB_AVERAGING_WINDOW

    losses = []

    # Handle both old format (train{epoch}) and new format (iterations dict)
    if isinstance(record_dict, dict):
        # New format: record_dict has 'iterations' key with iteration numbers as keys
        for j in range(1, window + 1):
            iteration = (task_id + 1) * epochs - j
            if iteration in record_dict:
                record = record_dict[iteration]
                if isinstance(record, dict) and 'losses' in record:
                    losses.append(record['losses'].get('V', 0))
                elif isinstance(record, tuple):
                    # Old tuple format: (V, dV, dV_dx, dV_dtheta, H, ...)
                    losses.append(record[0])

    # Fixed by Claude: If no losses found with standard iteration calculation,
    # fall back to last recorded iterations (handles AWB offset mismatches)
    if not losses and record_dict:
        # Get last `window` iterations that exist
        sorted_iters = sorted(record_dict.keys(), reverse=True)
        for iteration in sorted_iters[:window]:
            record = record_dict[iteration]
            if isinstance(record, dict) and 'losses' in record:
                losses.append(record['losses'].get('V', 0))
            elif isinstance(record, tuple):
                losses.append(record[0])

    if not losses:
        return float('inf')
    return np.mean(losses)


def should_change_arch(trainWLoss, end_last,
                       threshold_high=None, min_delta=None):
    """Decide if architecture change is needed based on loss ratio.

    The decision logic:
        - If ratio > threshold_high: change_arch = True
        - Otherwise: change_arch = False

    Args:
        trainWLoss: Current training loss after preliminary training
        end_last: Loss at end of previous task (for task 1, this is task 0's optimal loss)
        threshold_high: High threshold for loss ratio (default: 0.45)
        min_delta: Deprecated, kept for backward compatibility

    Returns:
        Boolean indicating whether architecture should be changed
    """
    if threshold_high is None:
        threshold_high = DEFAULT_AWB_CHANGE_THRESHOLD_HIGH

    # Added by Claude: Compare current preliminary loss to previous task's final loss
    # Simplified logic: only use ratio, not min_delta condition
    ratio = trainWLoss / end_last

    return ratio > threshold_high


def compute_ab_threshold(trainWLoss, end_last, base_threshold=None):
    """Compute dynamic threshold for AB training convergence.

    The threshold adapts based on the loss ratio to allow more iterations
    when the loss is significantly higher than baseline.

    Args:
        trainWLoss: Current training loss after preliminary training
        end_last: Loss at end of previous task
        base_threshold: Base threshold value (default: 0.6)

    Returns:
        Computed threshold for AB training convergence
    """
    if base_threshold is None:
        base_threshold = DEFAULT_AWB_AB_THRESHOLD_BASE

    ratio = trainWLoss / end_last if end_last > 0 else 1.0

    if ratio > 3.0:
        threshold = max(1 / ratio, 0.45)
    elif 2.0 <= ratio < 3.0:
        threshold = min(1 / ratio, 0.6)
    elif 1.0 <= ratio < 2.0:
        threshold = min(1 / ratio, 0.75)
    else:
        threshold = 0.8

    return threshold


# Added by Claude: Generic AWB functions that delegate to model interface
def apply_V_transformation(model):
    """Generic V = A @ W @ B.T transformation using model's interface.

    Works with any model implementing apply_V_transformation() method.
    Falls back to model-specific logic for backward compatibility.

    Args:
        model: AWB-enabled model (MLP, CNN, CNN3D, or GCN)

    Returns:
        Model with transformed weights V = A @ W @ B.T
    """
    if hasattr(model, 'apply_V_transformation'):
        return model.apply_V_transformation()
    else:
        # Backward compatibility: use old model-specific functions
        from ..models.mlp import MLP
        if isinstance(model, MLP):
            return compute_V_from_AWB(model)
        else:
            return compute_V_from_AWB_gcn(model)


def partition_model_for_AB_training(model):
    """Generic partition for A/B training using model's interface.

    Args:
        model: AWB-enabled model

    Returns:
        Tuple of (diff_model, static_model)
    """
    if hasattr(model, 'partition_for_AB_training'):
        return model.partition_for_AB_training()
    else:
        # Backward compatibility
        return partition_for_AB_training(model)


def partition_model_for_standard_training(model):
    """Generic partition for standard training using model's interface.

    Args:
        model: AWB-enabled model

    Returns:
        Tuple of (params, static)
    """
    if hasattr(model, 'partition_for_standard_training'):
        return model.partition_for_standard_training()
    else:
        # Backward compatibility
        return partition_for_standard_training(model)


def initialize_AB_matrices(model, original_arch, new_arch, seed=5):
    """Generic A/B matrix initialization using model's interface.

    Args:
        model: AWB-enabled model
        original_arch: Original architecture specification
        new_arch: New architecture specification
        seed: Random seed

    Returns:
        Model with initialized A/B matrices
    """
    if hasattr(model, 'with_new_AB_matrices'):
        return model.with_new_AB_matrices(original_arch, new_arch, seed)
    else:
        # Backward compatibility
        return set_new_AB_matrices(model, original_arch, new_arch, seed)


# ============================================================================
# DEPRECATED: Legacy model-specific functions (kept for backward compatibility)
# New code should use the generic functions above or model interface methods
# ============================================================================

def _create_identity_like_matrix(new_size, old_size, key=None):
    """Create identity-like transformation matrix for AWB.

    Creates a matrix that preserves the original weights in the overlap region:
    - If new_size >= old_size: Identity in upper-left, zeros elsewhere
    - If new_size < old_size: Truncated identity

    This ensures A @ W @ B^T ≈ W initially for smooth knowledge transfer.

    Args:
        new_size: Target dimension
        old_size: Source dimension
        key: Optional PRNG key for small noise (unused, kept for API compatibility)

    Returns:
        JAX array of shape (new_size, old_size)
    """
    # Create identity-like matrix
    min_size = min(new_size, old_size)
    matrix = jnp.zeros((new_size, old_size))
    # Set diagonal to 1 for the overlapping region
    indices = jnp.arange(min_size)
    matrix = matrix.at[indices, indices].set(1.0)
    return matrix


def set_new_AB_matrices(model, original_arch, new_arch, seed=5):
    """Initialize A/B matrices for architecture transition (MLP only).

    DEPRECATED: Use initialize_AB_matrices() or model.with_new_AB_matrices() instead.

    When the architecture changes from original_arch to new_arch,
    we create transformation matrices A and B such that
    the forward pass becomes: A @ W @ B.T

    Uses identity-like initialization so that A @ W @ B^T ≈ W initially,
    preserving learned weights in the overlap region for smooth transfer.

    Args:
        model: Current equinox model (MLP)
        original_arch: Original architecture sizes list [in, h1, h2, ..., out]
        new_arch: New architecture sizes list [in, h1', h2', ..., out]
        seed: Random seed for initializer (unused, kept for API compatibility)

    Returns:
        Updated model with new A, B matrices and sizes
    """
    # A matrices: transform output dimensions [new_out, old_out]
    # Identity-like: A @ old_output ≈ old_output (with padding/truncation)
    A_list = [
        _create_identity_like_matrix(y_new, y_old)
        for y_old, y_new in zip(original_arch[1:], new_arch[1:])
    ]

    # B matrices: transform input dimensions [new_in, old_in]
    # Identity-like: B @ old_input ≈ old_input (with padding/truncation)
    B_list = [
        _create_identity_like_matrix(x_new, x_old)
        for x_old, x_new in zip(original_arch[:-1], new_arch[:-1])
    ]

    # Update model with new sizes and A/B matrices
    model = eqx.tree_at(lambda x: x.sizes, model, new_arch)
    model = eqx.tree_at(lambda x: x.A, model, A_list)
    model = eqx.tree_at(lambda x: x.B, model, B_list)

    return model


def compute_V_from_AWB(model):
    """Compute new weights V = A @ W @ B.T for all layers (MLP only).

    DEPRECATED: Use apply_V_transformation() or model.apply_V_transformation() instead.

    This is STEP 4 of the AWB algorithm: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Args:
        model: Equinox MLP model with trained A, B matrices

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    for j in range(len(model.sizes) - 1):
        # Compute transformed weight: V = A @ W @ B.T
        Vw = model.A[j] @ model.layers[j].weight @ jnp.transpose(model.B[j])
        # Compute transformed bias: Vb = bias @ A.T
        Vb = model.layers[j].bias @ model.A[j].T

        # Update model with new weights
        model = eqx.tree_at(lambda x: x.layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.layers[j].bias, model, Vb)

    return model


def partition_for_AB_training(model):
    """Partition model for A/B training (freeze W, train A/B) - MLP only.

    DEPRECATED: Use partition_model_for_AB_training() or model.partition_for_AB_training() instead.

    This creates a filter spec where only A and B are trainable,
    used in STEP 3b of the AWB algorithm.

    Args:
        model: Equinox MLP model

    Returns:
        Tuple of (diff_model, static_model) where:
            - diff_model: Contains only A and B (trainable)
            - static_model: Contains everything else (frozen)
    """
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A, x.B),
        filter_spec,
        replace=(True, True)
    )
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


def partition_for_standard_training(model):
    """Partition model for standard training (freeze A/B, train W) - MLP only.

    DEPRECATED: Use partition_model_for_standard_training() or model.partition_for_standard_training() instead.

    This creates the standard partition where A and B are frozen
    and only the layer weights are trainable.

    Args:
        model: Equinox MLP model

    Returns:
        Tuple of (params, static) where:
            - params: Contains trainable arrays (weights, biases)
            - static: Contains A, B matrices (frozen)
    """
    params, static = eqx.partition(model, eqx.is_array)

    # Move A and B to static (frozen)
    static = eqx.tree_at(
        lambda x: (x.A, x.B),
        static,
        replace=(model.A, model.B)
    )

    # Remove A and B from params (set to None)
    params = eqx.tree_at(
        lambda x: (x.A, x.B),
        params,
        replace=(None, None)
    )

    return params, static


def partition_for_AB_training_cnn(model):
    """Partition CNN model for A/B training (freeze W, train A/B).

    Args:
        model: Equinox CNN model with A_conv, B_conv, A_feed, B_feed

    Returns:
        Tuple of (diff_model, static_model)
    """
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        filter_spec,
        replace=(True, True, True, True)
    )
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


def partition_for_standard_training_cnn(model):
    """Partition CNN model for standard training (freeze A/B, train W).

    Note: A_conv, B_conv, A_feed, B_feed are lists of arrays.
    We must preserve the list structure by replacing each element with None,
    not replacing the entire list with None (which would change the PyTree structure
    and cause optimizer state mismatch errors).

    Args:
        model: Equinox CNN model

    Returns:
        Tuple of (params, static)
    """
    params, static = eqx.partition(model, eqx.is_array)

    static = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        static,
        replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed)
    )

    # Preserve list structure by replacing each element with None
    params = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        params,
        replace=([None]*len(model.A_conv), [None]*len(model.B_conv),
                 [None]*len(model.A_feed), [None]*len(model.B_feed))
    )

    return params, static


# Added by Claude: CNN3D versions for two-conv-layer models (CIFAR)
def partition_for_AB_training_cnn3d(model):
    """Partition CNN3D model for A/B training (freeze W, train A/B).

    Args:
        model: Equinox CNN3D model with A_conv1, B_conv1, A_conv2, B_conv2, A_feed, B_feed

    Returns:
        Tuple of (diff_model, static_model)
    """
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
        filter_spec,
        replace=(True, True, True, True, True, True)
    )
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


def partition_for_standard_training_cnn3d(model):
    """Partition CNN3D model for standard training (freeze A/B, train W).

    Note: A_conv1, B_conv1, A_conv2, B_conv2 are lists of lists of arrays.
    A_feed, B_feed are lists of arrays.
    We must preserve the nested list structure by replacing each element with None,
    not replacing the entire list with None (which would change the PyTree structure
    and cause optimizer state mismatch errors).

    Args:
        model: Equinox CNN3D model

    Returns:
        Tuple of (params, static)
    """
    params, static = eqx.partition(model, eqx.is_array)

    static = eqx.tree_at(
        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
        static,
        replace=(model.A_conv1, model.B_conv1, model.A_conv2, model.B_conv2, model.A_feed, model.B_feed)
    )

    # Preserve nested list structure by replacing each element with None
    none_conv1 = [[None for _ in row] for row in model.A_conv1]
    none_conv2 = [[None for _ in row] for row in model.A_conv2]
    none_feed = [None] * len(model.A_feed)
    params = eqx.tree_at(
        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
        params,
        replace=(none_conv1, none_conv1, none_conv2, none_conv2, none_feed, none_feed)
    )

    return params, static


def partition_for_AB_training_gnn(model):
    """Partition GNN model for A/B training (freeze W, train A/B).

    Args:
        model: Equinox GNN model with A_gcn, B_gcn, A_feed, B_feed

    Returns:
        Tuple of (diff_model, static_model)
    """
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        filter_spec,
        replace=(True, True, True, True)
    )
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


def partition_for_standard_training_gnn(model):
    """Partition GNN model for standard training (freeze A/B, train W).

    Args:
        model: Equinox GNN model

    Returns:
        Tuple of (params, static)
    """
    params, static = eqx.partition(model, eqx.is_array)

    static = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        static,
        replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed)
    )

    params = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        params,
        replace=(None, None, None, None)
    )

    return params, static


def save_layer_weights(model):
    """Save current layer weights and biases before architecture search.

    Used to restore weights if architecture search doesn't find improvement.

    Args:
        model: Equinox MLP model

    Returns:
        Tuple of (weight_list, bias_list)
    """
    weight_list = [model.layers[j].weight for j in range(len(model.layers))]
    bias_list = [model.layers[j].bias for j in range(len(model.layers))]
    return weight_list, bias_list


def restore_layer_weights(model, weight_list, bias_list):
    """Restore saved layer weights and biases to model.

    Args:
        model: Equinox MLP model
        weight_list: List of weight matrices
        bias_list: List of bias vectors

    Returns:
        Model with restored weights
    """
    for j in range(len(weight_list)):
        model = eqx.tree_at(lambda x: x.layers[j].weight, model, weight_list[j])
        model = eqx.tree_at(lambda x: x.layers[j].bias, model, bias_list[j])
    return model


def compute_V_from_AWB_gcn(model):
    """Compute new weights V = A @ W @ B.T for GCN model.

    This is STEP 4 of the AWB algorithm for GCN: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Based on train_model_graph from run_AWB_ALL_functions.py.

    Args:
        model: Equinox GCN model with trained A_gcn, B_gcn, A_feed, B_feed matrices

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    # Transform GCN layer weights: V = A @ W @ B.T
    for k in range(len(model.gcn_layers)):
        Vw = model.A_gcn[k] @ model.gcn_layers[k].weight @ jnp.transpose(model.B_gcn[k])
        Vb = model.gcn_layers[k].bias @ model.B_gcn[k].T

        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].weight, model, Vw)
        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].bias, model, Vb)

    # Transform feed layer weights: V = (A @ W.T @ B.T).T = B @ W @ A.T
    # Note: feed_layers use Linear3 with weight shape (out_size, in_size)
    # Forward pass is: x @ W.T + bias
    for j in range(len(model.feed_layers)):
        # Original: x @ W.T + bias -> x @ (A @ W.T @ B.T) + bias @ B.T
        Vw = (model.A_feed[j] @ model.feed_layers[j].weight.T @ jnp.transpose(model.B_feed[j])).T
        Vb = model.feed_layers[j].bias @ model.B_feed[j].T

        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].weight, model, Vw)
        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].bias, model, Vb)

    return model


def save_gcn_layer_weights(model):
    """Save current GCN layer weights and biases before architecture search.

    Args:
        model: Equinox GCN model

    Returns:
        Tuple of (gcn_weights, gcn_biases, mlp_weights, mlp_biases)
    """
    gcn_weights = [model.gcn_layers[k].weight for k in range(len(model.gcn_layers))]
    gcn_biases = [model.gcn_layers[k].bias for k in range(len(model.gcn_layers))]
    mlp_weights = [model.feed_layers[j].weight for j in range(len(model.feed_layers))]
    mlp_biases = [model.feed_layers[j].bias for j in range(len(model.feed_layers))]
    return gcn_weights, gcn_biases, mlp_weights, mlp_biases


def restore_gcn_layer_weights(model, gcn_weights, gcn_biases, mlp_weights, mlp_biases):
    """Restore saved GCN layer weights and biases to model.

    Args:
        model: Equinox GCN model
        gcn_weights: List of GCN weight matrices
        gcn_biases: List of GCN bias vectors
        mlp_weights: List of MLP weight matrices
        mlp_biases: List of MLP bias vectors

    Returns:
        Model with restored weights
    """
    for k in range(len(gcn_weights)):
        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].weight, model, gcn_weights[k])
        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].bias, model, gcn_biases[k])
    for j in range(len(mlp_weights)):
        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].weight, model, mlp_weights[j])
        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].bias, model, mlp_biases[j])
    return model


def create_optimizer_for_phase(phase, learning_rate=1e-4):
    """Create appropriate optimizer for each training phase.

    Args:
        phase: One of 'standard', 'ab_training', 'v_training'
        learning_rate: Learning rate for optimizer

    Returns:
        Optax optimizer
    """
    if phase == 'ab_training':
        return optax.adam(learning_rate)
    elif phase == 'v_training':
        return optax.adam(1e-3)  # Higher LR for V training
    else:  # standard
        return optax.adam(learning_rate)


# Added by Claude: CNN-specific AWB functions (moved from classification.py)
def save_cnn_layer_weights(model):
    """Save current CNN layer weights and biases before architecture search.

    Args:
        model: Equinox CNN model

    Returns:
        Tuple of (conv_weights, feed_weights, feed_biases)
    """
    conv_weights = [model.conv_layers[j].weight for j in range(len(model.conv_layers))]
    feed_weights = [model.feed_layers[j].weight for j in range(len(model.feed_layers))]
    feed_biases = [model.feed_layers[j].bias for j in range(len(model.feed_layers))]
    return conv_weights, feed_weights, feed_biases


def restore_cnn_layer_weights(model, conv_weights, feed_weights, feed_biases):
    """Restore saved CNN layer weights and biases to model.

    Args:
        model: Equinox CNN model
        conv_weights: List of conv weight matrices
        feed_weights: List of feed weight matrices
        feed_biases: List of feed bias vectors

    Returns:
        Model with restored weights
    """
    for j in range(len(conv_weights)):
        model = eqx.tree_at(lambda x: x.conv_layers[j].weight, model, conv_weights[j])
    for j in range(len(feed_weights)):
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, feed_weights[j])
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, feed_biases[j])
    return model


def compute_V_from_AWB_cnn(model):
    """Compute new weights V = A @ W @ B.T for CNN model.

    This is STEP 4 of the AWB algorithm for CNN: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Args:
        model: Equinox CNN model with trained A_conv, B_conv, A_feed, B_feed matrices

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    import jax.numpy as jnp

    # Transform conv layer weights
    new_conv_weights = []
    for i in range(model.channel_out):
        # For single input channel (MNIST): weight[i][0] is [H, W]
        transformed = model.A_conv[i] @ model.conv_layers[0].weight[i][0] @ jnp.transpose(model.B_conv[i])
        new_conv_weights.append([transformed])

    model = eqx.tree_at(lambda x: x.conv_layers[0].weight, model, jnp.array(new_conv_weights))

    # Transform feed layer weights
    for j in range(len(model.feed_sizes) - 1):
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        Vb = model.A_feed[j] @ model.feed_layers[j].bias
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)

    return model


def compute_V_from_AWB_cnn3d(model):
    """Compute new weights V = A @ W @ B.T for CNN3D model.

    This is STEP 4 of the AWB algorithm for CNN3D: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Args:
        model: Equinox CNN3D model with trained A_conv1, B_conv1, A_conv2, B_conv2, A_feed, B_feed

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    import jax.numpy as jnp

    # Transform conv layer 1 weights
    new_conv1_weights = []
    for i in range(model.channel_out):
        channel_weights = []
        for c in range(model.channel_in):
            transformed = model.A_conv1[i][c] @ model.conv_layers[0].weight[i][c] @ jnp.transpose(model.B_conv1[i][c])
            channel_weights.append(transformed)
        new_conv1_weights.append(channel_weights)

    model = eqx.tree_at(lambda x: x.conv_layers[0].weight, model, jnp.array(new_conv1_weights))

    # Transform conv layer 2 weights
    new_conv2_weights = []
    for i in range(model.channel_out * 2):
        channel_weights = []
        for c in range(model.channel_out):
            transformed = model.A_conv2[i][c] @ model.conv_layers[1].weight[i][c] @ jnp.transpose(model.B_conv2[i][c])
            channel_weights.append(transformed)
        new_conv2_weights.append(channel_weights)

    model = eqx.tree_at(lambda x: x.conv_layers[1].weight, model, jnp.array(new_conv2_weights))

    # Transform feed layer weights
    for j in range(len(model.feed_sizes) - 1):
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        Vb = model.A_feed[j] @ model.feed_layers[j].bias
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)

    return model


def set_new_AB_matrices_cnn(model, original_feed_sizes, new_feed_sizes, original_filter, new_filter):
    """Set new A/B matrices for CNN (single conv layer) architecture transition.

    Args:
        model: CNN model (with old weights to be transformed)
        original_feed_sizes: Original feed layer sizes
        new_feed_sizes: New feed layer sizes
        original_filter: Original filter size
        new_filter: New filter size

    Returns:
        Model with updated A/B matrices (W_old preserved)
    """
    from ..arch_search.cnn_search import prepABs

    A_feed, B_feed, A_conv, B_conv = prepABs(model, original_feed_sizes, original_filter,
                                              new_feed_sizes, new_filter)

    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.A_conv, model, A_conv)
    model = eqx.tree_at(lambda x: x.B_conv, model, B_conv)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed_sizes)
    model = eqx.tree_at(lambda x: x.filter_size, model, new_filter)

    return model


def set_new_AB_matrices_cnn3d(model, original_feed_sizes, new_feed_sizes, original_filter, new_filter):
    """Set new A/B matrices for CNN3D (two conv layers) architecture transition.

    Args:
        model: CNN3D model (with old weights to be transformed)
        original_feed_sizes: Original feed layer sizes
        new_feed_sizes: New feed layer sizes
        original_filter: Original filter size
        new_filter: New filter size

    Returns:
        Model with updated A/B matrices (W_old preserved)
    """
    from ..arch_search.cnn_search import prepABs_CNN3D

    A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2 = prepABs_CNN3D(
        model, original_feed_sizes, original_filter, new_feed_sizes, new_filter)

    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.A_conv1, model, A_conv1)
    model = eqx.tree_at(lambda x: x.B_conv1, model, B_conv1)
    model = eqx.tree_at(lambda x: x.A_conv2, model, A_conv2)
    model = eqx.tree_at(lambda x: x.B_conv2, model, B_conv2)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed_sizes)
    model = eqx.tree_at(lambda x: x.filter_size, model, new_filter)

    return model

# Added by Claude: GCN-specific AWB function (moved from graph_classification.py)
def set_new_AB_matrices_gcn(model, prev_gcn_sizes, prev_feed_sizes, opt_gcn, opt_mlp):
    """Set new A/B matrices for GCN architecture transition.

    Args:
        model: GCN model
        prev_gcn_sizes: Previous GCN layer sizes
        prev_feed_sizes: Previous feed layer sizes
        opt_gcn: Optimal GCN layer sizes
        opt_mlp: Optimal MLP/feed layer sizes

    Returns:
        Model with updated A/B matrices and architecture
    """
    from ..arch_search.gcn_search import prepABs_GCN

    # Update architecture
    model = eqx.tree_at(lambda x: x.gcn_sizes, model, opt_gcn)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_mlp)

    # Get transformation matrices
    A_feed, B_feed, A_gcn, B_gcn = prepABs_GCN(model, prev_feed_sizes, prev_gcn_sizes)

    model = eqx.tree_at(
        lambda x: (x.A_feed, x.B_feed, x.A_gcn, x.B_gcn),
        model,
        replace=(A_feed, B_feed, A_gcn, B_gcn)
    )

    return model
