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

from contlearn.config.constants import (
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_CHANGE_THRESHOLD_HIGH,
    DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA,
    DEFAULT_AWB_AB_THRESHOLD_BASE,
)


def compute_avg_loss(record_dict, task_id, epochs, window=None):
    """
    Compute average loss over last `window` epochs from training record.

    Args:
        record_dict: Dictionary containing training records with keys like "train{epoch}"
        task_id: Current task ID
        epochs: Total epochs per task
        window: Number of epochs to average (default: DEFAULT_AWB_AVERAGING_WINDOW)

    Returns:
        Average loss value over the specified window
    """
    if window is None:
        window = DEFAULT_AWB_AVERAGING_WINDOW

    losses = []
    for j in range(1, window + 1):
        key = "train" + str((task_id + 1) * epochs - j)
        if key in record_dict:
            # record_dict[key] is a tuple: (V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)
            # V (index 0) is the loss value
            losses.append(record_dict[key][0])

    if not losses:
        return float('inf')
    return np.mean(losses)


def should_change_arch(trainWLoss, end_last0, end_last,
                       threshold_high=None, min_delta=None):
    """
    Decide if architecture change is needed based on loss ratio thresholds.

    The decision logic (from example.py):
        - If ratio > threshold_high AND loss increased by min_delta: change_arch = True
        - If ratio > threshold_high AND loss did not increase: change_arch = False
        - If ratio <= threshold_high: change_arch = False

    Args:
        trainWLoss: Current training loss after preliminary training
        end_last0: Loss at end of task 0 (baseline)
        end_last: Loss at end of previous task
        threshold_high: High threshold for loss ratio (default: 0.45)
        min_delta: Minimum loss increase to trigger change (default: 0.01)

    Returns:
        Boolean indicating whether architecture should be changed
    """
    if threshold_high is None:
        threshold_high = DEFAULT_AWB_CHANGE_THRESHOLD_HIGH
    if min_delta is None:
        min_delta = DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA

    ratio = trainWLoss / end_last0

    if (ratio > threshold_high) and (end_last + min_delta <= trainWLoss):
        return True
    elif (ratio > threshold_high) and (end_last + min_delta > trainWLoss):
        return False
    else:  # ratio <= threshold_high
        return False


def compute_ab_threshold(trainWLoss, end_last, base_threshold=None):
    """
    Compute dynamic threshold for AB training convergence.

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


def set_new_AB_matrices(model, original_arch, new_arch, seed=5):
    """
    Initialize A/B matrices for architecture transition.

    When the architecture changes from original_arch to new_arch,
    we need to create transformation matrices A and B such that
    the forward pass becomes: A @ W @ B.T

    Args:
        model: Current equinox model (MLP)
        original_arch: Original architecture sizes list [in, h1, h2, ..., out]
        new_arch: New architecture sizes list [in, h1', h2', ..., out]
        seed: Random seed for initializer

    Returns:
        Updated model with new A, B matrices and sizes
    """
    initializer = jax.nn.initializers.glorot_uniform()

    # A matrices: transform output dimensions [new_out, old_out]
    A_list = [
        initializer(jax.random.PRNGKey(seed), (y_new, y_old))
        for y_old, y_new in zip(original_arch[1:], new_arch[1:])
    ]

    # B matrices: transform input dimensions [new_in, old_in]
    B_list = [
        initializer(jax.random.PRNGKey(seed), (x_new, x_old))
        for x_old, x_new in zip(original_arch[:-1], new_arch[:-1])
    ]

    # Update model with new sizes and A/B matrices
    model = eqx.tree_at(lambda x: x.sizes, model, new_arch)
    model = eqx.tree_at(lambda x: x.A, model, A_list)
    model = eqx.tree_at(lambda x: x.B, model, B_list)

    return model


def compute_V_from_AWB(model):
    """
    Compute new weights V = A @ W @ B.T for all layers.

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
    """
    Partition model for A/B training (freeze W, train A/B).

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
    """
    Partition model for standard training (freeze A/B, train W).

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


def save_layer_weights(model):
    """
    Save current layer weights and biases before architecture search.

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
    """
    Restore saved layer weights and biases to model.

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


def create_optimizer_for_phase(phase, learning_rate=1e-4):
    """
    Create appropriate optimizer for each training phase.

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
