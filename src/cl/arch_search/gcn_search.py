"""
Architecture search for GCN models.

Adapted from run_AWB_ALL_functions.py for cl_framework.
Searches local neighborhood for optimal GCN and MLP architecture dimensions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import List, Tuple, Dict, Any

from ..config.constants import (
    DEFAULT_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_LR,
    DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN,
    DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW,
    DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD,
    DEFAULT_NUM_CLASSES,
)


def _compute_search_loss(record_dict: Dict, task_id: int, epochs: int,
                         window: int = None) -> float:
    """Compute average loss from recent training iterations.

    Args:
        record_dict: Dictionary containing training records (iterations dict)
        task_id: Current task ID
        epochs: Number of epochs trained
        window: Number of recent epochs to average (default: 10)

    Returns:
        Average loss value
    """
    if window is None:
        window = DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW

    iterations = record_dict.get('iterations', {})
    if not iterations:
        return float('inf')

    # Get losses from recent iterations
    # Added by Claude: Use integer keys, not string keys with "iter_" prefix
    recent_losses = []
    for j in range(1, min(window + 1, epochs + 1)):
        iteration = (task_id + 1) * epochs - j
        if iteration in iterations:
            record = iterations[iteration]
            if isinstance(record, dict) and 'losses' in record:
                loss_val = record['losses'].get('V', None)
                if loss_val is not None:
                    recent_losses.append(loss_val)
            elif isinstance(record, tuple):
                # Old tuple format: (V, dV, dV_dx, dV_dtheta, H, ...)
                recent_losses.append(record[0])

    if recent_losses:
        return np.mean(recent_losses)
    return float('inf')


def _initialize_gcn_weights(model, initializer):
    """Initialize GCN layer weights randomly.

    Args:
        model: GCN model
        initializer: JAX initializer function

    Returns:
        Model with newly initialized GCN weights
    """
    for k in range(len(model.gcn_layers)):
        weight_shape = (model.gcn_sizes[k], model.gcn_sizes[k + 1])
        bias_shape = (1, model.gcn_sizes[k + 1])

        new_weight = initializer(jax.random.PRNGKey(5 + k), weight_shape)
        new_bias = initializer(jax.random.PRNGKey(5 + k + 100), bias_shape)

        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].weight, model, new_weight)
        model = eqx.tree_at(lambda x, idx=k: x.gcn_layers[idx].bias, model, new_bias)

    return model


def _initialize_mlp_weights(model, initializer, feed_sizes):
    """Initialize MLP layer weights randomly.

    Args:
        model: GCN model with feed layers
        initializer: JAX initializer function
        feed_sizes: List of feed layer sizes

    Returns:
        Model with newly initialized MLP weights
    """
    for j in range(len(model.feed_layers)):
        # Weight shape: (out_size, in_size) for Linear3
        weight_shape = (feed_sizes[j + 1], feed_sizes[j])
        bias_shape = (1, feed_sizes[j + 1])

        new_weight = initializer(jax.random.PRNGKey(5 + j + 200), weight_shape)
        new_bias = initializer(jax.random.PRNGKey(5 + j + 300), bias_shape)

        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].weight, model, new_weight)
        model = eqx.tree_at(lambda x, idx=j: x.feed_layers[idx].bias, model, new_bias)

    return model


def _partition_for_standard_training_gcn(model):
    """Partition GCN model for standard training (freeze A/B, train W).

    Args:
        model: GCN model

    Returns:
        Tuple of (params, static)
    """
    params, static = eqx.partition(model, eqx.is_array)

    # Move AWB matrices to static
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


def arch_search_GCN(original_gcn: List[int], original_mlp: List[int],
                    task: int, trainW_loss: float, og_epochs: int,
                    config: Dict[str, Any], train_loader, mem_train_loader, test_loader,
                    trainer=None, model=None) -> Tuple[List[int], List[int]]:
    """
    Architecture search for GCN and MLP layers (graph classification).

    Searches a local neighborhood to find optimal architecture dimensions.
    Based on arch_search_GCN from run_AWB_ALL_functions.py.

    Args:
        original_gcn: Original GCN architecture sizes [in_size, hidden, ...]
        original_mlp: Original MLP architecture sizes [gcn_out, hidden1, hidden2, n_class]
        task: Current task ID
        trainW_loss: Loss from preliminary training
        og_epochs: Number of epochs per search iteration
        config: Configuration dictionary
        train_loader: Current task training DataLoader
        mem_train_loader: Memory/experience replay DataLoader
        test_loader: Test DataLoader
        trainer: Trainer instance (optional, will create if None)
        model: GCN model to use as template (optional)

    Returns:
        Tuple of (optimal_gcn_sizes, optimal_mlp_sizes)
    """
    from ..core.trainer import Trainer
    from ..models.gcn import GCN

    # Get search parameters from config
    search_epochs = config.get('arch_search_epochs', DEFAULT_ARCH_SEARCH_EPOCHS)
    search_lr = config.get('arch_search_lr', DEFAULT_ARCH_SEARCH_LR)
    step_gcn = config.get('arch_search_step_size_gcn', DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN)
    step_mlp = config.get('arch_search_step_size_mlp', DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP)
    max_iter = config.get('arch_search_max_iter', DEFAULT_ARCH_SEARCH_MAX_ITER)
    loss_threshold = config.get('arch_search_loss_threshold', DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD)
    averaging_window = config.get('arch_search_averaging_window', DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW)
    num_classes = config.get('n_class', DEFAULT_NUM_CLASSES)

    # Override og_epochs with search_epochs
    og_epochs = search_epochs

    # Create trainer if not provided
    if trainer is None:
        trainer = Trainer(loss='class', metric='class', problem='graph')

    # Create initial model for architecture search
    if model is None:
        # Get input size from data
        sample_batch = next(iter(train_loader))
        in_size = sample_batch.x.shape[1]

        arch_model = GCN(
            in_size=in_size,
            feed_sizes=list(original_mlp),
            gcn_sizes=list(original_gcn),
            node_num=sample_batch.x.shape[0],
            out_size=num_classes
        )
    else:
        arch_model = model

    # Set current architecture
    arch_model = eqx.tree_at(
        lambda x: (x.gcn_sizes, x.feed_sizes),
        arch_model,
        replace=(list(original_gcn), list(original_mlp))
    )

    # Initialize weights randomly
    initializer = jax.nn.initializers.glorot_uniform()
    arch_model = _initialize_gcn_weights(arch_model, initializer)
    arch_model = _initialize_mlp_weights(arch_model, initializer, original_mlp)

    # Partition for training (freeze A/B)
    arch_params, arch_static = _partition_for_standard_training_gcn(arch_model)

    # Initial training to get baseline loss
    record_dict_arch = trainer.initialize_record_dict(config, run_id=0)
    optim = optax.adamw(search_lr)
    opt_state = optim.init(arch_params)

    train_config = {
        'batch_size': config.get('batch', config.get('batch_size', 20)),
        'flag': config.get('flag', [1.0, 1.0]),
    }

    # Train on current architecture
    train_data = (train_loader, mem_train_loader, (test_loader, test_loader), (test_loader, test_loader))

    arch_params, arch_static, opt_state, record_dict_arch = trainer.train__CL(
        train_data, arch_params, arch_static, opt_state, optim,
        n_iter=og_epochs, save_iter=config.get('save_iter', 10),
        task_id=task, config=train_config, record_dict=record_dict_arch,
        problem_type='graph', loss_type='classification'
    )

    arch_model = eqx.combine(arch_params, arch_static)

    # Get baseline loss
    loss_orig = _compute_search_loss(record_dict_arch, task, og_epochs, averaging_window)
    print(f"trainWLoss: {trainW_loss:.4f} -- baseline loss: {loss_orig:.4f}")

    # Initialize optimal values
    opt_gcn = list(original_gcn)
    opt_mlp = list(original_mlp)
    loss_opt = loss_orig

    # Extract architecture parameters for search
    # Assume arch_gcn = [in_size, z] and arch_mlp = [z, x1, x2, n_class]
    z2 = original_gcn[1] if len(original_gcn) > 1 else original_gcn[0]
    x1 = original_mlp[1] if len(original_mlp) > 1 else original_mlp[0]
    x2 = original_mlp[2] if len(original_mlp) > 2 else x1

    # Search loop
    n = 1  # Controls neighborhood spread
    search_round = 1
    print(max_iter)
    while (n < 3) or (loss_opt < loss_threshold * loss_orig):
        # Search over GCN architecture
        for j in range(3):
            curr_gcn = [original_gcn[0], z2 + n * (j + 1) * step_gcn]

            # Update GCN architecture and reinitialize weights
            arch_model = eqx.tree_at(lambda x: x.gcn_sizes, arch_model, curr_gcn)
            arch_model = _initialize_gcn_weights(arch_model, initializer)

            # Search over MLP architecture
            for k in range(3):
                for r in range(3):
                    curr_mlp = [
                        curr_gcn[-1],
                        x1 + n * (k + 1) * step_mlp,
                        x2 + n * (r + 1) * step_mlp,
                        num_classes
                    ]

                    # Update MLP architecture and reinitialize weights
                    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, curr_mlp)
                    arch_model = _initialize_mlp_weights(arch_model, initializer, curr_mlp)

                    print(f"========= curr_gcn: {curr_gcn} ========== curr_mlp: {curr_mlp}")

                    # Partition and train
                    record_dict_arch = trainer.initialize_record_dict(config, run_id=0)
                    optim2 = optax.adamw(search_lr)
                    arch_params, arch_static = _partition_for_standard_training_gcn(arch_model)
                    opt_state2 = optim2.init(arch_params)

                    arch_params, arch_static, opt_state2, record_dict_arch = trainer.train__CL(
                        train_data, arch_params, arch_static, opt_state2, optim2,
                        n_iter=og_epochs, save_iter=config.get('save_iter', 10),
                        task_id=task, config=train_config, record_dict=record_dict_arch,
                        problem_type='graph', loss_type='classification'
                    )

                    arch_model = eqx.combine(arch_params, arch_static)

                    # Evaluate candidate architecture
                    loss_poll = _compute_search_loss(record_dict_arch, task, og_epochs, averaging_window)

                    if loss_poll < loss_opt:
                        opt_gcn = curr_gcn[:]
                        opt_mlp = curr_mlp[:]
                        loss_opt = loss_poll

                    search_round += 1
                    print(f"ROUND {search_round}: opt_gcn: {opt_gcn} ---- opt_mlp: {opt_mlp} ---opt_loss: {loss_opt:.4f}")

        n += 3  # Skip to next neighborhood

    return opt_gcn, opt_mlp


def prepABs_GCN(model, prev_feed_sizes: List[int], prev_gcn_sizes: List[int]):
    """
    Prepare A and B transformation matrices for GCN architecture transition.

    Based on logic from run_AWB_ALL_functions.py train_model_graph function.

    Args:
        model: GCN model with new architecture
        prev_feed_sizes: Previous feed layer sizes
        prev_gcn_sizes: Previous GCN layer sizes

    Returns:
        Tuple of (A_feed, B_feed, A_gcn, B_gcn) transformation matrices
    """
    opt_feed_sizes = model.feed_sizes
    opt_gcn_sizes = model.gcn_sizes
    initializer = jax.nn.initializers.glorot_uniform()

    # Added by Claude: Extract ACTUAL layer dimensions from model weights
    # The model.feed_sizes may have been updated, but the actual layer weights still have old dimensions
    actual_feed_sizes = [model.feed_layers[0].weight.shape[1]]  # First layer input size
    for layer in model.feed_layers:
        actual_feed_sizes.append(layer.weight.shape[0])  # Output size

    actual_gcn_sizes = [model.gcn_layers[0].weight.shape[0]]  # First GCN layer input size
    for layer in model.gcn_layers:
        actual_gcn_sizes.append(layer.weight.shape[1])  # Output size

    # Check what changed (compare actual current dimensions to desired new dimensions)
    feed_changed = (list(actual_feed_sizes[1:-1]) != list(opt_feed_sizes[1:-1]))
    gcn_changed = (list(actual_gcn_sizes[1:]) != list(opt_gcn_sizes[1:]))

    if feed_changed and gcn_changed:
        print("New feed AND gcn!!!------------------")
        # Both changed: need transformation matrices for all
        # Added by Claude: Use actual_* sizes (current layer dimensions) instead of prev_*
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                  for x, y in zip(actual_feed_sizes[:-1], opt_feed_sizes[:-1])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                  for x, y in zip(actual_feed_sizes[1:], opt_feed_sizes[1:])]
        A_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                 for x, y in zip(actual_gcn_sizes[:-1], opt_gcn_sizes[:-1])]
        B_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                 for x, y in zip(actual_gcn_sizes[1:], opt_gcn_sizes[1:])]

    elif feed_changed and not gcn_changed:
        print("New FEED ONLY!!!------------------")
        # Only feed changed
        # Added by Claude: Use actual_* sizes (current layer dimensions) instead of prev_*
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                  for x, y in zip(actual_feed_sizes[:-1], opt_feed_sizes[:-1])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x))
                  for x, y in zip(actual_feed_sizes[1:], opt_feed_sizes[1:])]
        # GCN matrices stay identity
        A_gcn = [jnp.eye(x, x) for x in actual_gcn_sizes[:-1]]
        B_gcn = [jnp.eye(x, x) for x in actual_gcn_sizes[1:]]

    elif not feed_changed and gcn_changed:
        print("New GCN ONLY!!!------------------")
        # Only GCN changed
        # Feed matrices stay identity (but first one may need to change if gcn output changed)
        # Added by Claude: Use actual_* sizes (current layer dimensions) instead of prev_*
        if actual_gcn_sizes[-1] != opt_gcn_sizes[-1]:
            # First feed layer input size changed
            A_feed = [initializer(jax.random.PRNGKey(5), (opt_feed_sizes[0], actual_feed_sizes[0]))]
            A_feed += [jnp.eye(x, x) for x in actual_feed_sizes[1:-1]]
            B_feed = [jnp.eye(x, x) for x in actual_feed_sizes[1:]]
        else:
            A_feed = [jnp.eye(x, x) for x in actual_feed_sizes[:-1]]
            B_feed = [jnp.eye(x, x) for x in actual_feed_sizes[1:]]

        A_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                 for x, y in zip(actual_gcn_sizes[:-1], opt_gcn_sizes[:-1])]
        B_gcn = [initializer(jax.random.PRNGKey(5), (y, x))
                 for x, y in zip(actual_gcn_sizes[1:], opt_gcn_sizes[1:])]

    else:
        print("No architecture change - using identity matrices")
        # No change: use identity matrices
        # Added by Claude: Use actual_* sizes (current layer dimensions) instead of prev_*
        A_feed = [jnp.eye(x, x) for x in actual_feed_sizes[:-1]]
        B_feed = [jnp.eye(x, x) for x in actual_feed_sizes[1:]]
        A_gcn = [jnp.eye(x, x) for x in actual_gcn_sizes[:-1]]
        B_gcn = [jnp.eye(x, x) for x in actual_gcn_sizes[1:]]

    return A_feed, B_feed, A_gcn, B_gcn
