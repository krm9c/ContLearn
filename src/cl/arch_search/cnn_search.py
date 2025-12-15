"""
Architecture search for CNN models.

Contains search functions for:
- CNN (single-channel images like MNIST)
- CNN3D (multi-channel images like CIFAR)
- Preparation functions for A/B transformation matrices
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from ..models.cnn import CNN, CNN3D
from ..models.layers import Linear2
from ..config.constants import (
    DEFAULT_CNN_ARCH_SEARCH_EPOCHS,
    DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_LR,
    DEFAULT_ARCH_SEARCH_BATCH_SIZE,
    DEFAULT_ARCH_SEARCH_EXP_REPLAY,
    DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD,
    DEFAULT_ARCH_SEARCH_HIDDEN_RANGE,
    DEFAULT_ARCH_SEARCH_FILTER_MIN,
    DEFAULT_ARCH_SEARCH_FILTER_MAX,
    DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT,
    DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL,
    DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_ITER_INCREMENT,
    DEFAULT_NUM_CLASSES,
    DEFAULT_INPUT_SIZE_CIFAR,
    DEFAULT_RANDOM_KEY_OFFSET_CONV2,
    DEFAULT_RANDOM_KEY_OFFSET_ACONV2,
    DEFAULT_RANDOM_KEY_OFFSET_BCONV2,
)


def arch_search_CNN_fresh(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                          dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp,
                          trainer=None):
    """Architecture search for CNN (BACKWARD COMPATIBLE).

    # Added by Claude: Now delegates to core.arch_search.search_architecture()
    This is a backward-compatible wrapper that uses the new generic search algorithm.

    Args:
        filter_size: Current filter size
        feed_sizes: Current feed layer sizes [feed_input, h1, h2, n_class]
        task: Current task ID
        trainW_loss: Training loss from preliminary training
        og_epochs: Base epochs (overridden by config)
        config: Configuration dictionary
        dataloader_curr: Current task training data
        dataloader_exp: Experience replay data
        test_loader_curr: Current task test data
        test_loader_exp: Experience replay test data
        trainer: Trainer instance (optional)

    Returns:
        Tuple of (opt_feed_sizes, opt_filter_size)
    """
    # Added by Claude: Use generic search_architecture from core module
    from ..core.arch_search import search_architecture

    # Get model configuration for creating reference model
    channel_out = config.get('channel_out', 3)
    input_size = config.get('input_size', 28)
    channel_in = config.get('channel_in', 1)

    # Calculate feed input size for current filter
    conv_output = input_size - filter_size + 1
    pool_output = conv_output // 2
    feed_input_size = channel_out * pool_output * pool_output

    # Adjust feed_sizes to have correct input dimension
    adjusted_feed_sizes = list(feed_sizes)
    adjusted_feed_sizes[0] = feed_input_size

    # Create reference CNN model for search interface
    current_model = CNN(
        key=jax.random.PRNGKey(0),
        filter_size=filter_size,
        feed_sizes=adjusted_feed_sizes,
        input_size=input_size,
        channel_in=channel_in,
        channel_out=channel_out,
    )

    # Baseline architecture: (filter_size, feed_sizes) tuple
    baseline_arch = (filter_size, adjusted_feed_sizes)

    # Delegate to generic core search function
    opt_arch = search_architecture(
        model=current_model,
        baseline_arch=baseline_arch,
        task_id=task,
        baseline_loss=trainW_loss,
        dataloader_curr=dataloader_curr,
        dataloader_exp=dataloader_exp,
        test_loader_curr=test_loader_curr,
        test_loader_exp=test_loader_exp,
        config=config,
        trainer=trainer,
        model_type='cnn'
    )

    # Unpack result: (filter_size, feed_sizes)
    opt_filter, opt_feed_sizes = opt_arch

    return opt_feed_sizes, opt_filter


# NOTE: Legacy functions arch_search_CNN and arch_search_CNN3D have been moved to
# legacy/cnn_search_legacy.py. These functions used hardcoded DEFAULT_* constants
# instead of reading from config files. The current codebase uses arch_search_CNN_fresh
# which properly reads all configuration parameters from config files.


def prepABs(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN architecture search.

    When filter size changes, the flattened conv output size (feed_sizes[0]) also changes.
    This function handles all three cases:
    1. Both feed hidden layers AND conv filter change
    2. Only feed hidden layers change (filter stays same)
    3. Only conv filter changes (feed hidden layers stay same, but feed_sizes[0] changes)

    Args:
        model: Current CNN model with new architecture
        prev_feed_sizes: Previous feed layer sizes
        prev_filter_size: Previous filter size

    Returns:
        Tuple of (A_feed, B_feed, A_conv, B_conv) transformation matrices
    """
    opt_MLParch = list(model.feed_sizes)  # Added by Claude: Convert to list for comparison
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    # Added by Claude: Check if hidden layers changed (indices 1:3), not including feed_sizes[0]
    # which changes when filter size changes
    # Convert to lists to ensure proper comparison (in case model.feed_sizes is JAX array)
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(opt_MLParch[1:3]))
    filter_changed = (opt_filter != prev_filter_size)

    if hidden_changed and filter_changed:
        print("New feed AND conv!!!------------------")
        # Both changed: need transformation matrices for feed layers
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Added by Claude: When filter size changes, model recreated with new size - use identity
        B_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        A_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
    elif hidden_changed and not filter_changed:
        print("New FEED ONLY!!!------------------")
        # Only hidden layers changed: need transformation for feed layers, identity for conv
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        B_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        A_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
    else:
        # Added by Claude: Only filter changed - hidden layers stay same but feed_sizes[0] changes
        # because the flattened conv output size depends on filter size
        print("New CONV ONLY!!!------------------")
        # Added by Claude: When filter size changes, model has been recreated with NEW filter size
        # Conv weights are already at new size, so use identity matrices (no transformation needed)
        # A/B training will then learn to transform these fresh weights to perform better
        B_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        A_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        # Added by Claude: For feed layers, hidden layers are unchanged so use identity,
        # BUT B_feed[0] maps from prev_feed_sizes[0] (old flattened size) to opt_MLParch[0] (new flattened size)
        # A_feed maps output dimensions, B_feed maps input dimensions
        # B_feed[i] shape: (new_input_size, old_input_size) for layer i
        # A_feed[i] shape: (new_output_size, old_output_size) for layer i
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]  # Output dims unchanged
        # B_feed[0] needs to transform from prev_feed_sizes[0] to opt_MLParch[0]
        B_feed = []
        B_feed.append(initializer(jax.random.PRNGKey(10), (opt_MLParch[0], prev_feed_sizes[0])))
        # Rest of B_feed are identity (hidden layer input dims unchanged)
        for x in prev_feed_sizes[1:-1]:
            B_feed.append(jnp.eye(x, x))

    return A_feed, B_feed, A_conv, B_conv


def prepABs_CNN3D(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN3D architecture search.
    Handles multi-channel conv layers (A_conv1, B_conv1, A_conv2, B_conv2).

    When filter size changes, the flattened conv output size (feed_sizes[0]) also changes.
    This function handles all three cases:
    1. Both feed hidden layers AND conv filter change
    2. Only feed hidden layers change (filter stays same)
    3. Only conv filter changes (feed hidden layers stay same, but feed_sizes[0] changes)

    Args:
        model: Current CNN3D model with new architecture
        prev_feed_sizes: Previous feed layer sizes
        prev_filter_size: Previous filter size

    Returns:
        Tuple of (A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2) transformation matrices
    """
    opt_MLParch = list(model.feed_sizes)  # Added by Claude: Convert to list for comparison
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    # Added by Claude: Check if hidden layers changed (indices 1:3), not including feed_sizes[0]
    # Convert to lists to ensure proper comparison (in case model.feed_sizes is JAX array)
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(opt_MLParch[1:3]))
    filter_changed = (opt_filter != prev_filter_size)

    if hidden_changed and filter_changed:
        print("CNN3D: New feed AND conv!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Added by Claude: When filter size changes, model recreated with new size - use identity
        A_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]

    elif hidden_changed and not filter_changed:
        print("CNN3D: New FEED ONLY!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        A_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]

    else:
        # Added by Claude: Only filter changed - hidden layers stay same but feed_sizes[0] changes
        print("CNN3D: New CONV ONLY!!!------------------")
        # Added by Claude: When filter size changes, model has been recreated with NEW filter size
        # Conv weights are already at new size, so use identity matrices (no transformation needed)
        # A/B training will then learn to transform these fresh weights to perform better
        A_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        # Added by Claude: For feed layers, hidden layers are unchanged so use identity,
        # BUT B_feed[0] maps from prev_feed_sizes[0] (old flattened size) to opt_MLParch[0] (new flattened size)
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]  # Output dims unchanged
        # B_feed[0] needs to transform from prev_feed_sizes[0] to opt_MLParch[0]
        B_feed = []
        B_feed.append(initializer(jax.random.PRNGKey(10), (opt_MLParch[0], prev_feed_sizes[0])))
        # Rest of B_feed are identity (hidden layer input dims unchanged)
        for x in prev_feed_sizes[1:-1]:
            B_feed.append(jnp.eye(x, x))

    return A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2
