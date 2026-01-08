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
    """Architecture search for CNN and CNN3D (BACKWARD COMPATIBLE).

    # Added by Claude: Now delegates to core.arch_search.search_architecture()
    This is a backward-compatible wrapper that uses the new generic search algorithm.
    Automatically detects CNN vs CNN3D based on channel_in config.

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

    # Added by Claude: Detect CNN3D based on channel_in or data type
    # CNN3D is used for multi-channel images (CIFAR-10/100 with 3 RGB channels)
    is_cnn3d = (
        channel_in > 1 or
        config.get('data', '') in ['cifar10', 'cifar100'] or
        config.get('network', '') == 'cnn3d'
    )

    if is_cnn3d:
        # CNN3D: Two conv layers, need to calculate feed input size differently
        # First conv: (input_size - filter_size + 1) // 2 after pooling
        # Second conv: (first_out - filter_size + 1) // 2 after pooling
        first_conv_out = (input_size - filter_size + 1) // 2
        second_conv_out = (first_conv_out - filter_size + 1) // 2
        # Second conv outputs channel_out * 2 channels
        feed_input_size = (channel_out * 2) * second_conv_out * second_conv_out
    else:
        # CNN: Single conv layer
        conv_output = input_size - filter_size + 1
        pool_output = conv_output // 2
        feed_input_size = channel_out * pool_output * pool_output

    # Adjust feed_sizes to have correct input dimension
    adjusted_feed_sizes = list(feed_sizes)
    adjusted_feed_sizes[0] = feed_input_size

    # Added by Claude: Create appropriate model type based on channel_in
    if is_cnn3d:
        current_model = CNN3D(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=adjusted_feed_sizes,
            input_size=input_size,
            channel_in=channel_in,
            channel_out=channel_out,
        )
    else:
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


def prepABs(model, prev_feed_sizes, prev_filter_size, new_feed_sizes, new_filter_size):
    """
    Prepare A and B transformation matrices for CNN architecture transition.

    The AWB algorithm transforms OLD weights W_old using: A @ W_old @ B.T
    - A transforms output dimensions: shape (new_out, old_out)
    - B transforms input dimensions: shape (new_in, old_in)

    W_old stays in the model - we do NOT recreate the model.

    Args:
        model: CNN model with OLD weights (W_old)
        prev_feed_sizes: Previous/old feed layer sizes
        prev_filter_size: Previous/old filter size
        new_feed_sizes: New/target feed layer sizes
        new_filter_size: New/target filter size

    Returns:
        Tuple of (A_feed, B_feed, A_conv, B_conv) transformation matrices
    """
    initializer = jax.nn.initializers.glorot_uniform()

    # Check what changed
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(new_feed_sizes[1:3]))
    filter_changed = (new_filter_size != prev_filter_size)

    print(f"  prepABs: filter {prev_filter_size}→{new_filter_size}, "
          f"feed {prev_feed_sizes}→{new_feed_sizes}")

    # A_conv/B_conv: Transform conv filter from old→new size
    # A_conv @ W_old @ B_conv.T: (new, old) @ (old, old) @ (old, new) = (new, new)
    # Stored as stacked 3D arrays: (channel_out, new_filter_size, prev_filter_size)
    if filter_changed:
        print(f"  Conv filter changed: A_conv/B_conv shape = ({new_filter_size}, {prev_filter_size})")
        A_conv = jnp.stack([initializer(jax.random.PRNGKey(5 + i), (new_filter_size, prev_filter_size))
                            for i in range(model.channel_out)])
        B_conv = jnp.stack([initializer(jax.random.PRNGKey(6 + i), (new_filter_size, prev_filter_size))
                            for i in range(model.channel_out)])
    else:
        # No filter change - identity matrices stacked
        A_conv = jnp.stack([jnp.eye(prev_filter_size) for _ in range(model.channel_out)])
        B_conv = jnp.stack([jnp.eye(prev_filter_size) for _ in range(model.channel_out)])

    # A_feed/B_feed: Transform feed layers from old→new dimensions
    # A_feed[i] shape: (new_out, old_out) for layer i
    # B_feed[i] shape: (new_in, old_in) for layer i
    # A_feed @ W_old @ B_feed.T: (new_out, old_out) @ (old_out, old_in) @ (old_in, new_in) = (new_out, new_in)

    # A_feed transforms output dimensions (indices 1: of feed_sizes)
    A_feed = [initializer(jax.random.PRNGKey(7), (new_out, old_out))
              for old_out, new_out in zip(prev_feed_sizes[1:], new_feed_sizes[1:])]

    # B_feed transforms input dimensions (indices :-1 of feed_sizes)
    B_feed = [initializer(jax.random.PRNGKey(8), (new_in, old_in))
              for old_in, new_in in zip(prev_feed_sizes[:-1], new_feed_sizes[:-1])]

    return A_feed, B_feed, A_conv, B_conv


def prepABs_CNN3D(model, prev_feed_sizes, prev_filter_size, new_feed_sizes, new_filter_size):
    """
    Prepare A and B transformation matrices for CNN3D architecture transition.

    The AWB algorithm transforms OLD weights W_old using: A @ W_old @ B.T
    - A transforms output dimensions: shape (new_out, old_out)
    - B transforms input dimensions: shape (new_in, old_in)

    W_old stays in the model - we do NOT recreate the model.
    CNN3D has two conv layers, so we need A/B for both conv1 and conv2.

    Args:
        model: CNN3D model with OLD weights (W_old)
        prev_feed_sizes: Previous/old feed layer sizes
        prev_filter_size: Previous/old filter size
        new_feed_sizes: New/target feed layer sizes
        new_filter_size: New/target filter size

    Returns:
        Tuple of (A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2) transformation matrices
    """
    initializer = jax.nn.initializers.glorot_uniform()

    # Check what changed
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(new_feed_sizes[1:3]))
    filter_changed = (new_filter_size != prev_filter_size)

    print(f"  prepABs_CNN3D: filter {prev_filter_size}→{new_filter_size}, "
          f"feed {prev_feed_sizes}→{new_feed_sizes}")

    # A_conv/B_conv: Transform conv filter from old→new size
    # Shape: (new_filter, old_filter) to transform W_old of shape (old_filter, old_filter)
    # Stored as stacked 4D arrays: (channel_out, channel_in, new_filter_size, prev_filter_size)
    if filter_changed:
        print(f"  Conv filter changed: A_conv/B_conv shape = ({new_filter_size}, {prev_filter_size})")
        # Conv1: (channel_out, channel_in, new_filter, old_filter)
        A_conv1 = jnp.stack([
            jnp.stack([initializer(jax.random.PRNGKey(5 + j * model.channel_in + c), (new_filter_size, prev_filter_size))
                       for c in range(model.channel_in)])
            for j in range(model.channel_out)
        ])
        B_conv1 = jnp.stack([
            jnp.stack([initializer(jax.random.PRNGKey(1000 + j * model.channel_in + c), (new_filter_size, prev_filter_size))
                       for c in range(model.channel_in)])
            for j in range(model.channel_out)
        ])
        # Conv2: (channel_out*2, channel_out, new_filter, old_filter)
        A_conv2 = jnp.stack([
            jnp.stack([initializer(jax.random.PRNGKey(2000 + j * model.channel_out + c), (new_filter_size, prev_filter_size))
                       for c in range(model.channel_out)])
            for j in range(model.channel_out * 2)
        ])
        B_conv2 = jnp.stack([
            jnp.stack([initializer(jax.random.PRNGKey(3000 + j * model.channel_out + c), (new_filter_size, prev_filter_size))
                       for c in range(model.channel_out)])
            for j in range(model.channel_out * 2)
        ])
    else:
        # No filter change - identity matrices stacked
        A_conv1 = jnp.stack([jnp.stack([jnp.eye(prev_filter_size) for c in range(model.channel_in)]) for j in range(model.channel_out)])
        B_conv1 = jnp.stack([jnp.stack([jnp.eye(prev_filter_size) for c in range(model.channel_in)]) for j in range(model.channel_out)])
        A_conv2 = jnp.stack([jnp.stack([jnp.eye(prev_filter_size) for c in range(model.channel_out)]) for j in range(model.channel_out * 2)])
        B_conv2 = jnp.stack([jnp.stack([jnp.eye(prev_filter_size) for c in range(model.channel_out)]) for j in range(model.channel_out * 2)])

    # A_feed/B_feed: Transform feed layers from old→new dimensions
    # A_feed[i] shape: (new_out, old_out) for layer i
    # B_feed[i] shape: (new_in, old_in) for layer i

    # A_feed transforms output dimensions (indices 1: of feed_sizes)
    A_feed = [initializer(jax.random.PRNGKey(9), (new_out, old_out))
              for old_out, new_out in zip(prev_feed_sizes[1:], new_feed_sizes[1:])]

    # B_feed transforms input dimensions (indices :-1 of feed_sizes)
    B_feed = [initializer(jax.random.PRNGKey(10), (new_in, old_in))
              for old_in, new_in in zip(prev_feed_sizes[:-1], new_feed_sizes[:-1])]

    return A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2
