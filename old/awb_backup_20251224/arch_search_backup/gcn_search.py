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


def arch_search_GCN(original_gcn: List[int], original_mlp: List[int],
                    task: int, trainW_loss: float, og_epochs: int,
                    config: Dict[str, Any], train_loader, mem_train_loader, test_loader,
                    trainer=None, model=None) -> Tuple[List[int], List[int]]:
    """Architecture search for GCN (BACKWARD COMPATIBLE).

    # Added by Claude: Now delegates to core.arch_search.search_architecture()
    This is a backward-compatible wrapper that uses the new generic search algorithm.

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
        trainer: Trainer instance (optional)
        model: GCN model to use as template (optional)

    Returns:
        Tuple of (optimal_gcn_sizes, optimal_mlp_sizes)
    """
    # Added by Claude: Use generic search_architecture from core module
    from ..core.arch_search import search_architecture
    from ..models.gcn import GCN

    # Create reference GCN model if not provided
    if model is None:
        # Get input size from data
        sample_batch = next(iter(train_loader))
        in_size = sample_batch.x.shape[1]
        num_classes = config.get('n_class', DEFAULT_NUM_CLASSES)

        model = GCN(
            in_size=in_size,
            feed_sizes=list(original_mlp),
            gcn_sizes=list(original_gcn),
            node_num=sample_batch.x.shape[0],
            out_size=num_classes
        )

    # Baseline architecture: (gcn_sizes, mlp_sizes) tuple
    baseline_arch = (original_gcn, original_mlp)

    # Delegate to generic core search function
    opt_arch = search_architecture(
        model=model,
        baseline_arch=baseline_arch,
        task_id=task,
        baseline_loss=trainW_loss,
        dataloader_curr=train_loader,
        dataloader_exp=mem_train_loader,
        test_loader_curr=test_loader,
        test_loader_exp=test_loader,
        config=config,
        trainer=trainer,
        model_type='gcn'
    )

    # Unpack result: (gcn_sizes, mlp_sizes)
    opt_gcn, opt_mlp = opt_arch

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
