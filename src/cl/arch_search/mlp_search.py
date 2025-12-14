"""
Architecture search for MLP models.

Provides architecture search algorithms for finding optimal MLP architectures
during AWB (Adaptive Weight Basis) continual learning.

The architecture search explores different hidden layer sizes to find
configurations that minimize loss on the current task while maintaining
compatibility with the AWB transformation pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import Dict, Any, List, Tuple, Optional

from ..models.mlp import MLP, create_mlp
from ..core.trainer import Trainer
from ..config.constants import (
    DEFAULT_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_THRESHOLD,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_MLP_INCREMENT,
    DEFAULT_ARCH_SEARCH_LARGE_INCREMENT,
    DEFAULT_ARCH_SEARCH_RANGE,
    DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW,
    DEFAULT_ARCH_SEARCH_LR,
    DEFAULT_BATCH_SIZE_VECTOR,
    DEFAULT_REPLAY_BUFFER_VECTOR,
)


def _create_search_model(arch: List[int], seed: int = 0, awb_enabled: bool = False) -> MLP:
    """Create an MLP model with specified architecture for search.

    Args:
        arch: Architecture sizes list [input, h1, h2, ..., output]
        seed: Random seed for weight initialization
        awb_enabled: Whether to enable AWB matrices

    Returns:
        Initialized MLP model
    """
    key = jax.random.PRNGKey(seed)
    return MLP(sizes=arch, key=key, awb_enabled=awb_enabled)


def _reinitialize_weights(model: MLP, seed: int = 0) -> MLP:
    """Reinitialize model weights with glorot uniform initialization.

    Used before each architecture search trial to ensure fair comparison
    between different architectures.

    Args:
        model: MLP model to reinitialize
        seed: Random seed for initialization

    Returns:
        Model with reinitialized weights
    """
    initializer = jax.nn.initializers.glorot_uniform()

    # Reinitialize each layer's weights and biases
    for j in range(len(model.sizes) - 1):
        in_size = model.sizes[j]
        out_size = model.sizes[j + 1]

        weight = initializer(jax.random.PRNGKey(seed + j), (out_size, in_size))
        bias = initializer(jax.random.PRNGKey(seed + j + 100), (1, out_size))

        model = eqx.tree_at(lambda x: x.layers[j].weight, model, weight)
        model = eqx.tree_at(lambda x: x.layers[j].bias, model, bias)

    return model


def _compute_search_loss(record_dict: Dict, task_id: int, epochs: int,
                         window: int = None) -> float:
    """Compute average loss over last `window` epochs for architecture comparison.

    Args:
        record_dict: Dictionary containing training records
        task_id: Current task ID
        epochs: Total epochs trained
        window: Number of epochs to average (default: DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW)

    Returns:
        Average loss value
    """
    if window is None:
        window = DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW

    losses = []
    iterations = record_dict.get('iterations', record_dict)

    for j in range(1, window + 1):
        iteration = (task_id + 1) * epochs - j
        if iteration in iterations:
            record = iterations[iteration]
            if isinstance(record, dict) and 'losses' in record:
                losses.append(record['losses'].get('V', 0))
            elif isinstance(record, tuple):
                losses.append(record[0])

    if not losses:
        return float('inf')
    return np.mean(losses)


def _train_candidate_architecture(
    model: MLP,
    trainer: Trainer,
    task_id: int,
    epochs: int,
    config: Dict[str, Any],
    dataloader_curr,
    dataloader_exp,
    test_loader_curr,
    test_loader_exp,
    averaging_window: int = None,
) -> Tuple[MLP, float]:
    """Train a candidate architecture and return its final loss.

    Args:
        model: MLP model with candidate architecture
        trainer: Trainer instance
        task_id: Current task ID
        epochs: Number of training epochs
        config: Training configuration
        dataloader_curr: Current task training data
        dataloader_exp: Experience replay data
        test_loader_curr: Current task test data
        test_loader_exp: Experience replay test data
        averaging_window: Number of epochs to average for loss computation

    Returns:
        Tuple of (trained_model, average_loss)
    """
    # Partition model for standard training (A/B frozen if present)
    params, static = eqx.partition(model, eqx.is_array)

    # If AWB enabled, move A/B to static
    if model.awb_enabled and model.A is not None:
        static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
        params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

    # Added by Claude: Get arch search LR from config with fallback to default
    search_lr = config.get('arch_search_lr', DEFAULT_ARCH_SEARCH_LR)

    # Create optimizer and initialize state
    optim = optax.adam(search_lr)
    opt_state = optim.init(params)

    # Initialize record dict for this search trial
    poll_dict = trainer.initialize_record_dict(config, run_id=0)

    # Training config for search
    train_config = {
        'batch_size': config.get('arch_search_batch_size', config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR)),
        'problem': config.get('problem', 'vectors'),
        'data_id': config.get('data', 'sine'),
        'flag': config.get('flag', [1.0, 1.0]),
        'len_exp_replay': config.get('arch_search_exp_replay', config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR)),
        'network': config.get('network', 'fcnn'),
    }

    # Prepare training data tuple
    train_data = (
        dataloader_curr, dataloader_exp,
        (test_loader_curr, test_loader_exp),
        (test_loader_curr, test_loader_exp)
    )

    # Train the model
    params, static, opt_state, poll_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=epochs, save_iter=config.get('save_iter', 10),
        task_id=task_id, config=train_config, record_dict=poll_dict,
        problem_type='vectors', loss_type='regression'
    )

    # Compute average loss
    avg_loss = _compute_search_loss(poll_dict, task_id, epochs, window=averaging_window)

    # Combine back to model
    model = eqx.combine(params, static)

    return model, avg_loss


def arch_search_MLP(
    original_arch: List[int],
    task_id: int,
    trainW_loss: float,
    og_epochs: int,
    config: Dict[str, Any],
    dataloader_curr,
    dataloader_exp,
    test_loader_curr,
    test_loader_exp,
    current_model: MLP = None,
) -> List[int]:
    """Architecture search for MLP regression models.

    Searches for an optimal architecture by training candidate architectures
    with different hidden layer sizes and selecting the one with lowest loss.

    The search process:
    1. Uses the current model's loss (trainW_loss) as baseline for comparison
    2. Iteratively explores architectures with incrementally larger hidden layers
    3. Uses a grid search over hidden layer sizes within a search range
    4. Returns the architecture with lowest loss (or original if no improvement)

    Args:
        original_arch: Original architecture sizes [input, h1, h2, ..., output]
        task_id: Current task ID
        trainW_loss: Training loss from preliminary training on current model (baseline)
        og_epochs: Base epochs (overridden by arch_search_epochs from config)
        config: Configuration dictionary with:
            - arch_search_epochs: Epochs per candidate (default: 100)
            - arch_search_threshold: Loss improvement threshold (default: 0.9)
            - arch_search_max_iter: Max search iterations (default: 10)
            - arch_search_mlp_increment: Hidden size increment (default: 15)
            - arch_search_large_increment: Large increment for restarts (default: 250)
            - arch_search_range: Grid search range (default: 5)
        dataloader_curr: Current task training data loader
        dataloader_exp: Experience replay data loader
        test_loader_curr: Current task test data loader
        test_loader_exp: Experience replay test data loader
        current_model: Current MLP model (used for baseline loss reference)

    Returns:
        Optimal architecture sizes list [input, h1', h2', ..., output]
    """
    print(f"  Starting MLP architecture search for task {task_id}")
    print(f"  Original architecture: {original_arch}")
    print(f"  Baseline loss (from current model): {trainW_loss:.6f}")

    # Get search hyperparameters from config
    search_epochs = config.get('arch_search_epochs', DEFAULT_ARCH_SEARCH_EPOCHS)
    threshold = config.get('arch_search_threshold', DEFAULT_ARCH_SEARCH_THRESHOLD)
    max_iter = config.get('arch_search_max_iter', DEFAULT_ARCH_SEARCH_MAX_ITER)
    mlp_increment = config.get('arch_search_mlp_increment', DEFAULT_ARCH_SEARCH_MLP_INCREMENT)
    large_increment = config.get('arch_search_large_increment', DEFAULT_ARCH_SEARCH_LARGE_INCREMENT)
    search_range = config.get('arch_search_range', DEFAULT_ARCH_SEARCH_RANGE)
    averaging_window = config.get('arch_search_averaging_window', DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW)

    # Create trainer for search
    trainer = Trainer(
        loss=config.get('loss', 'mse'),
        metric=config.get('metric', 'mse'),
        problem=config.get('problem', 'vectors'),
    )

    # Use current model's loss (trainW_loss) as baseline for comparison
    # This is the loss from preliminary training on the current model
    loss_orig = trainW_loss

    # Initialize search state
    # For a 4-layer MLP [input, h1, h2, output], we search over h1 and h2
    # Assumes at least 2 hidden layers
    if len(original_arch) < 4:
        print(f"  Architecture too small for search, returning original")
        return original_arch

    input_size = original_arch[0]
    output_size = original_arch[-1]
    x = original_arch[1]  # First hidden layer size
    y = original_arch[2]  # Second hidden layer size

    opt_loss = loss_orig
    opt_arch = list(original_arch)
    k = 0

    # Iterative grid search for better architectures
    while (opt_loss >= loss_orig * threshold) and (k < max_iter):
        print(f"  Search iteration {k + 1}/{max_iter}")
        found_improvement = False

        for n in range(search_range):
            for m in range(search_range):
                # Candidate architecture with incremented hidden sizes
                h1_candidate = x + mlp_increment * n
                h2_candidate = y + mlp_increment * m

                # Build candidate architecture
                # Preserve original structure but with new hidden sizes
                if len(original_arch) == 4:
                    # [input, h1, h2, output]
                    curr_arch = [input_size, h1_candidate, h2_candidate, output_size]
                else:
                    # More layers: keep middle layers same, adjust first two hidden
                    curr_arch = [input_size, h1_candidate, h2_candidate] + original_arch[3:]

                # Skip if same as original (already tested via trainW_loss)
                if curr_arch == list(original_arch):
                    continue

                # Create and train candidate model
                candidate_model = _create_search_model(curr_arch, seed=task_id + n * 100 + m)
                candidate_model = _reinitialize_weights(candidate_model, seed=task_id + n * 100 + m)

                _, poll_loss = _train_candidate_architecture(
                    candidate_model, trainer, task_id, search_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp,
                    averaging_window=averaging_window
                )

                # Track best architecture
                if poll_loss < opt_loss:
                    opt_loss = poll_loss
                    opt_arch = curr_arch
                    found_improvement = True
                    print(f"    Found better architecture: {curr_arch} with loss {poll_loss:.6f}")

        # If no improvement found in this iteration, try larger increment
        if opt_arch[1] == original_arch[1] and opt_arch[2] == original_arch[2]:
            x = x + large_increment
            y = y + large_increment
            print(f"    No improvement, expanding search to h1={x}, h2={y}")
        else:
            # Continue from best found architecture
            x = opt_arch[1]
            y = opt_arch[2]

        k += 1

    print(f"  Architecture search complete.")
    print(f"  Best architecture: {opt_arch} with loss {opt_loss:.6f}")

    return opt_arch
