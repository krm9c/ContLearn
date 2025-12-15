"""
Architecture search utilities for continual learning.

This module provides generic architecture search functionality that works
across all model types (MLP, CNN, GCN) following the same pattern as AWB
utilities in awb.py.

# Added by Claude: Core module for architecture-agnostic search
The search algorithm follows a generic delegate pattern where:
- Core search loop is model-agnostic (this module)
- Model-specific logic is injected via model interface methods
- Similar to AWB's apply_V_transformation() pattern

Models must implement the search interface:
    - generate_search_candidates(iteration, current_best, config)
    - create_with_architecture(arch_spec, seed)
    - reinitialize_weights(seed)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import Dict, Any, List, Tuple, Optional

from ..config.constants import (
    DEFAULT_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_THRESHOLD,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW,
    DEFAULT_ARCH_SEARCH_LR,
    DEFAULT_BATCH_SIZE_VECTOR,
    DEFAULT_BATCH_SIZE_CLASSIFICATION,
    DEFAULT_REPLAY_BUFFER_VECTOR,
    DEFAULT_REPLAY_BUFFER_GRAPH,
    DEFAULT_CNN_ARCH_SEARCH_EPOCHS,
    DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_BATCH_SIZE,
    DEFAULT_ARCH_SEARCH_EXP_REPLAY,
)


def load_search_config(config: Dict[str, Any], model_type: Optional[str] = None) -> Dict[str, Any]:
    """Load architecture search hyperparameters with defaults from constants.py.

    # Added by Claude: Consolidates config loading from all search functions
    Replaces the 20+ lines of config.get() calls that were duplicated in
    mlp_search.py (lines 242-248), cnn_search.py (lines 66-79), and
    gcn_search.py (lines 171-178).

    IMPORTANT: All defaults come from constants.py, NOT hardcoded.
    Uses config.get(key, DEFAULT_FROM_CONSTANTS) pattern.

    Args:
        config: Full configuration dictionary
        model_type: Optional model type ('mlp', 'cnn', 'cnn3d', 'gcn')
                   Used to select model-specific epoch defaults

    Returns:
        Dict with standardized search configuration keys:
            - search_epochs: Training epochs per candidate architecture
            - search_lr: Learning rate for candidate training
            - search_batch_size: Batch size for search
            - search_exp_replay: Experience replay buffer size
            - max_iter: Maximum search iterations
            - averaging_window: Window for loss averaging
            - threshold: Loss improvement threshold
            - ... (see return dict below)
    """
    # Model-specific epoch defaults
    if model_type == 'cnn':
        default_epochs = DEFAULT_CNN_ARCH_SEARCH_EPOCHS
    elif model_type == 'cnn3d':
        default_epochs = DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS
    else:
        # MLP, GCN, or unspecified
        default_epochs = DEFAULT_ARCH_SEARCH_EPOCHS

    # Build search config with all defaults from constants.py
    return {
        'search_epochs': config.get('arch_search_epochs', default_epochs),
        'search_lr': config.get('arch_search_lr', DEFAULT_ARCH_SEARCH_LR),
        'search_batch_size': config.get('arch_search_batch_size', DEFAULT_ARCH_SEARCH_BATCH_SIZE),
        'search_exp_replay': config.get('arch_search_exp_replay', DEFAULT_ARCH_SEARCH_EXP_REPLAY),
        'max_iter': config.get('arch_search_max_iter', DEFAULT_ARCH_SEARCH_MAX_ITER),
        'averaging_window': config.get('arch_search_averaging_window', DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW),
        'threshold': config.get('arch_search_threshold', DEFAULT_ARCH_SEARCH_THRESHOLD),
    }


def compute_search_loss(
    record_dict: Dict,
    task_id: int,
    epochs: int,
    window: Optional[int] = None
) -> float:
    """Compute average loss from recent training iterations.

    # Added by Claude: Unified loss computation for all model types
    Replaces duplicated code in:
    - mlp_search.py _compute_search_loss() (lines 79-109)
    - gcn_search.py _compute_search_loss() (lines 27-64)
    - cnn_search.py inline loss extraction (lines 142-147, 206-211)

    Handles both dict and tuple record formats for compatibility.

    Args:
        record_dict: Training records dictionary with 'iterations' key
        task_id: Current task ID (0-indexed)
        epochs: Total epochs trained for this candidate
        window: Number of recent epochs to average (default: from constants)

    Returns:
        Average loss value over the window, or float('inf') if no data
    """
    if window is None:
        window = DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW

    losses = []
    iterations = record_dict.get('iterations', record_dict)

    # Extract losses from last `window` iterations
    for j in range(1, window + 1):
        iteration = (task_id + 1) * epochs - j
        if iteration in iterations:
            record = iterations[iteration]
            if isinstance(record, dict) and 'losses' in record:
                # New dict format: {'losses': {'V': ..., 'H': ..., ...}}
                losses.append(record['losses'].get('V', 0))
            elif isinstance(record, tuple):
                # Old tuple format: (V, dV, dV_dx, dV_dtheta, H, ...)
                losses.append(record[0])

    if not losses:
        return float('inf')

    return np.mean(losses)


def partition_for_search(model: eqx.Module) -> Tuple[eqx.Module, eqx.Module]:
    """Partition model for architecture search training (freeze A/B if AWB enabled).

    # Added by Claude: Generic delegate pattern like apply_V_transformation() in awb.py
    Delegates to model.partition_for_standard_training() if available (AWB models).
    Falls back to standard partitioning for models without AWB.

    This consolidates the partitioning logic that was duplicated in:
    - mlp_search.py _train_candidate_architecture() (lines 142-147)
    - cnn_search.py (lines 108-117, 181-190)
    - gcn_search.py _partition_for_standard_training_gcn() (lines 124-138)

    Args:
        model: Model instance (MLP, CNN, CNN3D, or GCN)

    Returns:
        Tuple of (params, static) ready for training
    """
    if hasattr(model, 'partition_for_standard_training'):
        # AWB-enabled model - use model's partition method
        return model.partition_for_standard_training()
    else:
        # Non-AWB model - standard partition (all arrays trainable)
        return eqx.partition(model, eqx.is_array)


def reinitialize_weights(model: eqx.Module, seed: int = 0) -> eqx.Module:
    """Reinitialize model weights for fair architecture comparison.

    # Added by Claude: Generic delegate pattern
    Calls model.reinitialize_weights(seed) if available.
    Ensures fair comparison between candidate architectures by starting
    from fresh random initialization.

    Args:
        model: Model instance to reinitialize
        seed: Random seed for reproducibility

    Returns:
        Model with freshly initialized weights

    Raises:
        NotImplementedError: If model doesn't implement reinitialize_weights()
    """
    if hasattr(model, 'reinitialize_weights'):
        return model.reinitialize_weights(seed)
    else:
        raise NotImplementedError(
            f"Model {type(model).__name__} doesn't implement reinitialize_weights() method. "
            f"Models must implement the search interface to use generic architecture search."
        )


def build_train_config(config: Dict[str, Any], search_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build training configuration dict for trainer.train__CL() calls during search.

    # Added by Claude: Standardizes train_config construction
    Replaces inline config building in:
    - mlp_search.py _train_candidate_architecture() (lines 160-167)
    - cnn_search.py (lines 123-130, 193-196)
    - gcn_search.py (lines 223-226)

    Args:
        config: Full configuration dictionary
        search_cfg: Search-specific config from load_search_config()

    Returns:
        Training config dict compatible with trainer.train__CL()
    """
    # Get problem type to determine defaults
    problem = config.get('problem', 'vectors')
    prob = config.get('prob', 'regression')

    # Select appropriate defaults based on problem type
    if problem == 'graph':
        default_batch = config.get('batch_size', 32)  # Graph default
        default_replay = config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_GRAPH)
    elif prob == 'classification':
        default_batch = config.get('batch_size', DEFAULT_BATCH_SIZE_CLASSIFICATION)
        default_replay = config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR)  # Use vector default
    else:  # regression / vectors
        default_batch = config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR)
        default_replay = config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR)

    return {
        'batch_size': search_cfg.get('search_batch_size', default_batch),
        'problem': problem,
        'data_id': config.get('data', 'sine'),
        'flag': config.get('flag', [1.0, 1.0]),
        'len_exp_replay': search_cfg.get('search_exp_replay', default_replay),
        'network': config.get('network', 'fcnn'),
    }


def check_early_stopping(
    current_loss: float,
    best_loss: float,
    patience_counter: int,
    patience: int = 3,
    min_improvement: float = 1e-4
) -> Tuple[bool, int]:
    """Check if architecture search should terminate early.

    # Added by Claude: Speed optimization - NEW functionality
    Enables early termination when search converges or stagnates.

    Args:
        current_loss: Loss of current candidate
        best_loss: Best loss found so far
        patience_counter: Current patience count (iterations without improvement)
        patience: Max iterations without improvement before stopping
        min_improvement: Minimum delta to count as improvement

    Returns:
        Tuple of (should_stop, updated_patience_counter)
    """
    # Check if current candidate improved over best
    if current_loss < best_loss - min_improvement:
        # Improvement found - reset patience
        return False, 0
    else:
        # No improvement - increment patience
        updated_counter = patience_counter + 1
        should_stop = updated_counter >= patience
        return should_stop, updated_counter


def adapt_search_range(
    iteration: int,
    improvement_rate: float,
    base_range: int
) -> int:
    """Adapt search range based on convergence.

    # Added by Claude: Speed optimization - NEW functionality
    Reduces search space as optimal architecture is found.

    Args:
        iteration: Current search iteration
        improvement_rate: (best_loss - baseline_loss) / baseline_loss
        base_range: Initial search range

    Returns:
        Adapted search range (never less than 2)
    """
    if improvement_rate < 0.01:  # Converged (< 1% improvement)
        return max(2, base_range // 2)
    elif improvement_rate < 0.05:  # Converging (< 5% improvement)
        return max(3, base_range * 2 // 3)
    else:  # Still exploring
        return base_range


def should_evaluate_candidate(
    candidate_size: int,
    best_size: int,
    best_loss: float,
    baseline_loss: float,
    expansion_threshold: float = 1.5
) -> bool:
    """Decide if a candidate architecture is worth evaluating.

    # Added by Claude: Speed optimization - NEW functionality
    Prunes candidates unlikely to improve via heuristics.
    Skips very large expansions unless loss is still far from baseline.

    Args:
        candidate_size: Total parameters in candidate
        best_size: Total parameters in current best
        best_loss: Current best loss
        baseline_loss: Baseline loss from preliminary training
        expansion_threshold: Max size ratio to allow without loss check

    Returns:
        True if candidate should be evaluated, False to skip
    """
    if best_size == 0:
        return True  # First candidate

    size_ratio = candidate_size / best_size
    loss_ratio = best_loss / baseline_loss if baseline_loss > 0 else 1.0

    # Allow large expansions only if loss is still high
    if size_ratio > expansion_threshold:
        return loss_ratio > 0.9  # Still far from baseline

    return True  # Normal-sized candidate, evaluate it


def search_architecture(
    model: eqx.Module,
    baseline_arch,
    task_id: int,
    baseline_loss: float,
    dataloader_curr,
    dataloader_exp,
    test_loader_curr,
    test_loader_exp,
    config: Dict[str, Any],
    trainer=None,
    model_type: Optional[str] = None
):
    """Generic architecture search function for any model.

    # Added by Claude: Core generic search algorithm
    Works for ANY model implementing the search interface:
        - model.generate_search_candidates(iteration, current_best, config)
        - model.create_with_architecture(arch_spec, seed, awb_enabled)
        - model.reinitialize_weights(seed)

    This is the CORE generic search algorithm that replaces model-specific
    search functions in arch_search/*.py.

    Args:
        model: Current model instance (used to get awb_enabled state)
        baseline_arch: Baseline architecture (current best)
        task_id: Current task ID
        baseline_loss: Loss from preliminary training (baseline for comparison)
        dataloader_curr: Current task training data
        dataloader_exp: Experience replay data
        test_loader_curr: Current task test data
        test_loader_exp: Experience replay test data
        config: Configuration dictionary
        trainer: Optional Trainer instance (created if None)
        model_type: Optional model type ('mlp', 'cnn', 'gcn') for config defaults

    Returns:
        Optimal architecture found during search
    """
    print(f"  Starting architecture search for task {task_id}")
    print(f"  Baseline architecture: {baseline_arch}")
    print(f"  Baseline loss: {baseline_loss:.6f}")

    # Load search configuration
    search_cfg = load_search_config(config, model_type)

    # Create trainer if not provided
    if trainer is None:
        from .trainer import Trainer
        trainer = Trainer(
            loss=config.get('loss', 'mse'),
            metric=config.get('metric', 'mse'),
            problem=config.get('problem', 'vectors'),
        )

    # Build training config
    train_config = build_train_config(config, search_cfg)

    # Prepare training data tuple
    problem_type = config.get('problem', 'vectors')
    train_data = (
        dataloader_curr,
        dataloader_exp,
        (test_loader_curr, test_loader_exp),
        (test_loader_curr, test_loader_exp)
    )

    # Determine loss type from config
    prob = config.get('prob', 'regression')
    if prob == 'classification':
        loss_type = 'classification'
    else:
        loss_type = 'regression'

    # Initialize search state
    best_arch = baseline_arch
    best_loss = baseline_loss
    patience_counter = 0
    awb_enabled = getattr(model, 'awb_enabled', False)

    # Get search hyperparameters
    max_iter = search_cfg['max_iter']
    threshold = search_cfg['threshold']
    search_epochs = search_cfg['search_epochs']
    averaging_window = search_cfg['averaging_window']
    search_lr = search_cfg['search_lr']

    # Main search loop
    iteration = 0
    while (best_loss >= baseline_loss * threshold) and (iteration < max_iter):
        print(f"  Search iteration {iteration + 1}/{max_iter}")
        found_improvement = False

        # MODEL-SPECIFIC: Generate candidates via model interface
        candidates = model.generate_search_candidates(iteration, best_arch, config)

        if not candidates:
            print(f"    No candidates generated, stopping search")
            break

        for candidate_spec in candidates:
            # Skip if same as baseline (already tested)
            if candidate_spec == baseline_arch:
                continue

            # MODEL-SPECIFIC: Create model with candidate architecture
            candidate_model = model.create_with_architecture(
                candidate_spec,
                seed=task_id + iteration * 1000,
                awb_enabled=awb_enabled
            )

            # Reinitialize weights for fair comparison
            candidate_model = reinitialize_weights(
                candidate_model,
                seed=task_id + iteration * 1000
            )

            # GENERIC: Partition for training
            params, static = partition_for_search(candidate_model)

            # Create optimizer
            optim = optax.adam(search_lr)
            opt_state = optim.init(params)

            # Initialize record dict
            record_dict = trainer.initialize_record_dict(config, run_id=0)

            # GENERIC: Train candidate
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data,
                params,
                static,
                opt_state,
                optim,
                n_iter=search_epochs,
                save_iter=config.get('save_iter', 10),
                task_id=task_id,
                config=train_config,
                record_dict=record_dict,
                problem_type=problem_type,
                loss_type=loss_type
            )

            # GENERIC: Extract loss
            candidate_loss = compute_search_loss(
                record_dict,
                task_id,
                search_epochs,
                window=averaging_window
            )

            # Track best architecture
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_arch = candidate_spec
                found_improvement = True
                print(f"    Found better architecture: {candidate_spec} with loss {candidate_loss:.6f}")

        # Update search state for next iteration
        if not found_improvement:
            # No improvement in this iteration
            patience_counter += 1
            print(f"    No improvement found (patience: {patience_counter})")

            # Check early stopping
            early_stop_patience = config.get('arch_search_early_stop_patience', 3)
            if patience_counter >= early_stop_patience:
                print(f"    Early stopping triggered after {patience_counter} iterations without improvement")
                break
        else:
            # Reset patience on improvement
            patience_counter = 0

        iteration += 1

    print(f"  Architecture search complete.")
    print(f"  Best architecture: {best_arch} with loss {best_loss:.6f}")

    return best_arch


# Export all public functions
__all__ = [
    'search_architecture',
    'load_search_config',
    'compute_search_loss',
    'partition_for_search',
    'reinitialize_weights',
    'build_train_config',
    'check_early_stopping',
    'adapt_search_range',
    'should_evaluate_candidate',
]
