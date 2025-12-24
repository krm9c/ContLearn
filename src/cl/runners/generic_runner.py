"""
Generic unified runner for all problem types (regression, classification, graph).

This module consolidates regression.py, classification.py, and graph_classification.py
into a single generic runner using config-based dispatch.

Added by Claude: Layer-level AWB abstraction refactor.
"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, Any, Tuple

from ..core import Trainer
from ..core.awb import (
    compute_avg_loss,
    should_change_arch,
    initialize_AB_matrices,
    apply_V_transformation,
    partition_model_for_AB_training,
    partition_model_for_standard_training,
)

from ..config.constants import (
    DEFAULT_OPTIMIZER,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_AWB_V_LR_FACTOR,
    DEFAULT_AWB_V_WARMUP_EPOCHS,
    DEFAULT_AWB_TASK_LR_FACTOR,
    DEFAULT_AWB_TASK_WARMUP_EPOCHS,
    # Added by Claude: Task transition warmup constants
    DEFAULT_TASK_WARMUP_ENABLED,
    DEFAULT_TASK_WARMUP_EPOCHS,
    DEFAULT_TASK_WARMUP_LR_FACTOR,
    DEFAULT_WARMUP_GRAD_WEIGHTS,
    DEFAULT_MAIN_GRAD_WEIGHTS,
    # Added by Claude: Adaptive LR and gradient weight constants (loss-based)
    DEFAULT_ADAPTIVE_LR_MIN_ENABLED,
    DEFAULT_LR_MIN_BASE,
    DEFAULT_LR_MIN_MAX,
    DEFAULT_LR_MIN_LOSS_RATIO_THRESHOLD,
    DEFAULT_ADAPTIVE_GRAD_WEIGHTS_ENABLED,
    DEFAULT_GRAD_WEIGHTS_BASE,
    DEFAULT_GRAD_WEIGHTS_MAX_CURRENT,
    DEFAULT_GRAD_WEIGHTS_MIN_EXPERIENCE,
    DEFAULT_GRAD_WEIGHTS_LOSS_RATIO_THRESHOLD,
)

# ============================================================================
# Model Architecture Utilities (Added by Claude)
# ============================================================================

def get_model_architecture(model) -> dict:
    """Extract architecture information from any model type.

    Args:
        model: Equinox model (MLP, CNN, CNN3D, or GCN)

    Returns:
        Dict with architecture details (format depends on model type)
    """
    arch_info = {'model_type': type(model).__name__}

    # MLP: has 'sizes' attribute
    if hasattr(model, 'sizes'):
        arch_info['sizes'] = list(model.sizes)
        return arch_info

    # CNN/CNN3D: has feed_sizes and filter_size
    if hasattr(model, 'feed_sizes') and hasattr(model, 'filter_size'):
        arch_info['feed_sizes'] = list(model.feed_sizes)
        arch_info['filter_size'] = model.filter_size
        if hasattr(model, 'channel_out'):
            arch_info['channel_out'] = model.channel_out
        if hasattr(model, 'channel_in'):
            arch_info['channel_in'] = model.channel_in
        return arch_info

    # GCN: has gcn_sizes and feed_sizes
    if hasattr(model, 'gcn_sizes') and hasattr(model, 'feed_sizes'):
        arch_info['gcn_sizes'] = list(model.gcn_sizes)
        arch_info['feed_sizes'] = list(model.feed_sizes)
        return arch_info

    return arch_info


def save_model_weights(model):
    """Save model weights before architecture search.

    Args:
        model: Equinox model (MLP, CNN, CNN3D, or GCN)

    Returns:
        Saved weights structure (format depends on model type)
    """
    # MLP: save linear layer weights and biases
    if hasattr(model, 'layers'):
        weights = [layer.weight for layer in model.layers if hasattr(layer, 'weight')]
        biases = [layer.bias for layer in model.layers if hasattr(layer, 'bias')]
        return {'type': 'mlp', 'weights': weights, 'biases': biases}

    # CNN/CNN3D: save conv and feed layer weights
    if hasattr(model, 'conv_layers') and hasattr(model, 'feed_layers'):
        conv_weights = [layer.weight for layer in model.conv_layers]
        feed_weights = [layer.weight for layer in model.feed_layers]
        feed_biases = [layer.bias for layer in model.feed_layers]
        return {
            'type': 'cnn',
            'conv_weights': conv_weights,
            'feed_weights': feed_weights,
            'feed_biases': feed_biases
        }

    # GCN: save gcn and feed layer weights/biases
    if hasattr(model, 'gcn_layers') and hasattr(model, 'feed_layers'):
        gcn_weights = [layer.weight for layer in model.gcn_layers if hasattr(layer, 'weight')]
        gcn_biases = [layer.bias for layer in model.gcn_layers if hasattr(layer, 'bias')]
        feed_weights = [layer.weight for layer in model.feed_layers]
        feed_biases = [layer.bias for layer in model.feed_layers]
        return {
            'type': 'gcn',
            'gcn_weights': gcn_weights,
            'gcn_biases': gcn_biases,
            'feed_weights': feed_weights,
            'feed_biases': feed_biases
        }

    return None


def restore_model_weights(model, saved_weights):
    """Restore model weights after architecture search.

    Args:
        model: Equinox model
        saved_weights: Saved weights from save_model_weights()

    Returns:
        Model with restored weights
    """
    if saved_weights is None:
        return model

    weight_type = saved_weights.get('type')

    # MLP: restore linear layer weights
    if weight_type == 'mlp' and hasattr(model, 'layers'):
        weights = saved_weights['weights']
        biases = saved_weights['biases']
        for i, layer in enumerate(model.layers):
            if i < len(weights) and hasattr(layer, 'weight'):
                model = eqx.tree_at(lambda x: x.layers[i].weight, model, weights[i])
            if i < len(biases) and hasattr(layer, 'bias'):
                model = eqx.tree_at(lambda x: x.layers[i].bias, model, biases[i])
        return model

    # CNN: restore conv and feed layers
    if weight_type == 'cnn' and hasattr(model, 'conv_layers'):
        for i, weight in enumerate(saved_weights['conv_weights']):
            if i < len(model.conv_layers):
                model = eqx.tree_at(lambda x, idx=i: x.conv_layers[idx].weight, model, weight)
        for i, weight in enumerate(saved_weights['feed_weights']):
            if i < len(model.feed_layers):
                model = eqx.tree_at(lambda x, idx=i: x.feed_layers[idx].weight, model, weight)
        for i, bias in enumerate(saved_weights['feed_biases']):
            if i < len(model.feed_layers):
                model = eqx.tree_at(lambda x, idx=i: x.feed_layers[idx].bias, model, bias)
        return model

    # GCN: restore gcn and feed layers
    if weight_type == 'gcn' and hasattr(model, 'gcn_layers'):
        for i, weight in enumerate(saved_weights['gcn_weights']):
            if i < len(model.gcn_layers) and hasattr(model.gcn_layers[i], 'weight'):
                model = eqx.tree_at(lambda x, idx=i: x.gcn_layers[idx].weight, model, weight)
        for i, bias in enumerate(saved_weights['gcn_biases']):
            if i < len(model.gcn_layers) and hasattr(model.gcn_layers[i], 'bias'):
                model = eqx.tree_at(lambda x, idx=i: x.gcn_layers[idx].bias, model, bias)
        for i, weight in enumerate(saved_weights['feed_weights']):
            if i < len(model.feed_layers):
                model = eqx.tree_at(lambda x, idx=i: x.feed_layers[idx].weight, model, weight)
        for i, bias in enumerate(saved_weights['feed_biases']):
            if i < len(model.feed_layers):
                model = eqx.tree_at(lambda x, idx=i: x.feed_layers[idx].bias, model, bias)
        return model

    return model


# ============================================================================
# Optimizer and Learning Rate Utilities (Generic)
# ============================================================================

def create_optimizer(config: Dict[str, Any]) -> optax.GradientTransformationExtraArgs:
    """Create optimizer from config with injectable hyperparameters.

    Added by Claude: Wraps optimizer with optax.inject_hyperparams to enable
    dynamic learning rate adjustment per task.

    Reference:
    - https://optax.readthedocs.io/en/latest/api.html#optax.inject_hyperparams

    Args:
        config: Configuration dict with optimizer, lr, weight_decay, momentum

    Returns:
        Optax optimizer with injectable hyperparameters
    """
    optimizer_name = config.get('optimizer', DEFAULT_OPTIMIZER).lower()
    lr = config.get('lr', DEFAULT_LR )
    weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
    momentum = config.get('momentum', 0.99)
    # Added by Claude: Use inject_hyperparams to allow dynamic LR changes
    if optimizer_name == 'adam':
        base_optimizer = optax.inject_hyperparams(optax.adam)
        return base_optimizer(learning_rate=lr)
    elif optimizer_name == 'adamw':
        base_optimizer = optax.inject_hyperparams(optax.adamw)
        return base_optimizer(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        base_optimizer = optax.inject_hyperparams(optax.sgd)
        return base_optimizer(learning_rate=lr, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        base_optimizer = optax.inject_hyperparams(optax.rmsprop)
        return base_optimizer(learning_rate=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def update_learning_rate(opt_state, new_lr: float):
    """Update learning rate in optimizer state.

    Added by Claude: Uses optax.inject_hyperparams mechanism to dynamically
    adjust learning rate per task by mutating the optimizer state.

    Example:
        >>> opt = optax.inject_hyperparams(optax.adam)(learning_rate=1e-4)
        >>> opt_state = opt.init(params)
        >>> opt_state = update_learning_rate(opt_state, 1e-5)
        >>> # Next opt.update() will use new learning rate

    Args:
        opt_state: Current optimizer state (must have hyperparams attribute)
        new_lr: New learning rate

    Returns:
        Updated optimizer state with modified learning rate
    """
    # Added by Claude: Directly mutate the hyperparams in optimizer state
    if hasattr(opt_state, 'hyperparams'):
        opt_state.hyperparams['learning_rate'] = new_lr
    else:
        # Fallback: If no hyperparams (shouldn't happen with inject_hyperparams)
        print(f"Warning: opt_state has no hyperparams attribute. LR not updated.")

    return opt_state


def create_optimizer_with_lr(config: dict, lr: float):
    """Create optimizer with a specific learning rate.

    Added by Claude: Used for AWB Step 5 warmup where we want to start with
    a lower learning rate to reduce the loss spike after V transformation.

    Args:
        config: Configuration dict with optimizer settings
        lr: Specific learning rate to use (overrides config['lr'])

    Returns:
        Optax optimizer with injectable hyperparameters
    """
    optimizer_name = config.get('optimizer', DEFAULT_OPTIMIZER).lower()
    weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
    momentum = config.get('momentum', 0.99)

    if optimizer_name == 'adam':
        base_optimizer = optax.inject_hyperparams(optax.adam)
        return base_optimizer(learning_rate=lr)
    elif optimizer_name == 'adamw':
        base_optimizer = optax.inject_hyperparams(optax.adamw)
        return base_optimizer(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        base_optimizer = optax.inject_hyperparams(optax.sgd)
        return base_optimizer(learning_rate=lr, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        base_optimizer = optax.inject_hyperparams(optax.rmsprop)
        return base_optimizer(learning_rate=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def compute_adaptive_lr_min(config: Dict[str, Any], loss_ratio: float = 1.0) -> float:
    """Compute adaptive minimum learning rate based on loss ratio.

    Added by Claude: When the loss ratio (current_loss / previous_loss) is high,
    it indicates the model is struggling with the new task. We boost lr_min to
    ensure sufficient learning capacity even when the LR schedule has decayed.

    The lr_min scales linearly from lr_min_base to lr_min_max as loss_ratio
    increases from threshold to 2*threshold.

    Args:
        config: Configuration dict with adaptive_lr_min settings
        loss_ratio: Ratio of current task loss to previous task loss

    Returns:
        Adaptive lr_min based on loss ratio
    """
    adaptive_enabled = config.get('adaptive_lr_min_enabled', DEFAULT_ADAPTIVE_LR_MIN_ENABLED)

    if not adaptive_enabled:
        return config.get('lr_min', 1e-6)

    lr_min_base = config.get('lr_min_base', DEFAULT_LR_MIN_BASE)
    lr_min_max = config.get('lr_min_max', DEFAULT_LR_MIN_MAX)
    threshold = config.get('lr_min_loss_ratio_threshold', DEFAULT_LR_MIN_LOSS_RATIO_THRESHOLD)

    # If loss ratio is below threshold, use base lr_min
    if loss_ratio <= threshold:
        return lr_min_base

    # Linear interpolation from lr_min_base to lr_min_max
    # as loss_ratio goes from threshold to 2*threshold (capped)
    progress = min((loss_ratio - threshold) / threshold, 1.0)
    lr_min = lr_min_base + progress * (lr_min_max - lr_min_base)

    return float(lr_min)


def compute_adaptive_grad_weights(config: Dict[str, Any], loss_ratio: float = 1.0) -> list:
    """Compute adaptive gradient weights based on loss ratio.

    Added by Claude: When the loss ratio (current_loss / previous_loss) is high,
    it indicates the model is struggling with the new task. We shift weights
    toward the current task (increase alpha, decrease beta) to allow the model
    to learn the new distribution.

    The weights shift linearly from base weights toward max_current/min_experience
    as loss_ratio increases from threshold to 2*threshold.

    Args:
        config: Configuration dict with adaptive_grad_weights settings
        loss_ratio: Ratio of current task loss to previous task loss

    Returns:
        Adaptive gradient weights [alpha, beta, gamma] for this task
    """
    adaptive_enabled = config.get('adaptive_grad_weights_enabled', DEFAULT_ADAPTIVE_GRAD_WEIGHTS_ENABLED)

    if not adaptive_enabled:
        return config.get('main_grad_weights', DEFAULT_MAIN_GRAD_WEIGHTS)

    base_weights = config.get('grad_weights_base', DEFAULT_GRAD_WEIGHTS_BASE)
    max_current = config.get('grad_weights_max_current', DEFAULT_GRAD_WEIGHTS_MAX_CURRENT)
    min_experience = config.get('grad_weights_min_experience', DEFAULT_GRAD_WEIGHTS_MIN_EXPERIENCE)
    threshold = config.get('grad_weights_loss_ratio_threshold', DEFAULT_GRAD_WEIGHTS_LOSS_RATIO_THRESHOLD)

    # If loss ratio is below threshold, use base weights
    if loss_ratio <= threshold:
        return list(base_weights)

    # Linear interpolation as loss_ratio goes from threshold to 2*threshold (capped)
    progress = min((loss_ratio - threshold) / threshold, 1.0)

    # Shift from base weights toward [max_current, min_experience, base_reg]
    alpha = base_weights[0] + progress * (max_current - base_weights[0])
    beta = base_weights[1] + progress * (min_experience - base_weights[1])
    gamma = base_weights[2]  # Regularization stays constant

    return [alpha, beta, gamma]


def compute_task_lr(config: Dict[str, Any], task_id: int, loss_ratio: float = 1.0) -> float:
    """Compute learning rate for current task based on schedule and loss ratio.

    Added by Claude: All schedules now respect adaptive lr_min floor that is
    computed based on the loss ratio between current and previous task. When
    the loss ratio is high, lr_min is boosted to ensure sufficient learning.

    Args:
        config: Configuration dict with lr_schedule, lr, lr_min, lr_decay_factor
        task_id: Current task ID
        loss_ratio: Ratio of current task loss to previous task loss (default 1.0)

    Returns:
        Learning rate for this task (clamped to adaptive lr_min minimum)
    """
    base_lr = config.get('lr', DEFAULT_LR)
    # Added by Claude: Use adaptive lr_min based on loss ratio
    lr_min = compute_adaptive_lr_min(config, loss_ratio)
    schedule = config.get('lr_schedule', 'constant')
    n_tasks = config.get('n_task', 5)

    if schedule == 'constant':
        return base_lr
    elif schedule == 'step':
        decay_factor = config.get('lr_decay_factor', 0.99)
        lr = base_lr * (decay_factor ** task_id)
        return max(lr, lr_min)
    elif schedule == 'exponential':
        # Added by Claude: Exponential decay from base_lr towards lr_min
        decay_factor = config.get('lr_decay_factor', 0.5)
        lr = base_lr * (decay_factor ** task_id)
        return max(lr, lr_min)
    elif schedule == 'cosine':
        # Added by Claude: Cosine annealing from base_lr to adaptive lr_min
        progress = task_id / max(n_tasks - 1, 1)
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + jnp.cos(jnp.pi * progress))
        return float(max(lr, lr_min))
    elif schedule == 'linear':
        # Added by Claude: Linear decay from base_lr to lr_min
        progress = task_id / max(n_tasks - 1, 1)
        lr = base_lr - progress * (base_lr - lr_min)
        return max(lr, lr_min)
    else:
        return base_lr


# ============================================================================
# Architecture Search (Added by Claude)
# ============================================================================

def run_architecture_search(model, config, task_id, trainWLoss, preliminary_epochs,
                            trainloader, exploader, valloader, testloader, trainer):
    """Dispatch to appropriate architecture search based on model type.

    Args:
        model: Current model (MLP, CNN, CNN3D, or GCN)
        config: Configuration dictionary
        task_id: Current task ID
        trainWLoss: Loss after preliminary training
        preliminary_epochs: Number of preliminary epochs
        trainloader: Current task training loader
        exploader: Experience replay loader
        valloader: Validation loader
        testloader: Test loader
        trainer: Trainer instance

    Returns:
        Tuple of (original_arch, new_arch, candidates_list, search_time)

        Note: candidates_list contains only the final optimal architecture selected.
        The underlying search functions may test many architectures (100+ for MLP)
        but only return the best one. This keeps the record_dict concise.

        Returns (None, None, [], 0.0) if no search available
    """
    import time

    network = config.get('network', 'fcnn')
    search_start = time.time()

    # MLP: architecture search on 'sizes'
    if network == 'fcnn' or (hasattr(model, 'sizes') and not hasattr(model, 'gcn_sizes')):
        print("    Using MLP architecture search")
        from ..arch_search import arch_search_MLP

        original_arch = list(model.sizes)

        # Call MLP architecture search
        opt_arch = arch_search_MLP(
            original_arch=original_arch,
            task_id=task_id,
            trainW_loss=trainWLoss,
            og_epochs=preliminary_epochs,
            config=config,
            dataloader_curr=trainloader,
            dataloader_exp=exploader,
            test_loader_curr=valloader,
            test_loader_exp=testloader,
            current_model=model,
        )

        search_time = time.time() - search_start
        print(f"    MLP search completed in {search_time:.2f}s: {original_arch} → {opt_arch}")

        # Track candidates (arch_search_MLP returns single optimal arch)
        candidates = [
            {'arch': opt_arch, 'loss': trainWLoss, 'is_optimal': True}
        ]

        return original_arch, opt_arch, candidates, search_time

    # CNN/CNN3D: architecture search on feed_sizes and filter_size
    # Uses same arch_search_CNN_fresh for both CNN and CNN3D
    elif network in ['cnn', 'cnn3d'] and hasattr(model, 'conv_layers'):
        # Detect if CNN3D using multiple signals for robustness:
        # 1. Model type name
        # 2. Presence of CNN3D-specific attributes (A_conv1 vs A_conv)
        # 3. Config data type (cifar10/cifar100 uses CNN3D)
        # 4. Channel count (CNN3D uses 3 channels, CNN uses 1)
        is_cnn3d = (
            type(model).__name__ == 'CNN3D' or
            hasattr(model, 'A_conv1') or  # CNN3D has A_conv1, regular CNN has A_conv
            config.get('data', '') in ['cifar10', 'cifar100'] or
            (hasattr(model, 'channel_in') and model.channel_in > 1)
        )

        print(f"    Using CNN{'3D' if is_cnn3d else ''} architecture search")
        from ..arch_search import arch_search_CNN_fresh

        original_feed = list(model.feed_sizes)
        original_filter = model.filter_size

        # Call CNN architecture search (works for both CNN and CNN3D)
        opt_feed, opt_filter = arch_search_CNN_fresh(
            filter_size=original_filter,
            feed_sizes=original_feed,
            task=task_id,
            trainW_loss=trainWLoss,
            og_epochs=preliminary_epochs,
            config=config,
            dataloader_curr=trainloader,
            dataloader_exp=exploader,
            test_loader_curr=valloader,
            test_loader_exp=testloader,
            trainer=trainer
        )

        search_time = time.time() - search_start
        print(f"    CNN{'3D' if is_cnn3d else ''} search completed in {search_time:.2f}s")
        print(f"      Feed: {original_feed} → {opt_feed}")
        print(f"      Filter: {original_filter} → {opt_filter}")

        # Return as tuples for easier comparison
        original_arch = (original_feed, original_filter)
        new_arch = (opt_feed, opt_filter)

        candidates = [
            {
                'arch': {'feed_sizes': opt_feed, 'filter_size': opt_filter, 'is_cnn3d': is_cnn3d},
                'loss': trainWLoss,
                'is_optimal': True
            }
        ]

        return original_arch, new_arch, candidates, search_time

    # GCN: architecture search on gcn_sizes and feed_sizes
    elif network == 'gcn' and hasattr(model, 'gcn_sizes'):
        print("    Using GCN architecture search")
        from ..arch_search import arch_search_GCN

        original_gcn = list(model.gcn_sizes)
        original_feed = list(model.feed_sizes)

        # Call GCN architecture search (returns gcn_sizes, feed_sizes)
        opt_gcn, opt_feed = arch_search_GCN(
            original_gcn=original_gcn,
            original_mlp=original_feed,
            task=task_id,
            trainW_loss=trainWLoss,
            og_epochs=preliminary_epochs,
            config=config,
            train_loader=trainloader,
            mem_train_loader=exploader,
            test_loader=valloader,
            trainer=trainer,
            model=model
        )

        search_time = time.time() - search_start
        print(f"    GCN search completed in {search_time:.2f}s")
        print(f"      GCN: {original_gcn} → {opt_gcn}")
        print(f"      Feed: {original_feed} → {opt_feed}")

        # Return as tuples for easier comparison
        original_arch = (original_gcn, original_feed)
        new_arch = (opt_gcn, opt_feed)

        candidates = [
            {
                'arch': {'gcn_sizes': opt_gcn, 'feed_sizes': opt_feed},
                'loss': trainWLoss,
                'is_optimal': True
            }
        ]

        return original_arch, new_arch, candidates, search_time

    else:
        # No architecture search available for this model type
        print(f"    No architecture search available for network={network}")
        return None, None, [], 0.0


# ============================================================================
# Model and Dataset Loading (Config-based Dispatch)
# ============================================================================

def load_checkpoint(config: Dict[str, Any]):
    """Load model, dataset, trainer, and optimizer based on config.

    Dispatches to appropriate loaders based on config['problem'] (data structure)
    and config['prob'] (task type).

    Args:
        config: Configuration dict

    Returns:
        Tuple of (trainer, optimizer, dataset, model)
    """
    # Added by Claude: Check 'problem' field first to distinguish graph vs vector data
    problem = config.get('problem', 'vectors')  # 'graph' or 'vectors'
    prob = config.get('prob', 'regression')      # 'regression' or 'classification'

    # Dispatch based on data structure type (problem field)
    if problem == 'graph':
        # Graph-structured data (GCN)
        from .graph_classification import load_graph_checkpoint
        return load_graph_checkpoint(config)
    elif prob == 'regression':
        # Vector regression (MLP)
        from .regression import load_regression_checkpoint
        return load_regression_checkpoint(config)
    elif prob == 'classification':
        # Vector classification (CNN/MLP)
        from .classification import load_classification_checkpoint
        return load_classification_checkpoint(config)
    else:
        raise ValueError(f"Unknown problem configuration: problem={problem}, prob={prob}")


# ============================================================================
# Generic AWB Training Pipeline
# ============================================================================

def train_model(config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
    """Generic unified training function for all problem types.

    Works with regression, classification, and graph problems.
    Supports MLP, CNN, CNN3D, and GCN models with AWB.

    Args:
        config: Configuration dict with all hyperparameters
        run_id: Run identifier for multiple runs

    Returns:
        Dictionary of training records
    """
    # Load components based on config
    trainer, optim, data, model = load_checkpoint(config)

    # Initialize optimizer state
    params, static = partition_model_for_standard_training(model)
    opt_state = optim.init(params)

    # Extract config values
    n_tasks = config.get('n_task', 5)
    epochs_per_task = config.get('epochs_per_task', 100)
    awb_enabled = config.get('awb_enabled', False)
    problem_type = config.get('problem', 'vectors')  # 'vectors' or 'graph'
    prob = config.get('prob', 'regression')  # 'regression' or 'classification'
    loss_type = 'regression' if prob == 'regression' else 'classification'

    # Initialize record dictionary
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Added by Claude: Initialize architecture history tracking
    record_dict['architecture_history'] = {}

    # Added by Claude: Track loss from previous task for adaptive LR and grad weights
    previous_task_loss = None

    # Training loop over tasks
    for task_id in range(n_tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        print(f"{'='*60}")

        # Generate dataset for current task
        trainloader, exploader = data.generate_dataset(task_id,
                                                       config.get('batch_size', 64),
                                                       phase='training')
        valloader, val_exploader = data.generate_dataset(task_id,
                                                         config.get('batch_size', 64),
                                                         phase='testing')
        # Keep both current and experience test loaders as tuple for proper Te/Cur vs Te/Exp metrics
        test_curr_loader, test_exp_loader = data.generate_dataset(task_id,
                                                                  config.get('batch_size', 64),
                                                                  phase='testing')
        testloader = (test_curr_loader, test_exp_loader)

        # Added by Claude: Compute loss ratio for adaptive LR and gradient weights
        # For task 0, loss_ratio = 1.0 (no previous task)
        # For tasks 1+, compute initial loss on current task and compare to previous
        loss_ratio = 1.0
        if task_id > 0 and previous_task_loss is not None and previous_task_loss > 0:
            # Compute initial loss on new task before training
            model_temp = eqx.combine(params, static)
            initial_loss = 0.0
            n_batches = 0
            for batch in trainloader:
                if problem_type == 'graph':
                    # Graph data - use batch directly
                    pass  # Skip for now, handled differently
                else:
                    x, y = batch
                    x = jnp.asarray(x.numpy(), dtype=jnp.float64)
                    if loss_type == 'classification':
                        y = jnp.asarray(y.numpy(), dtype=jnp.int64)
                        preds = jax.vmap(model_temp)(x)
                        if preds.ndim == 3 and preds.shape[1] == 1:
                            preds = jnp.squeeze(preds, axis=1)
                        loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(preds) * jax.nn.one_hot(y, preds.shape[-1]), axis=-1))
                    else:
                        y = jnp.asarray(y.numpy(), dtype=jnp.float64)
                        preds = jax.vmap(model_temp)(x)
                        loss = jnp.mean((preds - y) ** 2)
                    initial_loss += float(loss)
                    n_batches += 1
                if n_batches >= 5:  # Sample first 5 batches for speed
                    break
            initial_loss = initial_loss / max(n_batches, 1)
            loss_ratio = initial_loss / previous_task_loss
            print(f"Initial loss on new task: {initial_loss:.4f}, previous: {previous_task_loss:.4f}, ratio: {loss_ratio:.2f}")

        # Compute learning rate for this task with loss-based adaptive lr_min
        task_lr = compute_task_lr(config, task_id, loss_ratio)

        # Added by Claude: Update optimizer state with task-specific learning rate
        opt_state = update_learning_rate(opt_state, task_lr)

        # Task 0 or AWB disabled: Standard CL training
        if task_id == 0 or not awb_enabled:
            # Added by Claude: Show adaptive lr_min and loss ratio for this task
            adaptive_lr_min = compute_adaptive_lr_min(config, loss_ratio)
            print(f"Standard CL training (lr={task_lr:.6f}, adaptive_lr_min={adaptive_lr_min:.2e}, loss_ratio={loss_ratio:.2f})")

            # Recombine model
            model = eqx.combine(params, static)

            # Added by Claude: Initialize task recording
            arch_info = get_model_architecture(model)
            trainer.initialize_task(record_dict, task_id, arch_info)

            # Added by Claude: Task transition warmup configuration
            warmup_enabled = config.get('task_warmup_enabled', DEFAULT_TASK_WARMUP_ENABLED)
            warmup_epochs = config.get('task_warmup_epochs', DEFAULT_TASK_WARMUP_EPOCHS)
            warmup_lr_factor = config.get('task_warmup_lr_factor', DEFAULT_TASK_WARMUP_LR_FACTOR)
            warmup_grad_weights = config.get('warmup_grad_weights', DEFAULT_WARMUP_GRAD_WEIGHTS)
            # Added by Claude: Use adaptive gradient weights based on loss ratio
            main_grad_weights = compute_adaptive_grad_weights(config, loss_ratio)

            # Task 0: No warmup needed (no previous knowledge to protect)
            # Tasks 1+: Warmup with new task samples only, then main training with experience
            if task_id > 0 and warmup_enabled and warmup_epochs > 0:
                # === WARMUP PHASE: New task samples only ===
                warmup_lr = task_lr * warmup_lr_factor
                print(f"  Warmup phase: {warmup_epochs} epochs at lr={warmup_lr:.2e}")
                print(f"    Using new task samples only (no experience replay)")
                print(f"    Grad weights: {warmup_grad_weights}")

                # Create warmup optimizer with reduced LR
                optim_warmup = create_optimizer_with_lr(config, warmup_lr)
                opt_state_warmup = optim_warmup.init(params)

                # Create warmup config with warmup grad weights
                warmup_config = config.copy()
                warmup_config['grad_weights'] = warmup_grad_weights

                # Warmup uses trainloader for both current and "experience" (new task only)
                # Note: record_training=False - standard CL warmup is not recorded
                params, static, opt_state_warmup, record_dict = trainer.train__CL(
                    train__=(trainloader, trainloader, valloader, testloader),  # Use trainloader as exploader
                    params=params,
                    static=static,
                    opt_state=opt_state_warmup,
                    optim=optim_warmup,
                    n_iter=warmup_epochs,
                    task_id=task_id,
                    config=warmup_config,
                    record_dict=record_dict,
                    notABTrain=True,
                    problem_type=problem_type,
                    loss_type=loss_type,
                    phase='warmup',
                    record_training=False,
                    global_iteration_offset=task_id * epochs_per_task
                )

                # === MAIN PHASE: Full training with experience replay ===
                main_epochs = epochs_per_task - warmup_epochs
                print(f"  Main phase: {main_epochs} epochs at lr={task_lr:.2e}")
                print(f"    Using experience replay")
                # Added by Claude: Format adaptive grad weights for display
                grad_weights_str = f"[{main_grad_weights[0]:.2f}, {main_grad_weights[1]:.2f}, {main_grad_weights[2]:.2f}]"
                print(f"    Adaptive grad weights: {grad_weights_str} (current, experience, regularization)")

                # Create main config with main grad weights
                main_config = config.copy()
                main_config['grad_weights'] = main_grad_weights

                # Reinitialize optimizer at full LR
                opt_state = optim.init(params)
                opt_state = update_learning_rate(opt_state, task_lr)

                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params,
                    static=static,
                    opt_state=opt_state,
                    optim=optim,
                    n_iter=main_epochs,
                    task_id=task_id,
                    config=main_config,
                    record_dict=record_dict,
                    notABTrain=True,
                    problem_type=problem_type,
                    loss_type=loss_type,
                    phase='main',
                    record_training=True,
                    global_iteration_offset=task_id * epochs_per_task + warmup_epochs
                )

                # Update phase info with warmup details
                record_dict['tasks'][task_id]['phase_info'] = {
                    'type': 'standard_with_warmup',
                    'warmup_epochs': warmup_epochs,
                    'warmup_lr': warmup_lr,
                    'warmup_grad_weights': warmup_grad_weights,
                    'main_epochs': main_epochs,
                    'main_grad_weights': main_grad_weights,
                    'total_epochs': epochs_per_task
                }
            else:
                # Task 0 or warmup disabled: Standard training without warmup
                # Use main_grad_weights for consistency (or default if not specified)
                training_config = config.copy()
                if task_id > 0:  # Only override grad_weights for tasks after 0
                    training_config['grad_weights'] = main_grad_weights

                global_iter_offset = task_id * epochs_per_task
                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params,
                    static=static,
                    opt_state=opt_state,
                    optim=optim,
                    n_iter=epochs_per_task,
                    task_id=task_id,
                    config=training_config,
                    record_dict=record_dict,
                    notABTrain=True,
                    problem_type=problem_type,
                    loss_type=loss_type,
                    phase='main',
                    record_training=True,
                    global_iteration_offset=global_iter_offset
                )

                # Update phase info
                record_dict['tasks'][task_id]['phase_info'] = {
                    'type': 'standard',
                    'total_epochs': epochs_per_task
                }

            # Added by Claude: Track architecture for this task
            model_combined = eqx.combine(params, static)

            # Extract final loss from main_training (last V value recorded)
            main_training = record_dict['tasks'][task_id]['main_training']
            optimal_loss = main_training['V'][-1] if main_training['V'] else None

            # Get previous task's architecture if it exists
            prev_arch = None
            if task_id > 0 and (task_id - 1) in record_dict['architecture_history']:
                prev_arch = record_dict['architecture_history'][task_id - 1]['final_arch']

            record_dict['architecture_history'][task_id] = {
                'task_id': task_id,
                'original_arch': prev_arch,
                'searched_candidates': [],
                'final_arch': get_model_architecture(model_combined),
                'preliminary_loss': None,
                'optimal_loss': optimal_loss,
                'arch_changed': False,
                'change_reason': 'baseline_task' if task_id == 0 else 'awb_disabled',
                'search_time': 0.0,
                'loss_ratio': loss_ratio,
                'adaptive_lr_min': compute_adaptive_lr_min(config, loss_ratio),
                'adaptive_grad_weights': main_grad_weights if task_id > 0 else None,
            }

            # Added by Claude: Save optimal loss for next task's loss ratio computation
            if optimal_loss is not None:
                previous_task_loss = optimal_loss

            # Append to experience replay
            data.append_to_experience(task_id)

            # Added by Claude: Optional per-task evaluation for CL metrics
            # After training task_id, test on ALL tasks from 0 to task_id
            # This builds the performance matrix A[j][i] = performance on task i after training task j
            per_task_eval_enabled = config.get('per_task_eval_enabled', False)
            if per_task_eval_enabled:
                print(f"\n  Evaluating on all tasks 0..{task_id} for CL metrics...")
                task_performances = {}
                model_eval = eqx.combine(params, static)

                for prev_task_id in range(task_id + 1):
                    # Get test loader for this specific task
                    test_loader = data.generate_test_loader(prev_task_id, config.get('batch_size', 64))

                    # Evaluate model on this task
                    test_metric = trainer.compute_test_metric(
                        params=params,
                        static=static,
                        testloader=test_loader,
                        problem_type=problem_type,
                        loss_type=loss_type
                    )

                    task_performances[prev_task_id] = float(test_metric)
                    print(f"    Task {prev_task_id}: {test_metric:.4f}")

                # Record performance matrix entry
                trainer.record_task_performance(record_dict, task_id, task_performances)

        else:
            # AWB 5-step pipeline for tasks 1+
            print(f"AWB pipeline (lr={task_lr:.6f})")

            # Added by Claude: Initialize task recording
            model = eqx.combine(params, static)
            arch_info = get_model_architecture(model)
            trainer.initialize_task(record_dict, task_id, arch_info)

            # Added by Claude: Task transition warmup to reduce loss spike between tasks
            task_lr_factor = config.get('awb_task_lr_factor', DEFAULT_AWB_TASK_LR_FACTOR)
            task_warmup_epochs = config.get('awb_task_warmup_epochs', DEFAULT_AWB_TASK_WARMUP_EPOCHS)
            base_lr = config.get('lr', DEFAULT_LR)
            warmup_lr = base_lr * task_lr_factor

            # STEP 1: Preliminary training (with optional warmup)
            awb_prelim_epochs = config.get('awb_preliminary_epochs', 50)
            print(f"  Step 1: Preliminary training ({awb_prelim_epochs} epochs)")

            # Phase 1a: Warmup with low LR at task transition
            if task_warmup_epochs > 0:
                print(f"    Task warmup: {task_warmup_epochs} epochs at LR={warmup_lr:.2e}")
                optim_warmup = create_optimizer_with_lr(config, warmup_lr)
                opt_state = optim_warmup.init(params)

                warmup_iters = min(task_warmup_epochs, awb_prelim_epochs)
                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params,
                    static=static,
                    opt_state=opt_state,
                    optim=optim_warmup,
                    n_iter=warmup_iters,
                    task_id=task_id,
                    config=config,
                    record_dict=record_dict,
                    notABTrain=True,
                    problem_type=problem_type,
                    loss_type=loss_type,
                    phase='preliminary',
                    record_training=True,
                    global_iteration_offset=0
                )
                # Subtract warmup epochs from total preliminary epochs
                remaining_prelim = awb_prelim_epochs - warmup_iters
                # Phase 1b: Continue with full LR for remaining preliminary epochs
                if remaining_prelim > 0:
                    print(f"    Full LR: {remaining_prelim} epochs at LR={base_lr:.2e}")
                    opt_state = update_learning_rate(opt_state, base_lr)
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train__=(trainloader, exploader, valloader, testloader),
                        params=params,
                        static=static,
                        opt_state=opt_state,
                        optim=optim_warmup,
                        n_iter=remaining_prelim,
                        task_id=task_id,
                        config=config,
                        record_dict=record_dict,
                        notABTrain=True,
                        problem_type=problem_type,
                        loss_type=loss_type,
                        phase='preliminary',
                        record_training=True,
                        global_iteration_offset=0
                    )
            else:
                # No warmup, run all preliminary epochs at full LR
                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params,
                    static=static,
                    opt_state=opt_state,
                    optim=optim,
                    n_iter=awb_prelim_epochs,
                    task_id=task_id,
                    config=config,
                    record_dict=record_dict,
                    notABTrain=True,
                    problem_type=problem_type,
                    loss_type=loss_type,
                    phase='preliminary',
                    record_training=True,
                    global_iteration_offset=0
                )

            # STEP 2: Decide if architecture change needed
            # Added by Claude: Use task_id=0 since preliminary training uses global_iteration_offset=0
            trainWLoss = compute_avg_loss(record_dict['iterations'], task_id=0, epochs=awb_prelim_epochs)
            # Added by Claude: Get previous task's optimal loss (for task 1, this is task 0)
            # Handle case where previous task's optimal_loss might not exist
            if (task_id - 1) in record_dict['architecture_history']:
                end_last = record_dict['architecture_history'][task_id - 1].get('optimal_loss')
                if end_last is None:
                    # Fallback: use compute_avg_loss on previous task
                    end_last = compute_avg_loss(record_dict['iterations'], task_id - 1, epochs_per_task)
            else:
                # Should not happen, but provide fallback
                end_last = trainWLoss

            change_arch = should_change_arch(trainWLoss, end_last)
            print(f"  Step 2: Architecture change decision: {change_arch}")
            print(f"    trainWLoss={trainWLoss:.6f}, prev_task_loss={end_last:.6f}, ratio={trainWLoss/end_last:.3f}")

            # Initialize history entry for this task
            model_temp = eqx.combine(params, static)
            history_entry = {
                'task_id': task_id,
                'original_arch': get_model_architecture(model_temp),
                'searched_candidates': [],
                'preliminary_loss': trainWLoss,
                'arch_changed': False,
                'change_reason': None,
            }

            if change_arch:
                print("  ARCHITECTURE CHANGE TRIGGERED!")
                history_entry['change_reason'] = 'should_change_arch=True'

                # Save weights before architecture search
                model = eqx.combine(params, static)
                saved_weights = save_model_weights(model)

                # STEP 3a: Architecture search
                print(f"  Step 3a: Architecture search")
                original_arch, new_arch, candidates, search_time = run_architecture_search(
                    model=model,
                    config=config,
                    task_id=task_id,
                    trainWLoss=trainWLoss,
                    preliminary_epochs=awb_prelim_epochs,
                    trainloader=trainloader,
                    exploader=exploader,
                    valloader=valloader,
                    testloader=testloader,
                    trainer=trainer
                )

                history_entry['search_time'] = search_time
                history_entry['searched_candidates'] = candidates

                # Restore weights after search
                model = restore_model_weights(model, saved_weights)

                # Check if architecture actually changed
                if new_arch is not None and new_arch != original_arch:
                    # Initialize A/B matrices using model-specific functions
                    network = config.get('network', 'fcnn')

                    if network == 'fcnn':
                        # MLP: use generic function with flat lists
                        model = initialize_AB_matrices(model, original_arch, new_arch)

                    elif network in ['cnn', 'cnn3d'] and hasattr(model, 'conv_layers'):
                        # CNN/CNN3D: use CNN-specific function
                        # original_arch = (feed_sizes, filter_size)
                        # new_arch = (new_feed_sizes, new_filter_size)
                        from .classification import set_new_AB_matrices_cnn, set_new_AB_matrices_cnn3d

                        # Detect CNN3D
                        is_cnn3d = (
                            type(model).__name__ == 'CNN3D' or
                            hasattr(model, 'A_conv1') or
                            config.get('data', '') in ['cifar10', 'cifar100'] or
                            (hasattr(model, 'channel_in') and model.channel_in > 1)
                        )

                        original_feed, original_filter = original_arch
                        new_feed, new_filter = new_arch

                        if is_cnn3d:
                            model = set_new_AB_matrices_cnn3d(model, original_feed, new_feed,
                                                             original_filter, new_filter)
                        else:
                            model = set_new_AB_matrices_cnn(model, original_feed, new_feed,
                                                           original_filter, new_filter)

                    elif network == 'gcn' and hasattr(model, 'gcn_sizes'):
                        # GCN: use GCN-specific function
                        # original_arch = (gcn_sizes, feed_sizes)
                        # new_arch = (new_gcn_sizes, new_feed_sizes)
                        from .graph_classification import set_new_AB_matrices_gcn

                        original_gcn, original_feed = original_arch
                        new_gcn, new_feed = new_arch

                        model = set_new_AB_matrices_gcn(model, original_gcn, original_feed,
                                                       new_gcn, new_feed)
                    else:
                        raise ValueError(f"Unknown network type for AWB: {network}")

                    history_entry['arch_changed'] = True
                    print(f"    New architecture found!")

                    # Added by Claude: Check if transfer should be skipped (Condition 3)
                    awb_skip_transfer = config.get('awb_skip_transfer', False)

                    if awb_skip_transfer:
                        # CONDITION 3: Architecture change WITHOUT knowledge transfer
                        # Skip A/B training (Steps 3b-4), reinitialize with new architecture
                        print(f"  awb_skip_transfer=True: Skipping A/B training and V transformation")
                        print(f"  Reinitializing model with new architecture (random weights)")

                        # Get key for random initialization
                        key = jax.random.PRNGKey(task_id * 1000 + 42)

                        # Reinitialize model with new architecture
                        network = config.get('network', 'fcnn')
                        if network == 'fcnn':
                            from ..models.mlp import MLP
                            model = MLP(new_arch, key=key)
                        elif network in ['cnn', 'cnn3d']:
                            # CNN/CNN3D: new_arch = (feed_sizes, filter_size)
                            new_feed, new_filter = new_arch
                            if network == 'cnn3d':
                                from ..models.cnn import CNN3D
                                channel_in = config.get('channel_in', 3)
                                channel_out = config.get('channel_out', 10)
                                model = CNN3D(new_feed, new_filter, channel_in, channel_out, key=key)
                            else:
                                from ..models.cnn import CNN
                                channel_out = config.get('channel_out', 10)
                                model = CNN(new_feed, new_filter, channel_out, key=key)
                        elif network == 'gcn':
                            # GCN: new_arch = (gcn_sizes, feed_sizes)
                            new_gcn, new_feed = new_arch
                            from ..models.gcn import GCN
                            num_features = config.get('num_features', data.input_size)
                            num_classes = config.get('num_classes', data.output_size)
                            model = GCN(gcn_sizes=new_gcn, feed_sizes=new_feed,
                                       num_features=num_features, num_classes=num_classes, key=key)
                        else:
                            raise ValueError(f"Unknown network type for reinit: {network}")

                        history_entry['transfer_skipped'] = True
                        history_entry['reinitialized_with_new_arch'] = True

                        # Train with new architecture for remaining epochs
                        remaining_epochs = epochs_per_task - awb_prelim_epochs
                        print(f"  Training new architecture ({remaining_epochs} epochs)")

                        params, static = partition_model_for_standard_training(model)
                        opt_state = optim.init(params)
                        global_iter_offset = task_id * epochs_per_task

                        params, static, opt_state, record_dict = trainer.train__CL(
                            train__=(trainloader, exploader, valloader, testloader),
                            params=params,
                            static=static,
                            opt_state=opt_state,
                            optim=optim,
                            n_iter=remaining_epochs,
                            task_id=task_id,
                            config=config,
                            record_dict=record_dict,
                            notABTrain=True,
                            problem_type=problem_type,
                            loss_type=loss_type,
                            phase='main',
                            record_training=True,
                            global_iteration_offset=global_iter_offset
                        )

                        # Record final loss
                        optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, remaining_epochs)
                        history_entry['optimal_loss'] = optimal_loss

                    else:
                        # CONDITION 4: Full AWB with knowledge transfer
                        # STEP 3b: Train A/B matrices
                        awb_ab_epochs = config.get('awb_ab_training_epochs', 50)
                        print(f"  Step 3b: Train A/B matrices ({awb_ab_epochs} epochs)")

                        # Added by Claude: Initialize AB training recording
                        trainer.initialize_ab_training(record_dict, task_id)

                        diff_model, static_model = partition_model_for_AB_training(model)
                        # Added by Claude: Use inject_hyperparams for LR scheduling support
                        ab_optim = create_optimizer(config)
                        ab_opt_state = ab_optim.init(diff_model)
                        diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
                            train__=(trainloader, exploader, valloader, testloader),
                            params=diff_model,
                            static=static_model,
                            opt_state=ab_opt_state,
                            optim=ab_optim,
                            n_iter=awb_ab_epochs,
                            task_id=task_id,
                            config=config,
                            record_dict=record_dict,
                            notABTrain=False,
                            problem_type=problem_type,
                            loss_type=loss_type,
                            phase='ab',  # Added by Claude: AB training phase
                            record_training=True,  # Added by Claude: Record AB training
                            global_iteration_offset=0  # Added by Claude: Reset for AB phase
                        )

                        # Record A/B training loss
                        # Added by Claude: Use task_id=0 for AB training since global_iteration_offset=0
                        ab_loss = compute_avg_loss(record_dict['iterations'], task_id=0, epochs=awb_ab_epochs)
                        history_entry['ab_training_loss'] = ab_loss

                        # STEP 4: Compute V = A @ W @ B.T
                        print(f"  Step 4: Compute V transformation")
                        model = eqx.combine(diff_model, static_model)

                        # Use model-specific V transformation
                        network = config.get('network', 'fcnn')
                        if network == 'fcnn':
                            model = apply_V_transformation(model)
                        elif network in ['cnn', 'cnn3d'] and hasattr(model, 'conv_layers'):
                            from .classification import compute_V_from_AWB_cnn, compute_V_from_AWB_cnn3d
                            # Detect CNN3D
                            is_cnn3d = (
                                type(model).__name__ == 'CNN3D' or
                                hasattr(model, 'A_conv1') or
                                config.get('data', '') in ['cifar10', 'cifar100'] or
                                (hasattr(model, 'channel_in') and model.channel_in > 1)
                            )
                            if is_cnn3d:
                                model = compute_V_from_AWB_cnn3d(model)
                            else:
                                model = compute_V_from_AWB_cnn(model)
                        elif network == 'gcn' and hasattr(model, 'gcn_sizes'):
                            from ..core.awb import compute_V_from_AWB_gcn
                            model = compute_V_from_AWB_gcn(model)
                        else:
                            raise ValueError(f"Unknown network type for AWB V transformation: {network}")

                        # STEP 5: Train V with A/B frozen
                        # Added by Claude: Use warmup with low LR to reduce loss spike after V transformation
                        v_lr_factor = config.get('awb_v_lr_factor', DEFAULT_AWB_V_LR_FACTOR)
                        v_warmup_epochs = config.get('awb_v_warmup_epochs', DEFAULT_AWB_V_WARMUP_EPOCHS)
                        base_lr = config.get('lr', DEFAULT_LR)
                        warmup_lr = base_lr * v_lr_factor

                        # Calculate remaining epochs: total - preliminary - AB training
                        awb_ab_epochs = config.get('awb_ab_training_epochs', 50)
                        remaining_epochs = epochs_per_task - awb_prelim_epochs - awb_ab_epochs
                        remaining_epochs = max(0, remaining_epochs)  # Ensure non-negative
                        print(f"  Step 5: Train V with A/B frozen ({remaining_epochs} epochs)")
                        print(f"    Warmup: {v_warmup_epochs} epochs at LR={warmup_lr:.2e}, then LR={base_lr:.2e}")

                        params, static = partition_model_for_standard_training(model)

                        # Phase 1: Warmup with low LR
                        if v_warmup_epochs > 0:
                            optim_warmup = create_optimizer_with_lr(config, warmup_lr)
                            opt_state = optim_warmup.init(params)

                            warmup_iters = min(v_warmup_epochs, remaining_epochs)
                            global_iter_offset = task_id * epochs_per_task
                            params, static, opt_state, record_dict = trainer.train__CL(
                                train__=(trainloader, exploader, valloader, testloader),
                                params=params,
                                static=static,
                                opt_state=opt_state,
                                optim=optim_warmup,
                                n_iter=warmup_iters,
                                task_id=task_id,
                                config=config,
                                record_dict=record_dict,
                                notABTrain=True,
                                problem_type=problem_type,
                                loss_type=loss_type,
                                phase='main',
                                record_training=True,
                                global_iteration_offset=global_iter_offset
                            )
                            remaining_epochs -= 0

                        # Phase 2: Continue with full LR (preserves optimizer momentum)
                        if remaining_epochs > 0:
                            if v_warmup_epochs > 0:
                                # Update LR to full value while preserving optimizer state
                                opt_state = update_learning_rate(opt_state, base_lr)
                                optim_full = optim_warmup  # Reuse same optimizer (LR updated in state)
                            else:
                                # No warmup phase, create fresh optimizer with full LR
                                optim_full = create_optimizer(config)
                                opt_state = optim_full.init(params)
                                global_iter_offset = task_id * epochs_per_task

                            params, static, opt_state, record_dict = trainer.train__CL(
                                train__=(trainloader, exploader, valloader, testloader),
                                params=params,
                                static=static,
                                opt_state=opt_state,
                                optim=optim_full,
                                n_iter=remaining_epochs,
                                task_id=task_id,
                                config=config,
                                record_dict=record_dict,
                                notABTrain=True,
                                problem_type=problem_type,
                                loss_type=loss_type,
                                phase='main',
                                record_training=True,
                                global_iteration_offset=global_iter_offset
                            )

                        # Record final optimal loss and architecture
                        optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, epochs_per_task)
                        history_entry['optimal_loss'] = optimal_loss
                        history_entry['final_arch'] = get_model_architecture(eqx.combine(params, static))

                else:
                    # Architecture search returned same architecture
                    history_entry['arch_changed'] = False
                    history_entry['change_reason'] = 'search_found_same_arch'

                    remaining_epochs = epochs_per_task - awb_prelim_epochs

                    # Added by Claude: Handle case where all epochs were used in preliminary training
                    if remaining_epochs > 0:
                        print(f"    Architecture search found same architecture, continuing normal training for {remaining_epochs} epochs")

                        # Added by Claude: Reinitialize optimizer state to match current params structure
                        # This is needed because preliminary training may have used a different optimizer
                        opt_state = optim.init(params)
                        opt_state = update_learning_rate(opt_state, task_lr)

                        # Added by Claude: Compute global iteration offset for proper recording
                        # After preliminary training, we continue from the expected task offset
                        global_iter_offset = task_id * epochs_per_task

                        params, static, opt_state, record_dict = trainer.train__CL(
                            train__=(trainloader, exploader, valloader, testloader),
                            params=params,
                            static=static,
                            opt_state=opt_state,
                            optim=optim,
                            n_iter=remaining_epochs,
                            task_id=task_id,
                            config=config,
                            record_dict=record_dict,
                            notABTrain=True,
                            problem_type=problem_type,
                            loss_type=loss_type,
                            phase='main',
                            record_training=True,
                            global_iteration_offset=global_iter_offset
                        )

                        optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, remaining_epochs)
                    else:
                        # Added by Claude: No remaining epochs - use preliminary training loss
                        print(f"    Architecture search found same architecture, no remaining epochs (used all {awb_prelim_epochs} in preliminary training)")
                        optimal_loss = trainWLoss

                    history_entry['optimal_loss'] = optimal_loss
                    history_entry['final_arch'] = get_model_architecture(eqx.combine(params, static))
                    print(f"    Final optimal_loss: {optimal_loss:.6f}")

            else:
                # No architecture change decided
                history_entry['change_reason'] = 'should_change_arch=False'
                print(f"    No architecture change needed")

                # Calculate remaining epochs after preliminary training
                remaining_epochs = epochs_per_task - awb_prelim_epochs

                # Added by Claude: Handle case where all epochs were used in preliminary training
                if remaining_epochs > 0:
                    print(f"    Continuing training for {remaining_epochs} remaining epochs")

                    # Added by Claude: Reinitialize optimizer state to match current params structure
                    # This is needed because preliminary training may have used a different optimizer
                    # (optim_warmup) which creates incompatible state with the original optim
                    opt_state = optim.init(params)
                    opt_state = update_learning_rate(opt_state, task_lr)

                    # Added by Claude: Compute global iteration offset for proper recording
                    global_iter_offset = task_id * epochs_per_task

                    params, static, opt_state, record_dict = trainer.train__CL(
                        train__=(trainloader, exploader, valloader, testloader),
                        params=params,
                        static=static,
                        opt_state=opt_state,
                        optim=optim,
                        n_iter=remaining_epochs,
                        task_id=task_id,
                        config=config,
                        record_dict=record_dict,
                        notABTrain=True,
                        problem_type=problem_type,
                        loss_type=loss_type,
                        phase='main',
                        record_training=True,
                        global_iteration_offset=global_iter_offset
                    )

                    optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, remaining_epochs)
                else:
                    # Added by Claude: No remaining epochs - use preliminary training loss
                    print(f"    No remaining epochs (used all {awb_prelim_epochs} in preliminary training)")
                    optimal_loss = trainWLoss

                history_entry['optimal_loss'] = optimal_loss
                history_entry['final_arch'] = history_entry['original_arch']
                print(f"    Final optimal_loss: {optimal_loss:.6f}")

            # Added by Claude: Add loss ratio and adaptive values to history entry
            history_entry['loss_ratio'] = loss_ratio
            history_entry['adaptive_lr_min'] = compute_adaptive_lr_min(config, loss_ratio)
            history_entry['adaptive_grad_weights'] = compute_adaptive_grad_weights(config, loss_ratio)

            # Save history entry for this task
            record_dict['architecture_history'][task_id] = history_entry

            # Added by Claude: Save optimal loss for next task's loss ratio computation
            if history_entry.get('optimal_loss') is not None:
                previous_task_loss = history_entry['optimal_loss']

            # Append to experience replay
            data.append_to_experience(task_id)

            # Added by Claude: Optional per-task evaluation for CL metrics
            # After training task_id, test on ALL tasks from 0 to task_id
            per_task_eval_enabled = config.get('per_task_eval_enabled', False)
            if per_task_eval_enabled:
                print(f"\n  Evaluating on all tasks 0..{task_id} for CL metrics...")
                task_performances = {}
                model_eval = eqx.combine(params, static)

                for prev_task_id in range(task_id + 1):
                    # Get test loader for this specific task
                    test_loader = data.generate_test_loader(prev_task_id, config.get('batch_size', 64))

                    # Evaluate model on this task
                    test_metric = trainer.compute_test_metric(
                        params=params,
                        static=static,
                        testloader=test_loader,
                        problem_type=problem_type,
                        loss_type=loss_type
                    )

                    task_performances[prev_task_id] = float(test_metric)
                    print(f"    Task {prev_task_id}: {test_metric:.4f}")

                # Record performance matrix entry
                trainer.record_task_performance(record_dict, task_id, task_performances)

    # Added by Claude: Print comprehensive architecture evolution summary
    if awb_enabled and 'architecture_history' in record_dict:
        print("\n" + "="*70)
        print("ARCHITECTURE EVOLUTION SUMMARY")
        print("="*70)

        for tid in sorted(record_dict['architecture_history'].keys()):
            entry = record_dict['architecture_history'][tid]

            print(f"\nTask {tid}:")
            print(f"  Original:       {entry['original_arch']}")
            print(f"  Final:          {entry['final_arch']}")
            print(f"  Changed:        {entry['arch_changed']}")
            print(f"  Reason:         {entry.get('change_reason', 'N/A')}")

            if entry.get('preliminary_loss') is not None:
                print(f"  Prelim Loss:    {entry['preliminary_loss']:.6f}")
            if 'ab_training_loss' in entry:
                print(f"  AB Train Loss:  {entry['ab_training_loss']:.6f}")
            if entry.get('optimal_loss') is not None:
                print(f"  Optimal Loss:   {entry['optimal_loss']:.6f}")

            if entry.get('searched_candidates'):
                print(f"  Selected Architecture:")
                for cand in entry['searched_candidates']:
                    print(f"    {cand['arch']} → loss={cand['loss']:.6f}")

            if 'search_time' in entry and entry['search_time'] > 0:
                print(f"  Search Time:    {entry['search_time']:.2f}s")

        print("\n" + "="*70)

    # Save final model and records
    if config.get('model_path'):
        model = eqx.combine(params, static)

        # Add "_awb" suffix if AWB is enabled and not already in path
        base_path = config['model_path']
        if config.get('awb_enabled', False) and '_awb' not in base_path:
            base_path = f"{base_path}_awb"
        model_path = f"{base_path}_run{run_id}"

        # Create output directory if needed
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

        # Use equinox serialization for models (handles JAX pytrees correctly)
        eqx.tree_serialise_leaves(f"{model_path}.eqx", model)

        # Added by Claude: Use RecordingMixin's save_record_dict for consistent naming
        # This creates files like: regression_sine_fcnn_run0_records.pkl (non-AWB)
        #                          regression_sine_fcnn_awb_run0_records.pkl (AWB)
        trainer.save_record_dict(record_dict, model_path)

    return record_dict
