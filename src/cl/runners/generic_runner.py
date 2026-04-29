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
# Added by Claude: AWB pipeline refactoring
from ..core.awb_pipeline import run_awb_task
# Added by Claude: Profiling support
from ..core.profiling import enable_profiling
from ..models.mlp import MLPAWBOps, create_mlp, MLP
from ..models.cnn import CNNAWBOps, CNN, CNN3D
from ..models.gcn import GCNAWBOps, GCN
from ..models.transformer import TransformerAWBOps, create_transformer, TransformerEncoder
import pickle
from pathlib import Path
from ..datasets.sine import SineDataset
from ..datasets.mnist import MNISTDataset, PermutedMNISTDataset
from ..datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
from ..datasets.synthetic_graph import SyntheticGraphDataset, TaskShiftGraphDataset

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
    DEFAULT_SAVE_ITER,
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

    # Transformer: has embed_dim and n_heads
    if hasattr(model, 'embed_dim') and hasattr(model, 'n_heads'):
        arch_info['embed_dim'] = model.embed_dim
        arch_info['n_heads'] = model.n_heads
        arch_info['mlp_dim'] = model.mlp_dim
        arch_info['n_layers'] = model.n_layers
        arch_info['seq_len'] = model.seq_len
        arch_info['token_dim'] = model.input_dim
        if hasattr(model, 'output_dim'):
            arch_info['output_dim'] = model.output_dim
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
# Evaluation Helpers
# ============================================================================

def evaluate_on_loader(trainer, params, static, loader, problem_type='vectors'):
    """Evaluate model on a full DataLoader and return average metric.

    Added by Claude: Helper to properly iterate through DataLoader for evaluation.

    Args:
        trainer: Trainer instance with return_metric method
        params: Model parameters
        static: Static model components
        loader: DataLoader to evaluate on
        problem_type: 'vectors' or 'graph'

    Returns:
        float: Average metric across all batches
    """
    metrics = []

    # Added by Claude: Get graph transforms if needed (converts edge_index to adj)
    if problem_type == 'graph':
        from ..core.loops import get_graph_transforms
        transforms = get_graph_transforms()

    for batch in loader:
        # Convert PyTorch tensors to JAX arrays for vectors
        if problem_type == 'vectors':
            x, y = batch
            # Convert to numpy first, then to JAX
            # Match training dtype: float64 for x (always), infer y dtype from data
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            y_np = y.numpy() if hasattr(y, 'numpy') else y

            x = jnp.asarray(x_np, dtype=jnp.float64)
            # Infer y dtype: int64 for classification (integer labels), float64 for regression
            y_dtype = jnp.int64 if y_np.dtype.kind in ('i', 'u') else jnp.float64
            y = jnp.asarray(y_np, dtype=y_dtype)
            batch = (x, y)
        else:
            # For graphs: apply transforms to convert edge_index to adj
            batch = transforms(batch)

        metric_value = trainer.return_metric(
            params=params,
            statics=static,
            data=batch,
            notABTrain=True
        )
        metrics.append(float(metric_value))

    # Return average metric across all batches
    return sum(metrics) / len(metrics) if metrics else 0.0


# ============================================================================
# Model and Dataset Loading (Config-based Dispatch)
# ============================================================================

def load_checkpoint(config: Dict[str, Any]):
    """Initialize model, dataset, trainer, and optimizer based on config.

    Note: Despite the name "load_checkpoint", this function initializes fresh components
    (checkpoint loading is not yet implemented).

    Args:
        config: Configuration dict

    Returns:
        Tuple of (trainer, optimizer, dataset, model)
    """
    # Added by Claude: Unified initialization for all problem types
    from ..config.constants import DEFAULT_BATCH_SIZE_VECTOR, DEFAULT_REPLAY_BUFFER_VECTOR, DEFAULT_AWB_ENABLED

    problem = config.get('problem', 'vectors')  # 'graph' or 'vectors'
    data_type = config.get('data', 'sine')      # Dataset name
    network = config.get('network', 'fcnn')     # Network architecture

    # Create dataset based on data type
    dataset_config = {
        'delta': config.get('delta', 0.001),
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR),
        'len_exp_replay': config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR),
        'debug_mode': config.get('debug_mode', False),
        'debug_limit': config.get('debug_limit', 100),
        'n_task': config.get('n_task', 10),
        'problem': problem,
        'network': network,
    }

    if data_type == 'sine':
        dataset = SineDataset(dataset_config)
    elif data_type == 'mnist':
        dataset = MNISTDataset(dataset_config)
    elif data_type == 'permuted_mnist':
        dataset = PermutedMNISTDataset(dataset_config)
    elif data_type == 'cifar10':
        dataset = CIFAR10Dataset(dataset_config)
    elif data_type == 'cifar100':
        dataset = CIFAR100Dataset(dataset_config)
    elif data_type == 'synthetic':
        dataset = SyntheticGraphDataset(dataset_config)
    # Added by Claude: Task-shift synthetic graph dataset for domain-shift CL
    elif data_type == 'synthetic_taskshift':
        # Pass through task-shift specific config parameters
        dataset_config.update({
            'num_graphs': config.get('num_graphs', 2000),
            'num_channels': config.get('num_channels', 10),
            'avg_num_nodes': config.get('avg_num_nodes', 20),
            'num_classes': config.get('num_classes', 5),
            'feature_noise_base': config.get('feature_noise_base', 0.1),
            'edge_dropout_base': config.get('edge_dropout_base', 0.05),
            'feature_shift_base': config.get('feature_shift_base', 0.05),
        })
        dataset = TaskShiftGraphDataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {data_type}")

    # Create model based on network type
    model_config = {
        'input_size': dataset.input_size,
        'output_size': dataset.output_size,
        'awb_enabled': config.get('awb_enabled', DEFAULT_AWB_ENABLED),
    }

    if network == 'fcnn':
        # MLP for regression or vectors
        model_config.update({
            'n_layers': config.get('n_layers', 4),
            'hln': config.get('hln', 75),
        })
        model = create_mlp(model_config)
    elif network == 'cnn':
        # CNN for image classification (MNIST) - from old classification.py
        from ..config.constants import DEFAULT_INPUT_SIZE_MNIST
        channel_out = config.get('channel_out', 3)
        filter_size = config.get('filter_size', 4)  # Integer, not list
        input_size = config.get('input_size', DEFAULT_INPUT_SIZE_MNIST)  # Image dimension, not flattened
        num_classes = model_config['output_size']

        # Calculate flatten_size based on conv/pool output
        # After conv: (input_size - filter_size + 1)
        # After MaxPool2d(kernel_size=2, stride=2): conv_out // 2
        conv_output = (input_size - filter_size + 1)
        pool_output = conv_output // 2
        flatten_size = channel_out * pool_output * pool_output

        # Default feed sizes: [flatten_size, 512, 64, num_classes]
        feed_sizes = config.get('feed_sizes', [flatten_size, 512, 64, num_classes])
        feed_sizes[0] = flatten_size  # Always use calculated flatten size
        feed_sizes[-1] = num_classes  # Fixed by Claude: Always use correct num_classes

        model = CNN(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_in=config.get('channel_in', 1),
            channel_out=channel_out,
            awb_arch=config.get('awb_arch', None)
        )
    elif network == 'cnn3d':
        # CNN3D for multi-channel images (CIFAR) - from old classification.py
        from ..config.constants import DEFAULT_INPUT_SIZE_CIFAR, DEFAULT_CHANNEL_OUT_CNN3D
        channel_out = config.get('channel_out', DEFAULT_CHANNEL_OUT_CNN3D)
        channel_in = config.get('channel_in', 3)
        input_size = config.get('input_size', DEFAULT_INPUT_SIZE_CIFAR)
        filter_size = config.get('filter_size', 4)  # Integer, not list
        num_classes = model_config['output_size']

        # Calculate feed layer input size for CNN3D (two conv+pool layers)
        # After conv1: (input_size - filter_size + 1)
        # After pool1 (stride=2): conv1_out // 2
        # After conv2: (pool1_out - filter_size + 1)
        # After pool2 (stride=2): conv2_out // 2
        conv1_out = input_size - filter_size + 1
        pool1_out = conv1_out // 2
        conv2_out = pool1_out - filter_size + 1
        pool2_out = conv2_out // 2
        flatten_size = pool2_out * pool2_out * channel_out * 2  # channel_out * 2 after second conv

        # Default feed sizes for CIFAR
        feed_sizes = config.get('feed_sizes', [flatten_size, 512, 256, num_classes])
        feed_sizes[0] = flatten_size  # Always use calculated flatten size
        feed_sizes[-1] = num_classes  # Fixed by Claude: Always use correct num_classes (important for CIFAR-100)

        model = CNN3D(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_in=channel_in,
            channel_out=channel_out,
            num_classes=num_classes
        )
    elif network == 'gcn':
        # GCN for graph data - from old graph_classification.py
        gcn_sizes = config.get('gcn_sizes', [model_config['input_size'], 128])
        feed_sizes = config.get('feed_sizes', [128, 128, 128, model_config['output_size']])
        num_classes = model_config['output_size']

        # Get node_num from a sample batch
        sample_batch = next(iter(dataset.get_test_loader()))
        node_num = sample_batch.x.shape[0]

        model = GCN(
            in_size=model_config['input_size'],
            feed_sizes=feed_sizes,
            gcn_sizes=gcn_sizes,
            node_num=node_num,
            SEED=config.get('seed', 1234),
            out_size=num_classes,
            graph=True
        )
    elif network == 'transformer':
        transformer_config = model_config.copy()
        transformer_config.update({
            'transformer_seq_len': config.get('transformer_seq_len'),
            'transformer_token_dim': config.get('transformer_token_dim'),
            'transformer_embed_dim': config.get('transformer_embed_dim'),
            'transformer_n_heads': config.get('transformer_n_heads'),
            'transformer_mlp_dim': config.get('transformer_mlp_dim'),
            'transformer_n_layers': config.get('transformer_n_layers'),
            'awb_enabled': config.get('awb_enabled', DEFAULT_AWB_ENABLED),
        })
        model = create_transformer(transformer_config)
    else:
        raise ValueError(f"Unknown network: {network}")

    # Create trainer
    trainer = Trainer(
        loss=config.get('loss', 'mse' if problem == 'regression' else 'cross_entropy'),
        metric=config.get('metric', 'mse' if problem == 'regression' else 'accuracy'),
        problem=problem,
    )

    # Create optimizer
    optimizer = create_optimizer(config)

    return trainer, optimizer, dataset, model


def build_model_from_arch(arch_info: Dict[str, Any], config: Dict[str, Any]):
    awb_enabled = config.get('awb_enabled', False)
    model_type = arch_info.get('model_type', '')

    if 'sizes' in arch_info:
        return MLP.create_with_architecture(arch_info['sizes'], seed=0, awb_enabled=awb_enabled)

    if 'embed_dim' in arch_info and 'n_heads' in arch_info:
        arch_spec = {
            'seq_len': arch_info['seq_len'],
            'token_dim': arch_info['token_dim'],
            'embed_dim': arch_info['embed_dim'],
            'n_heads': arch_info['n_heads'],
            'mlp_dim': arch_info['mlp_dim'],
            'n_layers': arch_info['n_layers'],
            'output_dim': arch_info.get('output_dim', config.get('n_class', 10)),
        }
        return TransformerEncoder.create_with_architecture(arch_spec, seed=0, awb_enabled=awb_enabled)

    if 'gcn_sizes' in arch_info and 'feed_sizes' in arch_info:
        return GCN.create_with_architecture(
            (arch_info['gcn_sizes'], arch_info['feed_sizes']),
            seed=0,
            awb_enabled=awb_enabled
        )

    if 'feed_sizes' in arch_info and 'filter_size' in arch_info:
        key = jax.random.PRNGKey(0)
        filter_size = arch_info['filter_size']
        feed_sizes = arch_info['feed_sizes']
        input_size = arch_info.get('input_size')
        channel_in = arch_info.get('channel_in')
        channel_out = arch_info.get('channel_out')

        if model_type == 'CNN3D':
            return CNN3D(
                key=key,
                filter_size=filter_size,
                feed_sizes=feed_sizes,
                input_size=input_size,
                channel_in=channel_in,
                channel_out=channel_out
            )
        return CNN(
            key=key,
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_in=channel_in if channel_in is not None else 1,
            channel_out=channel_out,
            padding=arch_info.get('padding'),
            stride=arch_info.get('stride')
        )

    raise ValueError("Unsupported architecture info for resume")


def load_resume_checkpoint(config: Dict[str, Any], optim):
    resume_from = config.get('resume_from')
    resume_latest = config.get('resume_latest', False)

    if not resume_from and not resume_latest:
        return None

    if resume_from:
        record_path = Path(resume_from)
        if record_path.is_dir():
            record_paths = sorted(record_path.glob("*_records.pkl"), key=lambda p: p.stat().st_mtime)
            if not record_paths:
                raise FileNotFoundError(f"No checkpoint records found in {record_path}")
            record_path = record_paths[-1]
    else:
        checkpoint_dir = Path(f"{config.get('model_path', 'outputs/model')}_checkpoints")
        record_paths = sorted(checkpoint_dir.glob("*_records.pkl"), key=lambda p: p.stat().st_mtime)
        if not record_paths:
            raise FileNotFoundError(f"No checkpoint records found in {checkpoint_dir}")
        record_path = record_paths[-1]

    with open(record_path, 'rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and 'record_dict' in payload:
        record_dict = payload.get('record_dict', {})
        metadata = payload.get('metadata', {})
        model_path = payload.get('model_path')
        opt_state_path = payload.get('opt_state_path')
    else:
        record_dict = payload
        metadata = {}
        model_path = None
        opt_state_path = None

    if model_path is None:
        model_path = str(record_path).replace('_records.pkl', '.eqx')
    if opt_state_path is None:
        opt_state_path = str(record_path).replace('_records.pkl', '_opt.eqx')

    arch_info = metadata.get('arch_info')
    if not arch_info:
        raise ValueError("Checkpoint metadata missing arch_info; cannot resume")

    # Fixed by Claude: debug output for resume diagnostics
    print(f"[RESUME-DBG] record_path = {record_path}")
    print(f"[RESUME-DBG] model_path  = {model_path}")
    print(f"[RESUME-DBG] metadata keys: {list(metadata.keys())}")
    print(f"[RESUME-DBG]   phase             = {metadata.get('phase')}")
    print(f"[RESUME-DBG]   task_id           = {metadata.get('task_id')}")
    print(f"[RESUME-DBG]   epoch             = {metadata.get('epoch')}")
    print(f"[RESUME-DBG]   arch_info         = {metadata.get('arch_info')}")
    print(f"[RESUME-DBG]   awb_original_arch = {metadata.get('awb_original_arch')}")
    print(f"[RESUME-DBG]   awb_best_arch     = {metadata.get('awb_best_arch')}")
    if isinstance(record_dict, dict):
        hist = record_dict.get('architecture_history', {})
        print(f"[RESUME-DBG] architecture_history keys: {list(hist.keys()) if isinstance(hist, dict) else 'N/A'}")
        t = metadata.get('task_id', 0)
        if isinstance(hist, dict) and t in hist:
            print(f"[RESUME-DBG]   hist[{t}] = {hist[t]}")

    model_template = build_model_from_arch(arch_info, config)
    model = None
    try:
        model = eqx.tree_deserialise_leaves(model_path, model_template)
    except RuntimeError as exc:
        phase = metadata.get('phase')
        alt_arches = []
        if phase in ('awb_search', 'ab'):
            alt_arches = [metadata.get('awb_best_arch'), metadata.get('awb_original_arch')]

        for alt_arch in alt_arches:
            if not alt_arch or alt_arch == arch_info:
                continue
            try:
                model_template = build_model_from_arch(alt_arch, config)
                model = eqx.tree_deserialise_leaves(model_path, model_template)
                print(f"[WARN] Loaded checkpoint using alternate arch from metadata ({phase}).")
                break
            except RuntimeError:
                model = None

        # AB phase: model has old-arch weights + A/B matrices for the transition.
        # Build template with set_AB_matrices(original_arch, new_arch) to match.
        if model is None and phase == 'ab':
            orig_arch = metadata.get('awb_original_arch')
            new_arch = metadata.get('awb_best_arch') or arch_info

            # Recover original_arch from linear growth schedule if not in metadata
            if orig_arch is None and config.get('awb_arch_linear_growth', False):
                t = metadata.get('task_id', 1)
                embed_step = config.get('awb_arch_embed_step', 32)
                mlp_step = config.get('awb_arch_mlp_step', 32)
                base_embed = config.get('transformer_embed_dim', 128)
                base_mlp = config.get('transformer_mlp_dim', 256)
                base_n_layers = config.get('transformer_n_layers', 2)
                base_n_heads = config.get('transformer_n_heads', 4)

                # Previous task's arch = base + (task_id-1) * step
                prev_embed = base_embed + (t - 1) * embed_step
                prev_mlp = base_mlp + (t - 1) * mlp_step

                n_heads_schedule = config.get('awb_arch_n_heads_schedule')
                if n_heads_schedule and isinstance(n_heads_schedule, list) and t >= 2:
                    prev_heads = n_heads_schedule[min(t - 2, len(n_heads_schedule) - 1)]
                else:
                    prev_heads = base_n_heads

                n_layers_schedule = config.get('awb_arch_n_layers_schedule')
                if n_layers_schedule and isinstance(n_layers_schedule, list) and t >= 2:
                    prev_layers = n_layers_schedule[min(t - 2, len(n_layers_schedule) - 1)]
                else:
                    prev_layers = base_n_layers

                while prev_embed % prev_heads != 0:
                    prev_embed += embed_step

                orig_arch = {
                    'seq_len': config.get('transformer_seq_len', 784),
                    'token_dim': config.get('transformer_token_dim', 1),
                    'embed_dim': prev_embed,
                    'n_heads': prev_heads,
                    'mlp_dim': prev_mlp,
                    'n_layers': prev_layers,
                    'output_dim': config.get('output_dim', 10),
                }
                print(f"[RESUME] Recovered orig_arch from linear growth schedule: {orig_arch}")

            if orig_arch is not None and new_arch is not None:
                try:
                    # Build model from ORIGINAL arch (matches the old-sized W on disk),
                    # then set_AB_matrices creates the transition (old W + new-sized A/B).
                    template = build_model_from_arch(orig_arch, config)
                    awb_ops = create_awb_operations(template)
                    template = awb_ops.set_AB_matrices(template, orig_arch, new_arch)
                    model = eqx.tree_deserialise_leaves(model_path, template)
                    print(f"[WARN] Loaded AB checkpoint with transition A/B ({orig_arch} -> {new_arch}).")
                except Exception as inner:
                    print(f"[WARN] AB transition reconstruction failed: {inner}")
                    model = None

        # Fixed by Claude: phase='main' checkpoints were saved AFTER an arch change,
        # so A/B matrices in the file have the transition shape (new_size, old_size)
        # rather than the fresh square-identity shape a template gets from __init__.
        # Rebuild the template with A/B explicitly set to the transition shape, then
        # deserialize into it.
        if model is None and phase == 'main':
            orig_arch = metadata.get('awb_original_arch')
            new_arch = metadata.get('awb_best_arch') or arch_info

            # Fixed by Claude: older checkpoints had a bug where dict-shaped arches
            # (e.g., Transformer) were saved as list(dict) = keys only, dropping all
            # the integer values. Detect that corruption and discard -- we'll fall
            # through to the architecture_history fallback below.
            def _arch_corrupt(a):
                if a is None:
                    return True
                if isinstance(arch_info, dict) and not isinstance(a, dict):
                    return True
                return False

            if _arch_corrupt(orig_arch):
                print(f"[RESUME] Discarding corrupt metadata['awb_original_arch']={orig_arch}; "
                      f"will recover from architecture_history.")
                orig_arch = None
            if _arch_corrupt(new_arch):
                print(f"[RESUME] Discarding corrupt metadata['awb_best_arch']={new_arch}; "
                      f"will use arch_info.")
                new_arch = arch_info

            # Fallback: if metadata didn't carry the arches (older checkpoints),
            # try to recover them from record_dict['architecture_history'].
            # Important: during mid-Step-5 saves, record_dict['architecture_history']
            # has NOT yet been populated for the CURRENT task (the post-task write
            # happens at the end of train_model's task loop). So look at task_id-1's
            # final_arch as the original_arch for task_id.
            if orig_arch is None:
                hist = record_dict.get('architecture_history', {}) if isinstance(record_dict, dict) else {}
                t = metadata.get('task_id', 0)
                if isinstance(hist, dict):
                    if t in hist:
                        orig_arch = hist[t].get('original_arch')
                        new_arch = hist[t].get('final_arch') or new_arch
                    elif (t - 1) in hist:
                        # Current-task entry not written yet -> use previous task's
                        # final_arch as this task's original_arch.
                        orig_arch = hist[t - 1].get('final_arch')
                        print(f"[RESUME] Recovered orig_arch from hist[{t-1}]['final_arch'].")

            if orig_arch is not None and new_arch is not None:
                try:
                    template = build_model_from_arch(arch_info, config)
                    awb_ops = create_awb_operations(template)
                    template = awb_ops.set_AB_matrices(template, orig_arch, new_arch)
                    model = eqx.tree_deserialise_leaves(model_path, template)
                    print(f"[WARN] Loaded main checkpoint with transition-shape A/B "
                          f"({orig_arch} -> {new_arch}).")
                except Exception as inner:
                    print(f"[WARN] main-phase transition reconstruction failed: {inner}")
                    model = None
            else:
                print(f"[WARN] Cannot reconstruct A/B transition shape: "
                      f"orig_arch={orig_arch}, new_arch={new_arch}")

        if model is None and phase == 'awb_search':
            # AWB search checkpoints may be saved without A/B matrices
            config_no_awb = {**config, 'awb_enabled': False}
            model_template = build_model_from_arch(arch_info, config_no_awb)
            model = eqx.tree_deserialise_leaves(model_path, model_template)
            print("[WARN] Loaded awb_search checkpoint without A/B matrices; A/B will be reinitialized in Step 3b.")

        if model is None:
            raise exc

    params, static = partition_model_for_standard_training(model)
    opt_state_template = optim.init(params)
    if Path(opt_state_path).exists():
        opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state_template)
    else:
        opt_state = opt_state_template

    resume_state = {
        'task_id': metadata.get('task_id', 0),
        'epoch': metadata.get('epoch', 0),
        'phase': metadata.get('phase', 'main'),
        'notABTrain': metadata.get('notABTrain', True),
        'awb_best_arch': metadata.get('awb_best_arch'),
        'awb_original_arch': metadata.get('awb_original_arch'),
        'preliminary_loss': metadata.get('preliminary_loss'),
        'previous_task_loss': metadata.get('previous_task_loss'),
        'cumulative_compute': metadata.get('cumulative_compute'),
    }

    return model, opt_state, record_dict, resume_state


def create_awb_operations(model):
    """Create appropriate AWBOperations instance based on model type.

    Added by Claude: AWB pipeline refactoring helper.

    Args:
        model: Equinox model (MLP, CNN, CNN3D, or GCN)

    Returns:
        AWBOperations instance (MLPAWBOps, CNNAWBOps, or GCNAWBOps)
    """
    # MLP: has 'sizes' attribute
    if hasattr(model, 'sizes') and not hasattr(model, 'feed_sizes'):
        return MLPAWBOps()

    # CNN/CNN3D: has feed_sizes and filter_size
    elif hasattr(model, 'feed_sizes') and hasattr(model, 'filter_size'):
        # Detect CNN3D
        is_cnn3d = (
            type(model).__name__ == 'CNN3D' or
            hasattr(model, 'A_conv1') or  # CNN3D has A_conv1, regular CNN has A_conv
            (hasattr(model, 'channel_in') and model.channel_in > 1)
        )
        return CNNAWBOps(is_cnn3d=is_cnn3d)

    # GCN: has gcn_sizes and feed_sizes
    elif hasattr(model, 'gcn_sizes') and hasattr(model, 'feed_sizes'):
        return GCNAWBOps()

    # Transformer: has embed_dim and n_heads
    elif hasattr(model, 'embed_dim') and hasattr(model, 'n_heads'):
        return TransformerAWBOps()

    else:
        raise ValueError(f"Unknown model type: {type(model).__name__}")


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
    # Added by Claude: Enable profiling if requested
    enable_profiling(config.get('profiling_enabled', False))
    if config.get('profiling_enabled'):
        print(f"\n{'='*60}")
        print(f"PROFILING ENABLED - Detailed timing information will be shown")
        print(f"{'='*60}\n")

    # Load components based on config
    trainer, optim, data, model = load_checkpoint(config)

    # Optional resume
    resume_state = None
    resume_payload = load_resume_checkpoint(config, optim)
    if resume_payload is not None:
        model, opt_state, record_dict, resume_state = resume_payload
        params, static = partition_model_for_standard_training(model)
    else:
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
    if resume_state is None:
        record_dict = trainer.initialize_record_dict(config, run_id=run_id)
    else:
        if record_dict is None:
            record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Added by Claude: Initialize architecture history tracking
    if 'architecture_history' not in record_dict:
        record_dict['architecture_history'] = {}

    # Added by Claude: Track loss from previous task for adaptive LR and grad weights
    previous_task_loss = None

    # Training loop over tasks
    start_task = 0
    resume_epoch = 0
    resume_phase = 'main'
    if resume_state is not None:
        start_task = resume_state.get('task_id', 0)
        resume_epoch = resume_state.get('epoch', 0) + 1
        resume_phase = resume_state.get('phase', 'main')

        if awb_enabled and resume_phase not in ('main', 'awb_search', 'ab'):
            raise ValueError("Resume inside AWB phases is not supported. Resume from a main checkpoint or disable AWB.")

        if resume_phase in ('awb_search', 'ab'):
            resume_epoch = 0

        if resume_epoch >= epochs_per_task:
            start_task += 1
            resume_epoch = 0

        # Fixed by Claude: removed hard-fail for mid-Step-5 resume of AWB tasks.
        # AWB's Steps 1-4 already happened before the checkpoint; resuming means
        # skipping straight into Step 5 main training at resume_epoch. Dispatched
        # via resume_state -> run_awb_task (which handles resume_phase='main').

    for task_id in range(start_task, n_tasks):
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

        # Pre-training evaluation for Forward Transfer (FWT) metric.
        # FWT needs A[task_id-1][task_id]: accuracy on task_id BEFORE training it,
        # using the model trained through task_id-1.
        per_task_eval_enabled = config.get('per_task_eval_enabled', False)
        if per_task_eval_enabled and task_id >= 1:
            print(f"\n  FWT pre-eval: evaluating on task {task_id} before training...")
            test_loader_fwt = data.generate_test_loader(task_id, config.get('batch_size', 64))
            fwt_metric = evaluate_on_loader(
                trainer=trainer,
                params=params,
                static=static,
                loader=test_loader_fwt,
                problem_type=problem_type
            )
            print(f"    Task {task_id} (before training): {fwt_metric:.4f}")
            # Store as A[task_id-1][task_id] in the performance matrix
            if 'task_performance_matrix' not in record_dict:
                record_dict['task_performance_matrix'] = {}
            prev_key = task_id - 1
            if prev_key not in record_dict['task_performance_matrix']:
                record_dict['task_performance_matrix'][prev_key] = {}
            record_dict['task_performance_matrix'][prev_key][str(task_id)] = float(fwt_metric)

        # Task 0 or AWB disabled: Standard CL training
        if task_id == 0 or not awb_enabled:
            if resume_state is not None and task_id == start_task and resume_epoch > 0:
                print(f"Resuming task {task_id} at epoch {resume_epoch}")
                model = eqx.combine(params, static)

                save_iter = config.get('save_iter', DEFAULT_SAVE_ITER)

                remaining_epochs = max(0, epochs_per_task - resume_epoch)
                if remaining_epochs == 0:
                    continue

                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params, static=static, opt_state=opt_state, optim=optim,
                    n_iter=remaining_epochs, save_iter=save_iter,
                    task_id=task_id, config=config, record_dict=record_dict,
                    problem_type=problem_type, loss_type=loss_type,
                    phase='main', record_training=True,
                    global_iteration_offset=task_id * epochs_per_task + resume_epoch
                )
                model = eqx.combine(params, static)
                continue
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
            # Added by Claude: Cap warmup to leave at least 1 epoch for main training
            warmup_epochs = min(warmup_epochs, max(0, epochs_per_task - 1))
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
                    test_metric = evaluate_on_loader(
                        trainer=trainer,
                        params=params,
                        static=static,
                        loader=test_loader,
                        problem_type=problem_type
                    )

                    task_performances[prev_task_id] = float(test_metric)
                    print(f"    Task {prev_task_id}: {test_metric:.4f}")

                # Record performance matrix entry
                trainer.record_task_performance(record_dict, task_id, task_performances)

        else:
            # AWB 5-step pipeline for tasks 1+ (refactored to use awb_pipeline module)
            print(f"AWB pipeline (lr={task_lr:.6f})")

            # Initialize task recording
            model = eqx.combine(params, static)
            arch_info = get_model_architecture(model)
            trainer.initialize_task(record_dict, task_id, arch_info)

            # Get previous task loss for architecture change decision
            if (task_id - 1) in record_dict.get('architecture_history', {}):
                previous_task_loss = record_dict['architecture_history'][task_id - 1].get('optimal_loss')
            else:
                previous_task_loss = None

            # Create AWB operations for this model type
            awb_ops = create_awb_operations(model)

            # Run AWB 5-step pipeline
            train_data = (trainloader, exploader, valloader, testloader)
            model, optim, opt_state, record_dict = run_awb_task(
                task_id=task_id,
                trainer=trainer,
                model=model,
                train_data=train_data,
                optim=optim,
                opt_state=opt_state,
                config=config,
                record_dict=record_dict,
                awb_ops=awb_ops,
                problem_type=problem_type,
                loss_type=loss_type,
                previous_task_loss=previous_task_loss,
                # Fixed by Claude: also forward resume_state when phase='main' so
                # run_awb_task can jump straight into Step 5 main at resume_epoch.
                resume_state=resume_state if (resume_state is not None and resume_phase in ('awb_search', 'ab', 'main') and task_id == start_task) else None
            )

            # Extract final params and static
            params, static = eqx.partition(model, eqx.is_array)

            # Added by Claude: Extract architecture history from AWB task
            # The AWB pipeline stores architecture info in record_dict['tasks'][task_id]
            main_training = record_dict['tasks'][task_id]['main_training']
            optimal_loss = main_training['V'][-1] if main_training['V'] else None

            # Update architecture history
            if task_id in record_dict.get('tasks', {}):
                task_info = record_dict['tasks'][task_id]
                prev_arch = None
                if task_id > 0 and (task_id - 1) in record_dict['architecture_history']:
                    prev_arch = record_dict['architecture_history'][task_id - 1]['final_arch']

                record_dict['architecture_history'][task_id] = {
                    'task_id': task_id,
                    'original_arch': prev_arch,
                    'searched_candidates': task_info.get('searched_architectures', []),
                    'final_arch': get_model_architecture(model),
                    'preliminary_loss': task_info.get('preliminary_loss'),
                    'optimal_loss': optimal_loss,
                    'arch_changed': task_info.get('architecture_changed', False),
                    'change_reason': task_info.get('change_reason', 'unknown'),
                    'search_time': task_info.get('search_time', 0.0),
                    'loss_ratio': task_info.get('loss_ratio'),
                    'adaptive_lr_min': None,  # AWB uses different LR strategy
                    'adaptive_grad_weights': None,  # AWB uses different grad weight strategy
                }

            # Save optimal loss for next task's loss ratio computation
            if optimal_loss is not None:
                previous_task_loss = optimal_loss

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
                    test_metric = evaluate_on_loader(
                        trainer=trainer,
                        params=params,
                        static=static,
                        loader=test_loader,
                        problem_type=problem_type
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
