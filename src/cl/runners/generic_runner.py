"""
Generic unified runner for all problem types (regression, classification, graph).

This module consolidates regression.py, classification.py, and graph_classification.py
into a single generic runner using config-based dispatch.

Added by Claude: Layer-level AWB abstraction refactor.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pickle
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
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    momentum = config.get('momentum', 0.9)

    # Added by Claude: Use inject_hyperparams to allow dynamic LR changes
    if optimizer_name == 'adam':
        base_optimizer = optax.inject_hyperparams(optax.adamw)
        return base_optimizer(learning_rate=lr, weight_decay=weight_decay)
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


def compute_task_lr(config: Dict[str, Any], task_id: int) -> float:
    """Compute learning rate for current task based on schedule.

    Args:
        config: Configuration dict with lr_schedule
        task_id: Current task ID

    Returns:
        Learning rate for this task
    """
    base_lr = config.get('lr', 1e-4)
    schedule = config.get('lr_schedule', 'constant')

    if schedule == 'constant':
        return base_lr
    elif schedule == 'step':
        decay_factor = config.get('lr_decay_factor', 0.9)
        return base_lr * (decay_factor ** task_id)
    elif schedule == 'exponential':
        decay_factor = config.get('lr_decay_factor', 0.95)
        return base_lr * (decay_factor ** task_id)
    elif schedule == 'cosine':
        # Simple cosine decay over tasks
        n_tasks = config.get('n_task', 5)
        return base_lr * 0.5 * (1 + jnp.cos(jnp.pi * task_id / n_tasks))
    elif schedule == 'linear':
        n_tasks = config.get('n_task', 5)
        return base_lr * (1 - task_id / n_tasks)
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

    # Training loop over tasks
    for task_id in range(n_tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        print(f"{'='*60}")

        # Generate dataset for current task
        trainloader, exploader = data.generate_dataset(task_id,
                                                       config.get('batch_size', 64),
                                                       phase='training')
        valloader, _ = data.generate_dataset(task_id,
                                             config.get('batch_size', 64),
                                             phase='testing')
        testloader, _ = data.generate_dataset(task_id,
                                              config.get('batch_size', 64),
                                              phase='testing')

        # Compute learning rate for this task
        task_lr = compute_task_lr(config, task_id)

        # Added by Claude: Update optimizer state with task-specific learning rate
        opt_state = update_learning_rate(opt_state, task_lr)

        # Task 0 or AWB disabled: Standard CL training
        if task_id == 0 or not awb_enabled:
            print(f"Standard CL training (lr={task_lr:.6f})")

            # Recombine model
            model = eqx.combine(params, static)

            # Train with standard partition
            params, static, opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=params,
                static=static,
                opt_state=opt_state,
                optim=optim,
                n_iter=epochs_per_task,
                task_id=task_id,
                config=config,
                record_dict=record_dict,
                notABTrain=True,
                problem_type=problem_type,
                loss_type=loss_type
            )

            # Added by Claude: Track architecture for this task
            model_combined = eqx.combine(params, static)
            record_dict['architecture_history'][task_id] = {
                'task_id': task_id,
                'original_arch': record_dict['architecture_history'][task_id - 1]['final_arch'] if task_id > 0 else None,
                'searched_candidates': [],
                'final_arch': get_model_architecture(model_combined),
                'preliminary_loss': None,
                'optimal_loss': compute_avg_loss(record_dict['iterations'], task_id, epochs_per_task),
                'arch_changed': False,
                'change_reason': 'baseline_task' if task_id == 0 else 'awb_disabled',
                'search_time': 0.0,
            }

            # Append to experience replay
            data.append_to_experience(task_id)

        else:
            # AWB 5-step pipeline for tasks 1+
            print(f"AWB pipeline (lr={task_lr:.6f})")

            # STEP 1: Preliminary training
            awb_prelim_epochs = config.get('awb_preliminary_epochs', 10)
            print(f"  Step 1: Preliminary training ({awb_prelim_epochs} epochs)")

            model = eqx.combine(params, static)
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
                loss_type=loss_type
            )

            # STEP 2: Decide if architecture change needed
            trainWLoss = compute_avg_loss(record_dict['iterations'], task_id, awb_prelim_epochs)
            end_last0 = record_dict['architecture_history'][0]['optimal_loss']
            end_last = record_dict['architecture_history'][task_id - 1]['optimal_loss']

            change_arch = should_change_arch(trainWLoss, end_last0, end_last)
            print(f"  Step 2: Architecture change decision: {change_arch}")
            print(f"    trainWLoss={trainWLoss:.6f}, baseline={end_last0:.6f}, prev={end_last:.6f}")

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

                    # STEP 3b: Train A/B matrices
                    awb_ab_epochs = config.get('awb_ab_training_epochs', 50)
                    print(f"  Step 3b: Train A/B matrices ({awb_ab_epochs} epochs)")

                    diff_model, static_model = partition_model_for_AB_training(model)
                    ab_optim = optax.adam(config.get('lr', 1e-4))
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
                        loss_type=loss_type
                    )

                    # Record A/B training loss
                    ab_loss = compute_avg_loss(record_dict['iterations'], task_id, awb_ab_epochs)
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
                    remaining_epochs = epochs_per_task - awb_prelim_epochs - awb_ab_epochs
                    print(f"  Step 5: Train V with A/B frozen ({remaining_epochs} epochs)")

                    params, static = partition_model_for_standard_training(model)
                    opt_state = optim.init(params)

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
                        loss_type=loss_type
                    )

                    # Record final optimal loss and architecture
                    optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, remaining_epochs)
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
                            loss_type=loss_type
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

                remaining_epochs = epochs_per_task - awb_prelim_epochs

                # Added by Claude: Handle case where all epochs were used in preliminary training
                if remaining_epochs > 0:
                    print(f"    Continuing training for {remaining_epochs} remaining epochs")

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
                        loss_type=loss_type
                    )

                    optimal_loss = compute_avg_loss(record_dict['iterations'], task_id, remaining_epochs)
                else:
                    # Added by Claude: No remaining epochs - use preliminary training loss
                    print(f"    No remaining epochs (used all {awb_prelim_epochs} in preliminary training)")
                    optimal_loss = trainWLoss

                history_entry['optimal_loss'] = optimal_loss
                history_entry['final_arch'] = history_entry['original_arch']
                print(f"    Final optimal_loss: {optimal_loss:.6f}")

            # Save history entry for this task
            record_dict['architecture_history'][task_id] = history_entry

            # Append to experience replay
            data.append_to_experience(task_id)

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
                for i, cand in enumerate(entry['searched_candidates']):
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

        # Use equinox serialization for models (handles JAX pytrees correctly)
        eqx.tree_serialise_leaves(f"{model_path}.eqx", model)
        # Use pickle for record dictionary
        with open(f"{model_path}_records.pkl", 'wb') as f:
            pickle.dump(record_dict, f)

    return record_dict
