"""
Training runner for regression tasks.

Orchestrates the training loop for regression problems (e.g., sine wave)
with optional AWB (Adaptive Weight Basis) pipeline for architecture morphing.

The AWB 5-step algorithm:
    Task 0: Standard CL training
    Tasks 1+:
        STEP 1: Preliminary training on new task
        STEP 2: Decide if architecture change needed
        If change needed:
            STEP 3a: Architecture search
            STEP 3b: Train A/B with W frozen
            STEP 4: Compute V = A @ W @ B.T
            STEP 5: Train V with A/B frozen
        Else:
            Continue standard training
"""

import os
import numpy as np
import optax
import equinox as eqx
from typing import Dict, Any, Optional

from ..core.trainer import Trainer
from ..core.awb import (
    compute_avg_loss,
    should_change_arch,
    compute_ab_threshold,
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
    save_layer_weights,
    restore_layer_weights,
)
from ..models.mlp import MLP, create_mlp
from ..datasets.sine import SineDataset
from ..arch_search.mlp_search import arch_search_MLP
from ..config.constants import (
    DEFAULT_BATCH_SIZE_VECTOR,
    DEFAULT_REPLAY_BUFFER_VECTOR,
    DEFAULT_AWB_ENABLED,
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
)


# Added by Claude: Optimizer factory with inject_hyperparams for dynamic LR updates
def create_optimizer(config: Dict[str, Any]) -> optax.GradientTransformationExtraArgs:
    """Create optimizer from configuration with injectable hyperparameters.

    Uses optax.inject_hyperparams to allow dynamic learning rate updates
    without resetting the optimizer state (preserves momentum, etc.).

    Supports the following optimizers via config['optimizer']:
        - 'adam' (default): Adam optimizer
        - 'adamw': AdamW optimizer with weight decay
        - 'sgd': SGD with optional momentum
        - 'rmsprop': RMSprop optimizer

    Args:
        config: Configuration dictionary with:
            - optimizer: Optimizer type string (default: 'adam')
            - lr: Learning rate (default: 1e-4)
            - weight_decay: Weight decay for adamw (default: 1e-4)
            - momentum: Momentum for sgd (default: 0.9)

    Returns:
        optax.GradientTransformationExtraArgs: Configured optimizer with injectable LR
    """
    lr = config.get('lr', 1e-4)
    optimizer_name = config.get('optimizer', 'adam').lower()

    if optimizer_name == 'adam':
        return optax.inject_hyperparams(optax.adam)(learning_rate=lr)
    elif optimizer_name == 'adamw':
        weight_decay = config.get('weight_decay', 1e-4)
        return optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optax.inject_hyperparams(optax.sgd)(learning_rate=lr, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return optax.inject_hyperparams(optax.rmsprop)(learning_rate=lr)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}', using adam")
        return optax.inject_hyperparams(optax.adam)(learning_rate=lr)


def update_learning_rate(opt_state, new_lr: float):
    """Update the learning rate in optimizer state without resetting momentum.

    Args:
        opt_state: Current optimizer state from optax.inject_hyperparams optimizer
        new_lr: New learning rate value

    Returns:
        Updated optimizer state with new learning rate (or unchanged if not injectable)
    """
    # Check if optimizer was created with inject_hyperparams
    # inject_hyperparams creates a state with .hyperparams attribute
    if hasattr(opt_state, 'hyperparams') and 'learning_rate' in opt_state.hyperparams:
        opt_state.hyperparams['learning_rate'] = new_lr
    # If not an injectable optimizer, just return unchanged
    # (can't dynamically update LR for standard optax optimizers)
    return opt_state


# Added by Claude: Learning rate scheduler for task-based decay
def compute_task_lr(config: Dict[str, Any], task_id: int) -> float:
    """Compute learning rate for a given task with decay scheduling.

    Supports multiple decay strategies via config['lr_schedule']:
        - 'constant': No decay (default)
        - 'step': Step decay by lr_decay_factor every lr_decay_steps tasks
        - 'exponential': Exponential decay: lr * (lr_decay_factor ^ task_id)
        - 'cosine': Cosine annealing to lr_min over n_task tasks
        - 'linear': Linear decay from lr to lr_min over n_task tasks

    Args:
        config: Configuration dictionary with:
            - lr: Initial learning rate
            - lr_schedule: Decay schedule type (default: 'constant')
            - lr_decay_factor: Decay factor for step/exponential (default: 0.9)
            - lr_decay_steps: Tasks between decays for step schedule (default: 1)
            - lr_min: Minimum learning rate (default: 1e-6)
            - n_task: Total number of tasks (for cosine/linear schedules)
        task_id: Current task index (0-based)

    Returns:
        float: Learning rate for this task
    """
    base_lr = config.get('lr', 1e-4)
    schedule = config.get('lr_schedule', 'constant').lower()
    decay_factor = config.get('lr_decay_factor', 0.9)
    decay_steps = config.get('lr_decay_steps', 1)
    lr_min = config.get('lr_min', 1e-6)
    n_tasks = config.get('n_task', 40)

    if schedule == 'constant':
        return base_lr

    elif schedule == 'step':
        # Decay by factor every decay_steps tasks
        n_decays = task_id // decay_steps
        lr = base_lr * (decay_factor ** n_decays)
        return max(lr, lr_min)

    elif schedule == 'exponential':
        # Continuous exponential decay
        lr = base_lr * (decay_factor ** task_id)
        return max(lr, lr_min)

    elif schedule == 'cosine':
        # Cosine annealing from base_lr to lr_min
        progress = task_id / max(1, n_tasks - 1)
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * progress))
        return lr

    elif schedule == 'linear':
        # Linear decay from base_lr to lr_min
        progress = task_id / max(1, n_tasks - 1)
        lr = base_lr - progress * (base_lr - lr_min)
        return max(lr, lr_min)

    else:
        print(f"Warning: Unknown lr_schedule '{schedule}', using constant")
        return base_lr


def load_regression_checkpoint(config: Dict[str, Any]):
    """Load or create model, trainer, optimizer, and dataset for regression.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (trainer, optimizer, dataset, model)
    """
    # Create dataset
    dataset_config = {
        'delta': config.get('delta', 0.001),
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR),
        'len_exp_replay': config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR),
        'debug_mode': config.get('debug_mode', False),
        'debug_limit': config.get('debug_limit', 100),
        'n_task': config.get('n_task', 40),
        'data_path': config.get('data_path', 'Incremental_Sine1e^4.p'),
        'problem': config.get('problem', 'vectors'),
        'network': config.get('network', 'fcnn'),
    }
    dataset = SineDataset(dataset_config)

    # Create model
    model_config = {
        'input_size': dataset.input_size,
        'output_size': dataset.output_size,
        'n_layers': config.get('n_layers', 4),
        'hln': config.get('hln', 256),
        'awb_enabled': config.get('awb_enabled', DEFAULT_AWB_ENABLED),
    }
    model = create_mlp(model_config)

    # Create trainer
    trainer = Trainer(
        loss=config.get('loss', 'mse'),
        metric=config.get('metric', 'mse'),
        problem=config.get('problem', 'vectors'),
    )

    # Added by Claude: Create optimizer from config (supports adam, adamw, sgd, rmsprop)
    optimizer = create_optimizer(config)

    return trainer, optimizer, dataset, model


def train_model_reg(config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
    """Train model for regression task using unified training loop.

    When AWB is enabled (config['awb_enabled'] = True), uses the 5-step algorithm.
    When AWB is disabled (default), uses standard CL training for all tasks.

    Args:
        config: Configuration dictionary containing:
            - n_task: Number of tasks
            - epochs_per_task: Training epochs per task
            - batch_size: Batch size
            - lr: Learning rate
            - awb_enabled: Whether to use AWB pipeline
            - save_iter: Save metrics every N epochs
            - model_path: Path to save model
            - flag: Regularization flags [current_weight, experience_weight]
        run_id: Run identifier for logging

    Returns:
        record_dict: Dictionary containing training records
    """
    # Load checkpoint (creates model, trainer, dataset, optimizer)
    trainer, optim, data, model = load_regression_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Move A, B to static (frozen) for standard training if AWB enabled
    if config.get('awb_enabled', DEFAULT_AWB_ENABLED):
        static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
        params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

    # Initialize optimizer state
    opt_state = optim.init(params)

    # Check if AWB pipeline is enabled
    awb_enabled = config.get('awb_enabled', DEFAULT_AWB_ENABLED)

    # AWB configuration parameters
    preliminary_epochs = config.get('awb_preliminary_epochs', DEFAULT_AWB_PRELIMINARY_EPOCHS)
    ab_training_epochs = config.get('awb_ab_training_epochs', DEFAULT_AWB_AB_TRAINING_EPOCHS)
    ab_warmup_epochs = config.get('awb_ab_warmup_epochs', DEFAULT_AWB_AB_WARMUP_EPOCHS)
    ab_max_iterations = config.get('awb_ab_max_iterations', DEFAULT_AWB_AB_MAX_ITERATIONS)
    averaging_window = config.get('awb_averaging_window', DEFAULT_AWB_AVERAGING_WINDOW)

    # Training config shared across phases
    train_config = {
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR),
        'problem': config.get('problem', 'vectors'),
        'data_id': config.get('data', 'sine'),
        'flag': config.get('flag', [1.0, 1.0]),
        'len_exp_replay': config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_VECTOR),
        'network': config.get('network', 'fcnn'),
        # Added by Claude: Gradient combination weights [alpha, beta, gamma]
        'grad_weights': config.get('grad_weights', None),
    }

    # Track baseline losses for AWB decision logic
    end_last0 = None
    end_last = None
    mlp_arch_list = []

    for i in range(config['n_task']):
        print(f"\n{'='*50}")
        print(f"Task {i}")
        print(f"{'='*50}")

        # Generate dataloaders for current task
        dataloader_curr, dataloader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training'
        )
        test_loader_curr, test_loader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing'
        )
        train_data = (dataloader_curr, dataloader_exp,
                      (test_loader_curr, test_loader_exp), (test_loader_curr, test_loader_exp))

        if i == 0:
            # Task 0: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='regression'
            )
            end_last0 = compute_avg_loss(record_dict['iterations'], i,
                                         config['epochs_per_task'], averaging_window)
            end_last = end_last0
            print(f"Task 0 baseline loss: {end_last0:.6f}")

        elif awb_enabled:
            # AWB PIPELINE FOR TASKS 1+
            print(f"AWB Pipeline enabled for task {i}")

            # STEP 1: Preliminary training
            print(f"STEP 1: Preliminary training ({preliminary_epochs} epochs)")
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=preliminary_epochs, save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='regression'
            )

            model = eqx.combine(params, static)
            trainWLoss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                          preliminary_epochs, averaging_window)

            # STEP 2: Decide if architecture change is needed
            print(f"STEP 2: Checking architecture change (loss={trainWLoss:.6f})")
            change_arch = should_change_arch(trainWLoss, end_last0, end_last)
            original_arch = model.sizes
            mlp_weight_layer, mlp_bias_layer = save_layer_weights(model)

            if True:
                print("ARCHITECTURE CHANGE TRIGGERED!")

                # STEP 3a: Architecture search
                # Searches for optimal architecture by training candidate models
                # Uses trainWLoss (from current model) as baseline for comparison
                opt_arch = arch_search_MLP(
                    original_arch=original_arch,
                    task_id=i,
                    trainW_loss=trainWLoss,
                    og_epochs=preliminary_epochs,
                    config=config,
                    dataloader_curr=dataloader_curr,
                    dataloader_exp=dataloader_exp,
                    test_loader_curr=test_loader_curr,
                    test_loader_exp=test_loader_exp,
                    current_model=model,
                )
                print(f"Optimal Architecture: {opt_arch}")
                mlp_arch_list.append(opt_arch)
                # Restore weights after search (search may have modified model state)
                model = restore_layer_weights(model, mlp_weight_layer, mlp_bias_layer)

                if opt_arch != original_arch:
                    model = set_new_AB_matrices(model, original_arch, opt_arch)

                    # STEP 3b: Train A/B with W frozen
                    print(f"STEP 3b: Training A/B matrices with W frozen")
                    diff_model, static_model = partition_for_AB_training(model)
                    optim2 = optax.adam(1e-4)
                    opt_state2 = optim2.init(diff_model)
                    ab_threshold = compute_ab_threshold(trainWLoss, end_last)

                    diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                        train_data, diff_model, static_model, opt_state2, optim2,
                        n_iter=ab_training_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        notABTrain=False, problem_type='vectors', loss_type='regression'
                    )

                    AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                               ab_training_epochs, averaging_window)
                    ab_iter = 1
                    while (trainWLoss * ab_threshold < AB_loss) and (ab_iter < ab_max_iterations):
                        diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                            train_data, diff_model, static_model, opt_state2, optim2,
                            n_iter=ab_training_epochs, save_iter=config['save_iter'],
                            task_id=i, config=train_config, record_dict=record_dict,
                            notABTrain=False, problem_type='vectors', loss_type='regression'
                        )
                        AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                                   ab_training_epochs, averaging_window)
                        ab_iter += 1

                    model = eqx.combine(diff_model, static_model)

                    # STEP 4: Compute V = A @ W @ B.T
                    print("STEP 4: Computing V = A @ W @ B.T")
                    model = compute_V_from_AWB(model)
                    params, static = partition_for_standard_training(model)

                    # STEP 5: Train V with A/B frozen
                    print(f"STEP 5: Training with new weights V")
                    optim = optax.adam(1e-3)
                    opt_state = optim.init(params)

                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=ab_warmup_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='regression'
                    )
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='regression'
                    )
                else:
                    print("Architecture search found same architecture, continuing normal training")
                    mlp_arch_list.append(original_arch)
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='regression'
                    )
            else:
                print("Architecture did NOT change - continuing standard training")
                mlp_arch_list.append(original_arch)
                params, static, opt_state, record_dict = trainer.train__CL(
                    train_data, params, static, opt_state, optim,
                    n_iter=ab_warmup_epochs + config['epochs_per_task'],
                    save_iter=config['save_iter'], task_id=i,
                    config=train_config, record_dict=record_dict,
                    problem_type='vectors', loss_type='regression'
                )

            end_last = compute_avg_loss(record_dict.get('iterations', {}), i,
                                        config['epochs_per_task'], averaging_window)

            # Re-partition model for next iteration
            model = eqx.combine(params, static)
            params, static = eqx.partition(model, eqx.is_array)
            if model.awb_enabled:
                static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
                params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

        else:
            # AWB DISABLED: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='regression'
            )

        # Add current task to experience replay
        data.append_to_experience(i)

    # Print architecture history if AWB was used
    if awb_enabled and mlp_arch_list:
        print("\nArchitecture history:")
        for task_idx, arch in enumerate(mlp_arch_list):
            print(f"  Task {task_idx + 1}: {arch}")

    # Save model and records
    model = eqx.combine(params, static)
    model_path = config.get('model_path', 'outputs/model')

    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    eqx.tree_serialise_leaves(model_path + '.eqx', model)
    trainer.save_record_dict(record_dict, model_path)

    print(f"\nModel saved to: {model_path}.eqx")

    del model, params, static
    return record_dict
