"""
AWB (Adaptive Weight Basis) Pipeline Orchestrator.

This module implements the generic 5-step AWB algorithm for continual learning:
    STEP 1: Preliminary training on new task
    STEP 2: Decide if architecture change is needed
    STEP 3a: Architecture search (if change needed)
    STEP 3b: Train A/B matrices with W frozen
    STEP 4: Compute V = A @ W @ B^T
    STEP 5: Train V with A/B frozen (warmup + main training)

The orchestrator is model-agnostic - it calls model-specific operations via
the AWBOperations interface. This allows the same pipeline to work for:
- MLP (fully connected networks)
- CNN (convolutional networks)
- GCN (graph convolutional networks)
- Transformers (attention-based models) - future

Recording control ensures clean separation of training phases:
- Preliminary, warmup, and architecture search: NOT recorded
- AB training: Recorded separately in record_dict['tasks'][task_id]['ab_training']
- Main training: Recorded normally in record_dict['iterations']
"""

from typing import Dict, Any, Tuple
import equinox as eqx
import optax
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from .awb_operations import AWBOperations
from .awb import (
    should_change_arch,
    compute_avg_loss,
    compute_ab_threshold,
)
from ..config.constants import (
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_LR,
)


def create_balanced_validation_set(loader, validation_ratio=0.2, batch_size=64):
    """Create a balanced validation set from a data loader.

    Samples validation_ratio% of data from each class to ensure balanced representation.
    This is used for AWB architecture search to avoid using full training data.

    Args:
        loader: PyTorch DataLoader to sample from
        validation_ratio: Fraction of data to use for validation (default 0.2 = 20%)
        batch_size: Batch size for validation loader

    Returns:
        DataLoader with balanced validation set
    """
    # Collect all data from the loader
    all_x, all_y = [], []
    for batch_x, batch_y in loader:
        all_x.append(batch_x)
        all_y.append(batch_y)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # Group indices by class
    unique_classes = torch.unique(all_y)
    val_indices = []

    for cls in unique_classes:
        cls_indices = torch.where(all_y == cls)[0]
        n_samples = len(cls_indices)
        n_val = max(1, int(n_samples * validation_ratio))  # At least 1 sample per class

        # Randomly sample indices for this class
        perm = torch.randperm(n_samples)
        val_idx = cls_indices[perm[:n_val]]
        val_indices.append(val_idx)

    # Combine all validation indices
    val_indices = torch.cat(val_indices)

    # Create validation dataset
    val_x = all_x[val_indices]
    val_y = all_y[val_indices]

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False  # Keep all validation samples
    )

    print(f"  Created balanced validation set: {len(val_indices)} samples from {len(unique_classes)} classes")
    return val_loader


def run_awb_task(
    task_id: int,
    trainer,
    model: eqx.Module,
    train_data: Tuple,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    config: Dict[str, Any],
    record_dict: Dict[str, Any],
    awb_ops: AWBOperations,
    problem_type: str,
    loss_type: str,
    previous_task_loss: float = None
) -> Tuple[eqx.Module, optax.GradientTransformation, optax.OptState, Dict[str, Any]]:
    """Execute complete AWB 5-step pipeline for one task.

    Args:
        task_id: Current task ID (must be >= 1, task 0 uses standard CL)
        trainer: Trainer instance
        model: Current model (Equinox module)
        train_data: Tuple of (trainloader, exploader, valloader, testloader)
        optim: Optimizer
        opt_state: Optimizer state
        config: Configuration dictionary
        record_dict: Recording dictionary
        awb_ops: Model-specific AWB operations
        problem_type: 'vectors' or 'graph'
        loss_type: 'regression' or 'classification'
        previous_task_loss: Loss from previous task

    Returns:
        Tuple of (model, optim, opt_state, record_dict)
    """
    if task_id < 1:
        raise ValueError(f"AWB pipeline only for tasks >= 1, got task_id={task_id}")

    print(f"\n{'='*70}")
    print(f"AWB PIPELINE - Task {task_id}")
    print(f"{'='*70}")

    # Extract configuration
    preliminary_epochs = config.get('awb_preliminary_epochs', DEFAULT_AWB_PRELIMINARY_EPOCHS)
    ab_training_epochs = config.get('awb_ab_training_epochs', DEFAULT_AWB_AB_TRAINING_EPOCHS)
    ab_warmup_epochs = config.get('awb_ab_warmup_epochs', DEFAULT_AWB_AB_WARMUP_EPOCHS)
    ab_max_iterations = config.get('awb_ab_max_iterations', DEFAULT_AWB_AB_MAX_ITERATIONS)
    averaging_window = config.get('awb_averaging_window', DEFAULT_AWB_AVERAGING_WINDOW)
    epochs_per_task = config.get('epochs_per_task', 500)
    save_iter = config.get('save_iter', 50)

    trainloader, exploader, valloader, testloader = train_data

    # Added by Claude: Initialize task metadata for architecture history
    task_metadata = {
        'preliminary_loss': None,
        'architecture_changed': False,
        'change_reason': 'no_change',
        'searched_architectures': [],
        'loss_ratio': None,
        'search_time': 0.0,
    }

    # STEP 1: Preliminary Training
    print(f"\n[STEP 1] Preliminary training ({preliminary_epochs} epochs) - Recorded temporarily")
    params, static = awb_ops.partition_for_standard_training(model)

    # Added by Claude: Store current iteration count to compute preliminary offset
    current_iterations = len(record_dict.get('iterations', {}))

    params, static, opt_state, record_dict = trainer.train__CL(
        train__=(trainloader, exploader, valloader, testloader),
        params=params, static=static, opt_state=opt_state, optim=optim,
        n_iter=preliminary_epochs, save_iter=save_iter,
        task_id=task_id, config=config, record_dict=record_dict,
        problem_type=problem_type, loss_type=loss_type,
        phase='preliminary', record_training=True, global_iteration_offset=current_iterations
    )

    model = eqx.combine(params, static)

    # Added by Claude: Get preliminary loss from last recorded iteration
    iterations_dict = record_dict.get('iterations', {})
    last_prelim_iter = current_iterations + preliminary_epochs - 1
    if last_prelim_iter in iterations_dict:
        last_record = iterations_dict[last_prelim_iter]
        trainWLoss = last_record['losses']['V'] if isinstance(last_record, dict) and 'losses' in last_record else last_record[0]
    else:
        # Fallback: try to find the most recent iteration
        available_iters = sorted([k for k in iterations_dict.keys() if k >= current_iterations])
        if available_iters:
            last_record = iterations_dict[available_iters[-1]]
            trainWLoss = last_record['losses']['V'] if isinstance(last_record, dict) and 'losses' in last_record else last_record[0]
        else:
            trainWLoss = 0.0

    print(f"  Preliminary loss: {trainWLoss:.6f}")

    # Added by Claude: Store preliminary loss in metadata
    task_metadata['preliminary_loss'] = trainWLoss

    # STEP 2: Decision
    print(f"\n[STEP 2] Architecture change decision")
    if previous_task_loss is None:
        previous_task_loss = trainWLoss
        print(f"  WARNING: No previous task loss, using preliminary loss")

    # Added by Claude: Store loss ratio in metadata
    task_metadata['loss_ratio'] = trainWLoss / previous_task_loss if previous_task_loss > 0 else 1.0

    print(f"  Current: {trainWLoss:.6f}, Previous: {previous_task_loss:.6f}")

    # Added by Claude: Read threshold from config
    threshold_high = config.get('awb_loss_ratio_threshold')

    change_arch = should_change_arch(trainWLoss, previous_task_loss,
                                     threshold_high=threshold_high)
    print(f"  Decision: {'CHANGE' if change_arch else 'KEEP'}")

    original_arch = awb_ops.get_model_architecture(model)
    saved_weights = awb_ops.save_weights(model)

    if change_arch:
        # STEP 3a: Architecture Search
        print(f"\n[STEP 3a] Architecture search - NOT recorded")

        # Added by Claude: Create balanced validation sets (20% of experience replay) for arch search
        # This is more efficient than using full training data and prevents overfitting
        validation_ratio = config.get('awb_validation_ratio', 0.2)
        val_batch_size = config.get('batch_size', 64)

        val_trainloader = create_balanced_validation_set(trainloader, validation_ratio, val_batch_size)
        val_exploader = create_balanced_validation_set(exploader, validation_ratio, val_batch_size)

        # Fixed by Claude: Unpack testloader tuple (test_curr_loader, test_exp_loader)
        test_curr, test_exp = testloader

        # Use validation sets for architecture search instead of full training data
        new_arch = awb_ops.search_architecture(
            model, task_id, trainWLoss, val_trainloader, val_exploader,
            test_curr, test_exp, config, trainer
        )
        print(f"  Original: {original_arch}")
        print(f"  Optimal: {new_arch}")

        model = awb_ops.restore_weights(model, saved_weights)

        if new_arch != original_arch:
            # Added by Claude: Update metadata for architecture change
            task_metadata['architecture_changed'] = True
            task_metadata['change_reason'] = 'loss_ratio_threshold'

            # STEP 3b: Train A/B
            print(f"\n[STEP 3b] Train A/B matrices ({ab_training_epochs} epochs) - Recorded separately")
            model = awb_ops.set_AB_matrices(model, original_arch, new_arch)
            diff_model, static_model = awb_ops.partition_for_AB_training(model)
            trainer.initialize_ab_training(record_dict, task_id)

            ab_lr = config.get('awb_ab_lr', DEFAULT_LR)
            ab_optim = optax.adam(ab_lr)
            ab_opt_state = ab_optim.init(diff_model)

            diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=diff_model, static=static_model, opt_state=ab_opt_state, optim=ab_optim,
                n_iter=ab_training_epochs, save_iter=save_iter,
                task_id=task_id, config=config, record_dict=record_dict,
                notABTrain=False, problem_type=problem_type, loss_type=loss_type,
                phase='ab', record_training=True, global_iteration_offset=0
            )

            ab_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=0,
                                       epochs=ab_training_epochs, window=averaging_window)
            print(f"  AB loss: {ab_loss:.6f}")

            # Optional: Continue AB training
            ab_threshold = compute_ab_threshold(trainWLoss, previous_task_loss)
            ab_iter = 1
            while (trainWLoss * ab_threshold < ab_loss) and (ab_iter < ab_max_iterations):
                print(f"  Continuing AB training (iter {ab_iter + 1})")
                diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=diff_model, static=static_model, opt_state=ab_opt_state, optim=ab_optim,
                    n_iter=ab_training_epochs, save_iter=save_iter,
                    task_id=task_id, config=config, record_dict=record_dict,
                    notABTrain=False, problem_type=problem_type, loss_type=loss_type,
                    phase='ab', record_training=True, global_iteration_offset=0
                )
                ab_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=0,
                                           epochs=ab_training_epochs, window=averaging_window)
                ab_iter += 1

            model = eqx.combine(diff_model, static_model)

            # STEP 4: Compute V
            print(f"\n[STEP 4] Compute V = A @ W @ B^T")
            model = awb_ops.compute_V(model)
            params, static = awb_ops.partition_for_standard_training(model)

            # STEP 5: Train V
            print(f"\n[STEP 5] Train V: warmup {ab_warmup_epochs} + main {epochs_per_task}")
            from ..runners.generic_runner import create_optimizer
            optim = create_optimizer(config)
            opt_state = optim.init(params)

            if ab_warmup_epochs > 0:
                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params, static=static, opt_state=opt_state, optim=optim,
                    n_iter=ab_warmup_epochs, save_iter=save_iter,
                    task_id=task_id, config=config, record_dict=record_dict,
                    problem_type=problem_type, loss_type=loss_type,
                    phase='warmup', record_training=False,
                    global_iteration_offset=task_id * epochs_per_task
                )

            params, static, opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=params, static=static, opt_state=opt_state, optim=optim,
                n_iter=epochs_per_task, save_iter=save_iter,
                task_id=task_id, config=config, record_dict=record_dict,
                problem_type=problem_type, loss_type=loss_type,
                phase='main', record_training=True,
                global_iteration_offset=task_id * epochs_per_task
            )
            model = eqx.combine(params, static)

        else:
            # Added by Claude: Same architecture found by search
            task_metadata['architecture_changed'] = False
            task_metadata['change_reason'] = 'search_found_same'

            # Same architecture, standard training
            print(f"  Same architecture - standard training")
            params, static = awb_ops.partition_for_standard_training(model)
            from ..runners.generic_runner import create_optimizer
            optim = create_optimizer(config)
            opt_state = optim.init(params)

            total_epochs = ab_warmup_epochs + epochs_per_task
            params, static, opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=params, static=static, opt_state=opt_state, optim=optim,
                n_iter=total_epochs, save_iter=save_iter,
                task_id=task_id, config=config, record_dict=record_dict,
                problem_type=problem_type, loss_type=loss_type,
                phase='main', record_training=True,
                global_iteration_offset=task_id * epochs_per_task
            )
            model = eqx.combine(params, static)
    else:
        # Added by Claude: No architecture change needed
        task_metadata['architecture_changed'] = False
        task_metadata['change_reason'] = 'loss_ratio_below_threshold'

        # No change, standard training
        print(f"  No change - standard training")
        params, static = awb_ops.partition_for_standard_training(model)
        from ..runners.generic_runner import create_optimizer
        optim = create_optimizer(config)
        opt_state = optim.init(params)

        total_epochs = ab_warmup_epochs + epochs_per_task
        params, static, opt_state, record_dict = trainer.train__CL(
            train__=(trainloader, exploader, valloader, testloader),
            params=params, static=static, opt_state=opt_state, optim=optim,
            n_iter=total_epochs, save_iter=save_iter,
            task_id=task_id, config=config, record_dict=record_dict,
            problem_type=problem_type, loss_type=loss_type,
            phase='main', record_training=True,
            global_iteration_offset=task_id * epochs_per_task
        )
        model = eqx.combine(params, static)

    final_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=task_id,
                                   epochs=epochs_per_task, window=averaging_window)
    print(f"\n[AWB COMPLETE] Task {task_id} final loss: {final_loss:.6f}")
    print(f"{'='*70}\n")

    # Added by Claude: Store task metadata for architecture history tracking
    if 'tasks' not in record_dict:
        record_dict['tasks'] = {}
    if task_id not in record_dict['tasks']:
        record_dict['tasks'][task_id] = {}
    record_dict['tasks'][task_id].update(task_metadata)

    return model, optim, opt_state, record_dict


__all__ = ['run_awb_task']
