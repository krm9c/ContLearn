"""
Training runner for graph classification tasks.

Orchestrates the training loop for graph classification problems (e.g., MUTAG, synthetic)
with optional AWB (Adaptive Weight Basis) pipeline for architecture morphing.

Adapted from train_model_graph in run_AWB_ALL_functions.py for cl_framework.

The AWB 5-step algorithm for graphs:
    Task 0: Standard CL training
    Tasks 1+:
        STEP 1: Preliminary training on new task
        STEP 2: Decide if architecture change needed
        If change needed:
            STEP 3a: Architecture search (GCN + MLP sizes)
            STEP 3b: Train A/B with W frozen
            STEP 4: Compute V = A @ W @ B.T
            STEP 5: Train V with A/B frozen
        Else:
            Continue standard training
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Dict, Any

from ..core.trainer import Trainer
from ..core.awb import (
    compute_avg_loss,
    should_change_arch,
    compute_ab_threshold,
    partition_for_AB_training_gnn,
    partition_for_standard_training_gnn,
    compute_V_from_AWB_gcn,
    save_gcn_layer_weights,
    restore_gcn_layer_weights,
)
# Added by Claude: Import create_optimizer for consistent optimizer configuration
from .generic_runner import create_optimizer
from ..models.gcn import GCN
from ..datasets.synthetic_graph import load_graph_dataset
from ..arch_search.gcn_search import arch_search_GCN, prepABs_GCN
from ..config.constants import (
    DEFAULT_BATCH_SIZE_GRAPH,
    DEFAULT_REPLAY_BUFFER_GRAPH,
    DEFAULT_AWB_ENABLED,
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
    DEFAULT_GCN_SIZES,
    DEFAULT_GCN_MLP_SIZES,
    DEFAULT_NUM_CLASSES,
)


def create_graph_optimizer(config: Dict[str, Any]) -> optax.GradientTransformationExtraArgs:
    """Create optimizer from configuration for graph classification tasks.

    Args:
        config: Configuration dictionary with:
            - optimizer: Optimizer type (default: 'adamw')
            - lr: Learning rate (default: 1e-4)

    Returns:
        Configured optimizer
    """
    lr = config.get('lr', 1e-4)
    optimizer_name = config.get('optimizer', 'adamw').lower()

    if optimizer_name == 'adam':
        return optax.adam(lr)
    elif optimizer_name == 'adamw':
        weight_decay = config.get('weight_decay', 1e-4)
        return optax.adamw(lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optax.sgd(lr, momentum=momentum)
    else:
        return optax.adamw(lr)


def load_graph_checkpoint(config: Dict[str, Any]):
    """Load or create model, trainer, optimizer, and dataset for graph classification.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (trainer, optimizer, dataset, model)
    """
    # Create dataset based on data type
    dataset_config = {
        'data': config.get('data', 'synthetic'),
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_GRAPH),
        'n_class': config.get('n_class', DEFAULT_NUM_CLASSES),
        'class_per_task': config.get('class_per_task', 2),
        'debug_mode': config.get('debug_mode', False),
        'debug_limit': config.get('debug_limit', 100),
        'num_graphs': config.get('num_graphs', 1000),
        'num_channels': config.get('num_channels', 5),
        'avg_num_nodes': config.get('avg_num_nodes', 2),
        'num_classes': config.get('n_class', DEFAULT_NUM_CLASSES),
    }

    dataset = load_graph_dataset(dataset_config)

    # Get model configuration
    num_classes = config.get('n_class', DEFAULT_NUM_CLASSES)
    gcn_sizes = config.get('gcn_sizes', DEFAULT_GCN_SIZES.copy())
    feed_sizes = config.get('feed_sizes', DEFAULT_GCN_MLP_SIZES.copy())

    # Set input size from dataset
    gcn_sizes[0] = dataset.num_features

    # Ensure output size matches
    feed_sizes[-1] = num_classes

    # Ensure first feed layer input matches last GCN output
    feed_sizes[0] = gcn_sizes[-1]

    # Get sample batch to get node count
    sample_batch = next(iter(dataset.get_test_loader()))
    node_num = sample_batch.x.shape[0]

    # Create GCN model
    model = GCN(
        in_size=dataset.num_features,
        feed_sizes=feed_sizes,
        gcn_sizes=gcn_sizes,
        node_num=node_num,
        SEED=config.get('seed', 1234),
        out_size=num_classes,
        graph=True,
    )

    # Create trainer for graph classification
    trainer = Trainer(
        loss=config.get('loss', 'class'),
        metric=config.get('metric', 'class'),
        problem=config.get('problem', 'graph'),
    )

    # Create optimizer
    optimizer = create_graph_optimizer(config)

    return trainer, optimizer, dataset, model


def set_new_AB_matrices_gcn(model, prev_gcn_sizes, prev_feed_sizes, opt_gcn, opt_mlp):
    """Set new A/B matrices for GCN architecture transition.

    Args:
        model: GCN model
        prev_gcn_sizes: Previous GCN layer sizes
        prev_feed_sizes: Previous feed layer sizes
        opt_gcn: Optimal GCN layer sizes
        opt_mlp: Optimal MLP/feed layer sizes

    Returns:
        Model with updated A/B matrices and architecture
    """
    # Update architecture
    model = eqx.tree_at(lambda x: x.gcn_sizes, model, opt_gcn)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_mlp)

    # Use prepABs_GCN to get the transformation matrices
    A_feed, B_feed, A_gcn, B_gcn = prepABs_GCN(model, prev_feed_sizes, prev_gcn_sizes)

    model = eqx.tree_at(
        lambda x: (x.A_feed, x.B_feed, x.A_gcn, x.B_gcn),
        model,
        replace=(A_feed, B_feed, A_gcn, B_gcn)
    )

    return model


def train_model_graph(config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
    """Train model for graph classification task using unified training loop.

    When AWB is enabled (config['awb_enabled'] = True), uses the 5-step algorithm.
    When AWB is disabled (default), uses standard CL training for all tasks.

    Based on train_model_graph from run_AWB_ALL_functions.py.

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
    trainer, optim, data, model = load_graph_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Move AWB matrices to static (frozen) - GCN uses A_gcn, B_gcn, A_feed, B_feed
    static = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        static,
        replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed)
    )
    params = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        params,
        replace=(None, None, None, None)
    )

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

    train_config = {
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_GRAPH),
        'flag': config.get('flag', [1.0, 1.0]),
    }

    # Track baseline losses for AWB decision logic
    # Added by Claude: Removed end_last0, now just track end_last (previous task loss)
    end_last = None
    gcn_arch_history = []
    mlp_arch_history = []

    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE_GRAPH)

    for i in range(config['n_task']):
        print(f"\n{'='*50}")
        print(f"Task {i}")
        print(f"{'='*50}")

        # Generate dataloaders
        train_loader, mem_train_loader = data.generate_dataset(
            task_id=i, batch_size=batch_size, phase='training'
        )
        test_loader = data.get_test_loader(batch_size)

        # Prepare data tuple for trainer
        train_data = (train_loader, mem_train_loader, (test_loader, test_loader), (test_loader, test_loader))

        if i == 0:
            # Task 0: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='graph', loss_type='classification'
            )
            # Added by Claude: Set end_last for task 0 (used as baseline for task 1 comparison)
            end_last = compute_avg_loss(record_dict.get('iterations', {}), i,
                                         config['epochs_per_task'], averaging_window)
            print(f"Task 0 baseline loss: {end_last:.6f}")

        elif awb_enabled:
            # AWB PIPELINE FOR TASKS 1+
            print(f"AWB Pipeline enabled for task {i}")

            # STEP 1: Preliminary training
            print(f"STEP 1: Preliminary training ({preliminary_epochs} epochs)")
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=preliminary_epochs, save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='graph', loss_type='classification'
            )

            model = eqx.combine(params, static)
            trainWLoss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                          preliminary_epochs, averaging_window)

            # STEP 2: Decide if architecture change is needed
            # Added by Claude: Compare current preliminary loss to previous task's final loss
            print(f"STEP 2: Checking architecture change (loss={trainWLoss:.6f}, prev={end_last:.6f})")
            change_arch = should_change_arch(trainWLoss, end_last)

            prev_gcn_sizes = list(model.gcn_sizes)
            prev_feed_sizes = list(model.feed_sizes)
            gcn_weights, gcn_biases, mlp_weights, mlp_biases = save_gcn_layer_weights(model)

            if True:
                print("ARCHITECTURE CHANGE TRIGGERED!")

                # STEP 3a: Architecture search for GCN and MLP
                opt_gcn, opt_mlp = arch_search_GCN(
                    original_gcn=prev_gcn_sizes,
                    original_mlp=prev_feed_sizes,
                    task=i,
                    trainW_loss=trainWLoss,
                    og_epochs=preliminary_epochs,
                    config=config,
                    train_loader=train_loader,
                    mem_train_loader=mem_train_loader,
                    test_loader=test_loader,
                    trainer=trainer,
                    model=model
                )
                print(f"Optimal GCN Architecture: {opt_gcn}")
                print(f"Optimal MLP Architecture: {opt_mlp}")
                gcn_arch_history.append(opt_gcn)
                mlp_arch_history.append(opt_mlp)

                # Restore weights after search (search uses fresh models)
                model = restore_gcn_layer_weights(model, gcn_weights, gcn_biases, mlp_weights, mlp_biases)

                if opt_mlp != prev_feed_sizes or opt_gcn != prev_gcn_sizes:
                    # Set new A/B matrices for the transition
                    model = set_new_AB_matrices_gcn(model, prev_gcn_sizes, prev_feed_sizes, opt_gcn, opt_mlp)

                    # STEP 3b: Train A/B with W frozen
                    print(f"STEP 3b: Training A/B matrices with W frozen")
                    diff_model, static_model = partition_for_AB_training_gnn(model)
                    optim2 = optax.adamw(1e-4)
                    opt_state2 = optim2.init(diff_model)
                    ab_threshold = compute_ab_threshold(trainWLoss, end_last)

                    diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                        train_data, diff_model, static_model, opt_state2, optim2,
                        n_iter=ab_training_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        notABTrain=False, problem_type='graph', loss_type='classification'
                    )

                    AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                               ab_training_epochs, averaging_window)
                    ab_iter = 1
                    ab_conv_list = [AB_loss]

                    # Iteratively train A/B until convergence or max iterations
                    while (trainWLoss * ab_threshold < AB_loss) and (ab_iter < ab_max_iterations):
                        diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                            train_data, diff_model, static_model, opt_state2, optim2,
                            n_iter=ab_training_epochs, save_iter=config['save_iter'],
                            task_id=i, config=train_config, record_dict=record_dict,
                            notABTrain=False, problem_type='graph', loss_type='classification'
                        )
                        AB_loss_new = compute_avg_loss(record_dict.get('iterations', {}), i,
                                                       ab_training_epochs, averaging_window)
                        print(f"AB ROUND {ab_iter}: trainWLoss={trainWLoss:.4f}, AB_loss={AB_loss_new:.4f}")

                        ab_conv_list.append(AB_loss_new)

                        # Check if loss is increasing (early stopping)
                        if AB_loss_new > AB_loss:
                            if len(ab_conv_list) >= 2 and ab_conv_list[-2] < AB_loss_new:
                                print("AB training loss increasing, stopping early")
                                break

                        # Check convergence
                        if ab_iter > 7 and len(ab_conv_list) >= 3:
                            conv_diff = np.mean(ab_conv_list[-3:])
                            if np.abs(conv_diff - AB_loss_new) <= 0.01:
                                print("AB_loss converged, stopping")
                                break

                        AB_loss = AB_loss_new
                        ab_iter += 1

                    model = eqx.combine(diff_model, static_model)

                    # STEP 4: Compute V = A @ W @ B.T
                    print("STEP 4: Computing V = A @ W @ B.T")
                    model = compute_V_from_AWB_gcn(model)
                    params, static = partition_for_standard_training_gnn(model)

                    # STEP 5: Train V with A/B frozen
                    print(f"STEP 5: Training with new weights V")
                    # Added by Claude: Use create_optimizer for consistent optimizer configuration
                    optim = create_optimizer(config)
                    opt_state = optim.init(params)

                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='graph', loss_type='classification'
                    )
                else:
                    print("Architecture search found same architecture, continuing normal training")
                    gcn_arch_history.append(prev_gcn_sizes)
                    mlp_arch_history.append(prev_feed_sizes)
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='graph', loss_type='classification'
                    )
            else:
                print("Architecture did NOT change - continuing standard training")
                gcn_arch_history.append(prev_gcn_sizes)
                mlp_arch_history.append(prev_feed_sizes)
                params, static, opt_state, record_dict = trainer.train__CL(
                    train_data, params, static, opt_state, optim,
                    n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                    task_id=i, config=train_config, record_dict=record_dict,
                    problem_type='graph', loss_type='classification'
                )

            end_last = compute_avg_loss(record_dict.get('iterations', {}), i,
                                        config['epochs_per_task'], averaging_window)

            # Re-partition model for next iteration
            model = eqx.combine(params, static)
            params, static = eqx.partition(model, eqx.is_array)
            static = eqx.tree_at(
                lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
                static,
                replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed)
            )
            params = eqx.tree_at(
                lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
                params,
                replace=(None, None, None, None)
            )

        else:
            # AWB DISABLED: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='graph', loss_type='classification'
            )

    # Print architecture history if AWB was used
    if awb_enabled and gcn_arch_history:
        print("\nArchitecture history:")
        for task_idx, (gcn, mlp) in enumerate(zip(gcn_arch_history, mlp_arch_history)):
            print(f"  Task {task_idx + 1}: gcn_sizes={gcn}, mlp_sizes={mlp}")

    # Save model and records
    model = eqx.combine(params, static)
    model_path = config.get('model_path', 'outputs/graph_model')

    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    eqx.tree_serialise_leaves(model_path + '.eqx', model)
    trainer.save_record_dict(record_dict, model_path)

    print(f"\nModel saved to: {model_path}.eqx")

    del model, params, static
    return record_dict