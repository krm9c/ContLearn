"""
Training runners for different problem types.

Contains the main training loop functions for:
- Graph classification
- Regression (with optional AWB pipeline)
- Classification
"""

import numpy as np
import optax
import equinox as eqx

from .checkpoint import load_checkpoint
from .awb_utils import (
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
from contlearn.data.loaders import continuum_Graph_classification
from contlearn.arch_search.mlp_search import arch_search_MLP
from contlearn.config.constants import (
    DEFAULT_BATCH_SIZE_VECTOR,
    DEFAULT_REPLAY_BUFFER_GRAPH,
    DEFAULT_REPLAY_BUFFER_VECTOR,
    DEFAULT_AWB_ENABLED,
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
)


def train_model_graph(config, run_id=0):
    """Train model for graph classification task"""
    trainer, optim, data, test, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)
    memory_train = []

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

    for i in range(config['n_task']):
        train_loader, mem_train_loader, memory_train = continuum_Graph_classification(
            data, memory_train,
            n_class=config['n_class'],
            select=config['class_per_task']
        )

        # Unified format: (train_current, train_exp, (test_current, test_exp), (test_current, test_exp))
        # For graphs, test is the same for both current and experience
        train_data = (train_loader, mem_train_loader, (test, test), (test, test))

        if i == 0:
            og_epochs = config['epochs_per_task']
            params, static, optim, record_dict = trainer.train__CL__graph(
                train_data, params, static, optim,
                n_iter=og_epochs, save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                record_dict=record_dict
            )
        else:
            # Architecture search and adaptive training (simplified for merged version)
            params, static, optim, record_dict = trainer.train__CL__graph(
                train_data, params, static, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                record_dict=record_dict
            )

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

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)

    # Save unified recording dictionary
    trainer.save_record_dict(record_dict, config['model_path'])

    del model, params, static

    return record_dict


def train_model_reg(config, run_id=0):
    """
    Train model for regression task with optional AWB (Adaptive Weight Basis) pipeline.

    When AWB is enabled (config['awb_enabled'] = True), uses the 5-step algorithm:
        Task 0: Standard CL training
        Tasks 1+:
            STEP 1: Train for preliminary epochs on new task
            STEP 2: Decide if architecture change needed (loss ratio thresholds)
            If change_arch == True:
                STEP 3a: Search for new architecture, set new A/B matrices
                STEP 3b: Train A/B with W frozen (notABTrain=False)
                STEP 4: Set new weights V = A @ W @ B.T
                STEP 5: Train V with A/B frozen (notABTrain=True)
            Else:
                Continue normal training

    When AWB is disabled (default), uses standard CL training for all tasks.
    """
    trainer, optim, data, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Move A, B to static (frozen) for standard training
    static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
    params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

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
        'batch_size': config.get('vector_batch_size', DEFAULT_BATCH_SIZE_VECTOR),
        'opt': 'Nash',
        'problem': config['problem'],
        'data_id': config['data'],
        'flag': config['flag'],
        'len_exp_replay': config.get('vector_replay_size', DEFAULT_REPLAY_BUFFER_VECTOR),
        'network': config['network']
    }

    # Track baseline losses for AWB decision logic
    end_last0 = None  # Loss at end of task 0
    end_last = None   # Loss at end of previous task
    mlp_arch_list = []  # Track architecture changes

    for i in range(config['n_task']):
        print(f"task-- {i}")

        # Generate data for current task
        dataloader_curr, dataloader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training'
        )
        test_loader_curr, test_loader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing'
        )

        if i == 0:
            # ==================== TASK 0: Standard Training ====================
            params, static, optim, record_dict = trainer.train__CL__reg(
                (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp), (test_loader_curr, test_loader_exp)),
                params, static, optim, n_iter=config['epochs_per_task'],
                save_iter=config['save_iter'], task_id=i,
                config=train_config, record_dict=record_dict
            )

            # Compute baseline loss for AWB decision logic
            end_last0 = compute_avg_loss(record_dict['iterations'], i,
                                         config['epochs_per_task'], averaging_window)
            end_last = end_last0
            print(f"Task 0 baseline loss: {end_last0}")

        elif awb_enabled:
            # ==================== AWB PIPELINE FOR TASKS 1+ ====================

            # STEP 1: Preliminary training on new task
            print(f"STEP 1: Training for {preliminary_epochs} preliminary epochs on task {i}")
            params, static, optim, record_dict = trainer.train__CL__reg(
                (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                 (test_loader_curr, test_loader_exp)),
                params, static, optim, n_iter=preliminary_epochs,
                save_iter=config['save_iter'], task_id=i,
                config=train_config, record_dict=record_dict
            )

            # Compute loss after preliminary training
            model = eqx.combine(params, static)
            trainWLoss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                          preliminary_epochs, averaging_window)

            # STEP 2: Decide if architecture change is needed
            print(f"STEP 2: Checking if architecture change needed for task {i}")
            print(f"  trainWLoss: {trainWLoss}, end_last0: {end_last0}, end_last: {end_last}")
            print(f"  ratio: {trainWLoss/end_last0 if end_last0 else 'N/A'}")

            change_arch = should_change_arch(trainWLoss, end_last0, end_last)

            # Save current weights in case we need to restore
            original_arch = model.sizes
            mlp_weight_layer, mlp_bias_layer = save_layer_weights(model)

            if change_arch:
                print("WE ARE CHANGING ARCHITECTURE!")

                # STEP 3a: Architecture search
                print(f"STEP 3a: Searching for optimal architecture for task {i}")
                opt_arch = arch_search_MLP(
                    original_arch, i, trainWLoss, preliminary_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp
                )
                print(f"NEW Architecture: {opt_arch}")
                mlp_arch_list.append(opt_arch)

                # Restore original weights before setting new A/B
                model = restore_layer_weights(model, mlp_weight_layer, mlp_bias_layer)

                if opt_arch != original_arch:
                    # Set new A/B matrices for the new architecture
                    model = set_new_AB_matrices(model, original_arch, opt_arch)

                    # STEP 3b: Train A/B with W frozen
                    print(f"STEP 3b: Training A/B matrices with W frozen for {ab_training_epochs} epochs")
                    diff_model, static_model = partition_for_AB_training(model)
                    optim2 = optax.adam(1e-4)

                    # Compute threshold for AB training convergence
                    ab_threshold = compute_ab_threshold(trainWLoss, end_last)
                    print(f"  AB threshold: {ab_threshold}, Goal: {trainWLoss * ab_threshold}")

                    # Initial AB training
                    diff_model, static_model, optim2, record_dict = trainer.train__CL__reg(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)),
                        diff_model, static_model, optim2, n_iter=ab_training_epochs,
                        save_iter=config['save_iter'], task_id=i,
                        config=train_config, record_dict=record_dict, notABTrain=False
                    )

                    AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                               ab_training_epochs, averaging_window)
                    print(f"AB training iteration 1: trainWLoss={trainWLoss}, AB_loss={AB_loss}")

                    # Continue AB training until convergence or max iterations
                    ab_iter = 1
                    while (trainWLoss * ab_threshold < AB_loss) and (ab_iter < ab_max_iterations):
                        diff_model, static_model, optim2, record_dict = trainer.train__CL__reg(
                            (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                             (test_loader_curr, test_loader_exp)),
                            diff_model, static_model, optim2, n_iter=ab_training_epochs,
                            save_iter=config['save_iter'], task_id=i,
                            config=train_config, record_dict=record_dict, notABTrain=False
                        )
                        AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                                   ab_training_epochs, averaging_window)
                        ab_iter += 1
                        print(f"AB training iteration {ab_iter}: AB_loss={AB_loss}")

                    # Combine after AB training
                    model = eqx.combine(diff_model, static_model)

                    # STEP 4: Set new weights V = A @ W @ B.T
                    print("STEP 4: Computing V = A @ W @ B.T")
                    model = compute_V_from_AWB(model)

                    # Re-partition for V training
                    params, static = partition_for_standard_training(model)

                    # STEP 5: Train V with A/B frozen
                    print(f"STEP 5: Training model with new weights V")
                    optim3 = optax.adam(1e-3)

                    # Warmup training
                    params, static, optim3, record_dict = trainer.train__CL__reg(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)),
                        params, static, optim3, n_iter=ab_warmup_epochs,
                        save_iter=config['save_iter'], task_id=i,
                        config=train_config, record_dict=record_dict
                    )

                    # Full training
                    params, static, optim3, record_dict = trainer.train__CL__reg(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)),
                        params, static, optim3, n_iter=config['epochs_per_task'],
                        save_iter=config['save_iter'], task_id=i,
                        config=train_config, record_dict=record_dict
                    )

                    # Update optimizer for next task
                    optim = optim3

                else:
                    # Architecture didn't change, continue with normal training
                    print("Architecture search found same architecture, continuing normal training")
                    mlp_arch_list.append(original_arch)
                    _train_standard_phase(trainer, params, static, optim, i, config,
                                          dataloader_curr, dataloader_exp,
                                          test_loader_curr, test_loader_exp,
                                          train_config, record_dict)
            else:
                # No architecture change needed
                print("ARCHITECTURE Did NOT change")
                mlp_arch_list.append(original_arch)

                # Continue with standard training for remaining epochs
                params, static, optim, record_dict = trainer.train__CL__reg(
                    (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                     (test_loader_curr, test_loader_exp)),
                    params, static, optim, n_iter=ab_warmup_epochs,
                    save_iter=config['save_iter'], task_id=i,
                    config=train_config, record_dict=record_dict
                )

                params, static, optim, record_dict = trainer.train__CL__reg(
                    (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                     (test_loader_curr, test_loader_exp)),
                    params, static, optim, n_iter=config['epochs_per_task'],
                    save_iter=config['save_iter'], task_id=i,
                    config=train_config, record_dict=record_dict
                )

            # Update end_last for next task
            end_last = compute_avg_loss(record_dict.get('iterations', {}), i,
                                        config['epochs_per_task'], averaging_window)

            # Re-partition model for next iteration
            model = eqx.combine(params, static)
            params, static = eqx.partition(model, eqx.is_array)
            static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
            params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

        else:
            # ==================== AWB DISABLED: Standard Training ====================
            params, static, optim, record_dict = trainer.train__CL__reg(
                (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                 (test_loader_curr, test_loader_exp)),
                params, static, optim, n_iter=config['epochs_per_task'],
                save_iter=config['save_iter'], task_id=i,
                config=train_config, record_dict=record_dict
            )

        # Append to experience replay
        data.append_to_experience(i)

    # Print architecture history if AWB was enabled
    if awb_enabled and mlp_arch_list:
        print("\nArchitecture history:")
        for task_idx, arch in enumerate(mlp_arch_list):
            print(f"  Task {task_idx + 1}: {arch}")

    # Save final model
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)

    # Save unified recording dictionary
    trainer.save_record_dict(record_dict, config['model_path'])

    del model, params, static

    return record_dict


def _train_standard_phase(trainer, params, static, optim, task_id, config,
                          dataloader_curr, dataloader_exp,
                          test_loader_curr, test_loader_exp,
                          train_config, record_dict):
    """Helper for standard training phase when arch doesn't change."""
    params, static, optim, record_dict = trainer.train__CL__reg(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)),
        params, static, optim, n_iter=config['epochs_per_task'],
        save_iter=config['save_iter'], task_id=task_id,
        config=train_config, record_dict=record_dict
    )
    return params, static, optim


def train_model_class(config, run_id=0):
    """Train model for classification task"""
    trainer, optim, data, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    static = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        static,
        replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed)
    )
    params = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        params,
        replace=(None, None, None, None)
    )

    for i in range(config['n_task']):
        print("task--", i)

        dataloader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training'
        )
        test_loader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing'
        )

        data.append_to_experience(i)

        params, static, optim, record_dict = trainer.train__CL__class(
            (dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),
             (test_loader_curr, test_loader_curr)),
            params, static, optim, n_iter=config['epochs_per_task'],
            save_iter=config['save_iter'], task_id=i,
            config={
                'batch_size': config['batch_size'],
                'opt': 'Nash',
                'problem': config['prob'],
                'data_id': config['data'],
                'len_exp_replay': config.get('class_replay_size', DEFAULT_REPLAY_BUFFER_GRAPH),
                'flag': config['flag'],
                'network': config['network']
            },
            record_dict=record_dict
        )

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'], model)

    # Save unified recording dictionary
    trainer.save_record_dict(record_dict, config['model_path'])

    del model, params, static

    return record_dict
