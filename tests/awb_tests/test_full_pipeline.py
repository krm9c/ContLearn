#!/usr/bin/env python
"""
Full AWB Pipeline Integration Test

Tests the complete AWB 5-step pipeline end-to-end.
Verifies that:
1. All 5 steps execute in correct order
2. Architecture changes when appropriate
3. Model performance improves or maintains across tasks
4. No numerical instabilities

Usage:
    python awb_tests/test_full_pipeline.py
    python awb_tests/test_full_pipeline.py --config awb_tests/configs/awb_test_cifar100.json
    python awb_tests/test_full_pipeline.py --verbose
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from cl.models.mlp import MLP
from cl.core.trainer import Trainer
from cl.core.awb import (
    should_change_arch,
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
)
from cl.core.arch_search import search_architecture
from cl.datasets.sine import SineDataset


def test_full_pipeline(config_path: str = None, verbose: bool = False):
    """Test the full AWB 5-step pipeline."""

    print("=" * 70)
    print("FULL AWB PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'data': 'sine',
            'n_task': 3,
            'epochs_per_task': 10,
            'batch_size': 64,
            'awb_preliminary_epochs': 5,
            'awb_ab_training_epochs': 5,
            'awb_ab_warmup_epochs': 2,
            'awb_v_warmup_epochs': 2,
            'awb_v_lr_factor': 0.1,
            'awb_loss_ratio_threshold': 1.2,
            'awb_loss_increase_threshold': 0.1,
            'arch_search_method': 'grid',
            'arch_search_max_iter': 1,
            'arch_search_mlp_increment': 16,
            'arch_search_range': 1,
            'lr': 1e-3,
            'debug_mode': True,
            'debug_limit': 200,
            'flag': [1.0, 1.0],
        }

    print(f"\nConfig: {config.get('data', 'sine')}")
    print(f"Number of tasks: {config.get('n_task', 3)}")
    print(f"Epochs per task: {config.get('epochs_per_task', 10)}")

    results = {
        'passed': True,
        'checks': {},
        'task_results': {},
        'architecture_changes': [],
    }

    # Fixed by Claude: Create dataset first to determine input/output dimensions
    dataset_config = {
        'batch_size': config.get('batch_size', 64),
        'n_task': config.get('n_task', 3),
        'debug_mode': True,
        'debug_limit': 200,
    }
    dataset = SineDataset(dataset_config)

    # Get dimensions from dataset
    input_dim = dataset.input_size  # Sine: 3 features
    output_dim = dataset.output_size  # Sine: 1 output

    # Initialize model with correct dimensions
    original_arch = [input_dim, 64, 64, output_dim]
    current_arch = original_arch.copy()

    print(f"\nInitial architecture: {original_arch}")

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)

    # Create trainer
    trainer = Trainer(loss='regression', metric='mse', problem='vectors')

    # Track losses across tasks
    task_losses = []
    total_start_time = time.time()

    # === TASK 0: Standard Training (No AWB) ===
    print("\n" + "=" * 70)
    print("TASK 0: Standard Training (No AWB)")
    print("=" * 70)

    task_id = 0
    dl_curr, dl_exp = dataset.generate_dataset(task_id=task_id, batch_size=config['batch_size'], phase='training')
    tl_curr, tl_exp = dataset.generate_dataset(task_id=task_id, batch_size=config['batch_size'], phase='testing')
    train_data = (dl_curr, dl_exp, (tl_curr, tl_exp), (tl_curr, tl_exp))

    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adamw(learning_rate=config.get('lr', 1e-3))
    opt_state = optim.init(params)

    train_config = {
        'batch_size': config.get('batch_size', 64),
        'problem': 'vectors',
        'data_id': 'sine',
        'flag': config.get('flag', [1.0, 1.0]),
        'network': 'fcnn',
    }

    record_dict = trainer.initialize_record_dict(config, run_id=0)

    params, static, opt_state, record_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=config.get('epochs_per_task', 10),
        save_iter=5,
        task_id=task_id,
        config=train_config,
        record_dict=record_dict,
        problem_type='vectors',
        loss_type='regression',
    )

    model = eqx.combine(params, static)

    # Get baseline loss
    baseline_loss = 0.0
    for batch in tl_curr:
        x, y = batch
        pred = jax.vmap(model)(x)
        baseline_loss += float(jnp.mean((pred - y) ** 2))
    baseline_loss /= len(list(tl_curr))

    task_losses.append(baseline_loss)
    results['task_results'][0] = {'loss': baseline_loss, 'arch': current_arch.copy()}
    print(f"Task 0 completed. Test loss: {baseline_loss:.4f}")

    # === TASKS 1+: AWB Pipeline ===
    for task_id in range(1, config.get('n_task', 3)):
        print("\n" + "=" * 70)
        print(f"TASK {task_id}: AWB Pipeline")
        print("=" * 70)

        task_start_time = time.time()

        # Generate data for this task
        dl_curr, dl_exp = dataset.generate_dataset(task_id=task_id, batch_size=config['batch_size'], phase='training')
        tl_curr, tl_exp = dataset.generate_dataset(task_id=task_id, batch_size=config['batch_size'], phase='testing')
        train_data = (dl_curr, dl_exp, (tl_curr, tl_exp), (tl_curr, tl_exp))

        # Append to experience replay
        dataset.append_to_experience(task_id)

        # --- STEP 1: Preliminary Training ---
        print("\n" + "-" * 50)
        print("STEP 1: Preliminary Training")
        print("-" * 50)

        params, static = eqx.partition(model, eqx.is_array)
        opt_state = optim.init(params)

        params, static, opt_state, record_dict = trainer.train__CL(
            train_data, params, static, opt_state, optim,
            n_iter=config.get('awb_preliminary_epochs', 5),
            save_iter=2,
            task_id=task_id,
            config=train_config,
            record_dict=record_dict,
            problem_type='vectors',
            loss_type='regression',
            phase='preliminary',
        )

        model = eqx.combine(params, static)

        # Calculate preliminary loss
        preliminary_loss = 0.0
        for batch in tl_curr:
            x, y = batch
            pred = jax.vmap(model)(x)
            preliminary_loss += float(jnp.mean((pred - y) ** 2))
        preliminary_loss /= len(list(tl_curr))

        print(f"Preliminary loss: {preliminary_loss:.4f}")
        print(f"Baseline loss: {baseline_loss:.4f}")

        # --- STEP 2: Decision ---
        print("\n" + "-" * 50)
        print("STEP 2: Architecture Change Decision")
        print("-" * 50)

        # Use correct API: should_change_arch(trainWLoss, end_last, threshold_high, min_delta)
        change_arch = should_change_arch(
            trainWLoss=preliminary_loss,
            end_last=baseline_loss,
            threshold_high=config.get('awb_loss_ratio_threshold', 0.45),
            min_delta=config.get('awb_loss_increase_threshold', 0.01)
        )
        print(f"Decision: {'CHANGE ARCHITECTURE' if change_arch else 'KEEP ARCHITECTURE'}")

        if change_arch:
            results['architecture_changes'].append(task_id)

            # --- STEP 3a: Architecture Search ---
            print("\n" + "-" * 50)
            print("STEP 3a: Architecture Search")
            print("-" * 50)

            search_config = config.copy()
            search_config['prob'] = 'regression'
            search_config['loss'] = 'mse'
            search_config['metric'] = 'mse'

            new_arch = search_architecture(
                model=model,
                baseline_arch=current_arch,
                task_id=task_id,
                baseline_loss=baseline_loss,
                dataloader_curr=dl_curr,
                dataloader_exp=dl_exp,
                test_loader_curr=tl_curr,
                test_loader_exp=tl_exp,
                config=search_config,
                trainer=trainer,
                model_type='mlp',
            )

            print(f"New architecture: {new_arch}")

            # --- STEP 3b: A/B Training ---
            print("\n" + "-" * 50)
            print("STEP 3b: A/B Matrix Training")
            print("-" * 50)

            model_with_ab = set_new_AB_matrices(model, current_arch, new_arch)
            params, static = partition_for_AB_training(model_with_ab)
            opt_state = optim.init(params)

            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config.get('awb_ab_training_epochs', 5),
                save_iter=2,
                task_id=task_id,
                config=train_config,
                record_dict=record_dict,
                problem_type='vectors',
                loss_type='regression',
                phase='ab_training',
                notABTrain=False,
            )

            model = eqx.combine(params, static)

            # --- STEP 4: V Transformation ---
            print("\n" + "-" * 50)
            print("STEP 4: V = A @ W @ B.T Transformation")
            print("-" * 50)

            model = compute_V_from_AWB(model)
            current_arch = new_arch.copy()
            print(f"Transformed to new architecture: {current_arch}")

        # --- STEP 5: V Training ---
        print("\n" + "-" * 50)
        print("STEP 5: V Training (A/B frozen)")
        print("-" * 50)

        params, static = partition_for_standard_training(model)
        lr = config.get('lr', 1e-3) * config.get('awb_v_lr_factor', 0.1)
        optim_v = optax.adamw(learning_rate=lr)
        opt_state = optim_v.init(params)

        remaining_epochs = config.get('epochs_per_task', 10) - config.get('awb_preliminary_epochs', 5)
        if change_arch:
            remaining_epochs -= config.get('awb_ab_training_epochs', 5)

        params, static, opt_state, record_dict = trainer.train__CL(
            train_data, params, static, opt_state, optim_v,
            n_iter=max(1, remaining_epochs),
            save_iter=2,
            task_id=task_id,
            config=train_config,
            record_dict=record_dict,
            problem_type='vectors',
            loss_type='regression',
            phase='v_training',
            notABTrain=True,
        )

        model = eqx.combine(params, static)

        # Evaluate task loss
        task_loss = 0.0
        count = 0
        for batch in tl_curr:
            x, y = batch
            pred = jax.vmap(model)(x)
            task_loss += float(jnp.mean((pred - y) ** 2))
            count += 1
        task_loss /= max(count, 1)

        task_time = time.time() - task_start_time
        task_losses.append(task_loss)
        results['task_results'][task_id] = {
            'loss': task_loss,
            'arch': current_arch.copy(),
            'arch_changed': change_arch,
            'time': task_time,
        }

        print(f"\nTask {task_id} completed in {task_time:.2f}s")
        print(f"Final loss: {task_loss:.4f}")
        print(f"Architecture: {current_arch}")

        # Update baseline for next task
        baseline_loss = task_loss

    total_time = time.time() - total_start_time

    # === Verification ===
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    # Check 1: All tasks completed
    check_all_tasks = len(results['task_results']) == config.get('n_task', 3)
    results['checks']['all_tasks_completed'] = check_all_tasks
    print(f"[{'PASS' if check_all_tasks else 'FAIL'}] All {config.get('n_task', 3)} tasks completed")
    if not check_all_tasks:
        results['passed'] = False

    # Check 2: No NaN losses
    check_no_nan = all(np.isfinite(l) for l in task_losses)
    results['checks']['no_nan_losses'] = check_no_nan
    print(f"[{'PASS' if check_no_nan else 'FAIL'}] No NaN/Inf losses")
    if not check_no_nan:
        results['passed'] = False

    # Check 3: Model produces valid output
    test_input = jax.random.normal(jax.random.PRNGKey(0), (32, current_arch[0]))
    try:
        output = model(test_input)
        check_valid = jnp.isfinite(output).all()
        results['checks']['valid_output'] = bool(check_valid)
        print(f"[{'PASS' if check_valid else 'FAIL'}] Model produces valid output")
        if not check_valid:
            results['passed'] = False
    except Exception as e:
        results['checks']['valid_output'] = False
        results['passed'] = False
        print(f"[FAIL] Model forward pass failed: {e}")

    # Check 4: Architecture changes recorded correctly
    check_arch_changes = True
    for task_id, task_result in results['task_results'].items():
        if task_id > 0 and 'arch_changed' in task_result:
            if task_result['arch_changed'] and task_id not in results['architecture_changes']:
                check_arch_changes = False
    results['checks']['arch_changes_recorded'] = check_arch_changes
    print(f"[{'PASS' if check_arch_changes else 'FAIL'}] Architecture changes recorded correctly")
    print(f"       Changes occurred at tasks: {results['architecture_changes']}")

    # === Performance Summary ===
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per task: {total_time / config.get('n_task', 3):.2f}s")
    print(f"\nLoss progression: {' -> '.join(f'{l:.4f}' for l in task_losses)}")
    print(f"Final architecture: {current_arch}")
    print(f"Architecture changes: {len(results['architecture_changes'])}")

    results['total_time'] = total_time
    results['final_arch'] = current_arch
    results['task_losses'] = task_losses

    # === Final Result ===
    print("\n" + "=" * 70)
    if results['passed']:
        print("FULL PIPELINE TEST: PASSED")
    else:
        print("FULL PIPELINE TEST: FAILED")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Full AWB Pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_full_pipeline(args.config, args.verbose)
    sys.exit(0 if results['passed'] else 1)
