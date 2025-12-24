#!/usr/bin/env python
"""
STEP 1: Preliminary Training Test

Tests the preliminary training phase of the AWB pipeline.
Verifies that:
1. Model trains correctly on new task data
2. Loss decreases during training
3. Gradients flow correctly through the network
4. Model outputs are valid (no NaN/Inf)

Usage:
    python awb_tests/test_step1_preliminary.py
    python awb_tests/test_step1_preliminary.py --config awb_tests/configs/awb_test_cifar100.json
"""

import argparse
import json
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from cl.models.cnn import CNN3D
from cl.core.trainer import Trainer
from cl.datasets.cifar import CIFAR100Dataset
from cl.config.constants import DEFAULT_BATCH_SIZE_CLASSIFICATION


def test_step1_preliminary(config_path: str = None, verbose: bool = False):
    """Test STEP 1: Preliminary training on new task."""

    print("=" * 70)
    print("STEP 1: Preliminary Training Test")
    print("=" * 70)

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default CIFAR-100 config
        config = {
            'data': 'cifar100',
            'n_task': 5,
            'epochs_per_task': 50,
            'batch_size': 256,
            'awb_preliminary_epochs': 10,
            'feed_sizes': [2304, 512, 256, 100],
            'filter_size': 3,
            'channel_out': 32,
            'lr': 1e-3,
            'debug_mode': True,
            'debug_limit': 500,
            'n_class': 100,
        }

    print(f"\nConfig: {config.get('data', 'cifar100')}")
    print(f"Preliminary epochs: {config.get('awb_preliminary_epochs', 10)}")
    print(f"Batch size: {config.get('batch_size', 256)}")

    # Create dataset
    dataset_config = {
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_CLASSIFICATION),
        'n_task': config.get('n_task', 5),
        'n_class': config.get('n_class', 100),
        'debug_mode': config.get('debug_mode', True),
        'debug_limit': config.get('debug_limit', 500),
    }
    dataset = CIFAR100Dataset(dataset_config)

    # Create model
    feed_sizes = config.get('feed_sizes', [2304, 512, 256, 100])
    model = CNN3D(
        filter_size=config.get('filter_size', 3),
        channel_out=config.get('channel_out', 32),
        feed_sizes=feed_sizes,
        key=jax.random.PRNGKey(42),
    )

    print(f"Model: CNN3D with feed_sizes={feed_sizes}")

    # Create trainer
    trainer = Trainer(loss='class', metric='class', problem='vectors')

    # Partition model
    params, static = eqx.partition(model, eqx.is_array)

    # Create optimizer
    optim = optax.adamw(learning_rate=config.get('lr', 1e-3))
    opt_state = optim.init(params)

    # Generate data for task 1 (task after initial training)
    task_id = 1
    dataloader_curr, dataloader_exp = dataset.generate_dataset(
        task_id=task_id, batch_size=config['batch_size'], phase='training'
    )
    test_loader_curr, test_loader_exp = dataset.generate_dataset(
        task_id=task_id, batch_size=config['batch_size'], phase='testing'
    )

    train_data = (dataloader_curr, dataloader_exp,
                  (test_loader_curr, test_loader_exp),
                  (test_loader_curr, test_loader_exp))

    train_config = {
        'batch_size': config.get('batch_size', 256),
        'problem': 'vectors',
        'data_id': 'cifar100',
        'flag': config.get('flag', [1.0, 1.0]),
        'network': 'cnn3d',
    }

    # Record initial loss
    record_dict = trainer.initialize_record_dict(config, run_id=0)

    # Time the preliminary training
    print(f"\nRunning preliminary training for {config.get('awb_preliminary_epochs', 10)} epochs...")
    start_time = time.time()

    params, static, opt_state, record_dict = trainer.train__CL(
        train_data, params, static, opt_state, optim,
        n_iter=config.get('awb_preliminary_epochs', 10),
        save_iter=config.get('save_iter', 5),
        task_id=task_id,
        config=train_config,
        record_dict=record_dict,
        problem_type='vectors',
        loss_type='classification',
        phase='preliminary',
    )

    elapsed_time = time.time() - start_time

    # Extract losses
    iterations = record_dict.get('iterations', {})
    losses = []
    for k, v in sorted(iterations.items()):
        if isinstance(v, dict) and 'losses' in v:
            losses.append(v['losses'].get('V', float('inf')))

    # Verification checks
    print("\n" + "=" * 50)
    print("Verification Results")
    print("=" * 50)

    results = {
        'passed': True,
        'checks': {},
        'elapsed_time': elapsed_time,
        'epochs': config.get('awb_preliminary_epochs', 10),
    }

    # Check 1: Loss recorded
    check1 = len(losses) > 0
    results['checks']['loss_recorded'] = check1
    print(f"[{'PASS' if check1 else 'FAIL'}] Loss values recorded: {len(losses)} entries")
    if not check1:
        results['passed'] = False

    # Check 2: No NaN in losses
    check2 = all(np.isfinite(l) for l in losses)
    results['checks']['no_nan_losses'] = check2
    print(f"[{'PASS' if check2 else 'FAIL'}] No NaN/Inf in losses")
    if not check2:
        results['passed'] = False

    # Check 3: Loss decreased (first vs last)
    if len(losses) >= 2:
        check3 = losses[-1] < losses[0]
        results['checks']['loss_decreased'] = check3
        results['initial_loss'] = float(losses[0])
        results['final_loss'] = float(losses[-1])
        print(f"[{'PASS' if check3 else 'FAIL'}] Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
        if not check3:
            results['passed'] = False
    else:
        results['checks']['loss_decreased'] = False
        print("[FAIL] Not enough loss values to check decrease")
        results['passed'] = False

    # Check 4: Model parameters updated
    model_after = eqx.combine(params, static)
    param_leaves_before = jax.tree_util.tree_leaves(model)
    param_leaves_after = jax.tree_util.tree_leaves(model_after)

    params_changed = False
    for p_before, p_after in zip(param_leaves_before, param_leaves_after):
        if isinstance(p_before, jnp.ndarray) and isinstance(p_after, jnp.ndarray):
            if not jnp.allclose(p_before, p_after):
                params_changed = True
                break

    results['checks']['params_updated'] = params_changed
    print(f"[{'PASS' if params_changed else 'FAIL'}] Model parameters updated")
    if not params_changed:
        results['passed'] = False

    # Check 5: Forward pass produces valid output
    # Fixed by Claude: CNN3D expects CHW format (3, 32, 32), not NHWC (1, 32, 32, 3)
    test_input = jnp.ones((3, 32, 32))  # CIFAR input shape in CHW format
    try:
        output = model_after(test_input)
        check5 = jnp.isfinite(output).all()
        results['checks']['valid_output'] = bool(check5)
        print(f"[{'PASS' if check5 else 'FAIL'}] Forward pass produces valid output")
        if not check5:
            results['passed'] = False
    except Exception as e:
        results['checks']['valid_output'] = False
        print(f"[FAIL] Forward pass failed: {e}")
        results['passed'] = False

    # Performance metrics
    print("\n" + "=" * 50)
    print("Performance Metrics")
    print("=" * 50)
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per epoch: {elapsed_time / config.get('awb_preliminary_epochs', 10):.2f}s")
    results['time_per_epoch'] = elapsed_time / config.get('awb_preliminary_epochs', 10)

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 1 TEST: PASSED")
    else:
        print("STEP 1 TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 1: Preliminary Training')
    parser.add_argument('--config', type=str, default='awb_tests/configs/awb_test_cifar100.json',
                        help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step1_preliminary(args.config, args.verbose)
    sys.exit(0 if results['passed'] else 1)
