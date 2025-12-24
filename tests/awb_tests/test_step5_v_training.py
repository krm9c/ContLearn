#!/usr/bin/env python
"""
STEP 5: V Training Test

Tests the V training phase of the AWB pipeline (A/B frozen, V trainable).
Verifies that:
1. V (weights) are trainable while A/B are frozen
2. Loss decreases during V training
3. A/B matrices remain unchanged
4. Model performs well on task data

Usage:
    python awb_tests/test_step5_v_training.py
    python awb_tests/test_step5_v_training.py --verbose
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
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_standard_training,
)
from cl.datasets.sine import SineDataset


def test_step5_v_training(config_path: str = None, verbose: bool = False):
    """Test STEP 5: V training with A/B frozen."""

    print("=" * 70)
    print("STEP 5: V Training Test (A/B frozen, V trainable)")
    print("=" * 70)

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'awb_v_warmup_epochs': 2,
            'epochs_per_task': 20,
            'lr': 1e-3,
            'awb_v_lr_factor': 0.1,
            'batch_size': 64,
            'debug_mode': True,
            'debug_limit': 200,
        }

    print(f"\nV training epochs: {config.get('epochs_per_task', 20)}")
    print(f"V learning rate factor: {config.get('awb_v_lr_factor', 0.1)}")

    results = {
        'passed': True,
        'checks': {},
    }

    # Create model, initialize A/B, and compute V
    original_arch = [10, 64, 64, 5]
    new_arch = [10, 96, 96, 5]

    print(f"\nOriginal architecture: {original_arch}")
    print(f"New architecture: {new_arch}")

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    model_with_ab = set_new_AB_matrices(model, original_arch, new_arch)
    model_with_v = compute_V_from_AWB(model_with_ab)

    # Store original A/B for comparison
    original_A = [a.copy() for a in model_with_v.A]
    original_B = [b.copy() for b in model_with_v.B]

    # Create dataset
    sine_config = {
        'batch_size': config.get('batch_size', 64),
        'n_task': 3,
        'debug_mode': True,
        'debug_limit': 200,
    }
    dataset = SineDataset(sine_config)
    dl_curr, dl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='training')
    tl_curr, tl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='testing')

    train_data = (dl_curr, dl_exp, (tl_curr, tl_exp), (tl_curr, tl_exp))

    # Test 1: Partition for V training (standard training)
    print("\n" + "-" * 50)
    print("Test 1: Partition for V Training")
    print("-" * 50)

    try:
        params, static = partition_for_standard_training(model_with_v)

        # Verify: A/B should be in static (frozen), layers should be in params (trainable)
        check_ab_frozen = True
        check_v_trainable = True

        # Check A/B are in static
        if static.A is None or static.B is None:
            check_ab_frozen = False
            print("[FAIL] A/B matrices should be in static (frozen)")

        # Check A/B are NOT in params
        if params.A is not None or params.B is not None:
            check_ab_frozen = False
            print("[FAIL] A/B matrices should NOT be in params")

        # Check layers are in params
        for i, layer in enumerate(params.layers):
            if layer.weight is None:
                check_v_trainable = False
                print(f"[FAIL] Layer {i} weight should be in params (trainable)")

        check_partition = check_ab_frozen and check_v_trainable
        results['checks']['partition_correct'] = check_partition
        print(f"[{'PASS' if check_partition else 'FAIL'}] Partition for V training is correct")
        if verbose:
            print(f"       A in static: {static.A is not None}")
            print(f"       B in static: {static.B is not None}")
            print(f"       A in params (should be None): {params.A}")
            print(f"       B in params (should be None): {params.B}")
        if not check_partition:
            results['passed'] = False

    except Exception as e:
        results['checks']['partition_correct'] = False
        results['passed'] = False
        print(f"[FAIL] Partition failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 2: V training reduces loss
    print("\n" + "-" * 50)
    print("Test 2: V Training Loss Reduction")
    print("-" * 50)

    try:
        trainer = Trainer(loss='regression', metric='mse', problem='vectors')

        # Create optimizer with V learning rate factor
        lr = config.get('lr', 1e-3) * config.get('awb_v_lr_factor', 0.1)
        optim = optax.adamw(learning_rate=lr)
        opt_state = optim.init(params)

        train_config = {
            'batch_size': config.get('batch_size', 64),
            'problem': 'vectors',
            'data_id': 'sine',
            'flag': [1.0, 1.0],
            'network': 'fcnn',
            'awb_enabled': True,
            'awb_arch': new_arch,
        }

        record_dict = trainer.initialize_record_dict(config, run_id=0)

        # Training loop
        n_epochs = config.get('epochs_per_task', 20)
        start_time = time.time()

        params, static, opt_state, record_dict = trainer.train__CL(
            train_data, params, static, opt_state, optim,
            n_iter=n_epochs,
            save_iter=5,
            task_id=1,
            config=train_config,
            record_dict=record_dict,
            problem_type='vectors',
            loss_type='regression',
            phase='v_training',
            notABTrain=True,  # Training V, not A/B
        )

        elapsed_time = time.time() - start_time

        # Extract losses
        iterations = record_dict.get('iterations', {})
        losses = []
        for k, v in sorted(iterations.items()):
            if isinstance(v, dict) and 'losses' in v:
                losses.append(v['losses'].get('V', float('inf')))

        if len(losses) >= 2:
            check_loss_decreased = losses[-1] < losses[0]
            results['checks']['loss_decreased'] = check_loss_decreased
            results['initial_loss'] = float(losses[0])
            results['final_loss'] = float(losses[-1])
            print(f"[{'PASS' if check_loss_decreased else 'FAIL'}] Loss decreased during V training")
            print(f"       Initial loss: {losses[0]:.4f}")
            print(f"       Final loss: {losses[-1]:.4f}")
            if not check_loss_decreased:
                results['passed'] = False
        else:
            results['checks']['loss_decreased'] = False
            print("[FAIL] Not enough loss values recorded")
            results['passed'] = False

        results['training_time'] = elapsed_time

    except Exception as e:
        results['checks']['loss_decreased'] = False
        results['passed'] = False
        print(f"[FAIL] V training failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 3: A/B matrices unchanged
    print("\n" + "-" * 50)
    print("Test 3: A/B Matrices Remain Frozen")
    print("-" * 50)

    try:
        # Combine params and static to get trained model
        model_trained = eqx.combine(params, static)

        check_a_unchanged = True
        check_b_unchanged = True

        for i, (orig_a, trained_a) in enumerate(zip(original_A, model_trained.A)):
            if not jnp.allclose(orig_a, trained_a, atol=1e-6):
                check_a_unchanged = False
                max_diff = jnp.max(jnp.abs(orig_a - trained_a))
                print(f"[FAIL] A[{i}] changed by {max_diff:.2e}")

        for i, (orig_b, trained_b) in enumerate(zip(original_B, model_trained.B)):
            if not jnp.allclose(orig_b, trained_b, atol=1e-6):
                check_b_unchanged = False
                max_diff = jnp.max(jnp.abs(orig_b - trained_b))
                print(f"[FAIL] B[{i}] changed by {max_diff:.2e}")

        check_ab_unchanged = check_a_unchanged and check_b_unchanged
        results['checks']['ab_unchanged'] = check_ab_unchanged
        print(f"[{'PASS' if check_ab_unchanged else 'FAIL'}] A/B matrices remain frozen during V training")
        if not check_ab_unchanged:
            results['passed'] = False

    except Exception as e:
        results['checks']['ab_unchanged'] = False
        results['passed'] = False
        print(f"[FAIL] A/B freeze check failed: {e}")

    # Test 4: V (weights) were updated
    print("\n" + "-" * 50)
    print("Test 4: V (Weights) Updated")
    print("-" * 50)

    try:
        original_weights = [layer.weight for layer in model_with_v.layers]

        check_v_updated = False
        for i, (orig_w, trained_layer) in enumerate(zip(original_weights, model_trained.layers)):
            trained_w = trained_layer.weight
            if not jnp.allclose(orig_w, trained_w, atol=1e-6):
                check_v_updated = True
                if verbose:
                    max_diff = jnp.max(jnp.abs(orig_w - trained_w))
                    print(f"       Layer {i} weight changed by max {max_diff:.4f}")

        results['checks']['v_updated'] = check_v_updated
        print(f"[{'PASS' if check_v_updated else 'FAIL'}] V (weights) were updated during training")
        if not check_v_updated:
            results['passed'] = False

    except Exception as e:
        results['checks']['v_updated'] = False
        results['passed'] = False
        print(f"[FAIL] V update check failed: {e}")

    # Test 5: Forward pass produces valid output
    print("\n" + "-" * 50)
    print("Test 5: Forward Pass Validity")
    print("-" * 50)

    try:
        test_input = jax.random.normal(jax.random.PRNGKey(0), (32, new_arch[0]))
        output = model_trained(test_input)

        check_valid = jnp.isfinite(output).all()
        check_shape = output.shape == (32, new_arch[-1])

        check_forward = check_valid and check_shape
        results['checks']['valid_forward'] = check_forward
        print(f"[{'PASS' if check_forward else 'FAIL'}] Forward pass produces valid output")
        print(f"       Output shape: {output.shape} (expected: (32, {new_arch[-1]}))")
        print(f"       All finite: {check_valid}")
        if not check_forward:
            results['passed'] = False

    except Exception as e:
        results['checks']['valid_forward'] = False
        results['passed'] = False
        print(f"[FAIL] Forward pass failed: {e}")

    # Test 6: Model performance on test data
    print("\n" + "-" * 50)
    print("Test 6: Model Performance on Test Data")
    print("-" * 50)

    try:
        # Evaluate on test data
        test_losses = []
        for batch in tl_curr:
            x, y = batch
            pred = jax.vmap(model_trained)(x)
            loss = jnp.mean((pred - y) ** 2)
            test_losses.append(float(loss))

        avg_test_loss = np.mean(test_losses)
        results['test_loss'] = avg_test_loss

        # Check that test loss is reasonable (not completely random)
        check_performance = avg_test_loss < 1.0  # Arbitrary threshold for sine regression
        results['checks']['reasonable_performance'] = check_performance
        print(f"[{'PASS' if check_performance else 'FAIL'}] Model achieves reasonable test performance")
        print(f"       Average test loss: {avg_test_loss:.4f}")
        if not check_performance:
            print("       (Note: Test loss > 1.0 may indicate training issues)")

    except Exception as e:
        results['checks']['reasonable_performance'] = False
        print(f"[FAIL] Performance evaluation failed: {e}")

    # Performance metrics
    print("\n" + "=" * 50)
    print("Performance Metrics")
    print("=" * 50)
    if 'training_time' in results:
        print(f"Total V training time: {results['training_time']:.2f}s")
        print(f"Time per epoch: {results['training_time'] / n_epochs:.2f}s")
        print(f"Effective learning rate: {lr:.2e}")

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 5 TEST: PASSED")
    else:
        print("STEP 5 TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 5: V Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step5_v_training(args.config, args.verbose)
    sys.exit(0 if results['passed'] else 1)
