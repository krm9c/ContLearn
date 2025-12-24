#!/usr/bin/env python
"""
STEP 3b: A/B Matrix Training Test

Tests the A/B matrix training phase of the AWB pipeline (W frozen, A/B trainable).
Verifies that:
1. A/B matrices are trainable while W is frozen
2. Loss decreases during A/B training
3. Original W weights remain unchanged
4. A/B matrices converge to stable values

Usage:
    python awb_tests/test_step3b_ab_training.py
    python awb_tests/test_step3b_ab_training.py --config awb_tests/configs/awb_test_cifar100.json
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
    partition_for_AB_training,
    partition_for_standard_training,
)
from cl.datasets.sine import SineDataset


def test_step3b_ab_training(config_path: str = None, verbose: bool = False):
    """Test STEP 3b: A/B matrix training with W frozen."""

    print("=" * 70)
    print("STEP 3b: A/B Matrix Training Test (W frozen, A/B trainable)")
    print("=" * 70)

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'awb_ab_training_epochs': 10,
            'awb_ab_warmup_epochs': 2,
            'lr': 1e-3,
            'batch_size': 64,
            'debug_mode': True,
            'debug_limit': 200,
        }

    print(f"\nA/B training epochs: {config.get('awb_ab_training_epochs', 10)}")
    print(f"Warmup epochs: {config.get('awb_ab_warmup_epochs', 2)}")

    results = {
        'passed': True,
        'checks': {},
    }

    # Fixed by Claude: Create dataset first to determine input/output dimensions
    dataset_config = {
        'batch_size': config.get('batch_size', 64),
        'n_task': 3,
        'debug_mode': True,
        'debug_limit': 200,
    }
    dataset = SineDataset(dataset_config)
    dl_curr, dl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='training')
    tl_curr, tl_exp = dataset.generate_dataset(task_id=1, batch_size=64, phase='testing')

    # Get input/output dimensions from dataset
    input_dim = dataset.input_size  # Sine: 3 features
    output_dim = dataset.output_size  # Sine: 1 output

    # Create model with AWB enabled - use dataset-appropriate dimensions
    original_arch = [input_dim, 64, 64, output_dim]
    new_arch = [input_dim, 96, 96, output_dim]  # Keep input/output, expand hidden

    print(f"\nOriginal architecture: {original_arch}")
    print(f"New architecture: {new_arch}")

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)

    # Initialize A/B matrices
    model_with_ab = set_new_AB_matrices(model, original_arch, new_arch)

    # Store original W weights for comparison
    original_weights = [layer.weight.copy() for layer in model_with_ab.layers]

    train_data = (dl_curr, dl_exp, (tl_curr, tl_exp), (tl_curr, tl_exp))

    # Test 1: Partition for A/B training
    print("\n" + "-" * 50)
    print("Test 1: Partition for A/B Training")
    print("-" * 50)

    try:
        params, static = partition_for_AB_training(model_with_ab)

        # Verify: W should be in static (frozen), A/B should be in params (trainable)
        check_w_frozen = True
        check_ab_trainable = True

        # Check A/B are in params
        if params.A is None or params.B is None:
            check_ab_trainable = False
            print("[FAIL] A/B matrices should be in params (trainable)")

        # Check W is in static (layers should be frozen)
        for i, layer in enumerate(static.layers):
            if layer.weight is None:
                check_w_frozen = False
                print(f"[FAIL] Layer {i} weight should be in static (frozen)")

        check_partition = check_w_frozen and check_ab_trainable
        results['checks']['partition_correct'] = check_partition
        print(f"[{'PASS' if check_partition else 'FAIL'}] Partition for A/B training is correct")
        if verbose:
            print(f"       A in params: {params.A is not None}")
            print(f"       B in params: {params.B is not None}")
        if not check_partition:
            results['passed'] = False

    except Exception as e:
        results['checks']['partition_correct'] = False
        results['passed'] = False
        print(f"[FAIL] Partition failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 2: A/B training reduces loss
    print("\n" + "-" * 50)
    print("Test 2: A/B Training Loss Reduction")
    print("-" * 50)

    try:
        trainer = Trainer(loss='regression', metric='mse', problem='vectors')

        # Create optimizer for A/B training
        optim = optax.adamw(learning_rate=config.get('lr', 1e-3))
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
        n_epochs = config.get('awb_ab_training_epochs', 10)
        start_time = time.time()

        params, static, opt_state, record_dict = trainer.train__CL(
            train_data, params, static, opt_state, optim,
            n_iter=n_epochs,
            save_iter=2,
            task_id=1,
            config=train_config,
            record_dict=record_dict,
            problem_type='vectors',
            loss_type='regression',
            phase='ab_training',
            notABTrain=False,  # Training A/B, not V
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
            print(f"[{'PASS' if check_loss_decreased else 'FAIL'}] Loss decreased during A/B training")
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
        print(f"[FAIL] A/B training failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 3: Original W weights unchanged
    print("\n" + "-" * 50)
    print("Test 3: W Weights Remain Frozen")
    print("-" * 50)

    try:
        # Combine params and static to get trained model
        model_trained = eqx.combine(params, static)

        check_w_unchanged = True
        for i, (orig_w, trained_layer) in enumerate(zip(original_weights, model_trained.layers)):
            trained_w = trained_layer.weight
            if not jnp.allclose(orig_w, trained_w, atol=1e-6):
                check_w_unchanged = False
                max_diff = jnp.max(jnp.abs(orig_w - trained_w))
                print(f"[FAIL] Layer {i} weights changed by {max_diff:.2e}")

        results['checks']['w_unchanged'] = check_w_unchanged
        print(f"[{'PASS' if check_w_unchanged else 'FAIL'}] W weights remain frozen during A/B training")
        if not check_w_unchanged:
            results['passed'] = False

    except Exception as e:
        results['checks']['w_unchanged'] = False
        results['passed'] = False
        print(f"[FAIL] W freeze check failed: {e}")

    # Test 4: A/B matrices were updated
    print("\n" + "-" * 50)
    print("Test 4: A/B Matrices Updated")
    print("-" * 50)

    try:
        # Get original A/B (identity-like initialization)
        original_A = model_with_ab.A
        original_B = model_with_ab.B

        # Get trained A/B
        trained_A = model_trained.A
        trained_B = model_trained.B

        check_a_updated = False
        check_b_updated = False

        for i, (orig_a, trained_a) in enumerate(zip(original_A, trained_A)):
            if not jnp.allclose(orig_a, trained_a, atol=1e-6):
                check_a_updated = True
                if verbose:
                    max_diff = jnp.max(jnp.abs(orig_a - trained_a))
                    print(f"       A[{i}] changed by max {max_diff:.4f}")

        for i, (orig_b, trained_b) in enumerate(zip(original_B, trained_B)):
            if not jnp.allclose(orig_b, trained_b, atol=1e-6):
                check_b_updated = True
                if verbose:
                    max_diff = jnp.max(jnp.abs(orig_b - trained_b))
                    print(f"       B[{i}] changed by max {max_diff:.4f}")

        check_ab_updated = check_a_updated and check_b_updated
        results['checks']['ab_updated'] = check_ab_updated
        print(f"[{'PASS' if check_ab_updated else 'FAIL'}] A/B matrices were updated during training")
        print(f"       A updated: {check_a_updated}, B updated: {check_b_updated}")
        if not check_ab_updated:
            results['passed'] = False

    except Exception as e:
        results['checks']['ab_updated'] = False
        results['passed'] = False
        print(f"[FAIL] A/B update check failed: {e}")

    # Test 5: A/B values are finite
    print("\n" + "-" * 50)
    print("Test 5: A/B Numerical Stability")
    print("-" * 50)

    try:
        all_finite = True
        for i, (a, b) in enumerate(zip(model_trained.A, model_trained.B)):
            if not jnp.isfinite(a).all():
                print(f"[FAIL] A[{i}] contains NaN/Inf")
                all_finite = False
            if not jnp.isfinite(b).all():
                print(f"[FAIL] B[{i}] contains NaN/Inf")
                all_finite = False

        results['checks']['ab_finite'] = all_finite
        print(f"[{'PASS' if all_finite else 'FAIL'}] All A/B values are finite")
        if not all_finite:
            results['passed'] = False

    except Exception as e:
        results['checks']['ab_finite'] = False
        results['passed'] = False
        print(f"[FAIL] Numerical check failed: {e}")

    # Performance metrics
    print("\n" + "=" * 50)
    print("Performance Metrics")
    print("=" * 50)
    if 'training_time' in results:
        print(f"Total A/B training time: {results['training_time']:.2f}s")
        print(f"Time per epoch: {results['training_time'] / n_epochs:.2f}s")

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 3b TEST: PASSED")
    else:
        print("STEP 3b TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 3b: A/B Matrix Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step3b_ab_training(args.config, args.verbose)
    sys.exit(0 if results['passed'] else 1)
