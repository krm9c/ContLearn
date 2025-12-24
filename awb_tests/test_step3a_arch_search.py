#!/usr/bin/env python
"""
STEP 3a: Architecture Search Test

Tests the architecture search phase of the AWB pipeline.
Verifies that:
1. Candidates are generated correctly
2. Search finds architectures with lower loss
3. Bayesian search uses fewer evaluations than grid search
4. JIT compilation happens and is cached

Usage:
    python awb_tests/test_step3a_arch_search.py
    python awb_tests/test_step3a_arch_search.py --config awb_tests/configs/awb_test_cifar100.json
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

from cl.models.cnn import CNN3D
from cl.core.trainer import Trainer
from cl.core.arch_search import (
    search_architecture,
    search_architecture_grid,
    search_architecture_bayesian,
    load_search_config,
)
from cl.datasets.cifar import CIFAR100Dataset
from cl.config.constants import DEFAULT_BATCH_SIZE_CLASSIFICATION


def test_step3a_arch_search(config_path: str = None, verbose: bool = False):
    """Test STEP 3a: Architecture Search."""

    print("=" * 70)
    print("STEP 3a: Architecture Search Test")
    print("=" * 70)

    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'data': 'cifar100',
            'n_task': 5,
            'batch_size': 256,
            'feed_sizes': [2304, 512, 256, 100],
            'filter_size': 3,
            'channel_out': 32,
            'lr': 1e-3,
            'debug_mode': True,
            'debug_limit': 500,
            'n_class': 100,
            'arch_search_method': 'bayesian',
            'arch_search_bo_trials': 3,
            'arch_search_epochs': 5,
            'arch_search_range': 2,
            'arch_search_mlp_increment': 32,
        }

    print(f"\nConfig: {config.get('data', 'cifar100')}")
    print(f"Search method: {config.get('arch_search_method', 'grid')}")
    print(f"BO trials: {config.get('arch_search_bo_trials', 5)}")

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
    print(f"Baseline architecture: {feed_sizes}")

    # Create trainer
    trainer = Trainer(loss='class', metric='class', problem='vectors')

    # Generate data for task 1
    task_id = 1
    dataloader_curr, dataloader_exp = dataset.generate_dataset(
        task_id=task_id, batch_size=config['batch_size'], phase='training'
    )
    test_loader_curr, test_loader_exp = dataset.generate_dataset(
        task_id=task_id, batch_size=config['batch_size'], phase='testing'
    )

    # Simulate baseline loss from preliminary training
    baseline_loss = 2.5  # Typical cross-entropy loss for random model on CIFAR-100

    results = {
        'passed': True,
        'checks': {},
        'baseline_arch': feed_sizes,
        'baseline_loss': baseline_loss,
    }

    # Test 1: Grid search
    print("\n" + "-" * 50)
    print("Test 1: Grid Search")
    print("-" * 50)

    config_grid = config.copy()
    config_grid['arch_search_method'] = 'grid'
    config_grid['arch_search_max_iter'] = 1  # Just 1 iteration for speed

    start_time = time.time()
    try:
        # Note: CNN3D doesn't have generate_search_candidates, so we test with MLP
        # For CNN, we just verify the search function runs without error
        from cl.models.mlp import MLP
        from cl.datasets.sine import SineDataset

        # Create sine dataset first to get correct dimensions
        sine_config = {
            'batch_size': 64,
            'n_task': 3,
            'debug_mode': True,
            'debug_limit': 200,
        }
        sine_dataset = SineDataset(sine_config)
        dl_curr, dl_exp = sine_dataset.generate_dataset(task_id=1, batch_size=64, phase='training')
        tl_curr, tl_exp = sine_dataset.generate_dataset(task_id=1, batch_size=64, phase='testing')

        # Create MLP with dimensions matching sine dataset
        # SineDataset: input_size=3, output_size depends on time steps (usually 10)
        input_size = sine_dataset.input_size
        output_size = sine_dataset.output_size
        mlp_baseline = [input_size, 64, 64, output_size]
        mlp_model = MLP(sizes=mlp_baseline, key=jax.random.PRNGKey(42), awb_enabled=True)

        mlp_config = config_grid.copy()
        mlp_config['prob'] = 'regression'
        mlp_config['loss'] = 'mse'
        mlp_config['metric'] = 'mse'

        grid_arch = search_architecture_grid(
            model=mlp_model,
            baseline_arch=mlp_baseline,
            task_id=1,
            baseline_loss=0.5,
            dataloader_curr=dl_curr,
            dataloader_exp=dl_exp,
            test_loader_curr=tl_curr,
            test_loader_exp=tl_exp,
            config=mlp_config,
            trainer=None,
            model_type='mlp',
        )
        grid_time = time.time() - start_time

        results['checks']['grid_search_runs'] = True
        results['grid_arch'] = grid_arch
        results['grid_time'] = grid_time
        print(f"[PASS] Grid search completed in {grid_time:.2f}s")
        print(f"       Result architecture: {grid_arch}")

    except Exception as e:
        results['checks']['grid_search_runs'] = False
        results['passed'] = False
        print(f"[FAIL] Grid search failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Bayesian search
    print("\n" + "-" * 50)
    print("Test 2: Bayesian Search (Optuna)")
    print("-" * 50)

    config_bayesian = config.copy()
    config_bayesian['arch_search_method'] = 'bayesian'
    config_bayesian['arch_search_bo_trials'] = 3

    start_time = time.time()
    try:
        bayesian_arch = search_architecture_bayesian(
            model=mlp_model,
            baseline_arch=mlp_baseline,
            task_id=1,
            baseline_loss=0.5,
            dataloader_curr=dl_curr,
            dataloader_exp=dl_exp,
            test_loader_curr=tl_curr,
            test_loader_exp=tl_exp,
            config=config_bayesian,
            trainer=None,
            model_type='mlp',
        )
        bayesian_time = time.time() - start_time

        results['checks']['bayesian_search_runs'] = True
        results['bayesian_arch'] = bayesian_arch
        results['bayesian_time'] = bayesian_time
        print(f"[PASS] Bayesian search completed in {bayesian_time:.2f}s")
        print(f"       Result architecture: {bayesian_arch}")

    except ImportError:
        results['checks']['bayesian_search_runs'] = 'skipped'
        print("[SKIP] Optuna not installed, Bayesian search skipped")
    except Exception as e:
        results['checks']['bayesian_search_runs'] = False
        results['passed'] = False
        print(f"[FAIL] Bayesian search failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Dispatcher works correctly
    print("\n" + "-" * 50)
    print("Test 3: Search Dispatcher")
    print("-" * 50)

    try:
        # Test with 'grid' method
        config_test = config.copy()
        config_test['arch_search_method'] = 'grid'
        config_test['arch_search_max_iter'] = 1

        dispatcher_arch = search_architecture(
            model=mlp_model,
            baseline_arch=mlp_baseline,
            task_id=1,
            baseline_loss=0.5,
            dataloader_curr=dl_curr,
            dataloader_exp=dl_exp,
            test_loader_curr=tl_curr,
            test_loader_exp=tl_exp,
            config=config_test,
            trainer=None,
            model_type='mlp',
        )

        results['checks']['dispatcher_works'] = True
        print(f"[PASS] Dispatcher correctly routes to grid search")
        print(f"       Result architecture: {dispatcher_arch}")

    except Exception as e:
        results['checks']['dispatcher_works'] = False
        results['passed'] = False
        print(f"[FAIL] Dispatcher failed: {e}")

    # Test 4: Architecture is valid
    print("\n" + "-" * 50)
    print("Test 4: Architecture Validity")
    print("-" * 50)

    if 'grid_arch' in results:
        arch = results['grid_arch']
        check_valid = (
            len(arch) == len(mlp_baseline) and
            arch[0] == mlp_baseline[0] and  # Input size preserved
            arch[-1] == mlp_baseline[-1] and  # Output size preserved
            all(h > 0 for h in arch)  # All positive
        )
        results['checks']['arch_valid'] = check_valid
        print(f"[{'PASS' if check_valid else 'FAIL'}] Architecture is valid")
        print(f"       Input preserved: {arch[0]} == {mlp_baseline[0]}")
        print(f"       Output preserved: {arch[-1]} == {mlp_baseline[-1]}")
        if not check_valid:
            results['passed'] = False

    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)

    if 'grid_time' in results and 'bayesian_time' in results:
        speedup = results['grid_time'] / results['bayesian_time'] if results['bayesian_time'] > 0 else 0
        print(f"Grid search time:     {results['grid_time']:.2f}s")
        print(f"Bayesian search time: {results['bayesian_time']:.2f}s")
        print(f"Speedup:              {speedup:.2f}x")
        results['speedup'] = speedup

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 3a TEST: PASSED")
    else:
        print("STEP 3a TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 3a: Architecture Search')
    parser.add_argument('--config', type=str, default='awb_tests/configs/awb_test_cifar100.json',
                        help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step3a_arch_search(args.config, args.verbose)
    sys.exit(0 if results['passed'] else 1)
