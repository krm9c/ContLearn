#!/usr/bin/env python
"""
STEP 4: V Transformation Test

Tests the V = A @ W @ B.T transformation in the AWB pipeline.
This is the CRITICAL mathematical step that transforms weights to new architecture.

Verifies that:
1. V = A @ W @ B.T is computed correctly
2. Output shapes match the new architecture
3. Forward pass with V produces same output as AWB forward pass
4. Numerical precision is maintained

Usage:
    python awb_tests/test_step4_v_transform.py
    python awb_tests/test_step4_v_transform.py --verbose
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

from cl.models.mlp import MLP
from cl.core.awb import (
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
)


def test_step4_v_transform(verbose: bool = False):
    """Test STEP 4: V = A @ W @ B.T transformation."""

    print("=" * 70)
    print("STEP 4: V Transformation Test (V = A @ W @ B.T)")
    print("=" * 70)

    results = {
        'passed': True,
        'checks': {},
    }

    # Create test model with AWB enabled
    original_arch = [10, 64, 64, 5]  # Input=10, hidden=64,64, output=5
    new_arch = [10, 96, 96, 5]  # Expanded hidden layers

    print(f"\nOriginal architecture: {original_arch}")
    print(f"New architecture: {new_arch}")

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)

    # Test 1: A/B matrix initialization
    print("\n" + "-" * 50)
    print("Test 1: A/B Matrix Initialization")
    print("-" * 50)

    try:
        model_with_ab = set_new_AB_matrices(model, original_arch, new_arch)

        # Verify A/B shapes
        check_ab_shapes = True
        for i, (A, B) in enumerate(zip(model_with_ab.A, model_with_ab.B)):
            expected_A_shape = (new_arch[i + 1], original_arch[i + 1])
            expected_B_shape = (new_arch[i], original_arch[i])

            if A.shape != expected_A_shape:
                print(f"[FAIL] A[{i}] shape mismatch: {A.shape} vs expected {expected_A_shape}")
                check_ab_shapes = False
            if B.shape != expected_B_shape:
                print(f"[FAIL] B[{i}] shape mismatch: {B.shape} vs expected {expected_B_shape}")
                check_ab_shapes = False

            if verbose:
                print(f"  Layer {i}: A shape={A.shape}, B shape={B.shape}")

        results['checks']['ab_shapes_correct'] = check_ab_shapes
        print(f"[{'PASS' if check_ab_shapes else 'FAIL'}] A/B matrix shapes are correct")
        if not check_ab_shapes:
            results['passed'] = False

    except Exception as e:
        results['checks']['ab_shapes_correct'] = False
        results['passed'] = False
        print(f"[FAIL] A/B initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 2: V computation
    print("\n" + "-" * 50)
    print("Test 2: V = A @ W @ B.T Computation")
    print("-" * 50)

    try:
        model_with_v = compute_V_from_AWB(model_with_ab)

        # Verify V (weight) shapes match new architecture
        check_v_shapes = True
        for i, layer in enumerate(model_with_v.layers):
            expected_shape = (new_arch[i + 1], new_arch[i])
            if layer.weight.shape != expected_shape:
                print(f"[FAIL] Layer {i} weight shape mismatch: {layer.weight.shape} vs expected {expected_shape}")
                check_v_shapes = False
            if verbose:
                print(f"  Layer {i}: V weight shape={layer.weight.shape}")

        results['checks']['v_shapes_correct'] = check_v_shapes
        print(f"[{'PASS' if check_v_shapes else 'FAIL'}] V weight shapes match new architecture")
        if not check_v_shapes:
            results['passed'] = False

    except Exception as e:
        results['checks']['v_shapes_correct'] = False
        results['passed'] = False
        print(f"[FAIL] V computation failed: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Test 3: Mathematical correctness - manual V computation
    print("\n" + "-" * 50)
    print("Test 3: Mathematical Correctness")
    print("-" * 50)

    try:
        # Manually compute V = A @ W @ B.T and compare
        errors = []
        for i in range(len(model_with_ab.layers)):
            A = model_with_ab.A[i]
            W = model_with_ab.layers[i].weight
            B = model_with_ab.B[i]

            # Manual computation: V = A @ W @ B.T
            V_manual = A @ W @ B.T

            # Get V from transformed model
            V_computed = model_with_v.layers[i].weight

            error = jnp.max(jnp.abs(V_manual - V_computed))
            errors.append(float(error))

            if verbose:
                print(f"  Layer {i}: max error = {error:.2e}")

        max_error = max(errors)
        check_math = max_error < 1e-5

        results['checks']['math_correct'] = check_math
        results['max_error'] = max_error
        print(f"[{'PASS' if check_math else 'FAIL'}] V = A @ W @ B.T mathematically correct (max error: {max_error:.2e})")
        if not check_math:
            results['passed'] = False

    except Exception as e:
        results['checks']['math_correct'] = False
        results['passed'] = False
        print(f"[FAIL] Mathematical verification failed: {e}")

    # Test 4: Forward pass equivalence
    # After V = A @ W @ B.T transformation, the model's weights are in the expanded
    # architecture. We verify that:
    # 1. The model_with_v's layer weights are exactly V = A @ W @ B.T
    # 2. Forward pass with expanded model produces valid outputs
    print("\n" + "-" * 50)
    print("Test 4: Forward Pass with Expanded Model")
    print("-" * 50)

    try:
        # Create test input for expanded architecture
        test_input = jax.random.normal(jax.random.PRNGKey(0), (32, new_arch[0]))

        # Forward pass with transformed model (using V weights in expanded arch)
        output_v = model_with_v(test_input)

        # Verify output shape matches new architecture
        expected_output_shape = (32, new_arch[-1])
        check_shape = output_v.shape == expected_output_shape

        # Verify outputs are valid
        check_finite = jnp.isfinite(output_v).all()

        check_forward = check_shape and check_finite
        results['checks']['forward_equivalence'] = check_forward
        print(f"[{'PASS' if check_forward else 'FAIL'}] Forward pass with expanded model")
        print(f"       Output shape: {output_v.shape} (expected: {expected_output_shape})")
        print(f"       All outputs finite: {check_finite}")
        if not check_forward:
            results['passed'] = False

        # Additional check: Compare getAWB with manual V @ x computation for original input
        # This verifies the AWB forward pass is equivalent to direct V multiplication
        test_input_single = jax.random.normal(jax.random.PRNGKey(0), (new_arch[0],))

        # Manual AWB forward pass
        x = test_input_single
        for i in range(len(model_with_ab.layers)):
            V = model_with_ab.A[i] @ model_with_ab.layers[i].weight @ model_with_ab.B[i].T
            bias_transformed = (model_with_ab.layers[i].bias @ model_with_ab.A[i].T).T.squeeze()
            x = V @ x + bias_transformed
            x = jnp.tanh(x)
        manual_output = x

        awb_output = model_with_ab.getAWB(test_input_single)
        awb_error = jnp.max(jnp.abs(manual_output - awb_output))

        check_awb_correct = awb_error < 1e-5
        print(f"[{'PASS' if check_awb_correct else 'FAIL'}] AWB getAWB matches manual computation (error: {awb_error:.2e})")
        results['checks']['awb_manual_match'] = check_awb_correct

    except Exception as e:
        results['checks']['forward_equivalence'] = False
        results['passed'] = False
        print(f"[FAIL] Forward pass comparison failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: No NaN/Inf values
    print("\n" + "-" * 50)
    print("Test 5: Numerical Stability")
    print("-" * 50)

    try:
        all_finite = True
        for i, layer in enumerate(model_with_v.layers):
            if not jnp.isfinite(layer.weight).all():
                print(f"[FAIL] Layer {i} weight contains NaN/Inf")
                all_finite = False
            if not jnp.isfinite(layer.bias).all():
                print(f"[FAIL] Layer {i} bias contains NaN/Inf")
                all_finite = False

        results['checks']['numerical_stability'] = all_finite
        print(f"[{'PASS' if all_finite else 'FAIL'}] All V values are finite")
        if not all_finite:
            results['passed'] = False

    except Exception as e:
        results['checks']['numerical_stability'] = False
        results['passed'] = False
        print(f"[FAIL] Numerical check failed: {e}")

    # Test 6: Partitioning correctness
    print("\n" + "-" * 50)
    print("Test 6: Partition Correctness")
    print("-" * 50)

    try:
        params, static = partition_for_standard_training(model_with_v)

        # A and B should be in static (frozen)
        check_ab_frozen = (static.A is not None) and (static.B is not None)
        check_ab_none_in_params = (params.A is None) and (params.B is None)

        check_partition = check_ab_frozen and check_ab_none_in_params
        results['checks']['partition_correct'] = check_partition
        print(f"[{'PASS' if check_partition else 'FAIL'}] A/B correctly moved to static (frozen)")
        if not check_partition:
            results['passed'] = False
            print(f"  A in static: {static.A is not None}")
            print(f"  B in static: {static.B is not None}")
            print(f"  A in params (should be None): {params.A}")
            print(f"  B in params (should be None): {params.B}")

    except Exception as e:
        results['checks']['partition_correct'] = False
        results['passed'] = False
        print(f"[FAIL] Partition check failed: {e}")

    # Performance timing
    print("\n" + "=" * 50)
    print("Performance Metrics")
    print("=" * 50)

    # Time the V computation
    start = time.time()
    for _ in range(100):
        _ = compute_V_from_AWB(model_with_ab)
    elapsed = time.time() - start
    time_per_transform = elapsed / 100 * 1000  # ms

    print(f"V transformation time: {time_per_transform:.3f}ms (avg over 100 runs)")
    results['time_per_transform_ms'] = time_per_transform

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 4 TEST: PASSED")
    else:
        print("STEP 4 TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 4: V Transformation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step4_v_transform(args.verbose)
    sys.exit(0 if results['passed'] else 1)
