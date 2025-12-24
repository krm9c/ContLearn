#!/usr/bin/env python
"""
Mathematical Correctness Tests for AWB Pipeline

Comprehensive tests to verify the mathematical properties of the AWB algorithm:
1. V transformation: V = A @ W @ B.T
2. Gradient flow correctness
3. Partition correctness (frozen vs trainable)
4. Shape consistency through pipeline
5. Output equivalence before/after transformation

Usage:
    python awb_tests/test_mathematical_correctness.py
    python awb_tests/test_mathematical_correctness.py --verbose
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from cl.models.mlp import MLP
from cl.models.cnn import CNN3D
from cl.core.awb import (
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
)
from cl.core.hamiltonian import _hamiltonian_core_mse_standard


class TestResult:
    """Container for test results."""
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.checks = {}
        self.errors = []

    def add_check(self, check_name, passed, message=None):
        self.checks[check_name] = passed
        if not passed:
            self.passed = False
            if message:
                self.errors.append(f"{check_name}: {message}")

    def summary(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"


def test_v_transformation_mlp():
    """Test V = A @ W @ B.T for MLP."""
    result = TestResult("V Transformation (MLP)")

    print("\n" + "=" * 60)
    print("Test: V = A @ W @ B.T Transformation (MLP)")
    print("=" * 60)

    # Setup
    original_arch = [10, 64, 64, 5]
    new_arch = [10, 96, 96, 5]

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    model_ab = set_new_AB_matrices(model, original_arch, new_arch)
    model_v = compute_V_from_AWB(model_ab)

    # Check 1: Shapes match
    for i, layer in enumerate(model_v.layers):
        expected = (new_arch[i + 1], new_arch[i])
        if layer.weight.shape != expected:
            result.add_check(f'layer_{i}_shape', False,
                           f"Expected {expected}, got {layer.weight.shape}")
        else:
            result.add_check(f'layer_{i}_shape', True)

    # Check 2: V = A @ W @ B.T mathematically
    max_error = 0
    for i in range(len(model_ab.layers)):
        V_expected = model_ab.A[i] @ model_ab.layers[i].weight @ model_ab.B[i].T
        V_actual = model_v.layers[i].weight
        error = float(jnp.max(jnp.abs(V_expected - V_actual)))
        max_error = max(max_error, error)

    result.add_check('v_equals_awbt', max_error < 1e-5,
                    f"Max error: {max_error:.2e}")

    print(f"  V = A @ W @ B.T max error: {max_error:.2e}")
    print(f"  {result.summary()}")

    return result


def test_forward_pass_equivalence():
    """Test that forward pass is equivalent before/after V transformation."""
    result = TestResult("Forward Pass Equivalence")

    print("\n" + "=" * 60)
    print("Test: Forward Pass Equivalence")
    print("=" * 60)

    original_arch = [10, 64, 64, 5]
    new_arch = [10, 96, 96, 5]

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    model_ab = set_new_AB_matrices(model, original_arch, new_arch)
    model_v = compute_V_from_AWB(model_ab)

    # Test input (batch of samples)
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 10))

    # Forward with AWB (getAWB expects unbatched)
    output_awb = jax.vmap(model_ab.getAWB)(x)

    # Forward with V weights (standard forward)
    output_v = model_v(x)

    error = float(jnp.max(jnp.abs(output_awb - output_v)))
    result.add_check('output_equivalence', error < 1e-4,
                    f"Max error: {error:.2e}")

    print(f"  Output difference: {error:.2e}")
    print(f"  {result.summary()}")

    return result


def test_gradient_flow_ab_training():
    """Test that gradients only flow to A/B during A/B training (W frozen)."""
    result = TestResult("Gradient Flow (A/B Training)")

    print("\n" + "=" * 60)
    print("Test: Gradient Flow During A/B Training")
    print("=" * 60)

    original_arch = [10, 64, 64, 5]
    new_arch = [10, 96, 96, 5]

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    model_ab = set_new_AB_matrices(model, original_arch, new_arch)

    # Partition for A/B training
    diff_model, static_model = partition_for_AB_training(model_ab)

    # Check that W is in static (frozen)
    w_in_static = all(
        layer.weight is not None
        for layer in static_model.layers
    )
    result.add_check('w_in_static', w_in_static)

    # Check that A/B are in diff (trainable)
    ab_in_diff = (diff_model.A is not None) and (diff_model.B is not None)
    result.add_check('ab_in_diff', ab_in_diff)

    # Compute gradient and verify W gradients are zero
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 5))

    def loss_fn(diff):
        model_combined = eqx.combine(diff, static_model)
        pred = jax.vmap(model_combined.getAWB)(x)
        return jnp.mean((pred - y) ** 2)

    grads = jax.grad(loss_fn)(diff_model)

    # A/B should have non-zero gradients
    a_grads_nonzero = any(jnp.any(g != 0) for g in grads.A)
    b_grads_nonzero = any(jnp.any(g != 0) for g in grads.B)

    result.add_check('a_gradients_nonzero', a_grads_nonzero)
    result.add_check('b_gradients_nonzero', b_grads_nonzero)

    # Layer weights should be None in diff (no gradients)
    w_grads_none = all(layer.weight is None for layer in grads.layers)
    result.add_check('w_gradients_frozen', w_grads_none)

    print(f"  W in static: {w_in_static}")
    print(f"  A/B in diff: {ab_in_diff}")
    print(f"  A gradients non-zero: {a_grads_nonzero}")
    print(f"  B gradients non-zero: {b_grads_nonzero}")
    print(f"  W gradients None: {w_grads_none}")
    print(f"  {result.summary()}")

    return result


def test_gradient_flow_v_training():
    """Test that gradients only flow to V during V training (A/B frozen)."""
    result = TestResult("Gradient Flow (V Training)")

    print("\n" + "=" * 60)
    print("Test: Gradient Flow During V Training")
    print("=" * 60)

    original_arch = [10, 64, 64, 5]
    new_arch = [10, 96, 96, 5]

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    model_ab = set_new_AB_matrices(model, original_arch, new_arch)
    model_v = compute_V_from_AWB(model_ab)

    # Partition for V training (A/B frozen)
    params, static = partition_for_standard_training(model_v)

    # Check A/B in static
    ab_in_static = (static.A is not None) and (static.B is not None)
    result.add_check('ab_in_static', ab_in_static)

    # Check A/B not in params
    ab_not_in_params = (params.A is None) and (params.B is None)
    result.add_check('ab_not_in_params', ab_not_in_params)

    # Compute gradient
    x = jax.random.normal(jax.random.PRNGKey(0), (32, new_arch[0]))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, new_arch[-1]))

    def loss_fn(p):
        model_combined = eqx.combine(p, static)
        pred = model_combined(x)
        return jnp.mean((pred - y) ** 2)

    grads = jax.grad(loss_fn)(params)

    # V (layer weights) should have non-zero gradients
    v_grads_nonzero = any(
        layer.weight is not None and jnp.any(layer.weight != 0)
        for layer in grads.layers
    )
    result.add_check('v_gradients_nonzero', v_grads_nonzero)

    # A/B should be None in grads
    ab_grads_none = (grads.A is None) and (grads.B is None)
    result.add_check('ab_gradients_frozen', ab_grads_none)

    print(f"  A/B in static: {ab_in_static}")
    print(f"  A/B not in params: {ab_not_in_params}")
    print(f"  V gradients non-zero: {v_grads_nonzero}")
    print(f"  A/B gradients None: {ab_grads_none}")
    print(f"  {result.summary()}")

    return result


def test_hamiltonian_with_awb():
    """Test that Hamiltonian gradient computation works with AWB model."""
    result = TestResult("Hamiltonian with AWB")

    print("\n" + "=" * 60)
    print("Test: Hamiltonian Gradient Computation")
    print("=" * 60)

    original_arch = [1, 64, 64, 1]

    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    params, static = eqx.partition(model, eqx.is_array)

    # Move A/B to static
    if model.awb_enabled:
        static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
        params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

    # Create dummy data
    batch_size = 32
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1))
    y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 1))
    exp_x = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1))
    exp_y = jax.random.normal(jax.random.PRNGKey(3), (batch_size, 1))
    deltax = jax.random.normal(jax.random.PRNGKey(4), (batch_size, 1)) * 0.01

    try:
        grad, losses = _hamiltonian_core_mse_standard(
            params, static, x, y, exp_x, exp_y, deltax,
            jnp.array(0.3), jnp.array(0.6), jnp.array(0.1),
            jnp.array(1000.0), jnp.array(1.0)
        )

        # Check gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grad)
        all_finite = all(
            jnp.isfinite(g).all() if g is not None else True
            for g in grad_leaves
        )
        result.add_check('gradients_finite', all_finite)

        # Check losses are finite
        losses_finite = all(jnp.isfinite(l) for l in losses if l is not None)
        result.add_check('losses_finite', losses_finite)

        print(f"  Gradients finite: {all_finite}")
        print(f"  Losses finite: {losses_finite}")
        print(f"  Loss V: {losses[0]:.4f}, dV: {losses[1]:.4f}, H: {losses[4]:.4f}")

    except Exception as e:
        result.add_check('hamiltonian_runs', False, str(e))
        print(f"  [FAIL] Hamiltonian failed: {e}")

    print(f"  {result.summary()}")
    return result


def test_shape_consistency_through_pipeline():
    """Test that shapes are consistent through the entire AWB pipeline."""
    result = TestResult("Shape Consistency Through Pipeline")

    print("\n" + "=" * 60)
    print("Test: Shape Consistency Through Pipeline")
    print("=" * 60)

    original_arch = [10, 64, 64, 5]
    new_arch = [10, 128, 128, 5]

    # Step 1: Original model
    model = MLP(sizes=original_arch, key=jax.random.PRNGKey(42), awb_enabled=True)
    print(f"  Original: {[l.weight.shape for l in model.layers]}")

    # Step 2: Add A/B matrices
    model_ab = set_new_AB_matrices(model, original_arch, new_arch)
    print(f"  After A/B: W shapes = {[l.weight.shape for l in model_ab.layers]}")
    print(f"             A shapes = {[a.shape for a in model_ab.A]}")
    print(f"             B shapes = {[b.shape for b in model_ab.B]}")

    # Verify A/B shapes
    for i in range(len(model_ab.layers)):
        expected_A = (new_arch[i + 1], original_arch[i + 1])
        expected_B = (new_arch[i], original_arch[i])

        if model_ab.A[i].shape != expected_A:
            result.add_check(f'A_{i}_shape', False,
                           f"Expected {expected_A}, got {model_ab.A[i].shape}")
        else:
            result.add_check(f'A_{i}_shape', True)

        if model_ab.B[i].shape != expected_B:
            result.add_check(f'B_{i}_shape', False,
                           f"Expected {expected_B}, got {model_ab.B[i].shape}")
        else:
            result.add_check(f'B_{i}_shape', True)

    # Step 3: Compute V
    model_v = compute_V_from_AWB(model_ab)
    print(f"  After V:  {[l.weight.shape for l in model_v.layers]}")

    # Verify V shapes match new architecture
    for i, layer in enumerate(model_v.layers):
        expected = (new_arch[i + 1], new_arch[i])
        if layer.weight.shape != expected:
            result.add_check(f'V_{i}_shape', False,
                           f"Expected {expected}, got {layer.weight.shape}")
        else:
            result.add_check(f'V_{i}_shape', True)

    # Step 4: Test forward pass
    x = jax.random.normal(jax.random.PRNGKey(0), (32, new_arch[0]))
    try:
        output = model_v(x)
        expected_output_shape = (32, new_arch[-1])
        if output.shape == expected_output_shape:
            result.add_check('output_shape', True)
        else:
            result.add_check('output_shape', False,
                           f"Expected {expected_output_shape}, got {output.shape}")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        result.add_check('forward_pass', False, str(e))

    print(f"  {result.summary()}")
    return result


def run_all_tests(verbose: bool = False):
    """Run all mathematical correctness tests."""

    print("\n" + "=" * 70)
    print("AWB MATHEMATICAL CORRECTNESS TEST SUITE")
    print("=" * 70)

    tests = [
        test_v_transformation_mlp,
        test_forward_pass_equivalence,
        test_gradient_flow_ab_training,
        test_gradient_flow_v_training,
        test_hamiltonian_with_awb,
        test_shape_consistency_through_pipeline,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            result = TestResult(test_fn.__name__)
            result.add_check('execution', False, str(e))
            results.append(result)
            print(f"  [FAIL] Test crashed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        print(result.summary())

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED")
        return True
    else:
        print("\nSOME TESTS FAILED")
        for result in results:
            if not result.passed:
                for error in result.errors:
                    print(f"  - {result.name}: {error}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWB Mathematical Correctness Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    success = run_all_tests(args.verbose)
    sys.exit(0 if success else 1)
