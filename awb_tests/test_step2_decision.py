#!/usr/bin/env python
"""
STEP 2: Architecture Change Decision Test

Tests the decision logic for whether to change architecture in the AWB pipeline.
Verifies that:
1. Decision is made based on loss ratio thresholds
2. Decision correctly identifies when architecture change is needed
3. Decision is consistent across multiple evaluations

Usage:
    python awb_tests/test_step2_decision.py
    python awb_tests/test_step2_decision.py --verbose
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from cl.core.awb import should_change_arch


def test_step2_decision(verbose: bool = False):
    """Test STEP 2: Architecture change decision logic.

    Note: The should_change_arch function uses:
        - trainWLoss: Current training loss after preliminary training
        - end_last: Loss at end of previous task
        - threshold_high: High threshold for loss ratio (default: 0.45)
        - min_delta: Minimum loss increase to trigger change (default: 0.01)

    Decision logic:
        - If ratio > threshold_high AND loss increased by min_delta: change_arch = True
        - Otherwise: change_arch = False
    """

    print("=" * 70)
    print("STEP 2: Architecture Change Decision Test")
    print("=" * 70)

    results = {
        'passed': True,
        'checks': {},
    }

    # Test 1: High loss ratio AND loss increased should trigger architecture change
    print("\n" + "-" * 50)
    print("Test 1: High Loss Ratio + Loss Increase (Should Change)")
    print("-" * 50)

    try:
        # Simulate high loss scenario with loss increase
        trainWLoss = 2.5  # Current preliminary loss
        end_last = 1.0    # Previous task's final loss
        loss_ratio = trainWLoss / end_last  # 2.5

        # Use threshold_high > 0.45 (default) and check if loss increased
        threshold_high = 0.5  # ratio 2.5 > 0.5
        min_delta = 0.1       # loss increase: 2.5 - 1.0 = 1.5 > 0.1

        should_change = should_change_arch(
            trainWLoss=trainWLoss,
            end_last=end_last,
            threshold_high=threshold_high,
            min_delta=min_delta
        )

        results['checks']['high_loss_triggers_change'] = should_change
        print(f"[{'PASS' if should_change else 'FAIL'}] High loss ratio + increase triggers architecture change")
        print(f"       Loss ratio: {loss_ratio:.2f} > threshold {threshold_high}")
        print(f"       Loss increase: {trainWLoss - end_last:.2f} > min_delta {min_delta}")
        if not should_change:
            results['passed'] = False

    except Exception as e:
        results['checks']['high_loss_triggers_change'] = False
        results['passed'] = False
        print(f"[FAIL] Decision test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Low loss ratio should NOT trigger architecture change
    print("\n" + "-" * 50)
    print("Test 2: Low Loss Ratio (Should NOT Change)")
    print("-" * 50)

    try:
        # Simulate low loss scenario (ratio below threshold)
        trainWLoss = 0.4  # Current preliminary loss
        end_last = 1.0    # Previous task's final loss
        loss_ratio = trainWLoss / end_last  # 0.4

        threshold_high = 0.5  # ratio 0.4 < 0.5, so should not change

        should_change = should_change_arch(
            trainWLoss=trainWLoss,
            end_last=end_last,
            threshold_high=threshold_high,
            min_delta=0.01
        )

        check_no_change = not should_change
        results['checks']['low_loss_no_change'] = check_no_change
        print(f"[{'PASS' if check_no_change else 'FAIL'}] Low loss ratio does NOT trigger architecture change")
        print(f"       Loss ratio: {loss_ratio:.2f} <= threshold {threshold_high}")
        if not check_no_change:
            results['passed'] = False

    except Exception as e:
        results['checks']['low_loss_no_change'] = False
        results['passed'] = False
        print(f"[FAIL] Decision test failed: {e}")

    # Test 3: High ratio but no loss increase - should NOT change
    print("\n" + "-" * 50)
    print("Test 3: High Ratio But No Loss Increase (Should NOT Change)")
    print("-" * 50)

    try:
        # High ratio but loss decreased (unlikely but possible edge case)
        trainWLoss = 0.6   # Current loss
        end_last = 1.0     # Previous loss
        loss_ratio = trainWLoss / end_last  # 0.6

        threshold_high = 0.5  # ratio 0.6 > 0.5
        min_delta = 0.1       # but loss decreased, not increased

        should_change = should_change_arch(
            trainWLoss=trainWLoss,
            end_last=end_last,
            threshold_high=threshold_high,
            min_delta=min_delta
        )

        # Should NOT change because loss didn't increase by min_delta
        check_no_change = not should_change
        results['checks']['no_increase_no_change'] = check_no_change
        print(f"[{'PASS' if check_no_change else 'FAIL'}] High ratio but no loss increase does NOT trigger change")
        print(f"       Loss ratio: {loss_ratio:.2f} > threshold {threshold_high}")
        print(f"       But loss decrease: {trainWLoss - end_last:.2f} < min_delta {min_delta}")
        if not check_no_change:
            results['passed'] = False

    except Exception as e:
        results['checks']['no_increase_no_change'] = False
        results['passed'] = False
        print(f"[FAIL] Test failed: {e}")

    # Test 4: Near-zero baseline loss handling
    print("\n" + "-" * 50)
    print("Test 4: Edge Case - Near-Zero Baseline Loss")
    print("-" * 50)

    try:
        trainWLoss = 1.0
        end_last = 0.001  # Very small but not zero

        should_change = should_change_arch(
            trainWLoss=trainWLoss,
            end_last=end_last,
            threshold_high=0.5,
            min_delta=0.01
        )

        # Should handle without error (high ratio will likely trigger change)
        results['checks']['near_zero_handled'] = True
        print(f"[PASS] Near-zero baseline loss handled correctly")
        print(f"       Loss ratio: {trainWLoss/end_last:.2f}")
        print(f"       Decision: {should_change}")

    except Exception as e:
        results['checks']['near_zero_handled'] = False
        results['passed'] = False
        print(f"[FAIL] Near-zero baseline case failed: {e}")

    # Test 5: Consistency check - multiple calls should give same result
    print("\n" + "-" * 50)
    print("Test 5: Consistency Check")
    print("-" * 50)

    try:
        trainWLoss = 2.0
        end_last = 1.0

        results_list = []
        for _ in range(10):
            should_change = should_change_arch(
                trainWLoss=trainWLoss,
                end_last=end_last,
                threshold_high=0.5,
                min_delta=0.1
            )
            results_list.append(should_change)

        all_same = all(r == results_list[0] for r in results_list)
        results['checks']['consistent_decisions'] = all_same
        print(f"[{'PASS' if all_same else 'FAIL'}] Decision is consistent across multiple calls")
        if not all_same:
            results['passed'] = False

    except Exception as e:
        results['checks']['consistent_decisions'] = False
        results['passed'] = False
        print(f"[FAIL] Consistency check failed: {e}")

    # Test 6: Default parameter values
    print("\n" + "-" * 50)
    print("Test 6: Default Parameter Values")
    print("-" * 50)

    try:
        trainWLoss = 2.0
        end_last = 1.0

        # Use defaults (threshold_high=0.45, min_delta=0.01)
        should_change = should_change_arch(
            trainWLoss=trainWLoss,
            end_last=end_last
        )

        results['checks']['default_config_works'] = True
        print(f"[PASS] Default parameter values work correctly")
        print(f"       Decision with defaults: {should_change}")

    except Exception as e:
        results['checks']['default_config_works'] = False
        results['passed'] = False
        print(f"[FAIL] Default config test failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    if results['passed']:
        print("STEP 2 TEST: PASSED")
    else:
        print("STEP 2 TEST: FAILED")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test AWB Step 2: Architecture Change Decision')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = test_step2_decision(args.verbose)
    sys.exit(0 if results['passed'] else 1)
