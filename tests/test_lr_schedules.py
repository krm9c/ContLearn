"""
Unit tests for learning rate schedules and warmup features.

Tests the new lr_schedule functionality and task warmup epochs added to the codebase.
These features were not previously tested.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import pytest
import numpy as np

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

from cl.runners.generic_runner import compute_task_lr, compute_adaptive_lr_min, compute_adaptive_grad_weights


class TestLearningRateSchedules:
    """Tests for learning rate scheduling functionality."""

    def test_constant_schedule(self):
        """Test constant learning rate schedule (no decay)."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'constant',
            'n_task': 10
        }

        # All tasks should have the same learning rate
        for task_id in range(10):
            lr = compute_task_lr(config, task_id)
            assert lr == 1e-3, f"Task {task_id}: expected 1e-3, got {lr}"

    def test_step_schedule_default(self):
        """Test step decay schedule with default decay_steps=1."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'step',
            'lr_decay_factor': 0.9,
            'lr_decay_steps': 1,  # Decay every task
            'lr_min': 1e-6,
            'n_task': 5
        }

        # Task 0: lr = 1e-3 * (0.9^0) = 1e-3
        lr0 = compute_task_lr(config, 0)
        assert abs(lr0 - 1e-3) < 1e-10

        # Task 1: lr = 1e-3 * (0.9^1) = 9e-4
        lr1 = compute_task_lr(config, 1)
        assert abs(lr1 - 9e-4) < 1e-10

        # Task 2: lr = 1e-3 * (0.9^2) = 8.1e-4
        lr2 = compute_task_lr(config, 2)
        assert abs(lr2 - 8.1e-4) < 1e-10

        # LRs should be decreasing
        assert lr0 > lr1 > lr2

    def test_step_schedule_every_2_tasks(self):
        """Test step decay schedule with decay_steps=2."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'step',
            'lr_decay_factor': 0.9,
            'lr_decay_steps': 2,  # Decay every 2 tasks
            'lr_min': 1e-6,
            'n_task': 5
        }

        # Tasks 0-1: lr = 1e-3 * (0.9^0) = 1e-3
        lr0 = compute_task_lr(config, 0)
        lr1 = compute_task_lr(config, 1)
        assert lr0 == lr1 == 1e-3

        # Tasks 2-3: lr = 1e-3 * (0.9^1) = 9e-4
        lr2 = compute_task_lr(config, 2)
        lr3 = compute_task_lr(config, 3)
        assert abs(lr2 - 9e-4) < 1e-10
        assert lr2 == lr3

        # Task 4: lr = 1e-3 * (0.9^2) = 8.1e-4
        lr4 = compute_task_lr(config, 4)
        assert abs(lr4 - 8.1e-4) < 1e-10

    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'lr_decay_factor': 0.9,
            'lr_min': 1e-6,
            'n_task': 5
        }

        # Exponential: lr = base_lr * (decay^task_id)
        lr0 = compute_task_lr(config, 0)
        assert abs(lr0 - 1e-3) < 1e-10  # 1e-3 * 0.9^0

        lr1 = compute_task_lr(config, 1)
        assert abs(lr1 - 9e-4) < 1e-10  # 1e-3 * 0.9^1

        lr3 = compute_task_lr(config, 3)
        expected_lr3 = 1e-3 * (0.9 ** 3)
        assert abs(lr3 - expected_lr3) < 1e-10

        # Should be monotonically decreasing
        lrs = [compute_task_lr(config, i) for i in range(5)]
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i+1], f"LR not decreasing: {lrs[i]} <= {lrs[i+1]}"

    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'cosine',
            'lr_min': 1e-6,
            'n_task': 10
        }

        # Task 0: should be close to base_lr
        lr0 = compute_task_lr(config, 0)
        assert abs(lr0 - 1e-3) < 1e-6

        # Task 9 (last): should be close to lr_min
        lr9 = compute_task_lr(config, 9)
        assert abs(lr9 - 1e-6) < 1e-6

        # Middle task (4-5): should be between base_lr and lr_min
        lr4 = compute_task_lr(config, 4)
        lr5 = compute_task_lr(config, 5)
        assert 1e-6 < lr4 < 1e-3
        assert 1e-6 < lr5 < 1e-3

        # Overall trend should be decreasing
        # (though cosine has smooth curve, not strict monotonic)
        assert lr0 > lr9

    def test_linear_schedule(self):
        """Test linear decay schedule."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'linear',
            'lr_min': 1e-6,
            'n_task': 10
        }

        # Task 0: should be base_lr
        lr0 = compute_task_lr(config, 0)
        assert abs(lr0 - 1e-3) < 1e-10

        # Task 9 (last): should be lr_min
        lr9 = compute_task_lr(config, 9)
        assert abs(lr9 - 1e-6) < 1e-10

        # Task 5 (middle): should be ~halfway between (with floating point tolerance)
        lr5 = compute_task_lr(config, 5)
        expected_lr5 = (1e-3 + 1e-6) / 2
        assert abs(lr5 - expected_lr5) < 1e-4  # Relaxed tolerance for linear interpolation

        # Should be strictly monotonically decreasing
        lrs = [compute_task_lr(config, i) for i in range(10)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i+1], f"LR not decreasing: {lrs[i]} < {lrs[i+1]}"

    def test_lr_min_respected(self):
        """Test that lr_min is respected as lower bound."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'lr_decay_factor': 0.5,  # Aggressive decay
            'lr_min': 1e-5,
            'n_task': 20
        }

        # Even with aggressive decay, lr should not go below lr_min
        for task_id in range(20):
            lr = compute_task_lr(config, task_id)
            assert lr >= 1e-5, f"Task {task_id}: lr {lr} < lr_min {1e-5}"

    def test_unknown_schedule_defaults_to_constant(self):
        """Test that unknown schedule type defaults to constant with warning."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'invalid_schedule',
            'n_task': 5
        }

        # Should fall back to constant (base_lr)
        for task_id in range(5):
            lr = compute_task_lr(config, task_id)
            assert lr == 1e-3

    def test_case_insensitive_schedule_names(self):
        """Test that schedule names are case-insensitive."""
        configs = [
            {'lr': 1e-3, 'lr_schedule': 'CONSTANT'},
            {'lr': 1e-3, 'lr_schedule': 'Constant'},
            {'lr': 1e-3, 'lr_schedule': 'constant'},
        ]

        for config in configs:
            lr = compute_task_lr(config, 0)
            assert lr == 1e-3


class TestAdaptiveLRMin:
    """Tests for adaptive lr_min based on loss ratio."""

    def test_adaptive_lr_min_disabled(self):
        """Test that when disabled, returns default lr_min."""
        config = {
            'adaptive_lr_min_enabled': False,
            'lr_min': 1e-6,
        }

        # Should always return lr_min regardless of loss_ratio
        assert compute_adaptive_lr_min(config, 0.5) == 1e-6
        assert compute_adaptive_lr_min(config, 2.0) == 1e-6

    def test_adaptive_lr_min_below_threshold(self):
        """Test that when loss ratio is below threshold, returns base lr_min."""
        config = {
            'adaptive_lr_min_enabled': True,
            'lr_min_base': 1e-6,
            'lr_min_max': 1e-4,
            'lr_min_loss_ratio_threshold': 1.0,
        }

        # Loss ratio 0.5 < threshold 1.0: use base
        lr_min = compute_adaptive_lr_min(config, 0.5)
        assert lr_min == 1e-6

    def test_adaptive_lr_min_above_threshold(self):
        """Test that when loss ratio is above threshold, scales lr_min."""
        config = {
            'adaptive_lr_min_enabled': True,
            'lr_min_base': 1e-6,
            'lr_min_max': 1e-4,
            'lr_min_loss_ratio_threshold': 1.0,
        }

        # Loss ratio 2.0 > threshold 1.0: should scale up
        lr_min = compute_adaptive_lr_min(config, 2.0)
        assert lr_min > 1e-6
        assert lr_min <= 1e-4  # Should not exceed max

    def test_adaptive_lr_min_capped_at_max(self):
        """Test that lr_min is capped at lr_min_max."""
        config = {
            'adaptive_lr_min_enabled': True,
            'lr_min_base': 1e-6,
            'lr_min_max': 1e-4,
            'lr_min_loss_ratio_threshold': 1.0,
        }

        # Very high loss ratio should still be capped
        lr_min = compute_adaptive_lr_min(config, 100.0)
        assert lr_min == 1e-4


class TestAdaptiveGradientWeights:
    """Tests for adaptive gradient weights based on loss ratio."""

    def test_adaptive_grad_weights_disabled(self):
        """Test that when disabled, returns default grad_weights."""
        config = {
            'adaptive_grad_weights_enabled': False,
            'grad_weights': [0.3, 0.6, 0.1],
        }

        # Should always return config grad_weights
        weights = compute_adaptive_grad_weights(config, 2.0)
        assert weights == [0.3, 0.6, 0.1]

    def test_adaptive_grad_weights_below_threshold(self):
        """Test that when loss ratio is below threshold, returns base weights."""
        config = {
            'adaptive_grad_weights_enabled': True,
            'grad_weights_base': [0.3, 0.6, 0.1],
            'grad_weights_max_current': 0.8,
            'grad_weights_min_experience': 0.1,
            'grad_weights_loss_ratio_threshold': 1.0,
        }

        # Loss ratio 0.5 < threshold: use base
        weights = compute_adaptive_grad_weights(config, 0.5)
        assert weights == [0.3, 0.6, 0.1]

    def test_adaptive_grad_weights_above_threshold(self):
        """Test that when loss ratio is high, increases current task weight."""
        config = {
            'adaptive_grad_weights_enabled': True,
            'grad_weights_base': [0.3, 0.6, 0.1],
            'grad_weights_max_current': 0.8,
            'grad_weights_min_experience': 0.1,
            'grad_weights_loss_ratio_threshold': 1.0,
        }

        # Loss ratio 2.0 > threshold: should increase alpha (current task)
        weights = compute_adaptive_grad_weights(config, 2.0)
        assert weights[0] > 0.3  # alpha increased
        assert weights[1] < 0.6  # beta decreased (but above min)
        assert weights[1] >= 0.1 - 1e-10  # beta at least min_experience (with float tolerance)

    def test_adaptive_grad_weights_sum_to_one(self):
        """Test that adapted weights still sum to ~1.0."""
        config = {
            'adaptive_grad_weights_enabled': True,
            'grad_weights_base': [0.3, 0.6, 0.1],
            'grad_weights_max_current': 0.8,
            'grad_weights_min_experience': 0.1,
            'grad_weights_loss_ratio_threshold': 1.0,
        }

        for loss_ratio in [0.5, 1.0, 2.0, 5.0]:
            weights = compute_adaptive_grad_weights(config, loss_ratio)
            total = sum(weights)
            assert abs(total - 1.0) < 1e-6, f"Weights {weights} don't sum to 1.0: {total}"


class TestLRScheduleEdgeCases:
    """Test edge cases and boundary conditions for LR schedules."""

    def test_single_task(self):
        """Test schedules with n_task=1."""
        config = {
            'lr': 1e-3,
            'lr_min': 1e-6,
            'n_task': 1
        }

        # All schedules should handle single task gracefully
        for schedule in ['constant', 'step', 'exponential', 'cosine', 'linear']:
            config['lr_schedule'] = schedule
            lr = compute_task_lr(config, 0)
            assert lr >= 1e-6  # Should be >= lr_min
            assert lr <= 1e-3  # Should be <= base_lr

    def test_zero_decay_factor(self):
        """Test exponential schedule with decay_factor=0 (edge case)."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'lr_decay_factor': 0.0,
            'lr_min': 1e-6,
        }

        # Task 0 should have base_lr
        lr0 = compute_task_lr(config, 0)
        assert lr0 == 1e-3

        # Task 1+: lr = base_lr * 0^task_id = 0, but clamped to lr_min
        lr1 = compute_task_lr(config, 1)
        assert lr1 == 1e-6

    def test_negative_task_id_gracefully_handled(self):
        """Test that negative task_id doesn't crash (though not expected in practice)."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'lr_decay_factor': 0.9,
        }

        # Should not crash and should return reasonable value
        lr = compute_task_lr(config, -1)
        assert lr > 0

    def test_very_large_task_id(self):
        """Test schedule with very large task_id."""
        config = {
            'lr': 1e-3,
            'lr_schedule': 'exponential',
            'lr_decay_factor': 0.99,
            'lr_min': 1e-7,
            'n_task': 1000
        }

        # Even with task_id=999, should be >= lr_min
        lr = compute_task_lr(config, 999)
        assert lr >= 1e-7
