"""
Unit tests for AWB (Adaptive Weight Basis) utility functions.
Tests the helper functions in training/awb_utils.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from training.awb_utils import (
    compute_avg_loss,
    should_change_arch,
    compute_ab_threshold,
    set_new_AB_matrices,
    compute_V_from_AWB,
    partition_for_AB_training,
    partition_for_standard_training,
    save_layer_weights,
    restore_layer_weights,
    create_optimizer_for_phase,
)
from utils.model import MLP


class TestComputeAvgLoss:
    """Tests for compute_avg_loss function."""

    def test_compute_avg_loss_basic(self):
        """Test basic loss computation from record dict."""
        # Create a mock record dict with loss values
        # Format: record_dict["train{epoch}"] = (V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)
        record_dict = {}
        task_id = 0
        epochs = 100

        # Populate with known values - loss (V) is index 0
        for i in range(epochs):
            record_dict[f"train{i}"] = (float(i) * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Compute average of last 10 epochs (90-99)
        avg_loss = compute_avg_loss(record_dict, task_id, epochs, window=10)

        # Expected: mean of [0.90, 0.91, 0.92, ..., 0.99] = 0.945
        expected = np.mean([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
        assert abs(avg_loss - expected) < 1e-6

    def test_compute_avg_loss_empty_dict(self):
        """Test with empty record dict returns inf."""
        record_dict = {}
        avg_loss = compute_avg_loss(record_dict, 0, 100, window=10)
        assert avg_loss == float('inf')

    def test_compute_avg_loss_partial_window(self):
        """Test when fewer entries than window size."""
        record_dict = {
            "train0": (1.0, 0, 0, 0, 0, 0, 0),
            "train1": (2.0, 0, 0, 0, 0, 0, 0),
            "train2": (3.0, 0, 0, 0, 0, 0, 0),
        }
        # Window of 10 but only 3 entries exist
        # compute_avg_loss looks for train{(task_id+1)*epochs - j} for j in range(1, window+1)
        # For task_id=0, epochs=3: looks for train2, train1, train0
        avg_loss = compute_avg_loss(record_dict, 0, 3, window=10)
        # Should average the existing entries: (1.0 + 2.0 + 3.0) / 3 = 2.0
        assert avg_loss == 2.0


class TestShouldChangeArch:
    """Tests for should_change_arch decision function."""

    def test_should_change_arch_high_ratio_loss_increased(self):
        """Test: high ratio AND loss increased -> should change."""
        trainWLoss = 1.0
        end_last0 = 0.5   # ratio = 1.0/0.5 = 2.0 > 0.45
        end_last = 0.8    # end_last + 0.01 = 0.81 <= 1.0

        result = should_change_arch(trainWLoss, end_last0, end_last)
        assert result == True

    def test_should_change_arch_high_ratio_loss_not_increased(self):
        """Test: high ratio but loss didn't increase significantly -> no change."""
        trainWLoss = 0.82
        end_last0 = 0.5   # ratio = 0.82/0.5 = 1.64 > 0.45
        end_last = 0.82   # end_last + 0.01 = 0.83 > 0.82

        result = should_change_arch(trainWLoss, end_last0, end_last)
        assert result == False

    def test_should_change_arch_low_ratio(self):
        """Test: low ratio -> no change regardless of other conditions."""
        trainWLoss = 0.2
        end_last0 = 0.5   # ratio = 0.2/0.5 = 0.4 <= 0.45
        end_last = 0.1

        result = should_change_arch(trainWLoss, end_last0, end_last)
        assert result == False

    def test_should_change_arch_custom_thresholds(self):
        """Test with custom threshold values."""
        trainWLoss = 0.6
        end_last0 = 1.0   # ratio = 0.6
        end_last = 0.5

        # With threshold_high=0.5, ratio (0.6) > 0.5
        # And end_last + 0.01 = 0.51 < 0.6
        result = should_change_arch(trainWLoss, end_last0, end_last,
                                    threshold_high=0.5, min_delta=0.01)
        assert result == True


class TestComputeAbThreshold:
    """Tests for compute_ab_threshold function."""

    def test_threshold_high_ratio(self):
        """Test threshold for ratio > 3.0."""
        threshold = compute_ab_threshold(4.0, 1.0)  # ratio = 4.0
        assert threshold == max(1/4.0, 0.45)
        assert threshold == 0.45

    def test_threshold_medium_high_ratio(self):
        """Test threshold for 2.0 <= ratio < 3.0."""
        threshold = compute_ab_threshold(2.5, 1.0)  # ratio = 2.5
        assert threshold == min(1/2.5, 0.6)
        assert threshold == 0.4

    def test_threshold_medium_ratio(self):
        """Test threshold for 1.0 <= ratio < 2.0."""
        threshold = compute_ab_threshold(1.5, 1.0)  # ratio = 1.5
        expected = min(1/1.5, 0.75)
        assert abs(threshold - expected) < 1e-6

    def test_threshold_low_ratio(self):
        """Test threshold for ratio < 1.0."""
        threshold = compute_ab_threshold(0.5, 1.0)  # ratio = 0.5
        assert threshold == 0.8


class TestSetNewABMatrices:
    """Tests for set_new_AB_matrices function."""

    def test_set_new_AB_matrices_shape(self):
        """Test that A/B matrices have correct shapes after architecture change."""
        # Create a simple MLP
        original_arch = [3, 10, 10, 2]
        model = MLP(sizes=original_arch)

        # New architecture
        new_arch = [3, 15, 20, 2]

        # Set new A/B matrices
        updated_model = set_new_AB_matrices(model, original_arch, new_arch)

        # Check sizes updated
        assert updated_model.sizes == new_arch

        # Check A matrix shapes: A[i] should be (new_out, old_out)
        assert updated_model.A[0].shape == (15, 10)  # Layer 0: (15, 10)
        assert updated_model.A[1].shape == (20, 10)  # Layer 1: (20, 10)
        assert updated_model.A[2].shape == (2, 2)    # Layer 2: (2, 2)

        # Check B matrix shapes: B[i] should be (new_in, old_in)
        assert updated_model.B[0].shape == (3, 3)    # Layer 0: (3, 3)
        assert updated_model.B[1].shape == (15, 10)  # Layer 1: (15, 10)
        assert updated_model.B[2].shape == (20, 10)  # Layer 2: (20, 10)

    def test_set_new_AB_matrices_same_arch(self):
        """Test when architecture doesn't change."""
        original_arch = [3, 10, 10, 2]
        model = MLP(sizes=original_arch)

        # Same architecture
        new_arch = [3, 10, 10, 2]

        updated_model = set_new_AB_matrices(model, original_arch, new_arch)

        # Sizes should remain the same
        assert updated_model.sizes == original_arch


class TestComputeVFromAWB:
    """Tests for compute_V_from_AWB function."""

    def test_compute_V_from_AWB_transformation(self):
        """Test that V = A @ W @ B.T is computed correctly."""
        # Create MLP with identity A and B matrices for easy verification
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes)

        # Set A and B to identity matrices
        A_list = [jnp.eye(3, 3), jnp.eye(2, 2)]
        B_list = [jnp.eye(2, 2), jnp.eye(3, 3)]

        model = eqx.tree_at(lambda x: x.A, model, A_list)
        model = eqx.tree_at(lambda x: x.B, model, B_list)

        # Store original weights
        original_w0 = model.layers[0].weight.copy()
        original_w1 = model.layers[1].weight.copy()

        # Compute V
        updated_model = compute_V_from_AWB(model)

        # With identity A and B, V should equal original W
        assert jnp.allclose(updated_model.layers[0].weight, original_w0)
        assert jnp.allclose(updated_model.layers[1].weight, original_w1)

    def test_compute_V_from_AWB_non_identity(self):
        """Test V computation with non-identity A and B."""
        sizes = [2, 2, 2]
        model = MLP(sizes=sizes)

        # Set specific A and B
        A_list = [2.0 * jnp.eye(2, 2), 2.0 * jnp.eye(2, 2)]
        B_list = [0.5 * jnp.eye(2, 2), 0.5 * jnp.eye(2, 2)]

        model = eqx.tree_at(lambda x: x.A, model, A_list)
        model = eqx.tree_at(lambda x: x.B, model, B_list)

        original_w0 = model.layers[0].weight.copy()

        updated_model = compute_V_from_AWB(model)

        # V = A @ W @ B.T = 2*I @ W @ 0.5*I.T = W (scaling cancels out)
        expected_v0 = A_list[0] @ original_w0 @ B_list[0].T
        assert jnp.allclose(updated_model.layers[0].weight, expected_v0)


class TestPartitionForABTraining:
    """Tests for partition_for_AB_training function."""

    def test_partition_for_AB_training(self):
        """Test that only A and B are trainable after partitioning."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes)

        diff_model, static_model = partition_for_AB_training(model)

        # A and B should be in diff_model (trainable)
        assert diff_model.A is not None
        assert diff_model.B is not None

        # Layers should be in static_model (frozen)
        # The layer weights should be None in diff_model
        for layer in diff_model.layers:
            assert layer.weight is None
            assert layer.bias is None


class TestPartitionForStandardTraining:
    """Tests for partition_for_standard_training function."""

    def test_partition_for_standard_training(self):
        """Test that A and B are frozen after standard partitioning."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes)

        params, static = partition_for_standard_training(model)

        # A and B should be in static (frozen)
        assert static.A is not None
        assert static.B is not None

        # A and B should be None in params
        assert params.A is None
        assert params.B is None


class TestSaveRestoreLayerWeights:
    """Tests for save_layer_weights and restore_layer_weights functions."""

    def test_save_and_restore_weights(self):
        """Test that weights can be saved and restored correctly."""
        sizes = [2, 3, 2]
        model = MLP(sizes=sizes)

        # Save original weights
        weight_list, bias_list = save_layer_weights(model)

        # Modify model weights
        new_weights = [jnp.zeros_like(w) for w in weight_list]
        for j, w in enumerate(new_weights):
            model = eqx.tree_at(lambda x: x.layers[j].weight, model, w)

        # Verify weights changed
        assert not jnp.allclose(model.layers[0].weight, weight_list[0])

        # Restore weights
        restored_model = restore_layer_weights(model, weight_list, bias_list)

        # Verify weights are restored
        for j in range(len(weight_list)):
            assert jnp.allclose(restored_model.layers[j].weight, weight_list[j])
            assert jnp.allclose(restored_model.layers[j].bias, bias_list[j])


class TestCreateOptimizerForPhase:
    """Tests for create_optimizer_for_phase function."""

    def test_create_optimizer_standard(self):
        """Test standard phase optimizer creation."""
        optim = create_optimizer_for_phase('standard', learning_rate=1e-4)
        assert optim is not None

    def test_create_optimizer_ab_training(self):
        """Test AB training phase optimizer creation."""
        optim = create_optimizer_for_phase('ab_training', learning_rate=1e-4)
        assert optim is not None

    def test_create_optimizer_v_training(self):
        """Test V training phase optimizer creation."""
        optim = create_optimizer_for_phase('v_training', learning_rate=1e-4)
        assert optim is not None


class TestAWBIntegration:
    """Integration tests for the complete AWB workflow."""

    def test_full_awb_workflow(self):
        """Test the complete AWB workflow without actual training."""
        # 1. Create initial model
        original_arch = [3, 10, 10, 2]
        model = MLP(sizes=original_arch)

        # 2. Simulate architecture change decision
        trainWLoss = 1.0
        end_last0 = 0.5
        end_last = 0.8
        change_arch = should_change_arch(trainWLoss, end_last0, end_last)
        assert change_arch == True

        # 3. Save original weights
        weight_list, bias_list = save_layer_weights(model)

        # 4. New architecture from "search"
        new_arch = [3, 15, 15, 2]

        # 5. Set new A/B matrices
        model = set_new_AB_matrices(model, original_arch, new_arch)
        assert model.sizes == new_arch

        # 6. Partition for AB training
        diff_model, static_model = partition_for_AB_training(model)
        assert diff_model.A is not None
        assert diff_model.B is not None

        # 7. After AB training, combine and compute V
        model = eqx.combine(diff_model, static_model)
        model = compute_V_from_AWB(model)

        # 8. Partition for standard training
        params, static = partition_for_standard_training(model)
        assert params.A is None  # A/B frozen
        assert params.B is None

        # 9. Verify model can still do forward pass
        x = jnp.ones((3,))  # Single sample
        final_model = eqx.combine(params, static)
        output = final_model(x)
        # Model output can be (2,) or (1, 2) depending on implementation
        assert output.shape[-1] == 2  # Last dimension should be output size

    def test_awb_disabled_path(self):
        """Test that AWB disabled path works correctly."""
        # When AWB is disabled, should_change_arch conditions shouldn't trigger
        trainWLoss = 0.2
        end_last0 = 0.5  # ratio = 0.4 < 0.45
        end_last = 0.3

        change_arch = should_change_arch(trainWLoss, end_last0, end_last)
        assert change_arch == False
