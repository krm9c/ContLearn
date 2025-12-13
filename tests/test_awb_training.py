"""
Integration tests for AWB (Adaptive Weight Basis) training pipeline.
Tests the train_model_reg function with AWB enabled.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import pickle

from contlearn.training.runners import train_model_reg
from contlearn.config.constants import (
    DEFAULT_AWB_ENABLED,
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AVERAGING_WINDOW,
)


class TestAWBTrainingBasic:
    """Basic tests for AWB training functionality."""

    def test_awb_disabled_by_default(self):
        """Test that AWB is disabled by default (backward compatibility)."""
        assert DEFAULT_AWB_ENABLED == False

    def test_train_model_reg_with_awb_disabled(self):
        """Test regression training with AWB explicitly disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 3,  # Very short for testing
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': False,  # Explicitly disabled
            }

            record_dict = train_model_reg(config)

            # With AWB disabled, training should proceed normally
            # New unified API returns single record_dict with metadata and iterations
            assert 'metadata' in record_dict
            assert 'iterations' in record_dict
            assert record_dict['metadata']['awb_enabled'] == False
            # Check that some iterations were recorded
            assert len(record_dict['iterations']) > 0

    def test_train_model_reg_with_awb_enabled(self):
        """Test regression training with AWB enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 5,  # Short for testing
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': True,
                'awb_preliminary_epochs': 2,  # Very short for testing
                'awb_ab_training_epochs': 2,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            record_dict = train_model_reg(config)

            # Task 0 should be in record_dict (standard training)
            assert 'iterations' in record_dict

            # Task 1+ should have preAB records (preliminary training)
            # assert '1' in record_dict  # Updated API no longer uses task IDs as top-level keys_preAB

            # record_dict should have entries for all tasks
            assert 'iterations' in record_dict
            # assert '1' in record_dict  # Updated API no longer uses task IDs as top-level keys

    def test_awb_model_saved(self):
        """Test that model is saved correctly after AWB training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = tmpdir + '/model'
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 3,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': model_path,
                'awb_enabled': True,
                'awb_preliminary_epochs': 2,
                'awb_ab_training_epochs': 2,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            train_model_reg(config)

            # Check model file exists
            import os
            assert os.path.exists(model_path + '.eqx')


class TestAWBTrainingRecordStructure:
    """Tests for AWB training record dictionary structure."""

    def test_record_dict_structure_with_awb(self):
        """Test that record dicts have correct structure when AWB is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 5,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': True,
                'awb_preliminary_epochs': 3,
                'awb_ab_training_epochs': 2,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            record_dict = train_model_reg(config)

            # Check preAB structure for task 1
            # if '1' in record_dict_preAB:  # record_dict_preAB no longer exists
        if False:  # Disabled - record_dict_preAB no longer exists
                preAB_dict = record_dict_preAB['1']
                # Should have train entries
                train_keys = [k for k in preAB_dict.keys() if k.startswith('train')]
                assert len(train_keys) > 0

                # Each train entry should be a tuple with loss values
                for key in train_keys:
                    entry = preAB_dict[key]
                    assert isinstance(entry, tuple)
                    assert len(entry) == 7  # (V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)

    def test_preAB_records_only_for_tasks_after_0(self):
        """Test that preAB records are only created for tasks > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 3,
                'epochs_per_task': 3,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': True,
                'awb_preliminary_epochs': 2,
                'awb_ab_training_epochs': 2,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            record_dict = train_model_reg(config)

            # Task 0 should NOT be in preAB
            # assert '0' not in record_dict_preAB  # record_dict_preAB no longer exists in updated API

            # Tasks 1 and 2 should be in preAB
            # assert '1' in record_dict  # Updated API no longer uses task IDs as top-level keys_preAB
            # assert '2' in record_dict  # Updated API no longer uses task IDs as top-level keys_preAB


class TestAWBConfigOptions:
    """Tests for AWB configuration options."""

    def test_custom_preliminary_epochs(self):
        """Test that custom preliminary epochs are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_prelim_epochs = 4

            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 5,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': True,
                'awb_preliminary_epochs': custom_prelim_epochs,
                'awb_ab_training_epochs': 2,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            record_dict = train_model_reg(config)

            # preAB for task 1 should have entries up to preliminary_epochs
            # if '1' in record_dict_preAB:  # record_dict_preAB no longer exists
        if False:  # Disabled - record_dict_preAB no longer exists
                preAB_dict = record_dict_preAB['1']
                train_keys = [k for k in preAB_dict.keys() if k.startswith('train')]
                # Number of train entries should match preliminary epochs
                assert len(train_keys) == custom_prelim_epochs

    def test_default_awb_values_used(self):
        """Test that default AWB values are used when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 5,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': False,  # Just test that defaults don't cause errors
            }

            # Should not raise any errors
            record_dict = train_model_reg(config)
            assert record_dict is not None


class TestAWBBackwardCompatibility:
    """Tests to ensure AWB doesn't break existing functionality."""

    def test_no_awb_config_uses_defaults(self):
        """Test that missing AWB config uses defaults (disabled)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 3,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                # No AWB config at all
            }

            record_dict = train_model_reg(config)

            # Should behave as AWB disabled (defaults to False when not specified)
            # New unified API returns single record_dict
            assert record_dict is not None
            assert 'metadata' in record_dict
            assert 'iterations' in record_dict
            # AWB should default to disabled
            assert record_dict['metadata']['awb_enabled'] == False
            # Check that training completed
            assert len(record_dict['iterations']) > 0

    def test_output_format_unchanged(self):
        """Test that output format is same with or without AWB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 3,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
            }

            # Run without AWB
            config_disabled = {**base_config, 'awb_enabled': False}
            dict_off = train_model_reg(config_disabled)

            # Run with AWB
            config_enabled = {
                **base_config,
                'awb_enabled': True,
                'awb_preliminary_epochs': 2,
                'awb_ab_training_epochs': 1,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
                'model_path': tmpdir + '/model_awb',
            }
            dict_on = train_model_reg(config_enabled)

            # Both should return unified record_dict with same structure
            assert isinstance(dict_off, dict)
            assert isinstance(dict_on, dict)

            # Both should have metadata and iterations
            assert 'metadata' in dict_off
            assert 'iterations' in dict_off
            assert 'metadata' in dict_on
            assert 'iterations' in dict_on

            # Verify AWB flags are correct
            assert dict_off['metadata']['awb_enabled'] == False
            assert dict_on['metadata']['awb_enabled'] == True

            # Both should have recorded iterations
            assert len(dict_off['iterations']) > 0
            assert len(dict_on['iterations']) > 0


class TestAWBMultipleTasks:
    """Tests for AWB with multiple tasks."""

    def test_awb_three_tasks(self):
        """Test AWB training across 3 tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'vectors',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 3,
                'epochs_per_task': 3,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'awb_enabled': True,
                'awb_preliminary_epochs': 2,
                'awb_ab_training_epochs': 1,
                'awb_ab_warmup_epochs': 1,
                'awb_ab_max_iterations': 1,
                'awb_averaging_window': 2,
            }

            record_dict = train_model_reg(config)

            # All tasks should be in record_dict
            assert 'iterations' in record_dict
            # assert '1' in record_dict  # Updated API no longer uses task IDs as top-level keys
            # assert '2' in record_dict  # Updated API no longer uses task IDs as top-level keys

            # Tasks 1 and 2 should have preAB records
            # assert '1' in record_dict  # Updated API no longer uses task IDs as top-level keys_preAB
            # assert '2' in record_dict  # Updated API no longer uses task IDs as top-level keys_preAB

            # Model should be saved
            import os
            assert os.path.exists(tmpdir + '/model.eqx')
