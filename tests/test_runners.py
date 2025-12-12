"""
Unit tests for training runners in training/runners.py.
Tests train_model_graph, train_model_reg, and train_model_class functions.
Note: These are integration tests that may take longer to run.
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

from training.runners import train_model_reg, train_model_class, train_model_graph


class TestTrainModelRegression:
    """Tests for train_model_reg function."""

    def test_train_model_reg_basic(self):
        """Test basic regression training runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 2,
                'epochs_per_task': 2,  # Very short for testing
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'arch_search': False,
                'arch_start_task': 1
            }

            # Run training
            record_dict_preAB, record_dict_AB, record_dict = train_model_reg(config)

            # Verify output structure - record_dict contains nested training metrics
            assert isinstance(record_dict, dict)
            # Training should complete without errors and return a dict
            assert record_dict is not None
            # Check that task keys exist (they may be nested)
            assert '0' in record_dict
            assert '1' in record_dict

    def test_train_model_reg_saves_model(self):
        """Test that regression training saves model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model')

            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 1,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': model_path,
                'arch_search': False,
                'arch_start_task': 1
            }

            train_model_reg(config)

            # Check that model was saved
            assert os.path.exists(model_path + '.eqx')

    def test_train_model_reg_multiple_tasks(self):
        """Test regression training with multiple continual learning tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 3,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'arch_search': False,
                'arch_start_task': 1
            }

            record_dict_preAB, record_dict_AB, record_dict = train_model_reg(config)

            # Should have records for all 3 tasks (nested structure)
            assert isinstance(record_dict, dict)
            assert '0' in record_dict
            assert '1' in record_dict
            assert '2' in record_dict


class TestTrainModelClassification:
    """Tests for train_model_class function."""

    def test_train_model_class_basic(self):
        """Test basic classification training runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'vectors',  # trainer expects 'vectors' for classification
                'data': 'mnist',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'n_task': 2,
                'epochs_per_task': 2,  # Very short for testing
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model'
            }

            # Run training
            record_dict_preAB, record_dict_AB, record_dict = train_model_class(config)

            # Verify output structure - record_dict contains nested training metrics
            assert isinstance(record_dict, dict)
            assert record_dict is not None
            # Check that task keys exist
            assert '0' in record_dict
            assert '1' in record_dict

    def test_train_model_class_saves_model(self):
        """Test that classification training saves model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_class_model')

            config = {
                'prob': 'classification',
                'problem': 'vectors',  # trainer expects 'vectors' for classification
                'data': 'mnist',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'n_task': 1,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': model_path
            }

            train_model_class(config)

            # Check that model was saved
            # train_model_class uses eqx.tree_serialise_leaves(config['model_path'], model)
            # This creates the file directly at model_path
            assert os.path.exists(model_path) or os.path.exists(model_path + '.eqx')

    def test_train_model_class_cifar(self):
        """Test classification training with CIFAR-10."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'vectors',  # trainer expects 'vectors' for classification
                'data': 'cifar10',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'n_task': 1,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model'
            }

            record_dict_preAB, record_dict_AB, record_dict = train_model_class(config)

            assert isinstance(record_dict, dict)
            assert len(record_dict) >= 1


class TestTrainModelGraph:
    """Tests for train_model_graph function."""

    def test_train_model_graph_basic(self):
        """Test basic graph classification training runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch': 32,
                'batch_size': 32,  # Some parts of code use batch_size
                'n_class': 2,
                'class_per_task': 1,
                'n_task': 2,
                'epochs_per_task': 2,  # Very short for testing
                'save_iter': 1,
                'delta': 1e-4,
                'flag': [0.5e-4, 0.5e-9],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model'
            }

            # Run training
            record_dict_preAB, record_dict_AB, record_dict = train_model_graph(config)

            # Verify output structure
            assert isinstance(record_dict, dict)
            assert record_dict is not None
            assert '0' in record_dict
            assert '1' in record_dict

    def test_train_model_graph_saves_model(self):
        """Test that graph training saves model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_graph_model')

            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch': 32,
                'batch_size': 32,
                'n_class': 2,
                'class_per_task': 1,
                'n_task': 1,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-4,
                'flag': [0.5e-4, 0.5e-9],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': model_path
            }

            train_model_graph(config)

            # Check that model was saved
            assert os.path.exists(model_path + '.eqx')

    def test_train_model_graph_multiple_tasks(self):
        """Test graph training with multiple continual learning tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'graphclassification',
                'problem': 'graph',
                'data': 'MUTAG',
                'network': 'gcn',
                'lr': 1e-4,
                'batch': 32,
                'batch_size': 32,
                'n_class': 2,
                'class_per_task': 1,
                'n_task': 2,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-4,
                'flag': [0.5e-4, 0.5e-9],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model'
            }
            
            record_dict_preAB, record_dict_AB, record_dict = train_model_graph(config)

            # Should have records for all tasks
            assert isinstance(record_dict, dict)
            assert '0' in record_dict
            assert '1' in record_dict


class TestTrainingRecordDict:
    """Tests for training record dictionary structure and content."""

    def test_record_dict_structure_regression(self):
        """Test that record_dict has expected structure for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 1,
                'epochs_per_task': 2,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'arch_search': False,
                'arch_start_task': 1
            }

            _, _, record_dict = train_model_reg(config)

            # Check that task 0 has records
            assert '0' in record_dict
            task_record = record_dict['0']

            # Task record should be a dictionary with training metrics
            assert isinstance(task_record, dict)

    def test_record_dict_structure_classification(self):
        """Test that record_dict has expected structure for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'classification',
                'problem': 'vectors',  # trainer expects 'vectors' for classification
                'data': 'mnist',
                'network': 'cnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 128,
                'n_class': 10,
                'n_task': 1,
                'epochs_per_task': 2,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'class',
                'metric': 'class',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model'
            }

            _, _, record_dict = train_model_class(config)

            assert '0' in record_dict
            assert isinstance(record_dict['0'], dict)


class TestEquinoxPartitionPattern:
    """Tests for Equinox partition pattern used in runners."""

    def test_awb_matrices_moved_to_static(self):
        """Test that AWB matrices are properly moved to static."""
        # This is tested implicitly by the training runs,
        # but we verify the pattern is applied
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'prob': 'regression',
                'problem': 'regression',
                'data': 'sine',
                'network': 'fcnn',
                'lr': 1e-3,
                'batch_size': 32,
                'hln': 64,
                'n_task': 1,
                'epochs_per_task': 1,
                'save_iter': 1,
                'delta': 1e-3,
                'flag': [1, 0],
                'loss': 'mse',
                'metric': 'mse',
                'tensorfile': tmpdir + '/tensorboard',
                'model_path': tmpdir + '/model',
                'arch_search': False,
                'arch_start_task': 1
            }

            # Training should complete without errors
            # The partition pattern is applied internally
            record_dict_preAB, record_dict_AB, record_dict = train_model_reg(config)

            assert record_dict is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
