"""
Integration tests for end-to-end training pipeline.
Tests the complete workflow from config to training to saving.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pickle
import pytest

# Added by Claude: Mark as unit test for test categorization
pytestmark = pytest.mark.unit

from cl.datasets import SineDataset
from cl.models import MLP, create_mlp
from cl.runners import train_model
from cl.core.trainer import Trainer


class TestTrainerMixins:
    """Tests for Trainer class that combines all mixins."""

    def test_trainer_initialization(self):
        """Test Trainer initializes correctly."""
        trainer = Trainer(loss='mse', problem='vectors', metric='mse')

        assert trainer.loss == 'mse'
        assert trainer.problem == 'vectors'
        assert trainer.metric == 'mse'

    def test_trainer_has_all_methods(self):
        """Test Trainer has methods from all mixins."""
        trainer = Trainer(loss='mse', problem='vectors', metric='mse')

        # From LossMixin
        assert hasattr(trainer, 'loss_fn_mse')
        assert hasattr(trainer, 'return_loss_grad')
        assert hasattr(trainer, 'return_metric')

        # From HamiltonianMixin
        assert hasattr(trainer, 'return_Hamiltonian_mse')
        assert hasattr(trainer, 'return_Hamiltonian_class')

        # From TrainingLoopsMixin
        assert hasattr(trainer, 'train__CL')

        # From RecordingMixin
        assert hasattr(trainer, 'record_metrics')
        assert hasattr(trainer, 'initialize_record_dict')


class TestSineRegressionPipeline:
    """Integration tests for sine regression training pipeline."""

    def test_basic_training_loop(self, test_sine_config, jax_key, tmp_path):
        """Test basic training loop execution using generic_runner."""
        # Added by Claude: Use generic runner's train_model() for end-to-end integration test
        config = test_sine_config.copy()
        config['model_path'] = str(tmp_path / "test_model")
        config['n_task'] = 1  # Single task for basic test

        # Run training through generic runner
        record_dict = train_model(config, run_id=0)

        # Verify training completed
        assert len(record_dict['iterations']) > 0
        assert 0 in record_dict['tasks']
        assert 'main_training' in record_dict['tasks'][0]

    def test_model_saving_and_loading(self, test_sine_config, jax_key, tmp_path):
        """Test model can be saved and loaded correctly."""
        config = test_sine_config.copy()
        model_path = str(tmp_path / "test_model.eqx")

        # Create dataset to get correct dimensions
        dataset = SineDataset(config)

        # Create and save model
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        # Use dataset properties to get correct dimensions from loaded data
        config['input_size'] = dataset.input_size
        config['output_size'] = dataset.output_size
        model = create_mlp(config)
        x = jnp.array(np.random.randn(3).astype(np.float32))
        output_before = model(x)

        eqx.tree_serialise_leaves(model_path, model)

        # Load model
        model_loaded = eqx.tree_deserialise_leaves(model_path, model)
        output_after = model_loaded(x)

        assert jnp.allclose(output_before, output_after)

    def test_multi_task_training(self, test_sine_config, jax_key, tmp_path):
        """Test training across multiple tasks using generic_runner."""
        # Added by Claude: Use generic runner's train_model() for end-to-end integration test
        config = test_sine_config.copy()
        config['model_path'] = str(tmp_path / "test_model")
        config['n_task'] = 2
        config['epochs_per_task'] = 2
        config['save_iter'] = 1

        # Run training through generic runner
        record_dict = train_model(config, run_id=0)

        # Verify multi-task training recorded
        assert len(record_dict['iterations']) > 0
        assert 0 in record_dict['tasks']
        assert 1 in record_dict['tasks']
        assert 'main_training' in record_dict['tasks'][0]
        assert 'main_training' in record_dict['tasks'][1]


class TestExperienceReplayIntegration:
    """Integration tests for experience replay functionality."""

    def test_experience_replay_in_training(self, test_exp_replay_config, jax_key, tmp_path):
        """Test experience replay is used during multi-task training."""
        config = test_exp_replay_config.copy()
        config['model_path'] = str(tmp_path / "test_model")

        dataset = SineDataset(config)

        # Train task 0
        dataset.load_task(0)
        dataset.append_to_experience(0)
        initial_exp_size = len(dataset.exp_x_train)

        # Train task 1
        dataset.load_task(1)
        dataset.append_to_experience(1)

        # Experience buffer should have grown
        assert len(dataset.exp_x_train) > initial_exp_size

        # Train task 2 - should trigger buffer limit
        dataset.load_task(2)
        dataset.append_to_experience(2)

        # Buffer should not exceed limit
        assert len(dataset.exp_x_train) <= config['len_exp_replay']


class TestRecordDictIntegration:
    """Integration tests for record dict structure after training."""

    def test_record_dict_structure(self, test_sine_config, jax_key, tmp_path):
        """Test record dict has correct structure after training."""
        # Added by Claude: Use generic runner's train_model() for end-to-end integration test
        config = test_sine_config.copy()
        config['model_path'] = str(tmp_path / "test_model")
        config['n_task'] = 1

        # Run training through generic runner
        record_dict = train_model(config, run_id=0)

        # Check new structure
        assert 'metadata' in record_dict
        assert 'iterations' in record_dict  # Backward compatibility
        assert 'tasks' in record_dict  # New task-based structure
        assert 'architecture_history' in record_dict

        # Check task structure
        assert 0 in record_dict['tasks']
        assert 'main_training' in record_dict['tasks'][0]
        task_data = record_dict['tasks'][0]['main_training']
        assert 'iterations' in task_data  # Global iterations
        assert 'epochs' in task_data      # Within-task epochs
        assert 'H' in task_data
        assert 'V' in task_data
        assert 'train_metric' in task_data

        # Check backward compatibility - iterations dict
        if len(record_dict['iterations']) > 0:
            sample_iter = list(record_dict['iterations'].values())[0]
            assert 'losses' in sample_iter
            assert 'gradients' in sample_iter
            assert 'metrics' in sample_iter

    def test_record_dict_save_and_load(self, test_sine_config, jax_key, tmp_path):
        """Test record dict can be saved and loaded."""
        config = test_sine_config.copy()
        config['model_path'] = str(tmp_path / "test_model")

        trainer = Trainer(loss='mse', problem='vectors', metric='mse')
        record_dict = trainer.initialize_record_dict(config, run_id=0)
        record_dict['iterations'][10] = {
            'losses': {'H': 1.0, 'V': 0.5},
            'metrics': {'train': 0.8}
        }

        # Save
        os.makedirs(tmp_path, exist_ok=True)
        filepath = trainer.save_record_dict(record_dict, config['model_path'])

        # Load and verify
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)

        assert loaded['metadata'] == record_dict['metadata']
        assert loaded['iterations'][10] == record_dict['iterations'][10]


class TestAWBPipelineIntegration:
    """Integration tests for AWB pipeline (without full training)."""

    def test_awb_model_creation(self, test_sine_awb_config, jax_key):
        """Test AWB-enabled model creation."""
        config = test_sine_awb_config.copy()
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        config['input_size'] = 3
        config['output_size'] = 10
        model = create_mlp(config)

        assert model.awb_enabled == True
        assert model.A is not None
        assert model.B is not None

    def test_awb_partitioning(self, test_sine_awb_config, jax_key):
        """Test AWB model can be partitioned for different training phases."""
        from cl.core.awb import partition_for_AB_training, partition_for_standard_training

        config = test_sine_awb_config.copy()
        # Added by Claude: create_mlp expects input_size/output_size in config dict
        config['input_size'] = 3
        config['output_size'] = 10
        model = create_mlp(config)

        # AB training partition
        diff_model, static_model = partition_for_AB_training(model)
        assert diff_model.A is not None
        assert diff_model.B is not None

        # Recombine
        model_combined = eqx.combine(diff_model, static_model)

        # Standard training partition
        params, static = partition_for_standard_training(model_combined)
        assert params.A is None
        assert params.B is None


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_debug_mode(self, test_sine_config):
        """Test debug mode is respected."""
        config = test_sine_config.copy()
        assert config['debug_mode'] == True
        assert config['debug_limit'] == 50

        dataset = SineDataset(config)
        dataset.load_task(0)

        # Data should be limited
        assert len(dataset.X_train) <= config['debug_limit']

    def test_config_batch_size(self, test_sine_config):
        """Test batch size is respected."""
        config = test_sine_config.copy()

        dataset = SineDataset(config)
        train_loader, _ = dataset.generate_dataset(
            task_id=0, batch_size=config['batch_size'], phase='training'
        )

        batch = next(iter(train_loader))
        x, y = batch
        # Batch size should be <= configured (last batch may be smaller)
        assert x.shape[0] <= config['batch_size']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
