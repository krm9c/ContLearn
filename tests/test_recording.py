"""
Unit tests for recording system in core/recording.py.
Tests RecordingMixin methods for metric tracking and serialization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pytest

from cl.core.recording import RecordingMixin
from cl.models import MLP


class MockTrainer(RecordingMixin):
    """Mock trainer class that inherits RecordingMixin for testing."""

    def __init__(self, n_iter=100):
        self.n_iter = n_iter


class TestRecordMetrics:
    """Tests for record_metrics method."""

    def test_record_metrics_basic(self, jax_key):
        """Test basic metric recording."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        losses = {'H': 1.0, 'V': 0.5, 'dV': 0.1, 'dV_dx': 0.01, 'dV_dtheta': 0.02}
        gradients = {'grad_norm': 0.1}
        metrics = {'train': 0.2, 'test_current': 0.3, 'test_experience': 0.25}

        record = trainer.record_metrics(
            iteration=10,
            step=10,
            task_id=0,
            losses=losses,
            gradients=gradients,
            metrics=metrics,
            model=model
        )

        assert record['step'] == 10
        assert record['task_id'] == 0
        assert record['losses'] == losses
        assert record['gradients'] == gradients
        assert record['metrics'] == metrics
        assert 'eigenvalues' in record

    def test_record_metrics_with_extra(self, jax_key):
        """Test metric recording with extra metrics."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        losses = {'H': 1.0, 'V': 0.5}
        gradients = {'grad_norm': 0.1}
        metrics = {'train': 0.2}
        extra = {'custom_metric': 42}

        record = trainer.record_metrics(
            iteration=10,
            step=10,
            task_id=0,
            losses=losses,
            gradients=gradients,
            metrics=metrics,
            model=model,
            extra_metrics=extra
        )

        assert 'extra_metrics' in record
        assert record['extra_metrics']['custom_metric'] == 42

    def test_record_metrics_global_step(self, jax_key):
        """Test global step computation."""
        trainer = MockTrainer(n_iter=100)
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        # Task 0, step 50
        record = trainer.record_metrics(
            iteration=50, step=50, task_id=0,
            losses={}, gradients={}, metrics={},
            model=model
        )
        assert record['global_step'] == 50

        # Task 1, step 50 -> global_step = 50 + 1*100 = 150
        record = trainer.record_metrics(
            iteration=150, step=50, task_id=1,
            losses={}, gradients={}, metrics={},
            model=model
        )
        assert record['global_step'] == 150


class TestComputeEigenvalues:
    """Tests for eigenvalue computation."""

    def test_compute_eigenvalues_no_awb(self, jax_key):
        """Test eigenvalue computation for model without AWB."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        eigenvalues = trainer._compute_eigenvalues(model)

        # Added by Claude: For non-AWB models, weight eigenvalues are stored in 'A' key
        # This is for compatibility with plotting code
        assert 'layer_0' in eigenvalues['A']  # Weight eigenvalues computed
        assert eigenvalues['B'] == {}  # B is empty for non-AWB

    def test_compute_eigenvalues_with_awb(self, jax_key):
        """Test eigenvalue computation for AWB-enabled model."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=True)

        eigenvalues = trainer._compute_eigenvalues(model)

        # Should have eigenvalues for each layer
        assert 'A' in eigenvalues
        assert 'B' in eigenvalues
        # Check that layer entries exist
        assert 'layer_0' in eigenvalues['A']
        assert 'layer_0' in eigenvalues['B']


class TestInitializeRecordDict:
    """Tests for initialize_record_dict method."""

    def test_initialize_record_dict_basic(self, test_sine_config):
        """Test basic record dict initialization."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=0)

        assert 'metadata' in record_dict
        assert 'iterations' in record_dict
        assert record_dict['iterations'] == {}

    def test_initialize_record_dict_metadata(self, test_sine_config):
        """Test record dict metadata contains config info."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=5)

        metadata = record_dict['metadata']
        assert metadata['problem'] == test_sine_config['problem']
        assert metadata['prob'] == test_sine_config['prob']
        assert metadata['dataset'] == test_sine_config['data']
        assert metadata['network'] == test_sine_config['network']
        assert metadata['n_tasks'] == test_sine_config['n_task']
        assert metadata['run_id'] == 5

    def test_initialize_record_dict_awb_flag(self, test_sine_awb_config):
        """Test record dict captures AWB enabled status."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_awb_config, run_id=0)

        assert record_dict['metadata']['awb_enabled'] == True


class TestSaveRecordDict:
    """Tests for save_record_dict method."""

    def test_save_record_dict(self, test_sine_config, tmp_path):
        """Test saving record dict to file."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=0)
        record_dict['iterations'][10] = {'losses': {'V': 0.5}, 'step': 10}

        base_path = str(tmp_path / "model")
        filepath = trainer.save_record_dict(record_dict, base_path)

        assert os.path.exists(filepath)
        assert filepath.endswith('.pkl')

    def test_save_record_dict_content(self, test_sine_config, tmp_path):
        """Test saved record dict can be loaded correctly."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=0)
        record_dict['iterations'][10] = {'losses': {'V': 0.5}, 'step': 10}

        base_path = str(tmp_path / "model")
        filepath = trainer.save_record_dict(record_dict, base_path)

        # Load and verify
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)

        assert loaded['metadata'] == record_dict['metadata']
        assert loaded['iterations'] == record_dict['iterations']

    def test_save_record_dict_filename_format(self, test_sine_config, tmp_path):
        """Test saved filename follows expected format."""
        trainer = MockTrainer()

        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=3)

        base_path = str(tmp_path / "outputs" / "model")
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        filepath = trainer.save_record_dict(record_dict, base_path)

        # Filename should be: {prob}_{dataset}_{network}_run{run_id}_records.pkl
        expected_name = "regression_sine_fcnn_run3_records.pkl"
        assert os.path.basename(filepath) == expected_name


class TestSaveAllRuns:
    """Tests for save_all_runs static method."""

    def test_save_all_runs(self, test_sine_config, tmp_path):
        """Test saving multiple runs to single file."""
        trainer = MockTrainer()

        # Create records for multiple runs
        all_runs = {}
        for run_id in range(3):
            record_dict = trainer.initialize_record_dict(test_sine_config, run_id=run_id)
            record_dict['iterations'][10] = {'losses': {'V': 0.5 + run_id * 0.1}}
            all_runs[run_id] = record_dict

        base_path = str(tmp_path / "outputs" / "model")
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        filepath = RecordingMixin.save_all_runs(all_runs, base_path, test_sine_config)

        assert os.path.exists(filepath)

        # Load and verify
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)

        assert 'runs' in loaded
        assert 'metadata' in loaded
        assert loaded['metadata']['total_runs'] == 3


class TestRecordStructureIntegration:
    """Integration tests for complete record structure."""

    def test_full_recording_workflow(self, test_sine_config, jax_key, tmp_path):
        """Test complete recording workflow."""
        trainer = MockTrainer(n_iter=100)
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=False)

        # Initialize
        record_dict = trainer.initialize_record_dict(test_sine_config, run_id=0)

        # Simulate recording multiple iterations
        for task_id in range(2):
            for epoch in range(1, 4):
                iteration = task_id * 100 + epoch * 10
                record_dict['iterations'][iteration] = trainer.record_metrics(
                    iteration=iteration,
                    step=epoch * 10,
                    task_id=task_id,
                    losses={'H': 1.0 - 0.1 * epoch, 'V': 0.5, 'dV': 0.1},
                    gradients={'grad_norm': 0.1},
                    metrics={'train': 0.8, 'test_current': 0.7, 'test_experience': 0.75},
                    model=model
                )

        # Save
        base_path = str(tmp_path / "model")
        filepath = trainer.save_record_dict(record_dict, base_path)

        # Load and verify structure
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)

        assert 'metadata' in loaded
        assert 'iterations' in loaded
        assert len(loaded['iterations']) == 6  # 2 tasks * 3 epochs

        # Verify iteration structure
        sample_record = loaded['iterations'][10]
        assert 'losses' in sample_record
        assert 'gradients' in sample_record
        assert 'metrics' in sample_record
        assert 'eigenvalues' in sample_record
        assert 'step' in sample_record
        assert 'task_id' in sample_record

    def test_record_with_awb_eigenvalues(self, test_sine_awb_config, jax_key, tmp_path):
        """Test recording with AWB eigenvalues."""
        trainer = MockTrainer()
        model = MLP(sizes=[3, 16, 10], key=jax_key, awb_enabled=True)

        record_dict = trainer.initialize_record_dict(test_sine_awb_config, run_id=0)

        record = trainer.record_metrics(
            iteration=10,
            step=10,
            task_id=0,
            losses={'H': 1.0, 'V': 0.5},
            gradients={'grad_norm': 0.1},
            metrics={'train': 0.8},
            model=model
        )

        record_dict['iterations'][10] = record

        # Eigenvalues should be computed
        assert 'eigenvalues' in record
        assert 'A' in record['eigenvalues']
        assert 'B' in record['eigenvalues']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
