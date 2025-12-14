"""
Unit tests for all experimental scripts.

Tests all config files with minimal settings to ensure scripts run without errors.
Uses debug mode with minimal epochs, iterations, and data points.

Usage:
    pytest scripts/test_scripts.py
    pytest scripts/test_scripts.py -v
    pytest scripts/test_scripts.py -k sine
    pytest scripts/test_scripts.py --tb=short
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cl.config import load_config
from cl.runners import train_model


# Test configuration overrides for fast minimal tests
# Added by Claude: These settings ensure tests run quickly while covering all code paths
MINIMAL_TEST_SETTINGS = {
    # Core test settings
    'debug_mode': True,
    'debug_limit': 50,  # Limit to 50 data points

    # Training parameters (minimal)
    'n_task': 2,
    'epochs_per_task': 2,  # Just 2 epochs per task
    'save_iter': 1,

    # Architecture search parameters (minimal)
    'arch_search_enabled': True,  # Enable to test the code path
    'arch_search_epochs': 1,  # Just 1 epoch per search
    'arch_search_max_iter': 1,  # Just 1 iteration
    'awb_preliminary_epochs': 1,  # Minimal preliminary training
    'awb_ab_training_epochs': 1,  # Minimal A/B training
    'awb_ab_max_iterations': 1,  # Minimal A/B iterations

    # Disable plots for speed
    'generate_plots': False,
}


class TestRunner:
    """Helper class to run tests and track failures."""

    def __init__(self):
        self.failures: List[Tuple[str, str, Exception]] = []
        self.successes: List[str] = []

    def run_config(self, config_name: str, config_path: str, overrides: Dict[str, Any] = None) -> bool:
        """Run a single config and track success/failure.

        Args:
            config_name: Name for reporting
            config_path: Path to config file
            overrides: Additional config overrides

        Returns:
            True if successful, False if failed
        """
        try:
            # Load base config
            config = load_config(config_path)

            # Apply test overrides
            config.update(MINIMAL_TEST_SETTINGS)

            # Apply additional overrides if provided
            if overrides:
                config.update(overrides)

            # Create temp directory for outputs
            with tempfile.TemporaryDirectory() as tmpdir:
                # Override output paths to temp directory
                config['model_path'] = os.path.join(tmpdir, 'model')
                config['figures_dir'] = os.path.join(tmpdir, 'figures')

                # Run the training
                record_dict = train_model(config, run_id=0)

                # Basic validation: check that training completed
                assert record_dict is not None, "No record_dict returned"
                assert 'iterations' in record_dict or 'tasks' in record_dict, "Invalid record_dict structure"

            self.successes.append(config_name)
            return True

        except Exception as e:
            self.failures.append((config_name, config_path, e))
            return False

    def get_summary(self) -> str:
        """Generate a summary of test results."""
        total = len(self.successes) + len(self.failures)
        summary = [
            f"\n{'='*70}",
            f"TEST SUMMARY: {len(self.successes)}/{total} passed",
            f"{'='*70}\n",
        ]

        if self.successes:
            summary.append(f"✓ PASSED ({len(self.successes)}):")
            for name in self.successes:
                summary.append(f"  ✓ {name}")
            summary.append("")

        if self.failures:
            summary.append(f"✗ FAILED ({len(self.failures)}):")
            for name, path, error in self.failures:
                summary.append(f"  ✗ {name}")
                summary.append(f"    Config: {path}")
                summary.append(f"    Error: {type(error).__name__}: {str(error)[:100]}")
            summary.append("")

        return "\n".join(summary)

    def print_todo_list(self):
        """Print TODO list for failures."""
        if not self.failures:
            return

        print("\n" + "="*70)
        print("TODO: Fix the following failures")
        print("="*70)
        for i, (name, path, error) in enumerate(self.failures, 1):
            print(f"\n{i}. {name}")
            print(f"   File: {path}")
            print(f"   Error: {type(error).__name__}")
            print(f"   Message: {str(error)[:200]}")


@pytest.fixture
def test_runner():
    """Fixture providing a test runner instance."""
    return TestRunner()


@pytest.fixture(scope="session")
def config_dir():
    """Fixture providing path to config directory."""
    return Path(__file__).parent.parent / 'config'


# ============================================================================
# Test Cases for Each Config
# ============================================================================

class TestRegressionScripts:
    """Tests for regression (sine wave) experiments."""

    def test_sine_standard(self, test_runner, config_dir):
        """Test sine wave regression with standard CL."""
        result = test_runner.run_config(
            'sine (standard CL)',
            str(config_dir / 'sine.json')
        )
        assert result, f"sine.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"

    def test_sine_awb(self, test_runner, config_dir):
        """Test sine wave regression with AWB enabled."""
        result = test_runner.run_config(
            'sine (AWB enabled)',
            str(config_dir / 'sine_awb.json')
        )
        assert result, f"sine_awb.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"


class TestMNISTScripts:
    """Tests for MNIST classification experiments."""

    def test_mnist_standard(self, test_runner, config_dir):
        """Test MNIST classification with standard CL."""
        result = test_runner.run_config(
            'MNIST (standard CL)',
            str(config_dir / 'mnist.json')
        )
        assert result, f"mnist.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"

    def test_mnist_awb(self, test_runner, config_dir):
        """Test MNIST classification with AWB enabled."""
        result = test_runner.run_config(
            'MNIST (AWB enabled)',
            str(config_dir / 'mnist_awb.json')
        )
        assert result, f"mnist_awb.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"


class TestCIFAR10Scripts:
    """Tests for CIFAR-10 classification experiments."""

    def test_cifar10_standard(self, test_runner, config_dir):
        """Test CIFAR-10 classification with standard CL."""
        result = test_runner.run_config(
            'CIFAR-10 (standard CL)',
            str(config_dir / 'cifar10.json')
        )
        assert result, f"cifar10.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"

    def test_cifar10_awb(self, test_runner, config_dir):
        """Test CIFAR-10 classification with AWB enabled."""
        result = test_runner.run_config(
            'CIFAR-10 (AWB enabled)',
            str(config_dir / 'cifar10_awb.json')
        )
        assert result, f"cifar10_awb.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"


class TestCIFAR100Scripts:
    """Tests for CIFAR-100 classification experiments."""

    def test_cifar100_standard(self, test_runner, config_dir):
        """Test CIFAR-100 classification with standard CL."""
        result = test_runner.run_config(
            'CIFAR-100 (standard CL)',
            str(config_dir / 'cifar100.json')
        )
        assert result, f"cifar100.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"

    def test_cifar100_awb(self, test_runner, config_dir):
        """Test CIFAR-100 classification with AWB enabled."""
        result = test_runner.run_config(
            'CIFAR-100 (AWB enabled)',
            str(config_dir / 'cifar100_awb.json')
        )
        assert result, f"cifar100_awb.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"


class TestGraphScripts:
    """Tests for graph classification experiments."""

    def test_synthetic_graph_standard(self, test_runner, config_dir):
        """Test synthetic graph classification with standard CL."""
        result = test_runner.run_config(
            'Synthetic Graph (standard CL)',
            str(config_dir / 'synthetic_graph.json')
        )
        assert result, f"synthetic_graph.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"

    def test_synthetic_graph_awb(self, test_runner, config_dir):
        """Test synthetic graph classification with AWB enabled."""
        result = test_runner.run_config(
            'Synthetic Graph (AWB enabled)',
            str(config_dir / 'synthetic_graph_awb.json')
        )
        assert result, f"synthetic_graph_awb.json failed: {test_runner.failures[-1][2] if test_runner.failures else ''}"


# ============================================================================
# Test Markers - for filtering tests
# ============================================================================

pytestmark = [
    pytest.mark.scripts,  # Mark all tests in this file as script tests
]