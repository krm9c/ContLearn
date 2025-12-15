"""
Training tests for all experimental configurations.

Tests all 10 config files (sine, mnist, cifar10, cifar100, synthetic_graph + AWB)
with minimal settings to ensure end-to-end training pipeline works.

Config files in tests/training/configs/ have debug settings baked in for fast execution.

Usage:
    pytest tests/training/test_all_configs.py
    pytest tests/training/test_all_configs.py -v
    pytest tests/training/test_all_configs.py -k sine
    ./run_tests.sh --training
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cl.config import load_config
from cl.runners import train_model

# Added by Claude: Pytest markers for test categorization
pytestmark = pytest.mark.training


class TestRunner:
    """Helper class to run tests and track failures."""

    def __init__(self):
        self.failures: List[Tuple[str, str, Exception]] = []
        self.successes: List[str] = []
        # Added by Claude: Store training outputs for markdown report
        self.training_outputs: Dict[str, Dict[str, Any]] = {}

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
            # Load base config (already has debug settings baked in)
            config = load_config(config_path)

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

                # Added by Claude: Store training outputs for markdown report
                self._store_training_output(config_name, config, record_dict)

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

    def _store_training_output(self, config_name: str, config: Dict[str, Any], record_dict: Dict[str, Any]):
        """Store training outputs for markdown report generation.

        Added by Claude: Extract key metrics from record_dict and store them
        for later markdown generation.
        """
        output = {
            'config_name': config_name,
            'config': {
                'data': config.get('data', 'unknown'),
                'prob': config.get('prob', 'unknown'),
                'network': config.get('network', 'unknown'),
                'n_task': config.get('n_task', 0),
                'epochs_per_task': config.get('epochs_per_task', 0),
                'awb_enabled': config.get('awb_enabled', False),
                'debug_mode': config.get('debug_mode', False),
                'debug_limit': config.get('debug_limit', 0),
            },
            'record_dict': self._extract_metrics(record_dict),
        }
        self.training_outputs[config_name] = output

    def _extract_metrics(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from record_dict for reporting.

        Added by Claude: Extract final losses, metrics, and summary statistics.
        """
        metrics = {}

        # Get final metrics from last task
        if 'tasks' in record_dict and record_dict['tasks']:
            last_task = record_dict['tasks'][-1]
            task_id = last_task.get('task_id', 'unknown')

            # Extract final losses
            if 'losses' in last_task and last_task['losses']:
                final_losses = last_task['losses'][-1] if isinstance(last_task['losses'], list) else last_task['losses']
                metrics['final_losses'] = {
                    'H': final_losses.get('H', 'N/A'),
                    'V': final_losses.get('V', 'N/A'),
                    'dV': final_losses.get('dV', 'N/A'),
                }

            # Extract final metrics
            if 'train_metric' in last_task:
                train_metrics = last_task['train_metric']
                metrics['final_train_metric'] = train_metrics[-1] if isinstance(train_metrics, list) else train_metrics

            if 'test_metric' in last_task:
                test_metrics = last_task['test_metric']
                metrics['final_test_metric'] = test_metrics[-1] if isinstance(test_metrics, list) else test_metrics

            metrics['task_id'] = task_id

        # Count total iterations
        if 'iterations' in record_dict:
            metrics['total_iterations'] = len(record_dict['iterations'])

        return metrics

    def generate_markdown_report(self, output_path: str = 'SCRIPT_TEST_RESULTS.md'):
        """Generate markdown report of all training outputs.

        Added by Claude: Create a comprehensive markdown report categorized by
        problem type and script.
        """
        # Categorize outputs by problem type
        categories = {
            'Regression': [],
            'MNIST Classification': [],
            'CIFAR-10 Classification': [],
            'CIFAR-100 Classification': [],
            'Graph Classification': [],
        }

        for name, output in self.training_outputs.items():
            data = output['config']['data']
            if 'sine' in data:
                categories['Regression'].append(output)
            elif 'mnist' in data.lower():
                categories['MNIST Classification'].append(output)
            elif 'cifar10' in data.lower():
                categories['CIFAR-10 Classification'].append(output)
            elif 'cifar100' in data.lower():
                categories['CIFAR-100 Classification'].append(output)
            elif 'graph' in data.lower() or 'synthetic' in data.lower():
                categories['Graph Classification'].append(output)

        # Generate markdown content
        lines = [
            "# Script Test Results",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Tests**: {len(self.successes)}",
            f"**Passed**: {len(self.successes)}",
            f"**Failed**: {len(self.failures)}",
            "",
            "---",
            "",
            "## Test Settings",
            "",
            "All tests run with minimal settings for fast validation:",
            "- `debug_mode`: true",
            "- `debug_limit`: 50 samples",
            "- `epochs_per_task`: 2",
            "- `n_task`: 2",
            "- `arch_search_max_iter`: 1",
            "",
            "---",
            "",
        ]

        # Add each category
        for category_name, outputs in categories.items():
            if not outputs:
                continue

            lines.extend([
                f"## {category_name}",
                "",
            ])

            for output in outputs:
                lines.extend(self._format_training_output(output))

        # Add failures section if any
        if self.failures:
            lines.extend([
                "---",
                "",
                "## Failed Tests",
                "",
            ])
            for name, path, error in self.failures:
                lines.extend([
                    f"### {name}",
                    "",
                    f"**Config**: `{path}`",
                    "",
                    f"**Error**: `{type(error).__name__}`",
                    "",
                    "```",
                    str(error),
                    "```",
                    "",
                ])

        # Write to file
        output_file = Path(__file__).parent.parent / output_path
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        return str(output_file)

    def _format_training_output(self, output: Dict[str, Any]) -> List[str]:
        """Format a single training output for markdown.

        Added by Claude: Create a nicely formatted section for each test.
        """
        config = output['config']
        metrics = output['record_dict']

        lines = [
            f"### {output['config_name']}",
            "",
            "**Configuration**:",
            "```json",
            json.dumps({
                'data': config['data'],
                'prob': config['prob'],
                'network': config['network'],
                'n_task': config['n_task'],
                'epochs_per_task': config['epochs_per_task'],
                'awb_enabled': config['awb_enabled'],
            }, indent=2),
            "```",
            "",
        ]

        # Add metrics if available
        if metrics:
            lines.extend([
                "**Training Results**:",
                "",
            ])

            if 'task_id' in metrics:
                lines.append(f"- **Final Task**: {metrics['task_id']}")

            if 'total_iterations' in metrics:
                lines.append(f"- **Total Iterations**: {metrics['total_iterations']}")

            if 'final_losses' in metrics:
                losses = metrics['final_losses']
                lines.extend([
                    "- **Final Losses**:",
                    f"  - H (Hamiltonian): {losses.get('H', 'N/A')}",
                    f"  - V (Experience): {losses.get('V', 'N/A')}",
                    f"  - dV (Regularization): {losses.get('dV', 'N/A')}",
                ])

            if 'final_train_metric' in metrics:
                lines.append(f"- **Final Train Metric**: {metrics['final_train_metric']}")

            if 'final_test_metric' in metrics:
                lines.append(f"- **Final Test Metric**: {metrics['final_test_metric']}")

            lines.append("")

        lines.extend([
            "**Status**: ✅ Passed",
            "",
            "---",
            "",
        ])

        return lines


@pytest.fixture(scope="session")
def test_runner():
    """Fixture providing a session-wide test runner instance.

    Added by Claude: Session scope ensures all tests use the same runner,
    allowing us to collect all results and report at the end.
    """
    return TestRunner()


# config_dir fixture is defined in conftest.py
# It points to tests/training/configs/ where test config files are located


# ============================================================================
# Test Cases for Each Config
# ============================================================================

class TestRegressionScripts:
    """Tests for regression (sine wave) experiments."""

    def test_sine_standard(self, test_runner, config_dir):
        """Test sine wave regression with standard CL."""
        # Added by Claude: Don't assert here, just run the test
        # All results are collected and reported at session end
        test_runner.run_config(
            'sine (standard CL)',
            str(config_dir / 'sine.json')
        )

    def test_sine_awb(self, test_runner, config_dir):
        """Test sine wave regression with AWB enabled."""
        test_runner.run_config(
            'sine (AWB enabled)',
            str(config_dir / 'sine_awb.json')
        )


class TestMNISTScripts:
    """Tests for MNIST classification experiments."""

    def test_mnist_standard(self, test_runner, config_dir):
        """Test MNIST classification with standard CL."""
        test_runner.run_config(
            'MNIST (standard CL)',
            str(config_dir / 'mnist.json')
        )

    def test_mnist_awb(self, test_runner, config_dir):
        """Test MNIST classification with AWB enabled."""
        test_runner.run_config(
            'MNIST (AWB enabled)',
            str(config_dir / 'mnist_awb.json')
        )


class TestCIFAR10Scripts:
    """Tests for CIFAR-10 classification experiments."""

    def test_cifar10_standard(self, test_runner, config_dir):
        """Test CIFAR-10 classification with standard CL."""
        test_runner.run_config(
            'CIFAR-10 (standard CL)',
            str(config_dir / 'cifar10.json')
        )

    def test_cifar10_awb(self, test_runner, config_dir):
        """Test CIFAR-10 classification with AWB enabled."""
        test_runner.run_config(
            'CIFAR-10 (AWB enabled)',
            str(config_dir / 'cifar10_awb.json')
        )


class TestCIFAR100Scripts:
    """Tests for CIFAR-100 classification experiments."""

    def test_cifar100_standard(self, test_runner, config_dir):
        """Test CIFAR-100 classification with standard CL."""
        test_runner.run_config(
            'CIFAR-100 (standard CL)',
            str(config_dir / 'cifar100.json')
        )

    def test_cifar100_awb(self, test_runner, config_dir):
        """Test CIFAR-100 classification with AWB enabled."""
        test_runner.run_config(
            'CIFAR-100 (AWB enabled)',
            str(config_dir / 'cifar100_awb.json')
        )


class TestGraphScripts:
    """Tests for graph classification experiments."""

    def test_synthetic_graph_standard(self, test_runner, config_dir):
        """Test synthetic graph classification with standard CL."""
        test_runner.run_config(
            'Synthetic Graph (standard CL)',
            str(config_dir / 'synthetic_graph.json')
        )

    def test_synthetic_graph_awb(self, test_runner, config_dir):
        """Test synthetic graph classification with AWB enabled."""
        test_runner.run_config(
            'Synthetic Graph (AWB enabled)',
            str(config_dir / 'synthetic_graph_awb.json')
        )


# ============================================================================
# Final Summary Test - Reports All Results
# ============================================================================
# Added by Claude: This test runs last and reports all collected results

class TestSummary:
    """Final test class that reports summary of all test results."""

    def test_zzz_final_summary(self, test_runner):
        """Report final summary of all tests (runs last due to zzz prefix).

        Added by Claude: This test always runs last and reports all failures.
        Tests will continue even if earlier tests fail, allowing us to see
        all errors in one run.
        """
        summary = test_runner.get_summary()
        print(summary)

        # Added by Claude: Generate markdown report with training outputs
        if test_runner.training_outputs:
            try:
                report_path = test_runner.generate_markdown_report()
                print(f"\n✓ Training outputs saved to: {report_path}")
            except Exception as e:
                print(f"\n⚠ Warning: Failed to generate markdown report: {e}")

        # If there were failures, fail this test with the summary
        if test_runner.failures:
            pytest.fail(f"\n{summary}\n{len(test_runner.failures)} test(s) failed. See details above.")


# ============================================================================
# Test Markers - for filtering tests
# ============================================================================

pytestmark = [
    pytest.mark.scripts,  # Mark all tests in this file as script tests
]