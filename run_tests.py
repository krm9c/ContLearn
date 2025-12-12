#!/usr/bin/env python
"""
Python-based test runner for ContLearn.
Provides a convenient interface for running different test suites.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="ContLearn Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --fast                   # Skip slow integration tests
  python run_tests.py --models --verbose       # Run model tests with verbose output
  python run_tests.py -k regression            # Run tests with 'regression' in name
  python run_tests.py --cov                    # Run with coverage report
  python run_tests.py --parallel               # Run tests in parallel
        """
    )

    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('-a', '--all', action='store_true',
                           help='Run all tests (default)')
    test_group.add_argument('-f', '--fast', action='store_true',
                           help='Run only fast tests (skip integration tests)')
    test_group.add_argument('-m', '--models', action='store_true',
                           help='Run model tests only')
    test_group.add_argument('-d', '--data', action='store_true',
                           help='Run data tests only')
    test_group.add_argument('-t', '--trainer', action='store_true',
                           help='Run trainer tests only')
    test_group.add_argument('-g', '--graph', action='store_true',
                           help='Run graph model tests only')
    test_group.add_argument('-c', '--checkpoint', action='store_true',
                           help='Run checkpoint tests only')
    test_group.add_argument('-r', '--runners', action='store_true',
                           help='Run training runner tests only')
    test_group.add_argument('-u', '--utils', action='store_true',
                           help='Run utility tests only')

    # Test execution options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Run with verbose output')
    parser.add_argument('-s', '--stdout', action='store_true',
                       help='Show print statements')
    parser.add_argument('-k', '--keyword', type=str, metavar='PATTERN',
                       help='Run tests matching PATTERN')
    parser.add_argument('--cov', action='store_true',
                       help='Run with coverage report')
    parser.add_argument('--cov-html', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel (requires pytest-xdist)')
    parser.add_argument('--markers', action='store_true',
                       help='List available test markers')

    args = parser.parse_args()

    # Check if pytest is installed
    try:
        subprocess.run(['pytest', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pytest is not installed.")
        print("Install it with: pip install pytest")
        return 1

    # Build pytest command
    cmd_parts = ['pytest']

    # Determine which tests to run
    if args.fast:
        cmd_parts.extend([
            'tests/',
            '-k', 'not (train_model_reg or train_model_class or train_model_graph)'
        ])
        description = "Running Fast Tests (Skipping Integration Tests)"
    elif args.models:
        cmd_parts.extend(['tests/test_models.py', 'tests/test_cnn3d.py'])
        description = "Running Model Tests"
    elif args.data:
        cmd_parts.append('tests/test_data.py')
        description = "Running Data Tests"
    elif args.trainer:
        cmd_parts.append('tests/test_trainer.py')
        description = "Running Trainer Tests"
    elif args.graph:
        cmd_parts.append('tests/test_graph_models.py')
        description = "Running Graph Model Tests"
    elif args.checkpoint:
        cmd_parts.extend(['tests/test_checkpoint.py', 'tests/test_config.py'])
        description = "Running Checkpoint Tests"
    elif args.runners:
        cmd_parts.append('tests/test_runners.py')
        description = "Running Training Runner Tests (Integration Tests)"
    elif args.utils:
        cmd_parts.append('tests/test_utils.py')
        description = "Running Utility Tests"
    else:
        # Default: run all tests
        cmd_parts.append('tests/')
        description = "Running All Tests"

    # Add optional flags
    if args.verbose:
        cmd_parts.append('-v')

    if args.stdout:
        cmd_parts.append('-s')

    if args.keyword and not args.fast:
        cmd_parts.extend(['-k', args.keyword])

    if args.cov:
        cmd_parts.extend([
            '--cov=utils',
            '--cov=training',
            '--cov=config',
            '--cov=data',
            '--cov-report=term-missing'
        ])

    if args.cov_html:
        cmd_parts.extend([
            '--cov=utils',
            '--cov=training',
            '--cov=config',
            '--cov=data',
            '--cov-report=html'
        ])

    if args.parallel:
        # Check if pytest-xdist is installed
        try:
            import xdist
            cmd_parts.extend(['-n', 'auto'])
        except ImportError:
            print("Warning: pytest-xdist not installed. Running tests sequentially.")
            print("Install it with: pip install pytest-xdist")

    if args.markers:
        return run_command('pytest --markers', "Available Test Markers")

    # Run the tests
    cmd = ' '.join(cmd_parts)
    exit_code = run_command(cmd, description)

    # Print summary
    print("\n" + "="*60)
    if exit_code == 0:
        print("✓ All tests passed!")

        if args.cov_html:
            print("\nCoverage report generated at: htmlcov/index.html")
            print("Open it with: open htmlcov/index.html (Mac) or xdg-open htmlcov/index.html (Linux)")
    else:
        print("✗ Some tests failed. See output above for details.")
    print("="*60 + "\n")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
