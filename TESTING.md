# Testing Guide for ContLearn

Quick reference for running tests in the ContLearn framework.

## Quick Start

```bash
# Run all tests
./run_tests.sh

# Or using Python
python run_tests.py --all

# Or using pytest directly
pytest tests/
```

## Test Runner Scripts

### Bash Script: `run_tests.sh`

The bash script provides a full-featured test runner with colored output:

```bash
# Run all tests
./run_tests.sh --all

# Run only fast tests (skip integration tests)
./run_tests.sh --fast

# Run specific test categories
./run_tests.sh --models          # Model tests only
./run_tests.sh --data            # Data tests only
./run_tests.sh --trainer         # Trainer tests only
./run_tests.sh --graph           # Graph model tests only
./run_tests.sh --checkpoint      # Checkpoint tests only
./run_tests.sh --runners         # Training runner tests only
./run_tests.sh --utils           # Utility tests only

# Run with options
./run_tests.sh --all --verbose              # Verbose output
./run_tests.sh --models --stdout            # Show print statements
./run_tests.sh --all --cov                  # With coverage report
./run_tests.sh --all --cov-html             # HTML coverage report
./run_tests.sh --all --parallel             # Parallel execution

# Pattern matching
./run_tests.sh -k regression                # Tests with 'regression' in name
./run_tests.sh -k "graph and not slow"      # Complex patterns

# Help
./run_tests.sh --help
```

### Python Script: `run_tests.py`

Cross-platform Python test runner:

```bash
# Run all tests
python run_tests.py --all

# Run specific categories
python run_tests.py --models
python run_tests.py --data
python run_tests.py --fast

# With options
python run_tests.py --all --verbose
python run_tests.py --models --cov
python run_tests.py -k regression

# Help
python run_tests.py --help
```

## Direct pytest Usage

For more control, use pytest directly:

```bash
# Run all tests
pytest tests/

# Run specific file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestMLP

# Run specific test
pytest tests/test_models.py::TestMLP::test_mlp_forward_pass

# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -s

# Pattern matching
pytest tests/ -k "regression"
pytest tests/ -k "graph and not integration"

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto

# Coverage report
pytest tests/ --cov=utils --cov=training --cov=config --cov=data

# HTML coverage report
pytest tests/ --cov=utils --cov=training --cov=config --cov=data --cov-report=html
```

## Test Categories

### Fast Unit Tests
Most tests complete in milliseconds:
- Model initialization and forward passes
- Data transformations
- Loss computations
- Configuration loading

### Integration Tests
Slower end-to-end tests (in `test_runners.py`):
- Full training loops
- Multi-task continual learning
- Model saving/loading

Skip integration tests for faster feedback:
```bash
./run_tests.sh --fast
# or
pytest tests/ -k "not (train_model_reg or train_model_class or train_model_graph)"
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_models.py           # MLP, CNN, CNN3D tests
├── test_graph_models.py     # GCN, GAT, myNN tests
├── test_data.py             # Data loading and experience replay
├── test_trainer.py          # Trainer class and loss functions
├── test_checkpoint.py       # Model initialization
├── test_runners.py          # Training orchestration (integration)
├── test_config.py           # Configuration management
├── test_utils.py            # Utility functions
├── test_cnn3d.py            # CNN3D specialized tests
└── README.md                # Detailed test documentation
```

## Common Workflows

### Development Workflow
```bash
# Quick check - fast tests only
./run_tests.sh --fast

# Test your changes
pytest tests/test_models.py -v

# Full validation before commit
./run_tests.sh --all --cov
```

### Debugging Failures
```bash
# Show print statements and stop on first failure
pytest tests/test_models.py -s -x

# Show local variables on failure
pytest tests/test_models.py -l

# Run single failing test
pytest tests/test_models.py::TestMLP::test_mlp_forward_pass -v
```

### Coverage Analysis
```bash
# Terminal coverage report
./run_tests.sh --all --cov

# HTML coverage report (better visualization)
./run_tests.sh --all --cov-html
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### CI/CD Pipeline
```bash
# Fast feedback
pytest tests/ -k "not (train_model_reg or train_model_class or train_model_graph)" -v

# Full test suite
pytest tests/ -v --cov=utils --cov=training --cov=config --cov=data --cov-report=xml
```

## Requirements

Install testing dependencies:

```bash
# Basic testing
pip install pytest

# Additional tools (recommended)
pip install pytest-cov           # Coverage reports
pip install pytest-xdist         # Parallel testing
pip install pytest-timeout       # Timeout handling
```

## Environment Setup

Some tests download datasets on first run:
- **Vision datasets**: MNIST, Omniglot, CIFAR-10, CIFAR-100 (via torchvision)
- **Graph datasets**: MUTAG, ENZYMES, PROTEINS (via torch_geometric)

These will be cached for subsequent runs.

## Troubleshooting

### Tests are slow
- Use `--fast` flag to skip integration tests
- Use `--parallel` for parallel execution
- Run specific test files instead of all tests

### Import errors
```bash
# Ensure you're in the project root
cd /home/kraghavan/current/ContLearn

# Install in development mode
pip install -e .
```

### GPU/CPU issues
```bash
# Force CPU for testing
export JAX_PLATFORM_NAME=cpu
pytest tests/
```

### Dataset download failures
- Check internet connection
- Some datasets require proxy settings (see script.sh for examples)
- Datasets are cached in `~/.cache/torch` and `~/.cache/torch_geometric`

## Writing New Tests

Follow the existing patterns:

```python
class TestMyFeature:
    """Tests for my new feature."""

    def test_feature_basic(self):
        """Test basic functionality."""
        # Setup
        input_data = ...

        # Execute
        result = my_function(input_data)

        # Assert
        assert result.shape == expected_shape
        assert not jnp.isnan(result).any()
```

See [tests/README.md](tests/README.md) for detailed guidelines.

## Additional Resources

- **Test Documentation**: [tests/README.md](tests/README.md)
- **Project README**: [README.md](README.md)
- **pytest Documentation**: https://docs.pytest.org/
