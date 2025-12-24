# Script Tests

Unit tests for all experimental scripts to ensure they run without errors.

## Overview

This test suite validates that all experimental configurations work correctly by running them with minimal settings:
- **Debug mode enabled**: Only 50 data points per dataset
- **Minimal epochs**: 2 epochs per task
- **Minimal architecture search**: 1 iteration only
- **Fast execution**: Tests complete in minutes, not hours

## Running Tests

### Run all script tests
```bash
./scripts/test_all_scripts.sh
```

### Using pytest directly
```bash
# All tests
pytest scripts/test_scripts.py -v

# Only regression tests
pytest scripts/test_scripts.py -v -m regression

# Only AWB tests
pytest scripts/test_scripts.py -v -m awb

# Only classification tests (MNIST/CIFAR)
pytest scripts/test_scripts.py -v -m classification

# Only graph tests
pytest scripts/test_scripts.py -v -m graph

# Specific test
pytest scripts/test_scripts.py -v -k sine

# With short traceback
pytest scripts/test_scripts.py --tb=short
```

## Test Coverage

### Regression (Sine Wave)
- ✓ `sine.json` - Standard CL
- ✓ `sine_awb.json` - AWB enabled

### Classification (MNIST)
- ✓ `mnist.json` - Standard CL
- ✓ `mnist_awb.json` - AWB enabled

### Classification (CIFAR-10)
- ✓ `cifar10.json` - Standard CL
- ✓ `cifar10_awb.json` - AWB enabled

### Classification (CIFAR-100)
- ✓ `cifar100.json` - Standard CL
- ✓ `cifar100_awb.json` - AWB enabled

### Graph Classification (Synthetic)
- ✓ `synthetic_graph.json` - Standard CL
- ✓ `synthetic_graph_awb.json` - AWB enabled

## Test Configuration

All tests use these minimal settings (defined in `test_scripts.py`):

```python
MINIMAL_TEST_SETTINGS = {
    'debug_mode': True,
    'debug_limit': 50,              # Only 50 samples
    'n_task': 2,                    # Just 2 tasks
    'epochs_per_task': 2,           # Just 2 epochs
    'arch_search_epochs': 1,        # 1 epoch for arch search
    'arch_search_max_iter': 1,      # 1 iteration
    'awb_preliminary_epochs': 1,    # Minimal preliminary
    'awb_ab_training_epochs': 1,    # Minimal A/B training
    'awb_ab_max_iterations': 1,     # 1 A/B iteration
}
```

## Failure Tracking

When tests fail, a TODO list is automatically generated showing:
1. Which config failed
2. The error type and message
3. Suggestions for fixing

Example output:
```
TODO: Fix the following script test failures
======================================================================

1. scripts/test_scripts.py::TestGraphScripts::test_synthetic_graph_awb
   Error: TypeError: dot_general requires contracting dimensions...

2. scripts/test_scripts.py::TestCIFAR100Scripts::test_cifar100_standard
   Error: ValueError: Shapes incompatible...
```

## Adding New Tests

To add a new experimental config to the test suite:

1. Add the config file to `config/`
2. Add a test method to the appropriate test class in `test_scripts.py`:

```python
def test_my_new_config(self, test_runner, config_dir):
    """Test description."""
    result = test_runner.run_config(
        'My Config Name',
        str(config_dir / 'my_config.json')
    )
    assert result, f"my_config.json failed"
```

3. Run the tests to verify it works

## Integration with CI/CD

These tests can be integrated into continuous integration:

```yaml
# .github/workflows/test.yml
- name: Test all experimental scripts
  run: |
    ./scripts/test_all_scripts.sh
```

## Performance

Typical execution times (on M1 Mac):
- **Regression tests**: ~10 seconds per test
- **Classification tests**: ~30 seconds per test
- **Graph tests**: ~15 seconds per test
- **Total suite**: ~5 minutes for all 10 tests

## Troubleshooting

### Tests timeout
Increase the timeout in pytest.ini or use `-o timeout=600`

### Import errors
Make sure you're running from the repository root and `src/` is in the Python path

### Config not found
Verify the config file exists in `config/` directory

### Model errors
Check that the model architecture matches the dataset requirements

## See Also

- Main test suite: `tests/`
- Test runner script: `run_tests.sh`
- Configuration guide: `config/TEMPLATE.md`
