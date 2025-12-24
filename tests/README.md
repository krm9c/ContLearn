# Test Suite Documentation

## Overview

This test suite provides comprehensive testing for the Continual Learning framework with JAX/Equinox. The suite is organized into unit tests (fast, ~30s) and training tests (comprehensive, ~5min).

## Recent Updates

### Fixes Applied (Dec 2024)
1. **AWB Test Errors Fixed**: Resolved dimension mismatches and shape errors in AWB pipeline tests
2. **Classification Metrics**: Added support for one-hot encoded labels in metric computation
3. **Test Configs**: Created self-contained configs in `tests/configs/` for all network types
4. **New Feature Tests**: Added `test_lr_schedules.py` covering learning rate schedules and adaptive features

### New Test Coverage
- **Learning Rate Schedules**: constant, step, exponential, cosine, linear (21 tests)
- **Adaptive Features**: adaptive lr_min, adaptive gradient weights
- **All Network Types**: FCNN (MLP), CNN, CNN3D, GCN with and without AWB

## Test Organization

### Unit Tests (`pytest -m unit`) - ~216 tests, ~30 seconds

| Test File | Purpose | Network Types |
|-----------|---------|---------------|
| `test_models.py` | Model architecture creation and forward pass | MLP, CNN, CNN3D, GCN |
| `test_layers.py` | Layer implementations (Linear, Conv, etc.) | All |
| `test_datasets.py` | Dataset loading and experience replay | All |
| `test_mnist.py` | MNIST-specific dataset tests | CNN |
| `test_cnn.py` | CNN/CNN3D architecture tests | CNN, CNN3D |
| `test_graph.py` | GCN architecture and graph processing | GCN |
| `test_losses.py` | Loss functions and metrics | All |
| `test_awb.py` | AWB utility functions | MLP, CNN, CNN3D, GCN |
| `test_recording.py` | Metric recording and eigenvalue tracking | All |
| `test_integration.py` | Component integration tests | All |
| **`test_lr_schedules.py`** | **Learning rate schedules & adaptive features** | **All** |

### Training Tests (`pytest -m training`) - ~11 tests, ~5 minutes

Located in `tests/training/`:
- Full end-to-end pipeline tests
- Test configs with debug mode for all 10 production configs
- Validates complete training workflow from initialization to saving

### AWB Pipeline Tests

Located in `tests/awb_tests/`:
- `test_step1_preliminary.py` - Preliminary training phase
- `test_step2_decision.py` - Architecture change decision logic
- `test_step3a_arch_search.py` - Architecture search algorithms
- `test_step3b_ab_training.py` - A/B matrix training (W frozen)
- `test_step4_v_transform.py` - V = A @ W @ B.T transformation
- `test_step5_v_training.py` - V training (A/B frozen)
- `test_full_pipeline.py` - Complete 5-step AWB pipeline
- `test_mathematical_correctness.py` - Mathematical properties verification
- `benchmark_performance.py` - Performance profiling

## Test Configurations

### Self-Contained Configs in `tests/configs/`

All test configs have:
- `debug_mode: true`
- `debug_limit: 50` (small dataset)
- `epochs_per_task: 2` (minimal training)
- `batch_size: 32` (or smaller for memory efficiency)

| Config File | Network | Dataset | AWB | Purpose |
|-------------|---------|---------|-----|---------|
| `test_sine.json` | FCNN (MLP) | Sine | No | Regression testing |
| `test_sine_awb.json` | FCNN (MLP) | Sine | Yes | Regression with AWB |
| `test_exp_replay.json` | FCNN (MLP) | Sine | No | Experience replay testing |
| `test_mnist.json` | CNN | MNIST | No | Classification testing |
| `test_mnist_awb.json` | CNN | MNIST | Yes | Classification with AWB |
| `test_cifar10.json` | CNN3D | CIFAR-10 | No | Multi-channel image classification |
| `test_cifar10_awb.json` | CNN3D | CIFAR-10 | Yes | Multi-channel with AWB |
| `test_cifar100.json` | CNN3D | CIFAR-100 | No | 100-class classification |
| `test_synthetic_graph.json` | GCN | Synthetic | No | Graph classification |
| `test_synthetic_graph_awb.json` | GCN | Synthetic | Yes | Graph classification with AWB |

### Loading Configs in Tests

Use fixtures from `conftest.py`:
```python
def test_example(test_sine_config):
    # test_sine_config automatically loads tests/configs/test_sine.json
    assert test_sine_config['data'] == 'sine'
```

Available fixtures:
- `test_sine_config()`
- `test_sine_awb_config()`
- `test_exp_replay_config()`
- `test_mnist_config()`
- `test_mnist_awb_config()`
- `test_cifar10_config()`
- `test_cifar10_awb_config()`
- `test_cifar100_config()`
- `test_synthetic_graph_config()`
- `test_graph_awb_config()` (alias for synthetic_graph_awb)

## Running Tests

### Quick Commands

```bash
# All unit tests (fast)
./run_tests.sh --unit

# Training tests (slow)
./run_tests.sh --training

# All tests
./run_tests.sh --all

# Specific category
./run_tests.sh --models
./run_tests.sh --datasets
./run_tests.sh --awb

# With coverage
./run_tests.sh --all --cov

# Pattern matching
./run_tests.sh -k "mlp"
./run_tests.sh -k "awb"
```

### Direct Pytest

```bash
# Single test file
pytest tests/test_lr_schedules.py -v

# Specific test
pytest tests/test_awb.py::TestComputeAvgLoss::test_compute_avg_loss_basic -v

# All unit tests
pytest tests/ -m unit

# Verbose with short traceback
pytest tests/ -v --tb=short
```

## Test Coverage Matrix

### Features Tested

| Feature | Test File | Status |
|---------|-----------|--------|
| MLP Forward Pass | test_models.py | ✓ |
| CNN Forward Pass | test_cnn.py | ✓ |
| CNN3D Forward Pass | test_cnn.py | ✓ |
| GCN Forward Pass | test_graph.py | ✓ |
| AWB A/B Matrices | test_awb.py | ✓ |
| AWB V Transformation | test_awb.py | ✓ |
| Experience Replay | test_datasets.py | ✓ |
| Loss Functions | test_losses.py | ✓ |
| Metric Recording | test_recording.py | ✓ |
| **LR Constant Schedule** | **test_lr_schedules.py** | **✓** |
| **LR Step Decay** | **test_lr_schedules.py** | **✓** |
| **LR Exponential Decay** | **test_lr_schedules.py** | **✓** |
| **LR Cosine Annealing** | **test_lr_schedules.py** | **✓** |
| **LR Linear Decay** | **test_lr_schedules.py** | **✓** |
| **Adaptive LR Min** | **test_lr_schedules.py** | **✓** |
| **Adaptive Grad Weights** | **test_lr_schedules.py** | **✓** |

### Network Types Tested

| Network | Standard CL | AWB | Test Files |
|---------|-------------|-----|------------|
| FCNN (MLP) | ✓ | ✓ | test_models.py, test_awb.py |
| CNN | ✓ | ✓ | test_cnn.py, test_mnist.py |
| CNN3D | ✓ | ✓ | test_cnn.py, test_integration.py |
| GCN | ✓ | ✓ | test_graph.py |

## Known Issues & Limitations

1. **MNIST DataLoader Test**: One test fails due to StopIteration with debug_limit=50 (too small dataset)
2. **Test Isolation**: Some tests require specific configs - use provided fixtures
3. **AWB Pipeline Tests**: Run from `tests/awb_tests/` directory for proper paths

## Adding New Tests

### 1. Create Test File

```python
"""Test description."""
import pytest

# Mark as unit test
pytestmark = pytest.mark.unit

class TestFeature:
    def test_basic_functionality(self):
        assert True
```

### 2. Use Existing Fixtures

```python
def test_with_config(test_sine_config):
    config = test_sine_config
    assert config['data'] == 'sine'
```

### 3. Add New Config (if needed)

Create `tests/configs/test_your_feature.json` with debug settings:
```json
{
    "debug_mode": true,
    "debug_limit": 50,
    "epochs_per_task": 2,
    ...
}
```

### 4. Add Fixture to conftest.py

```python
@pytest.fixture
def test_your_feature_config():
    """Load test_your_feature.json configuration."""
    return load_test_config('test_your_feature.json')
```

## Continuous Integration

Tests are designed to run in CI with:
- Fast unit tests run on every commit
- Training tests run nightly or on release branches
- Debug mode ensures tests complete quickly (<2 min for unit tests)

## Troubleshooting

### Config Not Found
- Ensure config exists in `tests/configs/`
- Check fixture name matches config filename
- Verify `TEST_CONFIG_DIR` in conftest.py points to correct location

### Import Errors
- Check `sys.path` modifications in test file
- Ensure `src/cl` package is importable
- Install package in editable mode: `pip install -e .`

### Memory Issues
- Reduce `debug_limit` in test configs
- Reduce `batch_size` in test configs
- Run fewer tests in parallel

### JAX/GPU Issues
- Unit tests don't require GPU
- Set `JAX_PLATFORM_NAME=cpu` for CPU-only testing
- GPU tests marked with `@pytest.mark.gpu`

## References

- Main README: `../README.md`
- Config Documentation: `../kkt_run/configs/README.md`
- AWB Documentation: `awb_tests/README.md`
- Test Runner: `../run_tests.sh`
