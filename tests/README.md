# ContLearn Test Suite

Comprehensive unit tests for the ContLearn continual learning framework.

## Test Organization

### Model Tests

**[test_models.py](test_models.py)** - Neural network architecture tests
- `TestMLP`: MLP initialization, forward pass, batch processing, AWB transformation
- `TestMLPorig`: Original MLP architecture tests
- `TestCNN`: CNN for 1-channel images (MNIST/Omniglot)
- `TestCNN3D`: CNN for 3-channel images (CIFAR-10/100)
- `TestLinearLayers`: Custom Linear, Linear2, Linear3 layer variants
- `TestModelSerialization`: Equinox serialization/deserialization

**[test_graph_models.py](test_graph_models.py)** - Graph neural network tests
- `TestGCNLayers`: Graph Convolutional Network layers (GCN, GCNorig)
- `TestGATLayers`: Graph Attention Network layers (SingleHeadGAT, MultiHeadGAT)
- `TestGraphPooling`: Sum, mean, max, and identity pooling operations
- `TestMyNN`: Combined GCN + MLP for graph classification
- `TestSparseMatmul`: Sparse matrix multiplication utilities
- `TestLinearVariants`: Linear layers used in graph models

**[test_cnn3d.py](test_cnn3d.py)** - Specialized CNN3D tests
- CNN3D forward pass with 3x32x32 inputs
- AWB transformation (get_AWBT)
- Output size calculations
- AWB matrix shape validation
- 3-channel vs 1-channel image handling
- Integration with Continual_Dataset

### Data Tests

**[test_data.py](test_data.py)** - Data loading and experience replay
- `TestDataReturn`: Dataset initialization for sine, MNIST, CIFAR-10
- `TestContinualDataset`: PyTorch DataLoader integration
- `TestDataTransformations`: Channel dimension handling (1-channel vs 3-channel)
- Experience replay buffer management
- Multiple continual learning tasks
- Dataset generation for regression/classification

### Training Tests

**[test_trainer.py](test_trainer.py)** - Trainer class functionality
- `TestTrainerInit`: Trainer initialization for different problem types
- `TestLossFunctions`: MSE loss, classification loss, batch processing
- `TestMetricFunctions`: Accuracy, MSE metrics, predictions
- `TestGraphLossFunctions`: Graph-specific loss functions
- `TestTrainerJIT`: JIT compilation verification
- `TestTrainerWithOptimizer`: Integration with Optax optimizers

**[test_checkpoint.py](test_checkpoint.py)** - Model initialization
- `TestCheckpointRegression`: Checkpoint loading for sine regression
- `TestCheckpointClassification`: MNIST, CIFAR-10, CIFAR-100 initialization
- `TestCheckpointGraphClassification`: Graph dataset initialization
- `TestCheckpointOptimizer`: Adam/AdamW optimizer setup
- `TestCheckpointDataset`: Dataset generation after loading
- `TestCheckpointSeed`: Reproducibility verification

**[test_runners.py](test_runners.py)** - Training orchestration (Integration tests)
- `TestTrainModelRegression`: Regression training pipeline
- `TestTrainModelClassification`: Classification training pipeline
- `TestTrainModelGraph`: Graph classification training pipeline
- `TestTrainingRecordDict`: Training metrics recording
- `TestEquinoxPartitionPattern`: AWB matrix partition verification

### Configuration Tests

**[test_config.py](test_config.py)** - Configuration management
- `TestParams`: JSON config loading, saving, updating
- Parameter attribute access
- Missing file and invalid JSON handling

### Utility Tests

**[test_utils.py](test_utils.py)** - Utility functions
- `TestSparseMatmul`: Sparse matrix operations
- `TestNormalization`: Row normalization, feature preprocessing
- `TestGraphPreprocessing`: Adjacency matrix normalization
- `TestVisualization`: Plotting distributions, gradient visualization
- `TestUtilsIntegration`: Graph preprocessing pipeline
- `TestEdgeCases`: Edge cases and error handling

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_models.py
pytest tests/test_data.py
pytest tests/test_trainer.py
```

### Run Specific Test Class
```bash
pytest tests/test_models.py::TestMLP
pytest tests/test_data.py::TestDataReturn
```

### Run Specific Test Function
```bash
pytest tests/test_models.py::TestMLP::test_mlp_forward_pass
pytest tests/test_checkpoint.py::TestCheckpointRegression::test_load_checkpoint_sine_regression
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run with Output from Print Statements
```bash
pytest tests/ -s
```

### Run Tests Matching Pattern
```bash
# Run all tests with "regression" in the name
pytest tests/ -k regression

# Run all tests with "graph" in the name
pytest tests/ -k graph

# Run all checkpoint tests
pytest tests/ -k checkpoint
```

### Run Tests and Show Coverage
```bash
pytest tests/ --cov=utils --cov=training --cov=config --cov=data
```

## Test Categories

### Fast Unit Tests
Most tests are fast unit tests that complete in milliseconds:
- Model initialization and forward passes
- Data transformations
- Loss function computations
- Configuration loading

### Integration Tests
Some tests are slower integration tests (marked in test_runners.py):
- `test_train_model_reg_*`: Full regression training loops
- `test_train_model_class_*`: Full classification training loops
- `test_train_model_graph_*`: Full graph training loops

To skip integration tests:
```bash
pytest tests/ -k "not (train_model_reg or train_model_class or train_model_graph)"
```

## Test Fixtures

Common fixtures are defined in [conftest.py](conftest.py):
- `temp_json_config`: Temporary JSON configuration file
- `regression_config`: Configuration for regression problems
- `classification_config`: Configuration for classification problems
- `graph_config`: Configuration for graph classification
- `jax_key`: JAX random key for reproducibility
- `dummy_mnist_batch`, `dummy_cifar_batch`, `dummy_regression_batch`: Test data batches

## Expected Test Behavior

### Tests Requiring Data Downloads
Some tests download datasets on first run:
- MNIST, Omniglot, CIFAR-10, CIFAR-100 (via torchvision)
- MUTAG, ENZYMES, PROTEINS (via torch_geometric)

These tests may be slower on first execution but will use cached data afterward.

### Tests Using Temporary Directories
Tests use Python's `tempfile.TemporaryDirectory()` to:
- Store TensorBoard logs
- Save/load model checkpoints
- Write temporary configurations

All temporary files are automatically cleaned up after tests.

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Organization**: Place tests in the appropriate file based on the module being tested
2. **Naming**: Use descriptive test names starting with `test_`
3. **Classes**: Group related tests in classes with `Test` prefix
4. **Docstrings**: Add clear docstrings explaining what each test validates
5. **Assertions**: Use informative assertion messages
6. **Fixtures**: Reuse fixtures from conftest.py when possible
7. **Cleanup**: Use context managers for temporary resources

Example:
```python
class TestNewFeature:
    """Tests for new feature functionality."""

    def test_feature_basic_case(self):
        """Test that feature works in basic case."""
        # Setup
        input_data = ...

        # Execute
        result = my_function(input_data)

        # Assert
        assert result.shape == expected_shape
        assert not jnp.isnan(result).any()
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- All tests use reproducible random seeds
- Temporary directories ensure no filesystem conflicts
- Tests are independent (can run in any order)
- Fast execution for quick feedback

## Test Coverage

Current test coverage by module:
- ✅ `utils/model.py`: MLP, CNN, CNN3D, GCN, GAT, myNN, Linear layers
- ✅ `utils/data.py`: data_return, Continual_Dataset, experience replay
- ✅ `utils/trainer.py`: Trainer, loss functions, metrics
- ✅ `utils/utils.py`: Sparse operations, preprocessing, visualization
- ✅ `config/params.py`: Params class
- ✅ `training/checkpoint.py`: load_checkpoint
- ✅ `training/runners.py`: train_model_reg, train_model_class, train_model_graph

## Known Issues

None currently. Report issues at the project repository.
