"""
Pytest fixtures and configuration for cl_framework tests.
"""

import sys
import os
import tempfile
import json

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


# Path to test config files (self-contained in tests/configs/)
TEST_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def load_test_config(config_name):
    """Load a test config file by name."""
    config_path = os.path.join(TEST_CONFIG_DIR, config_name)
    with open(config_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def test_sine_config():
    """Load test_sine.json configuration with debug limits."""
    return load_test_config('test_sine.json')


@pytest.fixture
def test_sine_awb_config():
    """Load test_sine_awb.json configuration with AWB enabled and debug limits."""
    return load_test_config('test_sine_awb.json')


@pytest.fixture
def test_exp_replay_config():
    """Load test_exp_replay.json configuration for experience replay testing."""
    return load_test_config('test_exp_replay.json')


@pytest.fixture
def regression_config():
    """Configuration for regression problems (inline for quick tests)."""
    return {
        "prob": "regression",
        "problem": "vectors",
        "data": "sine",
        "network": "fcnn",
        "lr": 1e-3,
        "batch_size": 32,
        "epochs_per_task": 3,
        "n_task": 2,
        "save_iter": 1,
        "hln": 32,
        "n_layers": 2,
        "delta": 0.001,
        "flag": [1.0, 0.0],
        "loss": "mse",
        "metric": "mse",
        "model_path": "outputs/test/test_model",
        "len_exp_replay": 200,
        "awb_enabled": False,
        "debug_mode": True,
        "debug_limit": 50
    }


@pytest.fixture
def jax_key():
    """JAX random key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_regression_batch():
    """Create a dummy regression batch (sine-like)."""
    # Sine data: x is [phase, amplitude, frequency], y is sine values
    x = torch.randn(32, 3)
    y = torch.randn(32, 10)
    return x, y


@pytest.fixture
def dummy_classification_batch():
    """Create a dummy classification batch."""
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    return x, y


@pytest.fixture
def mlp_sizes():
    """Standard MLP architecture sizes for testing."""
    return [3, 32, 32, 10]


@pytest.fixture
def small_mlp_sizes():
    """Small MLP architecture for quick tests."""
    return [3, 16, 10]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test artifacts."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def test_mnist_config():
    """Load test_mnist.json configuration with debug limits."""
    return load_test_config('test_mnist.json')


@pytest.fixture
def classification_config():
    """Configuration for classification problems (inline for quick tests)."""
    return {
        "prob": "classification",
        "problem": "vectors",
        "data": "mnist",
        "network": "cnn",
        "lr": 1e-3,
        "batch_size": 32,
        "epochs_per_task": 2,
        "n_task": 2,
        "save_iter": 1,
        "filter_size": 4,
        "channel_out": 3,
        "feed_sizes": [1875, 64, 10],
        "flag": [1.0, 0.0],
        "loss": "class",
        "metric": "class",
        "model_path": "outputs/test/test_cnn_model",
        "len_exp_replay": 200,
        "awb_enabled": False,
        "debug_mode": True,
        "debug_limit": 50
    }


@pytest.fixture
def dummy_image_batch():
    """Create a dummy image batch (MNIST-like)."""
    # MNIST: (batch, 1, 28, 28)
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    return x, y


@pytest.fixture
def cnn_config():
    """Configuration for creating test CNN models.

    feed_sizes[0] must match the flattened output after conv + pool:
    - Input: 28x28
    - After conv(4): (28 - 4 + 1) = 25
    - After eqx.nn.MaxPool2d(kernel_size=2, stride=2): 25 // 2 = 12
    - Flattened: channel_out * 12 * 12 = 3 * 144 = 432
    """
    return {
        "filter_size": 4,
        "channel_out": 3,
        "channel_in": 1,
        "input_size": 28,
        "feed_sizes": [432, 64, 10],  # 3 * 12 * 12 = 432 (stride=2)
    }


# Added by Claude: CNN3D config fixture for CIFAR-like models
@pytest.fixture
def cnn3d_config():
    """Configuration for creating test CNN3D models (CIFAR-like).

    feed_sizes[0] must match the flattened output after two conv+pool layers:
    - Input: 32x32
    - After conv1(3): (32 - 3 + 1) = 30
    - After pool1(stride=2): 30 // 2 = 15
    - After conv2(3): (15 - 3 + 1) = 13
    - After pool2(stride=2): 13 // 2 = 6
    - Flattened: channel_out * 2 * 6 * 6 = 32 * 2 * 36 = 2304
    """
    return {
        "filter_size": 3,
        "channel_out": 32,
        "channel_in": 3,
        "input_size": 32,
        "feed_sizes": [2304, 256, 10],
    }


@pytest.fixture
def dummy_cifar_batch():
    """Create a dummy CIFAR batch (3-channel 32x32 images)."""
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    return x, y


# Added by Claude: Graph model fixtures
@pytest.fixture
def gcn_config():
    """Configuration for creating test GCN models."""
    return {
        'in_size': 5,
        'gcn_sizes': [5, 64],
        'feed_sizes': [64, 32, 16, 10],
        'node_num': 10,
        'out_size': 10,
        'SEED': 42,
    }


@pytest.fixture
def graph_dataset_config():
    """Configuration for creating test graph datasets."""
    return {
        'data': 'synthetic',
        'batch_size': 4,
        'n_class': 10,
        'class_per_task': 2,
        'debug_mode': True,
        'debug_limit': 50,
        'num_graphs': 100,
        'num_channels': 5,
        'avg_num_nodes': 3,
        'num_classes': 10,
    }


@pytest.fixture
def test_graph_awb_config():
    """Load test_synthetic_graph_awb.json configuration."""
    return load_test_config('test_synthetic_graph_awb.json')


# Added by Claude: Additional test config fixtures for all network types
@pytest.fixture
def test_cifar10_config():
    """Load test_cifar10.json configuration (CNN3D without AWB)."""
    return load_test_config('test_cifar10.json')


@pytest.fixture
def test_cifar100_config():
    """Load test_cifar100.json configuration (CNN3D without AWB)."""
    return load_test_config('test_cifar100.json')


@pytest.fixture
def test_synthetic_graph_config():
    """Load test_synthetic_graph.json configuration (GCN without AWB)."""
    return load_test_config('test_synthetic_graph.json')


@pytest.fixture
def test_cifar10_awb_config():
    """Load test_cifar10_awb.json configuration (CNN3D with AWB)."""
    return load_test_config('test_cifar10_awb.json')


@pytest.fixture
def test_mnist_awb_config():
    """Load test_mnist_awb.json configuration (CNN with AWB)."""
    return load_test_config('test_mnist_awb.json')
