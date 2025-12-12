"""
Pytest fixtures and configuration for ContLearn tests.
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


@pytest.fixture
def temp_json_config():
    """Create a temporary JSON config file for testing."""
    config = {
        "prob": "classification",
        "problem": "classification",
        "data": "mnist",
        "network": "cnn",
        "lr": 1e-3,
        "batch_size": 32,
        "epochs_per_task": 10,
        "n_task": 2,
        "n_class": 10,
        "class_per_task": 2,
        "save_iter": 5,
        "hln": 128,
        "batch": 32,
        "delta": 0.01,
        "flag": [1, 1],
        "loss": "class",
        "metric": "class",
        "tensorfile": "runs/test",
        "model_path": "models/test_model"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def regression_config():
    """Configuration for regression problems."""
    return {
        "prob": "regression",
        "problem": "regression",
        "data": "sine",
        "network": "fcnn",
        "lr": 1e-3,
        "batch_size": 32,
        "epochs_per_task": 10,
        "n_task": 2,
        "save_iter": 5,
        "hln": 64,
        "delta": 0.01,
        "flag": [1, 1],
        "loss": "mse",
        "metric": "mse",
        "tensorfile": "runs/test_reg",
        "model_path": "models/test_reg_model",
        "len_exp_replay": 1000
    }


@pytest.fixture
def classification_config():
    """Configuration for classification problems."""
    return {
        "prob": "classification",
        "problem": "classification",
        "data": "mnist",
        "network": "cnn",
        "lr": 1e-3,
        "batch_size": 32,
        "epochs_per_task": 10,
        "n_task": 2,
        "n_class": 10,
        "class_per_task": 2,
        "save_iter": 5,
        "hln": 128,
        "batch": 32,
        "delta": 0.01,
        "flag": [1, 1],
        "loss": "class",
        "metric": "class",
        "tensorfile": "runs/test_class",
        "model_path": "models/test_class_model",
        "len_exp_replay": 1000
    }


@pytest.fixture
def graph_config():
    """Configuration for graph classification problems."""
    return {
        "prob": "graphclassification",
        "problem": "graph",
        "data": "MUTAG",
        "network": "gcn",
        "lr": 1e-4,
        "batch_size": 32,
        "epochs_per_task": 10,
        "n_task": 2,
        "n_class": 2,
        "class_per_task": 1,
        "save_iter": 5,
        "batch": 32,
        "delta": 0.01,
        "flag": [1, 1],
        "loss": "class",
        "metric": "class",
        "tensorfile": "runs/test_graph",
        "model_path": "models/test_graph_model",
        "len_exp_replay": 1000
    }


@pytest.fixture
def jax_key():
    """JAX random key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_mnist_batch():
    """Create a dummy MNIST-style batch."""
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    return x, y


@pytest.fixture
def dummy_cifar_batch():
    """Create a dummy CIFAR-style batch."""
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    return x, y


@pytest.fixture
def dummy_regression_batch():
    """Create a dummy regression batch."""
    x = torch.randn(32, 3)
    y = torch.randn(32, 10)
    return x, y
