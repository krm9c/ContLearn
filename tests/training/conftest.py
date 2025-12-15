"""
Pytest fixtures for training tests.

Provides session-scoped fixtures for full training pipeline tests.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def config_dir():
    """Path to test config directory.

    Returns the path to tests/training/configs/ which contains
    test-specific config files with debug settings baked in.
    """
    return Path(__file__).parent / 'configs'
