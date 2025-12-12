"""
Unit tests for config/params.py - Params class.
"""

import sys
import os
import json
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Params


class TestParams:
    """Tests for the Params configuration class."""

    def test_params_init_from_json(self, temp_json_config):
        """Test that Params loads correctly from a JSON file."""
        params = Params(temp_json_config)

        assert params.prob == "classification"
        assert params.lr == 1e-3
        assert params.batch_size == 32
        assert params.n_task == 2

    def test_params_dict_property(self, temp_json_config):
        """Test the dict property returns all parameters."""
        params = Params(temp_json_config)
        params_dict = params.dict

        assert isinstance(params_dict, dict)
        assert "prob" in params_dict
        assert "lr" in params_dict
        assert params_dict["prob"] == "classification"

    def test_params_save(self, temp_json_config):
        """Test saving parameters back to JSON."""
        params = Params(temp_json_config)

        # Modify a parameter
        params.lr = 0.01

        # Save to new file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = f.name

        try:
            params.save(save_path)

            # Load and verify
            with open(save_path) as f:
                saved_data = json.load(f)

            assert saved_data["lr"] == 0.01
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_params_update(self, temp_json_config):
        """Test updating parameters from another JSON file."""
        params = Params(temp_json_config)

        # Create update JSON
        update_config = {"lr": 0.001, "new_param": "test"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(update_config, f)
            update_path = f.name

        try:
            params.update(update_path)

            assert params.lr == 0.001
            assert params.new_param == "test"
            # Original params should still exist
            assert params.prob == "classification"
        finally:
            if os.path.exists(update_path):
                os.unlink(update_path)

    def test_params_missing_file(self):
        """Test that missing file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            Params("/nonexistent/path/config.json")

    def test_params_invalid_json(self):
        """Test that invalid JSON raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            invalid_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                Params(invalid_path)
        finally:
            if os.path.exists(invalid_path):
                os.unlink(invalid_path)

    def test_params_attribute_access(self, temp_json_config):
        """Test that parameters can be accessed as attributes."""
        params = Params(temp_json_config)

        # Access various types
        assert isinstance(params.prob, str)
        assert isinstance(params.lr, float)
        assert isinstance(params.n_task, int)
        assert isinstance(params.flag, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
