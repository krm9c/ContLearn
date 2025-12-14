"""
Configuration management for continual learning experiments.
"""

import json
from typing import Dict, Any


class Params:
    """Class that loads hyperparameters from a JSON file.

    Example:
        >>> params = Params('config/sine.json')
        >>> print(params.n_task)
        5
        >>> config = params.dict
        >>> print(config['epochs_per_task'])
        500
    """

    def __init__(self, json_path: str):
        """Load parameters from JSON file.

        Args:
            json_path: Path to JSON configuration file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path: str):
        """Save parameters to JSON file.

        Args:
            json_path: Path to save JSON file
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path: str):
        """Update parameters from another JSON file.

        Args:
            json_path: Path to JSON file with updates
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self) -> Dict[str, Any]:
        """Dict-like access to Params instance.

        Returns:
            Dictionary of all parameters
        """
        return self.__dict__


def load_config(json_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Convenience function that returns a dictionary directly.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        Dictionary of configuration parameters
    """
    with open(json_path) as f:
        return json.load(f)
