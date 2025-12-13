"""
Configuration management for continual learning experiments.
"""

import json


class Params:
    """Class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters back to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Update parameters from JSON file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Dict-like access to Params instance"""
        return self.__dict__
