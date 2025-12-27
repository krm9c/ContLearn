"""
Configuration management for continual learning experiments.

Added by Claude: Config loading with automatic defaults from constants.py
"""

import json
from typing import Dict, Any
from . import constants


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


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values from constants.py to config dictionary.

    Only applies defaults for parameters that are not already specified in config.
    This allows config files to only specify non-default values.

    Added by Claude: Automatic defaults for layer-level AWB refactor.

    Args:
        config: Configuration dictionary from JSON file

    Returns:
        Configuration dictionary with defaults applied
    """
    # Create a copy to avoid modifying original
    config = config.copy()

    # Helper function to set default if not present
    def set_default(key: str, value: Any):
        if key not in config:
            config[key] = value

    # ===== DATASET-DRIVEN AUTO-SELECTION =====
    # Get dataset (only required parameter from user)
    data = config.get('data', constants.DEFAULT_DATA)

    # Auto-select prob, problem, network, loss, metric based on dataset
    if data in constants.DATASET_CONFIG_MAP:
        dataset_config = constants.DATASET_CONFIG_MAP[data]
        set_default('prob', dataset_config['prob'])
        set_default('problem', dataset_config['problem'])
        set_default('network', dataset_config['network'])
        set_default('loss', dataset_config['loss'])
        set_default('metric', dataset_config['metric'])
    else:
        # Fallback to manual defaults if dataset not in map
        set_default('prob', constants.DEFAULT_PROB)
        set_default('problem', constants.DEFAULT_PROBLEM)
        set_default('network', constants.DEFAULT_NETWORK)
        set_default('loss', constants.DEFAULT_LOSS)
        set_default('metric', constants.DEFAULT_METRIC)

    # Ensure data is set
    set_default('data', constants.DEFAULT_DATA)

    # ===== TRAINING LOOP DEFAULTS =====
    set_default('n_task', constants.DEFAULT_N_TASK)
    set_default('epochs_per_task', constants.DEFAULT_EPOCHS_PER_TASK)
    set_default('save_iter', constants.DEFAULT_SAVE_ITER)
    # Added by Claude: Separate logging/evaluation intervals for performance optimization
    set_default('log_interval', constants.DEFAULT_LOG_INTERVAL)
    set_default('eval_interval', constants.DEFAULT_EVAL_INTERVAL)
    set_default('checkpoint_interval', constants.DEFAULT_CHECKPOINT_INTERVAL)
    set_default('model_path', constants.DEFAULT_MODEL_PATH)

    # Batch size: problem-specific defaults
    if 'batch_size' not in config:
        prob = config.get('prob', constants.DEFAULT_PROB)
        problem = config.get('problem', constants.DEFAULT_PROBLEM)
        if problem == 'graph':
            config['batch_size'] = constants.DEFAULT_BATCH_SIZE_GRAPH
        elif prob == 'classification':
            config['batch_size'] = constants.DEFAULT_BATCH_SIZE_CLASSIFICATION
        else:
            config['batch_size'] = constants.DEFAULT_BATCH_SIZE_REGRESSION

    # Experience replay: problem-specific defaults
    if 'len_exp_replay' not in config:
        problem = config.get('problem', constants.DEFAULT_PROBLEM)
        if problem == 'graph':
            config['len_exp_replay'] = constants.DEFAULT_REPLAY_BUFFER_GRAPH
        else:
            config['len_exp_replay'] = constants.DEFAULT_REPLAY_BUFFER_VECTOR

    # ===== OPTIMIZER DEFAULTS =====
    set_default('optimizer', constants.DEFAULT_OPTIMIZER)
    set_default('weight_decay', constants.DEFAULT_WEIGHT_DECAY)
    set_default('momentum', constants.DEFAULT_MOMENTUM)

    # Learning rate: problem-specific defaults
    if 'lr' not in config:
        prob = config.get('prob', constants.DEFAULT_PROB)
        problem = config.get('problem', constants.DEFAULT_PROBLEM)
        if problem == 'graph':
            config['lr'] = constants.DEFAULT_LR_GRAPH
        elif prob == 'classification':
            config['lr'] = constants.DEFAULT_LR_CLASSIFICATION
        else:
            config['lr'] = constants.DEFAULT_LR_REGRESSION

    # ===== LEARNING RATE SCHEDULE DEFAULTS =====
    set_default('lr_schedule', constants.DEFAULT_LR_SCHEDULE)
    set_default('lr_decay_factor', constants.DEFAULT_LR_DECAY_FACTOR)
    set_default('lr_decay_steps', constants.DEFAULT_LR_DECAY_STEPS)
    set_default('lr_min', constants.DEFAULT_LR_MIN)

    # ===== HAMILTONIAN GRADIENT DEFAULTS =====
    set_default('flag', constants.DEFAULT_FLAG)
    set_default('grad_weights', constants.DEFAULT_GRAD_WEIGHTS)

    # Added by Claude: dV normalization and gradient clipping (Priority 1 improvements)
    set_default('normalize_dV', constants.DEFAULT_NORMALIZE_DV)
    set_default('dV_scale_factor', constants.DEFAULT_DV_SCALE_FACTOR)
    set_default('gradient_clip_norm', constants.DEFAULT_GRADIENT_CLIP_NORM)

    # ===== DEBUG MODE DEFAULTS =====
    set_default('debug_mode', constants.DEFAULT_DEBUG_MODE)
    set_default('debug_limit', constants.DEFAULT_DEBUG_LIMIT)

    # ===== MODEL ARCHITECTURE DEFAULTS =====
    network = config.get('network', constants.DEFAULT_NETWORK)

    # MLP defaults
    if network == 'fcnn':
        set_default('n_layers', constants.DEFAULT_N_LAYERS)
        set_default('hln', constants.DEFAULT_HLN)

    # CNN defaults
    elif network in ['cnn', 'cnn3d']:
        set_default('filter_size', constants.DEFAULT_FILTER_SIZE)

        # Dataset-specific defaults
        data = config.get('data', constants.DEFAULT_DATA)
        if data in ['mnist', 'permuted_mnist']:
            set_default('channel_out', constants.DEFAULT_CHANNEL_OUT_CNN)
            set_default('channel_in', constants.DEFAULT_CHANNEL_IN_MNIST)
            set_default('input_size', constants.DEFAULT_INPUT_SIZE_MNIST)
            set_default('feed_sizes', constants.DEFAULT_CNN_MNIST_FEED)
        elif data in ['cifar10', 'cifar100']:
            set_default('channel_out', constants.DEFAULT_CHANNEL_OUT_CNN3D)
            set_default('channel_in', constants.DEFAULT_CHANNEL_IN_CIFAR)
            set_default('input_size', constants.DEFAULT_INPUT_SIZE_CIFAR)
            set_default('feed_sizes', constants.DEFAULT_CNN_CIFAR_FEED)

    # GCN defaults
    elif network == 'gcn':
        set_default('gcn_sizes', constants.DEFAULT_GCN_SIZES)
        set_default('feed_sizes', constants.DEFAULT_GCN_FEED_SIZES)

    # ===== CLASSIFICATION DEFAULTS =====
    if config.get('prob') == 'classification':
        set_default('n_class', constants.DEFAULT_NUM_CLASSES)

    # ===== DATASET-SPECIFIC DEFAULTS =====
    data = config.get('data', constants.DEFAULT_DATA)

    if data == 'sine':
        set_default('delta', constants.DEFAULT_SINE_DELTA)
        set_default('test_size', constants.DEFAULT_SINE_TEST_SIZE)

    elif data in ['mnist', 'permuted_mnist', 'cifar10', 'cifar100']:
        set_default('rotation_range', constants.DEFAULT_ROTATION_RANGE)
        set_default('scaling_range', constants.DEFAULT_SCALING_RANGE)
        if data == 'permuted_mnist':
            set_default('permutation_seed_multiplier', constants.DEFAULT_PERMUTATION_SEED_MULTIPLIER)

    elif data == 'synthetic':
        set_default('num_graphs', constants.DEFAULT_SYNTHETIC_NUM_GRAPHS)
        set_default('num_channels', constants.DEFAULT_SYNTHETIC_NUM_CHANNELS)
        set_default('avg_num_nodes', constants.DEFAULT_SYNTHETIC_AVG_NUM_NODES)
        set_default('num_classes', constants.DEFAULT_SYNTHETIC_NUM_CLASSES)
        set_default('class_per_task', constants.DEFAULT_CLASS_PER_TASK)

    # ===== AWB DEFAULTS =====
    set_default('awb_enabled', constants.DEFAULT_AWB_ENABLED)

    if config.get('awb_enabled', False):
        set_default('awb_preliminary_epochs', constants.DEFAULT_AWB_PRELIMINARY_EPOCHS)
        set_default('awb_ab_training_epochs', constants.DEFAULT_AWB_AB_TRAINING_EPOCHS)
        set_default('awb_ab_warmup_epochs', constants.DEFAULT_AWB_AB_WARMUP_EPOCHS)
        set_default('awb_ab_max_iterations', constants.DEFAULT_AWB_AB_MAX_ITERATIONS)
        set_default('awb_averaging_window', constants.DEFAULT_AWB_AVERAGING_WINDOW)

    # ===== ARCHITECTURE SEARCH DEFAULTS =====
    set_default('arch_search_enabled', constants.DEFAULT_ARCH_SEARCH_ENABLED)

    if config.get('arch_search_enabled', False) or config.get('awb_enabled', False):
        set_default('arch_search_epochs', constants.DEFAULT_ARCH_SEARCH_EPOCHS)
        set_default('arch_search_lr', constants.DEFAULT_ARCH_SEARCH_LR)
        set_default('arch_search_batch_size', constants.DEFAULT_ARCH_SEARCH_BATCH_SIZE)
        set_default('arch_search_exp_replay', constants.DEFAULT_ARCH_SEARCH_EXP_REPLAY)
        set_default('arch_search_max_iter', constants.DEFAULT_ARCH_SEARCH_MAX_ITER)
        set_default('arch_search_loss_threshold', constants.DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD)

        # CNN-specific arch search defaults
        if network in ['cnn', 'cnn3d']:
            set_default('arch_search_hidden_range', constants.DEFAULT_ARCH_SEARCH_HIDDEN_RANGE)
            set_default('arch_search_filter_min', constants.DEFAULT_ARCH_SEARCH_FILTER_MIN)
            set_default('arch_search_filter_max', constants.DEFAULT_ARCH_SEARCH_FILTER_MAX)

        # MLP/GCN-specific arch search defaults
        elif network in ['fcnn', 'gcn']:
            set_default('arch_search_step_size_mlp', constants.DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP)
            set_default('arch_search_range', constants.DEFAULT_ARCH_SEARCH_RANGE)

    return config


def load_config(json_path: str, apply_defaults_flag: bool = True) -> Dict[str, Any]:
    """Load configuration from JSON file with automatic defaults.

    Convenience function that returns a dictionary directly.
    Automatically applies defaults from constants.py unless disabled.

    Args:
        json_path: Path to JSON configuration file
        apply_defaults_flag: If True, apply defaults from constants.py (default: True)

    Returns:
        Dictionary of configuration parameters with defaults applied
    """
    with open(json_path) as f:
        config = json.load(f)

    if apply_defaults_flag:
        config = apply_defaults(config)

    return config
