#!/usr/bin/env python
"""
Entry point for multi-GPU continual learning experiments.

This script enables data-parallel training via JAX pmap for vector problems
(MLP/CNN). Graph workloads currently fall back to single-device execution.

Usage:
    python run_multi_gpu.py <config_file>
    python run_multi_gpu.py config/sine.json --runs 0
    python run_multi_gpu.py config/mnist.json --runs 1 --no-plots

Arguments:
    config_file: Path to JSON configuration file
    --runs: Run ID for this experiment (default: 0)
    --no-plots: Skip plot generation (default: generate plots)
    --figures-dir: Output directory for figures (default: figures)
"""
import argparse
import sys
import os
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Add run_files/scripts to path for plot_results import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'run_files', 'scripts'))

import jax
from cl.config import load_config
from cl.runners import train_model

# Suppress JAX MaxPool gradient warning (expected behavior for second-order derivatives)
warnings.filterwarnings('ignore', message='.*reduce-window min/max.*')


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Framework (Multi-GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--runs', type=int, default=0, help='Run ID for this experiment (default: 0)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Output directory for figures')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory for results')
    parser.add_argument('--model-suffix', type=str, default=None, help='Custom suffix for model/records files')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Enable multi-GPU pmap path in training loop
    config['multi_gpu'] = True
    config['multi_gpu_axis'] = config.get('multi_gpu_axis', 'devices')
    config['multi_gpu_debug'] = config.get('multi_gpu_debug', True)

    # Print device info
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()

    print(f"Loaded config from: {args.config}")
    print(f"Problem type: {config.get('prob', 'unknown')}")
    print(f"Dataset: {config.get('data', 'unknown')}")
    print(f"Tasks: {config.get('n_task', 'unknown')}")
    print(f"AWB enabled: {config.get('awb_enabled', False)}")
    print(f"Multi-GPU enabled: {config.get('multi_gpu', False)}")
    print()

    # Construct base path with AWB suffix if enabled and not already present
    if args.output_dir:
        base_model_path = os.path.join(args.output_dir, args.model_suffix or 'model')
    else:
        base_model_path = config.get('model_path', 'outputs/model')
        if config.get('awb_enabled', False) and '_awb' not in base_model_path:
            base_model_path = f"{base_model_path}_awb"

    if args.output_dir:
        config['model_path'] = base_model_path
        print(f"Output directory: {args.output_dir}")
        print(f"Model path: {base_model_path}")
        print()

    run_id = args.runs
    print(f"\n{'#'*60}")
    print(f"# Run ID: {run_id}")
    print(f"{'#'*60}")

    record_dict = train_model(config, run_id=run_id)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
