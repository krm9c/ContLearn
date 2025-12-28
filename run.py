#!/usr/bin/env python
"""
Entry point for continual learning experiments.

Usage:
    python scripts/run.py <config_file>
    python scripts/run.py config/sine.json
    python scripts/run.py config/sine.json --runs 3
    python scripts/run.py config/sine.json --runs 3 --no-plots

Arguments:
    config_file: Path to JSON configuration file
    --runs: Number of experiment runs (default: 1)
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
from cl.config import Params, load_config
# Added by Claude: Use new generic unified runner (layer-level AWB refactor)
from cl.runners import train_model
from cl.core.recording import RecordingMixin

# Added by Claude: Import plotting module for post-training visualization
from kkt_run.analysis.additional_python_scripts.scripts.plot_results import generate_plots

# Suppress JAX MaxPool gradient warning (expected behavior for second-order derivatives)
warnings.filterwarnings('ignore', message='.*reduce-window min/max.*')


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py kkt_run/config/sine.json
    python run.py kkt_run/config/sine.json --runs 3
    python run.py kkt_run/config/sine.json --no-plots
    python run.py kkt_run/config/sine.json --figures-dir outputs/figures
        """
    )
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--runs', type=int, default=1, help='Number of experiment runs')
    # Added by Claude: Options for plot generation
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Output directory for figures')
    # Added by Claude: Options for Polaris parallel runs
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory for results')
    parser.add_argument('--model-suffix', type=str, default=None, help='Custom suffix for model/records files')

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Added by Claude: Check JAX backend/device
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    print()

    print(f"Loaded config from: {args.config}")
    print(f"Problem type: {config.get('prob', 'unknown')}")
    print(f"Dataset: {config.get('data', 'unknown')}")
    print(f"Tasks: {config.get('n_task', 'unknown')}")
    print(f"AWB enabled: {config.get('awb_enabled', False)}")
    print()

    #Construct base path with AWB suffix if enabled and not already present
    if args.output_dir:
        # Use custom output directory (for Polaris runs)
        base_model_path = os.path.join(args.output_dir, args.model_suffix or 'model')
    else:
        # Default behavior
        base_model_path = config.get('model_path', 'outputs/model')
        if config.get('awb_enabled', False) and '_awb' not in base_model_path:
            base_model_path = f"{base_model_path}_awb"

    # Override config paths if custom output directory specified
    if args.output_dir:
        config['model_path'] = base_model_path
        print(f"Output directory: {args.output_dir}")
        print(f"Model path: {base_model_path}")
        print()

    # Added by Claude: Generic runner handles all problem types via config dispatch
    all_records = {}

    for run_id in range(args.runs):
        print(f"\n{'#'*60}")
        print(f"# Run {run_id + 1} / {args.runs}")
        print(f"{'#'*60}")

        # Generic unified runner works for all problem types
        record_dict = train_model(config, run_id=run_id)

        all_records[f'run_{run_id}'] = record_dict

        # Generate plots for each run (unless --no-plots)
        # Use AWB-aware figures directory
        if not args.no_plots:
            figures_dir = args.figures_dir if args.figures_dir != 'figures' or args.output_dir else args.figures_dir
            # If using default figures dir and no custom output, add AWB suffix
            if figures_dir == 'figures' and not args.output_dir and config.get('awb_enabled', False):
                dataset_name = config.get('data', 'unknown')
                figures_dir = f'figures/{dataset_name}_awb'
            generate_plots(record_dict, output_dir=figures_dir, run_id=str(run_id))

    # Save all runs if multiple
    if args.runs > 1:
        RecordingMixin.save_all_runs(all_records, base_model_path, config)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
