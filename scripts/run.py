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

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cl.config import Params, load_config
from cl.runners.regression import train_model_reg
from cl.runners.classification import train_model_class
from cl.runners.graph_classification import train_model_graph
from cl.core.recording import RecordingMixin

# Added by Claude: Import plotting module for post-training visualization
from plot_results import generate_plots


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run.py config/sine.json
    python scripts/run.py config/sine.json --runs 3
    python scripts/run.py config/sine.json --no-plots
    python scripts/run.py config/sine.json --figures-dir outputs/figures
        """
    )
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--runs', type=int, default=1, help='Number of experiment runs')
    # Added by Claude: Options for plot generation
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Output directory for figures')

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Problem type: {config.get('prob', 'unknown')}")
    print(f"Dataset: {config.get('data', 'unknown')}")
    print(f"Tasks: {config.get('n_task', 'unknown')}")
    print(f"AWB enabled: {config.get('awb_enabled', False)}")
    print()

    # Route to appropriate training function based on problem type
    prob_type = config.get('prob', 'regression')
    problem = config.get('problem', 'vectors')

    all_records = {}

    for run_id in range(args.runs):
        print(f"\n{'#'*60}")
        print(f"# Run {run_id + 1} / {args.runs}")
        print(f"{'#'*60}")

        # Added by Claude: Check problem field for graph classification
        if problem == 'graph':
            record_dict = train_model_graph(config, run_id=run_id)
        elif prob_type == 'regression':
            record_dict = train_model_reg(config, run_id=run_id)
        elif prob_type == 'classification':
            record_dict = train_model_class(config, run_id=run_id)
        else:
            print(f"Unknown problem type: {prob_type}, problem: {problem}")
            sys.exit(1)

        all_records[f'run_{run_id}'] = record_dict

        # Added by Claude: Generate plots for each run (unless --no-plots)
        if not args.no_plots:
            generate_plots(record_dict, output_dir=args.figures_dir, run_id=str(run_id))

    # Save all runs if multiple
    if args.runs > 1:
        RecordingMixin.save_all_runs(all_records, config.get('model_path', 'outputs/model'), config)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
