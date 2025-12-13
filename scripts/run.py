"""
Unified training script for continual learning experiments.

This script serves as the entry point for:
- Graph Classification
- Regression Problems
- Classification Problems

With architecture search and adaptive learning support.
"""

import argparse
import os
import pickle

from contlearn.config import Params
from contlearn.training import train_model_graph, train_model_reg, train_model_class
from contlearn.trainers import Trainer

import jax
print(jax.devices()) # Should list your GPU(s)
print(jax.default_backend()) # Should show 'gpu' or 'tpu'


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Unified training script for continual learning"
    )
    subparsers = parser.add_subparsers(help='', dest='command')

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("runs", default=1, help="the number of total runs")
    train_parser.add_argument("json", default=None, help="directory with configurations")

    basic_path = 'config/jsons/'
    args = parser.parse_args()
    json_path = os.path.join(basic_path + str(args.json))

    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"

    params = Params(json_path).dict

    if args.runs is not None:
        params['runs'] = int(args.runs)
    else:
        params['runs'] = 5

    if args.command == 'train':
        all_runs_records = {}

        for j in range(params['runs']):
            print(f"runs {j}, problem: {params['problem']}")

            if params['prob'] == 'regression':
                all_runs_records[str(j)] = train_model_reg(params, run_id=j)
            elif params['prob'] == 'classification':
                print("Starting classification training...")
                all_runs_records[str(j)] = train_model_class(params, run_id=j)
            elif params['problem'] == 'graph':
                all_runs_records[str(j)] = train_model_graph(params, run_id=j)

        # Save all runs together using the unified recording system
        if all_runs_records:
            # Use the save_all_runs static method
            Trainer.save_all_runs(all_runs_records, params.get('model_path', ''), params)

        # Legacy pickle save if 'file' parameter exists
        if 'file' in params:
            with open(str(params['file']) + '.pkl', 'wb') as f:
                pickle.dump(all_runs_records, f)
            print(f"Saved results to {params['file']}.pkl")


if __name__ == "__main__":
    main()
