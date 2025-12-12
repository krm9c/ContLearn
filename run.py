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

from config import Params
from training import train_model_graph, train_model_reg, train_model_class

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
        record_dict = {}
        record_dict_preAB = {}
        record_dict_AB = {}

        for j in range(params['runs']):
            print(f"runs {j}, problem: {params['problem']}")

            if params['prob'] == 'regression':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_reg(params)
            elif params['prob'] == 'classification':
                print("Starting classification training...")
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_class(params)
            elif params['problem'] == 'graph':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_graph(params)

        # Save results
        if 'file' in params:
            with open(str(params['file']) + '.pkl', 'wb') as f:
                pickle.dump(record_dict, f)
            print(f"Saved results to {params['file']}.pkl")


if __name__ == "__main__":
    main()
