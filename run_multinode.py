#!/usr/bin/env python
"""
Multi-node entry point using MPI + mpi4jax.

Each MPI rank pins to one GPU. Gradient averaging is done via
mpi4jax.allreduce after each batch.

Launch on Polaris:
    mpiexec -n $((NNODES * 4)) -ppn 4 python run_multinode.py <config>

This gives 4 ranks per node (one per GPU). Each rank computes gradients
on its own data shard; allreduce averages them across all ranks.
"""
import argparse
import sys
import os
import warnings

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
local_rank = int(os.environ.get('PMI_LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))

# Try CUDA_VISIBLE_DEVICES first (before JAX import)
os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'run_files', 'scripts'))

import jax
import jax.numpy as jnp

# If CUDA_VISIBLE_DEVICES didn't work (conda module pre-inits CUDA),
# use JAX's own device pinning: set the default device for this rank.
all_gpus = jax.devices("gpu")
if len(all_gpus) > 1 and local_rank < len(all_gpus):
    my_device = all_gpus[local_rank]
    jax.config.update("jax_default_device", my_device)
    print(f"[MPI] rank {rank} pinned to {my_device} (of {len(all_gpus)} visible)")
else:
    my_device = all_gpus[0]
    print(f"[MPI] rank {rank} using {my_device}")

from cl.config import load_config
from cl.runners import train_model

warnings.filterwarnings('ignore', message='.*reduce-window min/max.*')


def main():
    parser = argparse.ArgumentParser(description='Continual Learning (Multi-Node MPI)')
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--runs', type=int, default=0, help='Run ID')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model-suffix', type=str, default=None)
    args = parser.parse_args()

    is_main = (rank == 0)

    if is_main:
        print(f"JAX Backend:   {jax.default_backend()}")
        print(f"MPI world:     {world_size} ranks (1 GPU each)")
        print(f"Device:        {jax.devices()[0]}")
        print(f"Total GPUs:    {world_size}")
        print()

    if not os.path.exists(args.config):
        if is_main:
            print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    config['multi_node'] = True
    config['mpi_comm'] = comm
    config['mpi_rank'] = rank
    config['jax_device'] = my_device
    config['mpi_world_size'] = world_size
    config['multi_gpu'] = False

    if args.output_dir:
        base_model_path = os.path.join(args.output_dir, args.model_suffix or 'model')
        config['model_path'] = base_model_path

    if is_main:
        print(f"Loaded config from: {args.config}")
        print(f"Dataset: {config.get('data', 'unknown')}")
        print(f"Tasks: {config.get('n_task', 'unknown')}")
        print(f"AWB enabled: {config.get('awb_enabled', False)}")
        print()
        print(f"{'#' * 60}")
        print(f"# Run ID: {args.runs}")
        print(f"{'#' * 60}")

    record_dict = train_model(config, run_id=args.runs)

    if is_main:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
