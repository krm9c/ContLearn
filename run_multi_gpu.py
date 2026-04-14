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

# Added by Claude: Multi-node JAX initialization (must happen before any other JAX call).
#
# On Polaris (PALS launcher), jax.distributed's auto-discovery does NOT work -- we must
# derive coordinator_address from PBS_NODEFILE and pass num_processes/process_id from
# PMI_* env vars explicitly. This function also supports:
#   - SLURM clusters (SLURM_PROCID / SLURM_NTASKS / SLURM_STEP_NODELIST)
#   - Open MPI (OMPI_COMM_WORLD_RANK / _SIZE)
#   - Single-process launches (no init required)
#
# Launch patterns on Polaris:
#   1 rank per node (recommended, matches existing pmap code):
#       mpiexec -n $NNODES -ppn 1 python run_multi_gpu.py <config>
#     -> each rank sees all 4 local GPUs; pmap shards batch across them;
#        distributed collectives span all NNODES*4 GPUs.
#   N ranks per node (1 rank per GPU):
#       mpiexec -n $(NNODES*4) -ppn 4 python run_multi_gpu.py <config>
#     -> each rank pinned to 1 local GPU via local_device_ids=[PMI_LOCAL_RANK];
#        pmap axis still spans all NNODES*4 GPUs globally.
def _maybe_init_distributed():
    # Rank / size env vars (in order of preference)
    def _first_env(*names):
        for n in names:
            v = os.environ.get(n)
            if v is not None:
                return v
        return None

    rank_s = _first_env('PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK')
    size_s = _first_env('PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE')
    local_rank_s = _first_env('PMI_LOCAL_RANK', 'SLURM_LOCALID',
                               'OMPI_COMM_WORLD_LOCAL_RANK')
    local_size_s = _first_env('PMI_LOCAL_SIZE', 'SLURM_NTASKS_PER_NODE',
                               'OMPI_COMM_WORLD_LOCAL_SIZE')

    if rank_s is None or size_s is None:
        print("[JAX-DIST] No PMI/SLURM/OMPI env vars detected -- running single-process.")
        return False

    num_processes = int(size_s)
    process_id = int(rank_s)
    if num_processes == 1:
        print("[JAX-DIST] num_processes=1 -- single-process.")
        return False

    # Derive coordinator from the first hostname in PBS_NODEFILE
    # (Polaris/PBS). Fall back to SLURM_STEP_NODELIST on SLURM.
    coordinator_host = None
    nodefile = os.environ.get('PBS_NODEFILE')
    if nodefile and os.path.exists(nodefile):
        with open(nodefile) as f:
            for line in f:
                line = line.strip()
                if line:
                    coordinator_host = line
                    break
    if coordinator_host is None:
        # Fallback: SLURM's first node
        slurm_nodes = os.environ.get('SLURM_STEP_NODELIST') or os.environ.get('SLURM_JOB_NODELIST')
        if slurm_nodes:
            coordinator_host = slurm_nodes.split(',')[0].split('[')[0]
    if coordinator_host is None:
        print("[JAX-DIST] Could not determine coordinator host (PBS_NODEFILE missing); "
              "falling back to single-process.")
        return False

    # Port: fixed but jobid-derived to avoid cross-job collisions on shared nodes.
    port = 12345
    jobid = os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID')
    if jobid:
        # Keep port in the ephemeral range, deterministic per job.
        try:
            port = 20000 + (int(''.join(c for c in jobid if c.isdigit())[:5]) % 10000)
        except Exception:
            pass
    coordinator_address = f"{coordinator_host}:{port}"

    kwargs = dict(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )

    # If launching multiple ranks per node, pin each rank to its own GPU so
    # the local_device_ids don't overlap. Otherwise let each rank see all
    # local GPUs on its node.
    if local_size_s is not None and local_rank_s is not None and int(local_size_s) > 1:
        kwargs['local_device_ids'] = [int(local_rank_s)]

    try:
        jax.distributed.initialize(**kwargs)
    except Exception as e:
        print(f"[JAX-DIST] initialize({kwargs}) failed: {e}")
        raise

    print(f"[JAX-DIST] initialized | coordinator={coordinator_address} "
          f"process_id={process_id}/{num_processes} "
          f"local_device_ids={kwargs.get('local_device_ids', 'all')}")
    return True


_maybe_init_distributed()

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
    # Added by Claude: expose process_count / global device_count so multi-node launches
    # are visibly distinct from single-node (process_count > 1 means distributed is active).
    print(f"JAX Backend:        {jax.default_backend()}")
    print(f"JAX processes:      {jax.process_count()} (this is process_id={jax.process_index()})")
    print(f"Global device count:{jax.device_count()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"Local devices:      {jax.local_devices()}")
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
