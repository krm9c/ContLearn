#!/usr/bin/env python3
"""
AWB Pipeline Profiler - External profiling for ContLearn AWB pipeline.

This script profiles the AWB pipeline without modifying ContLearn code.
It wraps the training and collects timing data for each phase.

Usage:
    python scripts/profile_awb.py <config_path> [--output FILE]

Example:
    python scripts/profile_awb.py /path/to/ContLearn/tests/training/configs/mnist_awb.json

The script:
1. Loads ContLearn and the config
2. Wraps training with AWB hooks
3. Runs training with GPU monitoring
4. Outputs detailed analysis
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add jax-profiler to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))


def main():
    parser = argparse.ArgumentParser(description='Profile AWB pipeline')
    parser.add_argument('config', help='Path to ContLearn config file')
    parser.add_argument('--contlearn-path', help='Path to ContLearn repo',
                       default=str(Path(__file__).parent.parent.parent / 'ContLearn'))
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--gpu-interval', type=float, default=0.5,
                       help='GPU monitoring interval (seconds)')

    args = parser.parse_args()

    # Validate paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    contlearn_path = Path(args.contlearn_path)
    if not contlearn_path.exists():
        print(f"Error: ContLearn repo not found: {contlearn_path}")
        sys.exit(1)

    # Add ContLearn to path
    sys.path.insert(0, str(contlearn_path / 'src'))
    sys.path.insert(0, str(contlearn_path))

    print("="*70)
    print("AWB PIPELINE PROFILER")
    print("="*70)
    print(f"\nConfig: {config_path}")
    print(f"ContLearn: {contlearn_path}")

    # Import profiling tools
    from jax_profiler import GPUMonitor
    from jax_profiler.hooks import AWBPipelineHooks

    # Import ContLearn
    print("\nLoading ContLearn...")
    from cl.config import load_config
    from cl.runners import train_model

    # Load config
    config = load_config(str(config_path))

    print(f"\nConfig settings:")
    print(f"  data: {config.get('data')}")
    print(f"  network: {config.get('network')}")
    print(f"  awb_enabled: {config.get('awb_enabled')}")
    print(f"  n_task: {config.get('n_task')}")
    print(f"  epochs_per_task: {config.get('epochs_per_task')}")

    if not config.get('awb_enabled', False):
        print("\nWARNING: awb_enabled is False. Enable it to profile AWB pipeline.")

    # Initialize profiling
    awb_hooks = AWBPipelineHooks(gpu_monitoring=True)
    gpu_monitor = GPUMonitor(interval=args.gpu_interval)

    # Disable plot generation
    config['generate_plots'] = False

    print("\n" + "="*70)
    print("STARTING PROFILED TRAINING")
    print("="*70)

    # Start monitoring
    gpu_monitor.start()
    total_start = time.time()

    # Note: Since we can't easily inject hooks into the AWB pipeline without
    # modifying ContLearn, we'll run training and collect GPU stats.
    # For detailed phase-level profiling, the hooks would need to be called
    # from within the AWB pipeline code.
    try:
        result = train_model(config)
        success = True
    except Exception as e:
        import traceback
        print(f"\nTraining failed: {e}")
        traceback.print_exc()
        success = False
        result = {}

    total_time = time.time() - total_start
    gpu_stats = gpu_monitor.stop()

    # Print GPU summary
    print("\n" + "="*70)
    print("GPU MONITORING RESULTS")
    print("="*70)
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"GPU samples: {gpu_stats.samples}")
    print(f"GPU utilization: {gpu_stats.utilization_mean:.1f}% "
          f"(min={gpu_stats.utilization_min:.0f}%, max={gpu_stats.utilization_max:.0f}%)")
    print(f"Memory: {gpu_stats.memory_mean_mb:.0f}MB mean, {gpu_stats.memory_max_mb:.0f}MB peak")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if gpu_stats.utilization_mean < 30:
        print("\n[LOW GPU UTILIZATION DETECTED]")
        print(f"  Average: {gpu_stats.utilization_mean:.1f}%")
        print("\n  Likely causes for AWB pipeline:")
        print("  1. A/B training phase - computes A @ W @ B.T inside gradient")
        print("  2. Frequent JIT recompilation when switching training modes")
        print("  3. Architecture search overhead")
        print("  4. Data loading bottleneck")
        print("\n  Recommendations:")
        print("  - Reduce awb_ab_training_epochs")
        print("  - Use larger batch sizes")
        print("  - Consider Condition 3 (skip_transfer=True)")
        print("  - Profile with TensorBoard for JIT traces")

    # Save report
    if args.output:
        report = {
            'config_path': str(config_path),
            'total_time_sec': total_time,
            'success': success,
            'gpu': {
                'samples': gpu_stats.samples,
                'utilization_mean': gpu_stats.utilization_mean,
                'utilization_min': gpu_stats.utilization_min,
                'utilization_max': gpu_stats.utilization_max,
                'memory_mean_mb': gpu_stats.memory_mean_mb,
                'memory_max_mb': gpu_stats.memory_max_mb,
            },
            'config': {
                'data': config.get('data'),
                'network': config.get('network'),
                'awb_enabled': config.get('awb_enabled'),
                'n_task': config.get('n_task'),
                'epochs_per_task': config.get('epochs_per_task'),
                'awb_preliminary_epochs': config.get('awb_preliminary_epochs'),
                'awb_ab_training_epochs': config.get('awb_ab_training_epochs'),
            },
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)

    # Instructions for detailed profiling
    print("\nFor detailed phase-level profiling, you can:")
    print("1. Use TensorBoard profiler:")
    print(f"   jax-profile tensorboard /tmp/awb_traces python run.py {config_path}")
    print("2. Use Nsight Systems:")
    print(f"   nsys profile -o awb_profile python run.py {config_path}")
    print("3. Enable JAX compile logging:")
    print("   JAX_LOG_COMPILES=1 python run.py " + str(config_path))


if __name__ == '__main__':
    main()
