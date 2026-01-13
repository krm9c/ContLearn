#!/usr/bin/env python3
"""
AWB Pipeline Profiler - Run this to profile ContLearn AWB training.

Usage:
    python run_awb_profile.py                    # Uses default test config
    python run_awb_profile.py --config PATH      # Use specific config
    python run_awb_profile.py --quick            # Quick mode (fewer epochs)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
CONTLEARN_DIR = SCRIPT_DIR.parent / "ContLearn"

# Add to Python path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONTLEARN_DIR / "src"))
sys.path.insert(0, str(CONTLEARN_DIR))


def main():
    parser = argparse.ArgumentParser(description="Profile AWB Pipeline")
    parser.add_argument("--config", "-c", type=str,
                       help="Config file path (default: mnist_awb test config)")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode - reduce epochs for faster profiling")
    parser.add_argument("--output", "-o", type=str, default="awb_profile_results.json",
                       help="Output JSON file")
    parser.add_argument("--gpu-interval", type=float, default=0.5,
                       help="GPU sampling interval in seconds")
    args = parser.parse_args()

    # Default config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = CONTLEARN_DIR / "tests" / "training" / "configs" / "mnist_awb.json"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        print(f"\nAvailable test configs:")
        test_configs = list((CONTLEARN_DIR / "tests" / "training" / "configs").glob("*.json"))
        for c in test_configs[:10]:
            print(f"  {c}")
        sys.exit(1)

    print("=" * 70)
    print("AWB PIPELINE PROFILER")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"ContLearn: {CONTLEARN_DIR}")
    print(f"Output: {args.output}")

    # Import profiler
    from jax_profiler import GPUMonitor, TensorBoardProfiler
    from jax_profiler.hooks import AWBPipelineHooks

    # Import JAX and check backend
    import jax
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # Import ContLearn
    print("\nLoading ContLearn...")
    from cl.config import load_config
    from cl.runners import train_model

    # Load and modify config
    config = load_config(str(config_path))

    print(f"\nConfig settings:")
    print(f"  data: {config.get('data')}")
    print(f"  network: {config.get('network')}")
    print(f"  awb_enabled: {config.get('awb_enabled')}")
    print(f"  n_task: {config.get('n_task')}")
    print(f"  epochs_per_task: {config.get('epochs_per_task')}")
    print(f"  awb_preliminary_epochs: {config.get('awb_preliminary_epochs')}")
    print(f"  awb_ab_training_epochs: {config.get('awb_ab_training_epochs')}")

    if not config.get('awb_enabled', False):
        print("\nWARNING: awb_enabled is False in config!")
        print("Setting awb_enabled=True for profiling...")
        config['awb_enabled'] = True

    # Quick mode overrides
    if args.quick:
        print("\n[QUICK MODE] Reducing epochs...")
        config['n_task'] = min(config.get('n_task', 2), 2)
        config['epochs_per_task'] = min(config.get('epochs_per_task', 5), 3)
        config['awb_preliminary_epochs'] = min(config.get('awb_preliminary_epochs', 3), 2)
        config['awb_ab_training_epochs'] = min(config.get('awb_ab_training_epochs', 3), 2)
        config['debug_mode'] = True
        config['debug_limit'] = min(config.get('debug_limit', 1000), 500)
        print(f"  n_task: {config['n_task']}")
        print(f"  epochs_per_task: {config['epochs_per_task']}")
        print(f"  debug_limit: {config['debug_limit']}")

    # Disable plots
    config['generate_plots'] = False

    # Start GPU monitoring
    print("\n" + "=" * 70)
    print("STARTING PROFILED TRAINING")
    print("=" * 70)

    gpu_monitor = GPUMonitor(interval=args.gpu_interval)
    gpu_monitor.start()

    total_start = time.time()

    try:
        result = train_model(config)
        success = True
        error_msg = None
    except Exception as e:
        import traceback
        success = False
        error_msg = str(e)
        print(f"\nERROR: Training failed: {e}")
        traceback.print_exc()
        result = {}

    total_time = time.time() - total_start
    gpu_stats = gpu_monitor.stop()

    # Print results
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Success: {success}")

    print(f"\nGPU Statistics ({gpu_stats.samples} samples):")
    print(f"  Utilization: {gpu_stats.utilization_mean:.1f}% mean")
    print(f"               {gpu_stats.utilization_min:.0f}% min, {gpu_stats.utilization_max:.0f}% max")
    print(f"  Memory: {gpu_stats.memory_mean_mb:.0f}MB mean, {gpu_stats.memory_max_mb:.0f}MB peak")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    if gpu_stats.utilization_mean < 30:
        print(f"\n⚠️  LOW GPU UTILIZATION: {gpu_stats.utilization_mean:.1f}%")
        print("\nLikely causes in AWB pipeline:")
        print("  1. A/B training phase computes A @ W @ B.T inside gradient")
        print("     - This cannot be cached and is the primary bottleneck")
        print("  2. Architecture search evaluates multiple configurations")
        print("  3. JIT recompilation when switching training modes")
        print("  4. Small batch sizes or data loading overhead")
        print("\nRecommendations:")
        print("  - Reduce awb_ab_training_epochs")
        print("  - Increase batch_size")
        print("  - Use Condition 3 (awb_skip_transfer=True) to skip A/B training")
        print("  - Profile with TensorBoard for detailed XLA traces")
    elif gpu_stats.utilization_mean < 50:
        print(f"\n⚡ MODERATE GPU UTILIZATION: {gpu_stats.utilization_mean:.1f}%")
        print("  Consider increasing batch size or reducing eval frequency")
    else:
        print(f"\n✓ GOOD GPU UTILIZATION: {gpu_stats.utilization_mean:.1f}%")

    # Save report
    import json
    report = {
        'config_path': str(config_path),
        'total_time_sec': total_time,
        'success': success,
        'error': error_msg,
        'gpu': {
            'samples': gpu_stats.samples,
            'utilization_mean': gpu_stats.utilization_mean,
            'utilization_min': gpu_stats.utilization_min,
            'utilization_max': gpu_stats.utilization_max,
            'utilization_std': gpu_stats.utilization_std,
            'memory_mean_mb': gpu_stats.memory_mean_mb,
            'memory_max_mb': gpu_stats.memory_max_mb,
        },
        'config': {
            'data': config.get('data'),
            'network': config.get('network'),
            'awb_enabled': config.get('awb_enabled'),
            'n_task': config.get('n_task'),
            'epochs_per_task': config.get('epochs_per_task'),
            'batch_size': config.get('batch_size'),
            'awb_preliminary_epochs': config.get('awb_preliminary_epochs'),
            'awb_ab_training_epochs': config.get('awb_ab_training_epochs'),
        },
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path.absolute()}")

    # Next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR DETAILED PROFILING")
    print("=" * 70)
    print("\n1. TensorBoard profiling (XLA traces):")
    print(f"   cd {CONTLEARN_DIR}")
    print(f"   python -c \"import jax; jax.profiler.start_trace('/tmp/awb_traces')\" && \\")
    print(f"   python run.py {config_path} && \\")
    print(f"   python -c \"import jax; jax.profiler.stop_trace()\"")
    print("   tensorboard --logdir=/tmp/awb_traces")
    print("\n2. Enable JAX compile logging:")
    print(f"   JAX_LOG_COMPILES=1 python run.py {config_path}")
    print("\n3. NVIDIA Nsight profiling:")
    print(f"   nsys profile -o awb_report python run.py {config_path}")
    print("   nsys stats awb_report.nsys-rep")

    print("\n" + "=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
