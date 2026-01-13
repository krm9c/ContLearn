"""
Command-line interface for jax-profiler.

Usage:
    jax-profile gpu              # Show GPU stats
    jax-profile monitor          # Start GPU monitoring
    jax-profile tensorboard DIR  # Profile with TensorBoard
    jax-profile nsight SCRIPT    # Profile with Nsight Systems
"""

import argparse
import sys
import time
import json
from pathlib import Path


def cmd_gpu(args):
    """Show current GPU statistics."""
    from .standard.gpu_monitor import get_gpu_stats, get_gpu_memory_jax

    print("="*60)
    print("GPU STATUS")
    print("="*60)

    # nvidia-smi stats
    stats = get_gpu_stats()
    if stats:
        print(f"\nGPU: {stats['name']}")
        print(f"Memory: {stats['memory_used_mb']:.0f}MB / {stats['memory_total_mb']:.0f}MB "
              f"({stats['memory_free_mb']:.0f}MB free)")
        print(f"Utilization: {stats['utilization']}%")
        print(f"Temperature: {stats['temperature_c']}°C")
    else:
        print("\nnvidia-smi not available")

    # JAX stats
    jax_stats = get_gpu_memory_jax()
    if jax_stats:
        print(f"\nJAX Memory: {jax_stats['gb_in_use']:.2f}GB in use, "
              f"{jax_stats['peak_gb']:.2f}GB peak")

    print("="*60)


def cmd_monitor(args):
    """Monitor GPU utilization."""
    from .standard.gpu_monitor import GPUMonitor, print_gpu_summary

    print(f"Monitoring GPU (interval={args.interval}s, duration={args.duration}s)")
    print("Press Ctrl+C to stop early...")

    monitor = GPUMonitor(interval=args.interval)
    monitor.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass

    stats = monitor.stop()
    print_gpu_summary(stats)

    if args.output:
        output = {
            "duration_sec": stats.duration_sec,
            "samples": stats.samples,
            "utilization_mean": stats.utilization_mean,
            "utilization_min": stats.utilization_min,
            "utilization_max": stats.utilization_max,
            "memory_mean_mb": stats.memory_mean_mb,
            "memory_max_mb": stats.memory_max_mb,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


def cmd_tensorboard(args):
    """Profile with TensorBoard."""
    from .standard.tensorboard import TensorBoardProfiler

    log_dir = Path(args.log_dir)
    print(f"TensorBoard profiling to: {log_dir}")
    print(f"Run your training code, then view with:")
    print(f"  tensorboard --logdir={log_dir}")

    # If script provided, run it
    if args.script:
        import subprocess
        import os

        # Set up profiling
        env = os.environ.copy()
        env['JAX_PROFILER_LOG_DIR'] = str(log_dir)

        print(f"\nRunning: python {args.script} {' '.join(args.args)}")
        with TensorBoardProfiler(str(log_dir)):
            result = subprocess.run(
                ['python', args.script] + args.args,
                env=env
            )
        sys.exit(result.returncode)


def cmd_nsight(args):
    """Profile with Nsight Systems."""
    from .standard.nsight import NsightWrapper, print_nsight_instructions

    if args.help_nsight:
        print_nsight_instructions()
        return

    nsight = NsightWrapper(output_dir=args.output_dir)
    if not nsight.available:
        print("Nsight Systems (nsys) not found. Install from:")
        print("  https://developer.nvidia.com/nsight-systems")
        sys.exit(1)

    if args.script:
        report = nsight.profile_script(
            args.script,
            args=args.args,
            output_name=args.name or "profile"
        )
        if report and args.stats:
            print("\n" + "="*60)
            print("NSIGHT STATISTICS")
            print("="*60)
            stats = nsight.generate_stats(report)
            if stats:
                print(stats)


def main():
    parser = argparse.ArgumentParser(
        description="JAX Profiler - Industry-standard profiling toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jax-profile gpu                         # Show GPU stats
  jax-profile monitor -d 60               # Monitor GPU for 60s
  jax-profile tensorboard /tmp/traces     # Set up TensorBoard profiling
  jax-profile nsight train.py config.json # Profile with Nsight
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # gpu command
    gpu_parser = subparsers.add_parser("gpu", help="Show GPU statistics")

    # monitor command
    mon_parser = subparsers.add_parser("monitor", help="Monitor GPU utilization")
    mon_parser.add_argument("-i", "--interval", type=float, default=0.5,
                           help="Sampling interval (seconds)")
    mon_parser.add_argument("-d", "--duration", type=float, default=30,
                           help="Monitoring duration (seconds)")
    mon_parser.add_argument("-o", "--output", help="Output JSON file")

    # tensorboard command
    tb_parser = subparsers.add_parser("tensorboard", help="TensorBoard profiling")
    tb_parser.add_argument("log_dir", help="Directory for traces")
    tb_parser.add_argument("script", nargs="?", help="Script to profile")
    tb_parser.add_argument("args", nargs="*", help="Script arguments")

    # nsight command
    ns_parser = subparsers.add_parser("nsight", help="Nsight Systems profiling")
    ns_parser.add_argument("script", nargs="?", help="Script to profile")
    ns_parser.add_argument("args", nargs="*", help="Script arguments")
    ns_parser.add_argument("-o", "--output-dir", default="/tmp/nsight",
                          help="Output directory")
    ns_parser.add_argument("-n", "--name", help="Output name")
    ns_parser.add_argument("--stats", action="store_true",
                          help="Show statistics after profiling")
    ns_parser.add_argument("--help-nsight", action="store_true",
                          help="Show Nsight usage instructions")

    args = parser.parse_args()

    if args.command == "gpu":
        cmd_gpu(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "tensorboard":
        cmd_tensorboard(args)
    elif args.command == "nsight":
        cmd_nsight(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
