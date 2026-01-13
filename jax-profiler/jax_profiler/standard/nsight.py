"""
NVIDIA Nsight Systems wrapper for JAX profiling.

Nsight Systems provides detailed GPU kernel-level profiling:
- CUDA kernel execution times
- Memory transfer analysis
- CPU-GPU synchronization points
- Multi-stream execution visualization

Usage:
    # Command line (recommended):
    nsys profile -o report python train.py
    nsys stats report.nsys-rep

    # Programmatic wrapper:
    from jax_profiler import NsightWrapper

    nsight = NsightWrapper(output_dir="/tmp/nsight")
    nsight.profile_script("train.py", args=["config.json"])
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any


class NsightWrapper:
    """Wrapper for NVIDIA Nsight Systems profiling.

    This provides a Python interface to nsys commands.
    Nsight Systems must be installed separately.
    """

    def __init__(self, output_dir: str = "/tmp/nsight_profiles"):
        """Initialize Nsight wrapper.

        Args:
            output_dir: Directory to save profiling reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_nsight_available()

    def _check_nsight_available(self) -> bool:
        """Check if nsys is available."""
        self.available = shutil.which("nsys") is not None
        if not self.available:
            print("[NsightWrapper] WARNING: nsys not found in PATH")
            print("[NsightWrapper] Install NVIDIA Nsight Systems from:")
            print("  https://developer.nvidia.com/nsight-systems")
        return self.available

    def profile_script(
        self,
        script: str,
        args: Optional[List[str]] = None,
        output_name: str = "profile",
        capture_range: str = "cudaProfilerApi",
        trace: List[str] = None,
    ) -> Optional[Path]:
        """Profile a Python script using nsys.

        Args:
            script: Path to Python script
            args: Arguments to pass to script
            output_name: Name for output report (without extension)
            capture_range: When to capture (cudaProfilerApi, full, none)
            trace: What to trace (cuda, nvtx, osrt, etc.)

        Returns:
            Path to generated report or None if failed
        """
        if not self.available:
            print("[NsightWrapper] nsys not available, skipping profiling")
            return None

        args = args or []
        trace = trace or ["cuda", "nvtx", "osrt"]

        output_path = self.output_dir / output_name

        cmd = [
            "nsys", "profile",
            "-o", str(output_path),
            "--capture-range", capture_range,
            "--trace", ",".join(trace),
            "--force-overwrite", "true",
            "python", script,
        ] + args

        print(f"[NsightWrapper] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                report_path = output_path.with_suffix(".nsys-rep")
                print(f"[NsightWrapper] Report saved to: {report_path}")
                return report_path
            else:
                print(f"[NsightWrapper] Error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("[NsightWrapper] Profiling timed out")
            return None

    def generate_stats(self, report_path: Path) -> Optional[str]:
        """Generate statistics from a Nsight report.

        Args:
            report_path: Path to .nsys-rep file

        Returns:
            Statistics output as string
        """
        if not self.available:
            return None

        cmd = ["nsys", "stats", str(report_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"[NsightWrapper] Error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            return None

    def export_sqlite(self, report_path: Path) -> Optional[Path]:
        """Export Nsight report to SQLite for custom analysis.

        Args:
            report_path: Path to .nsys-rep file

        Returns:
            Path to SQLite database
        """
        if not self.available:
            return None

        sqlite_path = report_path.with_suffix(".sqlite")
        cmd = ["nsys", "export", "-t", "sqlite", "-o", str(sqlite_path), str(report_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return sqlite_path
        except subprocess.TimeoutExpired:
            pass
        return None


def print_nsight_instructions():
    """Print instructions for using Nsight Systems."""
    print("""
================================================================================
NVIDIA Nsight Systems - GPU Kernel Profiling
================================================================================

INSTALLATION:
  Download from: https://developer.nvidia.com/nsight-systems
  Or via conda: conda install -c nvidia nsight-systems

BASIC USAGE:
  # Profile a training script
  nsys profile -o report python train.py config.json

  # View statistics
  nsys stats report.nsys-rep

  # Open in GUI (if available)
  nsys-ui report.nsys-rep

USEFUL OPTIONS:
  --capture-range=cudaProfilerApi  # Only capture marked regions
  --trace=cuda,nvtx,osrt           # What to trace
  --gpu-metrics-device=0           # Capture GPU metrics
  --cudabacktrace=all              # CUDA call stacks

MARKING REGIONS IN CODE (optional):
  import jax

  # Start/stop programmatically
  jax.profiler.start_trace("/tmp/traces")
  # ... code to profile ...
  jax.profiler.stop_trace()

================================================================================
""")
