"""
GPU monitoring using nvidia-smi.

Provides real-time GPU utilization, memory, and temperature monitoring
using the industry-standard nvidia-smi tool.

Usage:
    from jax_profiler import GPUMonitor, get_gpu_stats

    # One-time stats
    stats = get_gpu_stats()
    print(f"GPU: {stats['name']}, Utilization: {stats['utilization']}%")

    # Continuous monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    # ... training code ...
    stats = monitor.stop()
    print(f"Avg utilization: {stats['utilization_mean']:.1f}%")
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from statistics import mean, stdev


@dataclass
class GPUSample:
    """Single GPU measurement sample."""
    timestamp: float
    utilization: float  # GPU utilization %
    memory_used: float  # MB
    memory_total: float  # MB
    temperature: Optional[float] = None  # Celsius
    power_draw: Optional[float] = None  # Watts


@dataclass
class GPUStats:
    """Aggregated GPU statistics."""
    samples: int = 0
    utilization_mean: float = 0.0
    utilization_min: float = 0.0
    utilization_max: float = 0.0
    utilization_std: float = 0.0
    memory_mean_mb: float = 0.0
    memory_max_mb: float = 0.0
    memory_total_mb: float = 0.0
    duration_sec: float = 0.0
    raw_samples: List[GPUSample] = field(default_factory=list)


def get_gpu_stats() -> Optional[Dict[str, Any]]:
    """Get current GPU statistics from nvidia-smi.

    Returns:
        Dict with GPU info or None if nvidia-smi not available.
        Keys: name, memory_total, memory_used, memory_free, utilization, temperature
    """
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(',')]
        if len(parts) >= 6:
            return {
                'name': parts[0],
                'memory_total_mb': float(parts[1]),
                'memory_used_mb': float(parts[2]),
                'memory_free_mb': float(parts[3]),
                'utilization': float(parts[4]),
                'temperature_c': float(parts[5]),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def get_gpu_memory_jax() -> Optional[Dict[str, float]]:
    """Get GPU memory stats from JAX.

    Returns:
        Dict with bytes_in_use, peak_bytes, or None if unavailable.
    """
    try:
        import jax
        from jax.lib import xla_bridge

        backend = xla_bridge.get_backend()
        devices = backend.devices()
        if not devices:
            return None

        device = devices[0]
        mem_stats = device.memory_stats()
        if mem_stats is None:
            return None

        return {
            'bytes_in_use': mem_stats.get('bytes_in_use', 0),
            'peak_bytes': mem_stats.get('peak_bytes_in_use', 0),
            'gb_in_use': mem_stats.get('bytes_in_use', 0) / (1024**3),
            'peak_gb': mem_stats.get('peak_bytes_in_use', 0) / (1024**3),
        }
    except Exception:
        return None


class GPUMonitor:
    """Background GPU monitoring thread.

    Samples GPU utilization at regular intervals using nvidia-smi.
    Thread-safe for use during training.

    Usage:
        monitor = GPUMonitor(interval=0.5)
        monitor.start()
        # ... training ...
        stats = monitor.stop()
    """

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        """Initialize GPU monitor.

        Args:
            interval: Sampling interval in seconds
            gpu_index: Which GPU to monitor (for multi-GPU systems)
        """
        self.interval = interval
        self.gpu_index = gpu_index
        self._samples: List[GPUSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

    def start(self):
        """Start background monitoring."""
        if self._thread is not None:
            return

        self._samples = []
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        """Stop monitoring and return aggregated statistics."""
        if self._thread is None:
            return GPUStats()

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

        return self._compute_stats()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            sample = self._sample_gpu()
            if sample:
                self._samples.append(sample)
            time.sleep(self.interval)

    def _sample_gpu(self) -> Optional[GPUSample]:
        """Take a single GPU sample."""
        try:
            result = subprocess.run(
                ['nvidia-smi',
                 f'--id={self.gpu_index}',
                 '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                return None

            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 3:
                return GPUSample(
                    timestamp=time.time() - self._start_time,
                    utilization=float(parts[0]),
                    memory_used=float(parts[1]),
                    memory_total=float(parts[2]),
                    temperature=float(parts[3]) if len(parts) > 3 and parts[3] != '[N/A]' else None,
                    power_draw=float(parts[4]) if len(parts) > 4 and parts[4] != '[N/A]' else None,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None

    def _compute_stats(self) -> GPUStats:
        """Compute aggregate statistics from samples."""
        if not self._samples:
            return GPUStats()

        utils = [s.utilization for s in self._samples]
        mems = [s.memory_used for s in self._samples]

        return GPUStats(
            samples=len(self._samples),
            utilization_mean=mean(utils),
            utilization_min=min(utils),
            utilization_max=max(utils),
            utilization_std=stdev(utils) if len(utils) > 1 else 0.0,
            memory_mean_mb=mean(mems),
            memory_max_mb=max(mems),
            memory_total_mb=self._samples[0].memory_total if self._samples else 0,
            duration_sec=self._samples[-1].timestamp if self._samples else 0,
            raw_samples=self._samples,
        )

    def get_current_stats(self) -> Optional[Dict[str, float]]:
        """Get current GPU stats without stopping monitor."""
        if not self._samples:
            return None
        latest = self._samples[-1]
        return {
            'utilization': latest.utilization,
            'memory_used_mb': latest.memory_used,
            'memory_total_mb': latest.memory_total,
            'temperature_c': latest.temperature,
        }


def print_gpu_summary(stats: GPUStats):
    """Print a formatted GPU statistics summary."""
    print("\n" + "="*60)
    print("GPU MONITORING SUMMARY")
    print("="*60)
    print(f"Duration: {stats.duration_sec:.1f}s ({stats.samples} samples)")
    print(f"Utilization: {stats.utilization_mean:.1f}% "
          f"(min={stats.utilization_min:.0f}%, max={stats.utilization_max:.0f}%, std={stats.utilization_std:.1f}%)")
    print(f"Memory: {stats.memory_mean_mb:.0f}MB mean, {stats.memory_max_mb:.0f}MB peak "
          f"/ {stats.memory_total_mb:.0f}MB total")
    print("="*60)
