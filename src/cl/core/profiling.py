"""
Non-intrusive profiling decorators for continual learning framework.

Activated only when config['profiling_enabled'] = True.
Uses Python decorators and global flag to minimize overhead when disabled.

Added by Claude: December 26, 2024 - CIFAR-10 AWB performance debugging
Enhanced by Claude: January 2025 - Detailed timing collection for bottleneck analysis

"""
import time
import functools
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

_PROFILING_ENABLED = False
_DETAILED_PROFILING = False  # Added by Claude: For fine-grained timing

# Added by Claude: Global timing collector for detailed profiling
_TIMING_COLLECTOR = None


@dataclass
class TimingSample:
    """Single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    epoch: int = 0
    batch: int = 0
    task_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUSample:
    """Single GPU utilization sample."""
    timestamp: float
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float] = None


class ProfileCollector:
    """Collects detailed timing and GPU metrics for analysis.

    Added by Claude: Thread-safe collector for fine-grained profiling data.
    Only active when detailed_profiling=True in config.
    """

    def __init__(self):
        self.timings: Dict[str, List[TimingSample]] = defaultdict(list)
        self.gpu_samples: List[GPUSample] = []
        self.start_time: float = time.time()
        self.config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Aggregated stats (computed on demand)
        self._stats_cache: Optional[Dict] = None

    def record_timing(self, name: str, duration_ms: float,
                      epoch: int = 0, batch: int = 0, task_id: int = 0,
                      metadata: Dict = None):
        """Record a timing measurement (thread-safe)."""
        with self._lock:
            self._stats_cache = None  # Invalidate cache
            sample = TimingSample(
                name=name,
                duration_ms=duration_ms,
                timestamp=time.time() - self.start_time,
                epoch=epoch,
                batch=batch,
                task_id=task_id,
                metadata=metadata or {}
            )
            self.timings[name].append(sample)

    def record_gpu(self, utilization: float, memory_used: float,
                   memory_total: float, temperature: float = None):
        """Record GPU utilization sample (thread-safe)."""
        with self._lock:
            sample = GPUSample(
                timestamp=time.time() - self.start_time,
                utilization_percent=utilization,
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                temperature_c=temperature
            )
            self.gpu_samples.append(sample)

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics for all timings."""
        with self._lock:
            if self._stats_cache is not None:
                return self._stats_cache

            stats = {}
            for name, samples in self.timings.items():
                if not samples:
                    continue
                durations = [s.duration_ms for s in samples]
                stats[name] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'mean_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'std_ms': self._std(durations),
                    # First vs rest (JIT warmup detection)
                    'first_ms': durations[0] if durations else 0,
                    'rest_mean_ms': sum(durations[1:]) / len(durations[1:]) if len(durations) > 1 else 0,
                }

            # GPU stats
            if self.gpu_samples:
                utils = [s.utilization_percent for s in self.gpu_samples]
                mems = [s.memory_used_mb for s in self.gpu_samples]
                stats['gpu'] = {
                    'samples': len(utils),
                    'utilization_mean': sum(utils) / len(utils),
                    'utilization_min': min(utils),
                    'utilization_max': max(utils),
                    'utilization_std': self._std(utils),
                    'memory_mean_mb': sum(mems) / len(mems),
                    'memory_max_mb': max(mems),
                }

            self._stats_cache = stats
            return stats

    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def generate_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        stats = self.get_stats()

        # Compute time breakdown
        total_time_ms = sum(s['total_ms'] for s in stats.values() if isinstance(s, dict) and 'total_ms' in s)

        breakdown = {}
        for name, s in stats.items():
            if isinstance(s, dict) and 'total_ms' in s:
                breakdown[name] = {
                    'percent': (s['total_ms'] / total_time_ms * 100) if total_time_ms > 0 else 0,
                    'total_ms': s['total_ms'],
                    'mean_ms': s['mean_ms'],
                    'count': s['count'],
                }

        # Sort by time percentage
        breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1]['percent'], reverse=True))

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_profiled_time_sec': total_time_ms / 1000,
            'wall_clock_time_sec': time.time() - self.start_time,
            'config': self.config,
            'metadata': self.metadata,
            'time_breakdown': breakdown,
            'detailed_stats': stats,
            'bottleneck_analysis': self._analyze_bottlenecks(stats, breakdown),
            'recommendations': self._generate_recommendations(stats, breakdown),
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[PROFILE] Report saved to: {output_path}")

        return report

    def _analyze_bottlenecks(self, stats: Dict, breakdown: Dict) -> Dict[str, Any]:
        """Analyze bottlenecks from timing data."""
        analysis = {
            'primary_bottleneck': None,
            'secondary_bottlenecks': [],
            'jit_warmup_detected': False,
            'data_loading_bottleneck': False,
            'gpu_underutilized': False,
        }

        # Find primary bottleneck (highest time %)
        if breakdown:
            top_item = list(breakdown.items())[0]
            analysis['primary_bottleneck'] = {
                'component': top_item[0],
                'percent': top_item[1]['percent'],
                'mean_ms': top_item[1]['mean_ms'],
            }

            # Secondary bottlenecks (> 10%)
            for name, data in list(breakdown.items())[1:]:
                if data['percent'] > 10:
                    analysis['secondary_bottlenecks'].append({
                        'component': name,
                        'percent': data['percent'],
                    })

        # JIT warmup detection
        for name, s in stats.items():
            if isinstance(s, dict) and 'first_ms' in s and 'rest_mean_ms' in s:
                if s['first_ms'] > 3 * s['rest_mean_ms'] and s['first_ms'] > 100:
                    analysis['jit_warmup_detected'] = True
                    break

        # Data loading bottleneck
        data_keys = ['batch_load', 'data_prefetch', 'jax_conversion']
        data_time = sum(breakdown.get(k, {}).get('percent', 0) for k in data_keys)
        if data_time > 30:
            analysis['data_loading_bottleneck'] = True

        # GPU underutilization
        if 'gpu' in stats:
            if stats['gpu']['utilization_mean'] < 50:
                analysis['gpu_underutilized'] = True

        return analysis

    def _generate_recommendations(self, stats: Dict, breakdown: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recs = []
        analysis = self._analyze_bottlenecks(stats, breakdown)

        if analysis['gpu_underutilized']:
            gpu_util = stats.get('gpu', {}).get('utilization_mean', 0)
            recs.append(f"GPU utilization is low ({gpu_util:.1f}%). Consider: "
                       "increasing batch size, optimizing data loading, or reducing sync points.")

        if analysis['data_loading_bottleneck']:
            recs.append("Data loading is a significant bottleneck (>30% of time). "
                       "Ensure use_jax_prefetch=true and try increasing prefetch_size.")

        if analysis['jit_warmup_detected']:
            recs.append("JIT compilation warmup detected. First iterations are slow. "
                       "This is expected behavior and amortizes over training.")

        # Check specific components
        if 'hamiltonian' in breakdown and breakdown['hamiltonian']['percent'] > 50:
            recs.append("Hamiltonian computation dominates (>50%). This is expected for "
                       "compute-bound training. Ensure GPU is well-utilized during this phase.")

        if 'test_metrics' in breakdown and breakdown['test_metrics']['percent'] > 20:
            recs.append("Test metric computation is significant (>20%). Consider increasing "
                       "eval_interval to reduce frequency.")

        if 'optimizer_step' in breakdown and breakdown['optimizer_step']['percent'] > 15:
            recs.append("Optimizer updates taking significant time. This is unusual - "
                       "check for unintended CPU sync or very large models.")

        if not recs:
            recs.append("No major bottlenecks detected. Training appears well-optimized.")

        return recs

    def print_summary(self):
        """Print a concise summary to console."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("PROFILING SUMMARY")
        print("="*70)

        # Time breakdown
        total_ms = sum(s['total_ms'] for s in stats.values() if isinstance(s, dict) and 'total_ms' in s)
        print(f"\nTotal profiled time: {total_ms/1000:.2f}s")
        print(f"Wall clock time: {time.time() - self.start_time:.2f}s")

        print("\nTime Breakdown:")
        print("-"*50)

        breakdown = []
        for name, s in stats.items():
            if isinstance(s, dict) and 'total_ms' in s:
                pct = (s['total_ms'] / total_ms * 100) if total_ms > 0 else 0
                breakdown.append((name, pct, s['mean_ms'], s['count']))

        breakdown.sort(key=lambda x: x[1], reverse=True)
        for name, pct, mean_ms, count in breakdown[:10]:
            bar = "#" * int(pct / 2)
            print(f"  {name:25s} {pct:5.1f}% {bar:25s} (mean: {mean_ms:.2f}ms, n={count})")

        # GPU stats
        if 'gpu' in stats:
            print("\nGPU Statistics:")
            print("-"*50)
            gpu = stats['gpu']
            print(f"  Utilization: {gpu['utilization_mean']:.1f}% "
                  f"(min: {gpu['utilization_min']:.1f}%, max: {gpu['utilization_max']:.1f}%)")
            print(f"  Memory: {gpu['memory_mean_mb']:.0f}MB mean, {gpu['memory_max_mb']:.0f}MB peak")

        print("="*70 + "\n")


def get_collector() -> Optional[ProfileCollector]:
    """Get the global profile collector (None if profiling disabled)."""
    return _TIMING_COLLECTOR


def init_collector(config: Dict = None) -> ProfileCollector:
    """Initialize global profile collector."""
    global _TIMING_COLLECTOR
    _TIMING_COLLECTOR = ProfileCollector()
    if config:
        _TIMING_COLLECTOR.config = config
    return _TIMING_COLLECTOR


def set_detailed_profiling(enabled: bool):
    """Enable/disable detailed profiling."""
    global _DETAILED_PROFILING
    _DETAILED_PROFILING = enabled


def is_detailed_profiling() -> bool:
    """Check if detailed profiling is enabled."""
    return _DETAILED_PROFILING and _PROFILING_ENABLED

# Added by Claude: GPU memory tracking
def get_gpu_memory_usage():
    """
    Get current GPU memory usage from JAX.

    Returns:
        Tuple of (used_gb, total_gb, percent) or None if unavailable
    """
    try:
        import jax
        from jax.lib import xla_bridge

        # Get GPU backend
        backend = xla_bridge.get_backend()

        # Get memory stats from first GPU device
        devices = backend.devices()
        if not devices:
            return None

        device = devices[0]

        # Get memory info
        mem_stats = device.memory_stats()
        if mem_stats is None:
            return None

        bytes_in_use = mem_stats.get('bytes_in_use', 0)
        # Try to get peak memory
        peak_bytes = mem_stats.get('peak_bytes_in_use', bytes_in_use)

        # Convert to GB
        used_gb = bytes_in_use / (1024**3)
        peak_gb = peak_bytes / (1024**3)

        # Try to estimate total memory (H200 has 144GB typically)
        # We'll use peak as a proxy since total isn't always available
        return (used_gb, peak_gb)

    except Exception as e:
        # Silently fail if JAX not available or memory stats unavailable
        return None

def format_memory_stats():
    """
    Format GPU memory usage as a string.

    Returns:
        Formatted string like "GPU: 2.3GB / 144GB (1.6%)" or "GPU: N/A"
    """
    mem = get_gpu_memory_usage()
    if mem is None:
        return "GPU: N/A"

    used_gb, peak_gb = mem
    return f"GPU: {used_gb:.2f}GB used, {peak_gb:.2f}GB peak"

def enable_profiling(enabled: bool):
    """
    Enable/disable profiling globally.

    Args:
        enabled: True to enable profiling output, False to disable
    """
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enabled

def profile(phase_name: str):
    """
    Decorator to time a function if profiling is enabled.

    When profiling is disabled, this has zero overhead (simple boolean check).
    When enabled, prints timing and GPU memory information for the decorated function.

    Args:
        phase_name: Human-readable name for this profiling phase

    Example:
        @profile("Dataset Loading")
        def load_data():
            # ... code ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _PROFILING_ENABLED:
                return func(*args, **kwargs)

            print(f"\n[PROFILE] {phase_name} starting... | {format_memory_stats()}")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s | {format_memory_stats()}")
            return result
        return wrapper
    return decorator

def profile_section(phase_name: str, enabled: bool = None):
    """
    Context manager for profiling a code section.

    Args:
        phase_name: Human-readable name for this profiling phase
        enabled: Optional override for profiling enabled status

    Example:
        with profile_section("JAX Pre-conversion"):
            # ... code to profile ...
    """
    class ProfileContext:
        def __enter__(self):
            if enabled is None:
                self.enabled = _PROFILING_ENABLED
            else:
                self.enabled = enabled

            if self.enabled:
                print(f"\n[PROFILE] {phase_name} starting... | {format_memory_stats()}")
                self.start_time = time.time()
            return self

        def __exit__(self, *args):
            if self.enabled:
                elapsed = time.time() - self.start_time
                print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s | {format_memory_stats()}")

    return ProfileContext()


# Added by Claude: Fine-grained timing context manager for detailed profiling
class timed_section:
    """Context manager for fine-grained timing that records to collector.

    Only records when detailed_profiling is enabled.
    Zero overhead when disabled.

    Example:
        with timed_section("hamiltonian", epoch=5, batch=10):
            # Code to time
    """

    def __init__(self, name: str, epoch: int = 0, batch: int = 0, task_id: int = 0):
        self.name = name
        self.epoch = epoch
        self.batch = batch
        self.task_id = task_id
        self.start_time = None

    def __enter__(self):
        if is_detailed_profiling():
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            collector = get_collector()
            if collector:
                collector.record_timing(
                    self.name, elapsed_ms,
                    epoch=self.epoch, batch=self.batch, task_id=self.task_id
                )


def timed(name: str):
    """Decorator for timing functions with detailed profiling.

    Only records when detailed_profiling is enabled.

    Example:
        @timed("optimizer_step")
        def update_params(grad, opt_state, params):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_detailed_profiling():
                return func(*args, **kwargs)

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            collector = get_collector()
            if collector:
                collector.record_timing(name, elapsed_ms)

            return result
        return wrapper
    return decorator


# Added by Claude: NVIDIA-SMI GPU monitoring utilities
def get_nvidia_gpu_stats():
    """Get GPU stats from nvidia-smi.

    Returns:
        List of dicts with utilization, memory, temperature per GPU.
        Returns None if nvidia-smi not available.
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 4:
                gpu_stats.append({
                    'utilization': float(parts[0].strip()),
                    'memory_used_mb': float(parts[1].strip()),
                    'memory_total_mb': float(parts[2].strip()),
                    'temperature_c': float(parts[3].strip()),
                })
            elif len(parts) >= 3:
                gpu_stats.append({
                    'utilization': float(parts[0].strip()),
                    'memory_used_mb': float(parts[1].strip()),
                    'memory_total_mb': float(parts[2].strip()),
                    'temperature_c': None,
                })
        return gpu_stats
    except Exception:
        return None


class GPUMonitor:
    """Background GPU monitoring thread.

    Samples GPU utilization at regular intervals and records to collector.

    Example:
        monitor = GPUMonitor(interval_sec=0.5)
        monitor.start()
        # ... training ...
        monitor.stop()
    """

    def __init__(self, interval_sec: float = 0.5):
        self.interval = interval_sec
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start monitoring in background thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _monitor_loop(self):
        """Background monitoring loop."""
        collector = get_collector()
        if collector is None:
            return

        while not self._stop_event.is_set():
            stats = get_nvidia_gpu_stats()
            if stats and len(stats) > 0:
                gpu = stats[0]  # First GPU
                collector.record_gpu(
                    utilization=gpu['utilization'],
                    memory_used=gpu['memory_used_mb'],
                    memory_total=gpu['memory_total_mb'],
                    temperature=gpu.get('temperature_c')
                )
            time.sleep(self.interval)
