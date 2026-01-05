"""
Non-intrusive profiling decorators for continual learning framework.

Activated only when config['profiling_enabled'] = True.
Uses Python decorators and global flag to minimize overhead when disabled.

Added by Claude: December 26, 2024 - CIFAR-10 AWB performance debugging

"""
import time
import functools

_PROFILING_ENABLED = False

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
