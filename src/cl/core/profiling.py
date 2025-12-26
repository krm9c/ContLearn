"""
Non-intrusive profiling decorators for continual learning framework.

Activated only when config['profiling_enabled'] = True.
Uses Python decorators and global flag to minimize overhead when disabled.

Added by Claude: December 26, 2024 - CIFAR-10 AWB performance debugging
"""
import time
import functools

_PROFILING_ENABLED = False

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
    When enabled, prints timing information for the decorated function.

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

            print(f"\n[PROFILE] {phase_name} starting...")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s")
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
                print(f"\n[PROFILE] {phase_name} starting...")
                self.start_time = time.time()
            return self

        def __exit__(self, *args):
            if self.enabled:
                elapsed = time.time() - self.start_time
                print(f"[PROFILE] {phase_name} complete: {elapsed:.2f}s")

    return ProfileContext()
