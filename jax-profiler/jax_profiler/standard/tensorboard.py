"""
TensorBoard profiler integration for JAX.

Uses JAX's built-in TensorBoard profiler for:
- XLA compilation traces
- Memory allocation tracking
- Kernel execution timing
- Device placement visualization

Usage:
    from jax_profiler import TensorBoardProfiler

    # Context manager
    with TensorBoardProfiler("/tmp/traces"):
        # ... training code ...

    # Manual control
    profiler = TensorBoardProfiler("/tmp/traces")
    profiler.start()
    # ... training code ...
    profiler.stop()

    # View results
    # tensorboard --logdir=/tmp/traces
"""

import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class TensorBoardProfiler:
    """Wrapper for JAX's TensorBoard profiler.

    This uses jax.profiler which integrates with TensorBoard's
    profiler plugin for visualization.
    """

    def __init__(self, log_dir: str, create_perfetto_trace: bool = True):
        """Initialize TensorBoard profiler.

        Args:
            log_dir: Directory to save profiling traces
            create_perfetto_trace: Also create Perfetto trace file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.create_perfetto = create_perfetto_trace
        self._active = False

    def start(self):
        """Start profiling trace."""
        if self._active:
            return

        import jax

        jax.profiler.start_trace(str(self.log_dir))
        self._active = True

    def stop(self) -> Path:
        """Stop profiling and save trace.

        Returns:
            Path to the saved trace directory
        """
        if not self._active:
            return self.log_dir

        import jax

        jax.profiler.stop_trace()
        self._active = False

        print(f"[TensorBoardProfiler] Trace saved to: {self.log_dir}")
        print(f"[TensorBoardProfiler] View with: tensorboard --logdir={self.log_dir}")

        return self.log_dir

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


@contextmanager
def profile_section(name: str, log_dir: Optional[str] = None):
    """Profile a specific code section.

    Args:
        name: Name for this profiling section
        log_dir: Directory to save traces (default: /tmp/jax_traces/{name})

    Usage:
        with profile_section("training_loop"):
            # ... code to profile ...
    """
    if log_dir is None:
        log_dir = f"/tmp/jax_traces/{name}"

    profiler = TensorBoardProfiler(log_dir)
    try:
        profiler.start()
        yield profiler
    finally:
        profiler.stop()


def enable_compile_logging():
    """Enable JAX compilation logging.

    This prints a message every time JAX JIT-compiles a function,
    useful for identifying unexpected recompilation.
    """
    import jax

    jax.config.update("jax_log_compiles", True)
    print("[JAX] Compile logging enabled - will print on each JIT compilation")


def disable_compile_logging():
    """Disable JAX compilation logging."""
    import jax

    jax.config.update("jax_log_compiles", False)
