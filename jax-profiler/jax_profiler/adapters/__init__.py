"""Project-specific adapters for jax-profiler."""

from .contlearn import ContLearnProfiler, setup_contlearn_hooks

__all__ = ["ContLearnProfiler", "setup_contlearn_hooks"]
