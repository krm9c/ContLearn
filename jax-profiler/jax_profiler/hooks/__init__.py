"""Custom hook infrastructure for domain-specific profiling."""

from .base import Hook, TimingHook, CounterHook, MetricHook
from .registry import HookRegistry
from .awb import AWBPipelineHooks, create_awb_wrapper

__all__ = [
    "Hook", "TimingHook", "CounterHook", "MetricHook", "HookRegistry",
    "AWBPipelineHooks", "create_awb_wrapper",
]
