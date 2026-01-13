"""
Base hook classes for custom profiling.

Hooks provide a way to inject profiling logic at specific points
in your code without modifying the core implementation.

Usage:
    from jax_profiler.hooks import TimingHook, HookRegistry

    registry = HookRegistry()
    timing = TimingHook()
    registry.register("train_step", timing)

    # In training loop
    registry.trigger("train_step", "before")
    # ... training step ...
    registry.trigger("train_step", "after")

    # Get results
    print(timing.get_stats())
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from statistics import mean, stdev


class Hook(ABC):
    """Abstract base class for profiling hooks."""

    @abstractmethod
    def before(self, context: Optional[Dict[str, Any]] = None):
        """Called before the hooked operation."""
        pass

    @abstractmethod
    def after(self, context: Optional[Dict[str, Any]] = None):
        """Called after the hooked operation."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return collected statistics."""
        pass

    @abstractmethod
    def reset(self):
        """Reset collected data."""
        pass


@dataclass
class TimingSample:
    """Single timing measurement."""
    duration_ms: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


class TimingHook(Hook):
    """Hook for measuring execution time.

    Usage:
        hook = TimingHook()
        hook.before()
        # ... code to time ...
        hook.after()
        print(hook.get_stats())
    """

    def __init__(self, name: str = "timing"):
        self.name = name
        self._samples: List[TimingSample] = []
        self._start_time: Optional[float] = None
        self._start_timestamp: float = time.time()

    def before(self, context: Optional[Dict[str, Any]] = None):
        """Start timing."""
        self._start_time = time.perf_counter()
        self._context = context or {}

    def after(self, context: Optional[Dict[str, Any]] = None):
        """Stop timing and record sample."""
        if self._start_time is None:
            return

        duration_ms = (time.perf_counter() - self._start_time) * 1000
        merged_context = {**self._context, **(context or {})}

        self._samples.append(TimingSample(
            duration_ms=duration_ms,
            timestamp=time.time() - self._start_timestamp,
            context=merged_context,
        ))
        self._start_time = None

    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self._samples:
            return {"name": self.name, "count": 0}

        durations = [s.duration_ms for s in self._samples]
        return {
            "name": self.name,
            "count": len(durations),
            "total_ms": sum(durations),
            "mean_ms": mean(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "std_ms": stdev(durations) if len(durations) > 1 else 0.0,
            "first_ms": durations[0],
            "last_ms": durations[-1],
        }

    def reset(self):
        """Reset timing data."""
        self._samples = []
        self._start_time = None
        self._start_timestamp = time.time()

    def get_samples(self) -> List[TimingSample]:
        """Get raw samples."""
        return self._samples


class CounterHook(Hook):
    """Hook for counting occurrences.

    Usage:
        hook = CounterHook()
        hook.after({"type": "cache_hit"})
        hook.after({"type": "cache_miss"})
        print(hook.get_stats())
    """

    def __init__(self, name: str = "counter"):
        self.name = name
        self._counts: Dict[str, int] = {}
        self._total = 0

    def before(self, context: Optional[Dict[str, Any]] = None):
        """No-op for counter."""
        pass

    def after(self, context: Optional[Dict[str, Any]] = None):
        """Increment counter."""
        self._total += 1
        if context and "type" in context:
            key = context["type"]
            self._counts[key] = self._counts.get(key, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get counter statistics."""
        return {
            "name": self.name,
            "total": self._total,
            "by_type": self._counts.copy(),
        }

    def reset(self):
        """Reset counters."""
        self._counts = {}
        self._total = 0


class MetricHook(Hook):
    """Hook for recording numeric metrics.

    Usage:
        hook = MetricHook()
        hook.after({"loss": 0.5, "accuracy": 0.9})
        hook.after({"loss": 0.3, "accuracy": 0.95})
        print(hook.get_stats())
    """

    def __init__(self, name: str = "metrics"):
        self.name = name
        self._metrics: Dict[str, List[float]] = {}

    def before(self, context: Optional[Dict[str, Any]] = None):
        """No-op for metrics."""
        pass

    def after(self, context: Optional[Dict[str, Any]] = None):
        """Record metrics from context."""
        if not context:
            return

        for key, value in context.items():
            if isinstance(value, (int, float)):
                if key not in self._metrics:
                    self._metrics[key] = []
                self._metrics[key].append(float(value))

    def get_stats(self) -> Dict[str, Any]:
        """Get metric statistics."""
        stats = {"name": self.name, "metrics": {}}

        for key, values in self._metrics.items():
            if values:
                stats["metrics"][key] = {
                    "count": len(values),
                    "mean": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }

        return stats

    def reset(self):
        """Reset metrics."""
        self._metrics = {}
