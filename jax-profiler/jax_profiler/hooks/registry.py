"""
Hook registry for managing multiple hooks.

Usage:
    from jax_profiler.hooks import HookRegistry, TimingHook

    registry = HookRegistry()

    # Register hooks
    registry.register("train_step", TimingHook("train_step"))
    registry.register("eval", TimingHook("eval"))

    # In code
    registry.trigger("train_step", "before")
    # ... training step ...
    registry.trigger("train_step", "after", {"batch": 0, "epoch": 1})

    # Get all stats
    print(registry.get_all_stats())

    # Context manager
    with registry.hook("train_step"):
        # ... training step ...
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from .base import Hook


class HookRegistry:
    """Registry for managing multiple profiling hooks.

    Provides a centralized way to register, trigger, and collect
    statistics from multiple hooks.
    """

    def __init__(self, enabled: bool = True):
        """Initialize hook registry.

        Args:
            enabled: If False, all hook operations are no-ops
        """
        self._hooks: Dict[str, List[Hook]] = {}
        self.enabled = enabled

    def register(self, name: str, hook: Hook):
        """Register a hook for a named point.

        Args:
            name: Name of the hook point (e.g., "train_step", "eval")
            hook: Hook instance to register
        """
        if name not in self._hooks:
            self._hooks[name] = []
        self._hooks[name].append(hook)

    def unregister(self, name: str, hook: Optional[Hook] = None):
        """Unregister hook(s) from a named point.

        Args:
            name: Name of the hook point
            hook: Specific hook to remove, or None to remove all
        """
        if name not in self._hooks:
            return

        if hook is None:
            del self._hooks[name]
        else:
            self._hooks[name] = [h for h in self._hooks[name] if h is not hook]

    def trigger(self, name: str, phase: str, context: Optional[Dict[str, Any]] = None):
        """Trigger hooks at a named point.

        Args:
            name: Name of the hook point
            phase: "before" or "after"
            context: Optional context dict passed to hooks
        """
        if not self.enabled or name not in self._hooks:
            return

        for hook in self._hooks[name]:
            if phase == "before":
                hook.before(context)
            elif phase == "after":
                hook.after(context)

    @contextmanager
    def hook(self, name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for triggering before/after hooks.

        Args:
            name: Name of the hook point
            context: Optional context dict

        Usage:
            with registry.hook("train_step", {"epoch": 1}):
                # ... code ...
        """
        self.trigger(name, "before", context)
        try:
            yield
        finally:
            self.trigger(name, "after", context)

    def get_stats(self, name: str) -> List[Dict[str, Any]]:
        """Get statistics from all hooks at a named point.

        Args:
            name: Name of the hook point

        Returns:
            List of stats dicts from each hook
        """
        if name not in self._hooks:
            return []
        return [hook.get_stats() for hook in self._hooks[name]]

    def get_all_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get statistics from all registered hooks.

        Returns:
            Dict mapping hook names to lists of stats
        """
        return {name: self.get_stats(name) for name in self._hooks}

    def reset(self, name: Optional[str] = None):
        """Reset hook data.

        Args:
            name: Specific hook point to reset, or None for all
        """
        if name is not None:
            if name in self._hooks:
                for hook in self._hooks[name]:
                    hook.reset()
        else:
            for hooks in self._hooks.values():
                for hook in hooks:
                    hook.reset()

    def print_summary(self):
        """Print a summary of all hook statistics."""
        print("\n" + "="*60)
        print("PROFILING HOOK SUMMARY")
        print("="*60)

        for name, stats_list in self.get_all_stats().items():
            print(f"\n[{name}]")
            for stats in stats_list:
                if "mean_ms" in stats:  # Timing hook
                    print(f"  {stats.get('name', 'timing')}: "
                          f"{stats['count']} calls, "
                          f"{stats['mean_ms']:.2f}ms mean, "
                          f"{stats['total_ms']:.0f}ms total")
                elif "total" in stats:  # Counter hook
                    print(f"  {stats.get('name', 'counter')}: "
                          f"{stats['total']} total")
                    for key, count in stats.get("by_type", {}).items():
                        print(f"    {key}: {count}")
                elif "metrics" in stats:  # Metric hook
                    print(f"  {stats.get('name', 'metrics')}:")
                    for key, mstats in stats.get("metrics", {}).items():
                        print(f"    {key}: {mstats['mean']:.4f} mean "
                              f"(min={mstats['min']:.4f}, max={mstats['max']:.4f})")

        print("="*60)


# Global registry for convenience
_global_registry: Optional[HookRegistry] = None


def get_global_registry() -> HookRegistry:
    """Get or create the global hook registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry


def reset_global_registry():
    """Reset the global hook registry."""
    global _global_registry
    _global_registry = None
