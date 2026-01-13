"""
ContLearn-specific profiling adapter.

Provides domain-specific hooks for continual learning:
- AWB pipeline phases (preliminary, arch_decision, ab_training, v_transform, v_training)
- Task transitions
- Hamiltonian gradient components
- Experience replay efficiency

The AWB pipeline has these phases causing poor GPU utilization:
- STEP 1: Preliminary training (standard, decent GPU usage)
- STEP 2: Architecture decision (fast, not a bottleneck)
- STEP 3a: Architecture search (expensive, evaluates multiple configs)
- STEP 3b: A/B matrix training (MAJOR BOTTLENECK - computes A@W@B.T in gradient)
- STEP 4: V computation (fast transformation)
- STEP 5: V training (standard, decent GPU usage)

Usage (external - no ContLearn modification):
    from jax_profiler import GPUMonitor, TensorBoardProfiler
    from jax_profiler.adapters import ContLearnProfiler

    # Simple GPU monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()

    # Import and run ContLearn
    from cl.runners import train_model
    train_model(config)

    stats = monitor.stop()
    print(f"GPU: {stats.utilization_mean}%")

    # Or with TensorBoard for XLA traces
    with TensorBoardProfiler("/tmp/traces"):
        train_model(config)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time

from ..hooks import HookRegistry, TimingHook, CounterHook, MetricHook
from ..standard.gpu_monitor import GPUMonitor


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_id: int
    start_time: float = 0.0
    end_time: float = 0.0
    phases: Dict[str, float] = field(default_factory=dict)  # phase -> duration_ms
    train_loss_final: Optional[float] = None
    test_accuracy_final: Optional[float] = None


class ContLearnProfiler:
    """Profiler for ContLearn continual learning framework.

    Tracks:
    - AWB pipeline phases
    - Task-level timing
    - Hamiltonian gradient computation
    - Experience replay
    - GPU utilization
    """

    # AWB pipeline phases
    AWB_PHASES = [
        "preliminary",      # Initial training before arch decision
        "arch_decision",    # Architecture search decision
        "ab_training",      # A/B matrix training
        "v_transform",      # V = A @ W @ B.T transformation
        "v_training",       # Training with transformed weights
    ]

    def __init__(self, gpu_monitoring: bool = True, monitor_interval: float = 0.5):
        """Initialize ContLearn profiler.

        Args:
            gpu_monitoring: Enable GPU utilization monitoring
            monitor_interval: GPU sampling interval in seconds
        """
        self.registry = HookRegistry()
        self._setup_hooks()

        self.gpu_monitoring = gpu_monitoring
        self.monitor_interval = monitor_interval
        self._gpu_monitor: Optional[GPUMonitor] = None

        self._task_metrics: List[TaskMetrics] = []
        self._current_task: Optional[TaskMetrics] = None
        self._current_phase: Optional[str] = None
        self._phase_start_time: float = 0.0
        self._start_time: float = 0.0

    def _setup_hooks(self):
        """Set up profiling hooks."""
        # Timing hooks for main components
        self.registry.register("hamiltonian", TimingHook("hamiltonian"))
        self.registry.register("optimizer_step", TimingHook("optimizer_step"))
        self.registry.register("data_prep", TimingHook("data_prep"))
        self.registry.register("evaluation", TimingHook("evaluation"))
        self.registry.register("experience_replay", TimingHook("experience_replay"))

        # Counter for events
        self.registry.register("jit_compile", CounterHook("jit_compile"))
        self.registry.register("gradient_clip", CounterHook("gradient_clip"))

        # Metrics
        self.registry.register("losses", MetricHook("losses"))
        self.registry.register("gradients", MetricHook("gradients"))

    def start(self):
        """Start profiling."""
        self._start_time = time.time()
        self._task_metrics = []
        self._current_task = None
        self.registry.reset()

        if self.gpu_monitoring:
            self._gpu_monitor = GPUMonitor(interval=self.monitor_interval)
            self._gpu_monitor.start()

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return report."""
        if self._gpu_monitor:
            gpu_stats = self._gpu_monitor.stop()
            self._gpu_monitor = None
        else:
            gpu_stats = None

        return self.get_report(gpu_stats)

    def on_task_start(self, task_id: int):
        """Called when a new task starts."""
        if self._current_task is not None:
            self._current_task.end_time = time.time()
            self._task_metrics.append(self._current_task)

        self._current_task = TaskMetrics(
            task_id=task_id,
            start_time=time.time(),
        )

    def on_task_end(self, task_id: int, train_loss: float = None, test_accuracy: float = None):
        """Called when a task ends."""
        if self._current_task and self._current_task.task_id == task_id:
            self._current_task.end_time = time.time()
            self._current_task.train_loss_final = train_loss
            self._current_task.test_accuracy_final = test_accuracy
            self._task_metrics.append(self._current_task)
            self._current_task = None

    def on_phase_start(self, phase: str):
        """Called when an AWB phase starts."""
        self._current_phase = phase
        self._phase_start_time = time.perf_counter()

    def on_phase_end(self, phase: str):
        """Called when an AWB phase ends."""
        if self._current_phase == phase and self._current_task:
            duration_ms = (time.perf_counter() - self._phase_start_time) * 1000
            self._current_task.phases[phase] = duration_ms
        self._current_phase = None

    def on_hamiltonian(self, duration_ms: float, losses: Dict[str, float] = None):
        """Record hamiltonian gradient computation."""
        self.registry.trigger("hamiltonian", "after", {"duration_ms": duration_ms})
        if losses:
            self.registry.trigger("losses", "after", losses)

    def on_optimizer_step(self, duration_ms: float, grad_norm: float = None):
        """Record optimizer step."""
        self.registry.trigger("optimizer_step", "after", {"duration_ms": duration_ms})
        if grad_norm is not None:
            self.registry.trigger("gradients", "after", {"grad_norm": grad_norm})

    def get_report(self, gpu_stats=None) -> Dict[str, Any]:
        """Generate profiling report."""
        total_time = time.time() - self._start_time if self._start_time else 0

        report = {
            "total_time_sec": total_time,
            "num_tasks": len(self._task_metrics),
            "tasks": [],
            "components": self.registry.get_all_stats(),
        }

        # Task-level breakdown
        for task in self._task_metrics:
            task_time = task.end_time - task.start_time if task.end_time else 0
            report["tasks"].append({
                "task_id": task.task_id,
                "duration_sec": task_time,
                "phases": task.phases,
                "train_loss": task.train_loss_final,
                "test_accuracy": task.test_accuracy_final,
            })

        # GPU stats
        if gpu_stats:
            report["gpu"] = {
                "utilization_mean": gpu_stats.utilization_mean,
                "utilization_min": gpu_stats.utilization_min,
                "utilization_max": gpu_stats.utilization_max,
                "memory_mean_mb": gpu_stats.memory_mean_mb,
                "memory_max_mb": gpu_stats.memory_max_mb,
                "samples": gpu_stats.samples,
            }

        return report

    def print_summary(self):
        """Print profiling summary."""
        report = self.get_report()

        print("\n" + "="*70)
        print("CONTLEARN PROFILING SUMMARY")
        print("="*70)

        print(f"\nTotal time: {report['total_time_sec']:.1f}s")
        print(f"Tasks completed: {report['num_tasks']}")

        # Task breakdown
        if report["tasks"]:
            print("\nTask Breakdown:")
            print("-"*50)
            for task in report["tasks"]:
                print(f"  Task {task['task_id']}: {task['duration_sec']:.1f}s")
                if task["phases"]:
                    for phase, dur in task["phases"].items():
                        print(f"    {phase}: {dur:.0f}ms")

        # Component timing
        print("\nComponent Timing:")
        print("-"*50)
        for name, stats_list in report["components"].items():
            for stats in stats_list:
                if "mean_ms" in stats:
                    print(f"  {name}: {stats['mean_ms']:.2f}ms mean, "
                          f"{stats['total_ms']:.0f}ms total ({stats['count']} calls)")

        # GPU
        if "gpu" in report:
            gpu = report["gpu"]
            print(f"\nGPU Utilization: {gpu['utilization_mean']:.1f}% "
                  f"(min={gpu['utilization_min']:.0f}%, max={gpu['utilization_max']:.0f}%)")

        print("="*70)


def setup_contlearn_hooks(registry: HookRegistry):
    """Set up standard ContLearn profiling hooks.

    Args:
        registry: HookRegistry to configure

    Returns:
        Configured registry
    """
    # Standard timing hooks
    registry.register("hamiltonian", TimingHook("hamiltonian"))
    registry.register("optimizer_step", TimingHook("optimizer_step"))
    registry.register("data_prep", TimingHook("data_prep"))
    registry.register("evaluation", TimingHook("evaluation"))

    # AWB-specific
    for phase in ContLearnProfiler.AWB_PHASES:
        registry.register(f"awb_{phase}", TimingHook(f"awb_{phase}"))

    return registry
