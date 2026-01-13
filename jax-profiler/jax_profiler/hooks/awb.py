"""
AWB Pipeline-specific profiling hooks.

The AWB (Adaptive Weight Basis) pipeline has 5 phases that cause significant
computational load and poor GPU utilization:

STEP 1: Preliminary training - Standard training to estimate task difficulty
STEP 2: Architecture decision - Compare losses to decide if change needed
STEP 3a: Architecture search - Evaluate multiple architectures (expensive)
STEP 3b: A/B matrix training - Train transformation matrices (W frozen)
STEP 4: V computation - V = A @ W @ B.T transformation
STEP 5: V training - Train with transformed weights

Key bottlenecks identified:
- A/B training uses different partition (get_AWBT computes A @ W @ B.T inside gradient)
- Architecture search evaluates multiple configurations
- JIT recompilation when switching between standard/AWB training modes

Usage:
    from jax_profiler.hooks.awb import AWBPipelineHooks

    hooks = AWBPipelineHooks()
    hooks.start()

    # Instrument AWB pipeline (external wrapper)
    hooks.on_phase_start("preliminary")
    # ... preliminary training ...
    hooks.on_phase_end("preliminary")

    hooks.on_phase_start("ab_training")
    # ... A/B training ...
    hooks.on_hamiltonian_call(is_awb=True, duration_ms=50.5)
    hooks.on_phase_end("ab_training")

    report = hooks.get_report()
    hooks.print_analysis()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from statistics import mean, stdev
from collections import defaultdict

from .base import Hook, TimingHook
from .registry import HookRegistry


@dataclass
class PhaseTiming:
    """Timing data for a single AWB phase execution."""
    phase: str
    task_id: int
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    hamiltonian_calls: int = 0
    hamiltonian_total_ms: float = 0.0
    optimizer_calls: int = 0
    optimizer_total_ms: float = 0.0
    jit_compiles: int = 0
    gpu_utilization_samples: List[float] = field(default_factory=list)


@dataclass
class HamiltonianTiming:
    """Timing for a single Hamiltonian gradient call."""
    is_awb_mode: bool  # True = A/B training, False = standard
    duration_ms: float
    task_id: int
    phase: str
    batch_idx: int = 0


class AWBPipelineHooks:
    """Comprehensive profiling hooks for AWB pipeline.

    Tracks:
    - Phase-level timing (5 AWB phases)
    - Hamiltonian gradient computation (standard vs AWB mode)
    - JIT compilation events
    - GPU utilization per phase
    - Optimizer step timing
    """

    # AWB pipeline phases
    PHASES = [
        "preliminary",      # STEP 1: Initial training
        "arch_decision",    # STEP 2: Loss comparison
        "arch_search",      # STEP 3a: Architecture evaluation
        "ab_training",      # STEP 3b: A/B matrix training
        "v_transform",      # STEP 4: V = A @ W @ B.T
        "v_warmup",         # STEP 5a: Warmup training
        "v_training",       # STEP 5b: Main training
    ]

    def __init__(self, gpu_monitoring: bool = True):
        """Initialize AWB pipeline hooks.

        Args:
            gpu_monitoring: Enable GPU utilization sampling during phases
        """
        self.gpu_monitoring = gpu_monitoring

        # Phase tracking
        self._phase_history: List[PhaseTiming] = []
        self._current_phase: Optional[PhaseTiming] = None
        self._current_task: int = 0

        # Hamiltonian tracking
        self._hamiltonian_timings: List[HamiltonianTiming] = []

        # JIT tracking
        self._jit_compiles: List[Dict[str, Any]] = []

        # Overall timing
        self._start_time: float = 0.0
        self._total_tasks: int = 0

    def start(self):
        """Start profiling session."""
        self._start_time = time.time()
        self._phase_history = []
        self._hamiltonian_timings = []
        self._jit_compiles = []
        self._current_phase = None
        self._current_task = 0
        self._total_tasks = 0

    def on_task_start(self, task_id: int):
        """Called when a new task starts."""
        self._current_task = task_id
        self._total_tasks = max(self._total_tasks, task_id + 1)

    def on_phase_start(self, phase: str):
        """Called when an AWB phase starts."""
        if phase not in self.PHASES:
            print(f"[AWBHooks] Warning: Unknown phase '{phase}'")

        self._current_phase = PhaseTiming(
            phase=phase,
            task_id=self._current_task,
            start_time=time.perf_counter(),
        )

    def on_phase_end(self, phase: str):
        """Called when an AWB phase ends."""
        if self._current_phase is None or self._current_phase.phase != phase:
            print(f"[AWBHooks] Warning: Phase mismatch on end: expected {self._current_phase.phase if self._current_phase else 'None'}, got {phase}")
            return

        self._current_phase.end_time = time.perf_counter()
        self._current_phase.duration_ms = (self._current_phase.end_time - self._current_phase.start_time) * 1000

        self._phase_history.append(self._current_phase)
        self._current_phase = None

    def on_hamiltonian_call(self, is_awb: bool, duration_ms: float, batch_idx: int = 0):
        """Called after each Hamiltonian gradient computation.

        Args:
            is_awb: True if AWB mode (A/B training), False if standard
            duration_ms: Duration of the call in milliseconds
            batch_idx: Current batch index
        """
        timing = HamiltonianTiming(
            is_awb_mode=is_awb,
            duration_ms=duration_ms,
            task_id=self._current_task,
            phase=self._current_phase.phase if self._current_phase else "unknown",
            batch_idx=batch_idx,
        )
        self._hamiltonian_timings.append(timing)

        # Update current phase stats
        if self._current_phase:
            self._current_phase.hamiltonian_calls += 1
            self._current_phase.hamiltonian_total_ms += duration_ms

    def on_optimizer_step(self, duration_ms: float):
        """Called after each optimizer step."""
        if self._current_phase:
            self._current_phase.optimizer_calls += 1
            self._current_phase.optimizer_total_ms += duration_ms

    def on_jit_compile(self, function_name: str, duration_ms: float):
        """Called when a JIT compilation occurs."""
        self._jit_compiles.append({
            'function': function_name,
            'duration_ms': duration_ms,
            'task_id': self._current_task,
            'phase': self._current_phase.phase if self._current_phase else "unknown",
        })

        if self._current_phase:
            self._current_phase.jit_compiles += 1

    def on_gpu_sample(self, utilization: float):
        """Record GPU utilization sample."""
        if self._current_phase:
            self._current_phase.gpu_utilization_samples.append(utilization)

    def get_phase_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by phase."""
        summary = {}

        for phase_name in self.PHASES:
            phase_timings = [p for p in self._phase_history if p.phase == phase_name]

            if not phase_timings:
                continue

            durations = [p.duration_ms for p in phase_timings]
            ham_totals = [p.hamiltonian_total_ms for p in phase_timings]
            ham_counts = [p.hamiltonian_calls for p in phase_timings]

            summary[phase_name] = {
                'count': len(phase_timings),
                'total_ms': sum(durations),
                'mean_ms': mean(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'hamiltonian_calls': sum(ham_counts),
                'hamiltonian_total_ms': sum(ham_totals),
                'hamiltonian_mean_ms': sum(ham_totals) / sum(ham_counts) if sum(ham_counts) > 0 else 0,
            }

            # GPU utilization
            all_gpu = []
            for p in phase_timings:
                all_gpu.extend(p.gpu_utilization_samples)
            if all_gpu:
                summary[phase_name]['gpu_mean'] = mean(all_gpu)
                summary[phase_name]['gpu_min'] = min(all_gpu)
                summary[phase_name]['gpu_max'] = max(all_gpu)

        return summary

    def get_hamiltonian_analysis(self) -> Dict[str, Any]:
        """Analyze Hamiltonian gradient computation times."""
        standard = [t for t in self._hamiltonian_timings if not t.is_awb_mode]
        awb = [t for t in self._hamiltonian_timings if t.is_awb_mode]

        analysis = {
            'total_calls': len(self._hamiltonian_timings),
            'standard_mode': {},
            'awb_mode': {},
        }

        if standard:
            durations = [t.duration_ms for t in standard]
            analysis['standard_mode'] = {
                'calls': len(standard),
                'total_ms': sum(durations),
                'mean_ms': mean(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'std_ms': stdev(durations) if len(durations) > 1 else 0,
            }

        if awb:
            durations = [t.duration_ms for t in awb]
            analysis['awb_mode'] = {
                'calls': len(awb),
                'total_ms': sum(durations),
                'mean_ms': mean(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'std_ms': stdev(durations) if len(durations) > 1 else 0,
            }

        # Compute slowdown factor
        if standard and awb:
            std_mean = analysis['standard_mode']['mean_ms']
            awb_mean = analysis['awb_mode']['mean_ms']
            analysis['awb_slowdown_factor'] = awb_mean / std_mean if std_mean > 0 else 0

        return analysis

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Identify bottlenecks in the AWB pipeline."""
        phase_summary = self.get_phase_summary()
        ham_analysis = self.get_hamiltonian_analysis()

        total_time = sum(s['total_ms'] for s in phase_summary.values())

        bottlenecks = []

        # Phase bottlenecks
        for phase, stats in phase_summary.items():
            pct = (stats['total_ms'] / total_time * 100) if total_time > 0 else 0
            if pct > 20:  # Phases taking >20% of time
                bottlenecks.append({
                    'type': 'phase',
                    'name': phase,
                    'percent': pct,
                    'total_ms': stats['total_ms'],
                    'suggestion': self._get_phase_suggestion(phase, stats),
                })

        # AWB mode slowdown
        if 'awb_slowdown_factor' in ham_analysis and ham_analysis['awb_slowdown_factor'] > 2:
            bottlenecks.append({
                'type': 'awb_mode',
                'name': 'AWB Hamiltonian',
                'slowdown': ham_analysis['awb_slowdown_factor'],
                'suggestion': 'AWB mode is significantly slower due to A@W@B.T computation inside gradient. Consider caching or restructuring.',
            })

        # JIT recompilation
        if len(self._jit_compiles) > self._total_tasks * 5:  # More than 5 compiles per task
            bottlenecks.append({
                'type': 'jit',
                'name': 'JIT Recompilation',
                'count': len(self._jit_compiles),
                'suggestion': 'Excessive JIT recompilation. Check for shape changes or dynamic operations.',
            })

        return {
            'total_time_ms': total_time,
            'bottlenecks': bottlenecks,
            'recommendations': self._get_recommendations(bottlenecks),
        }

    def _get_phase_suggestion(self, phase: str, stats: Dict[str, Any]) -> str:
        """Get optimization suggestion for a phase."""
        suggestions = {
            'preliminary': 'Reduce preliminary_epochs if loss stabilizes early',
            'arch_search': 'Reduce search space or use cached architecture decisions',
            'ab_training': 'A/B training is compute-intensive. Consider fewer epochs or larger batches',
            'v_warmup': 'Reduce warmup epochs if convergence is fast',
            'v_training': 'Standard training - optimize batch size and data loading',
        }
        return suggestions.get(phase, 'Profile in detail to identify specific bottleneck')

    def _get_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Get overall recommendations based on bottlenecks."""
        recommendations = []

        has_ab_bottleneck = any(b['name'] == 'ab_training' for b in bottlenecks if b['type'] == 'phase')
        has_awb_slowdown = any(b['type'] == 'awb_mode' for b in bottlenecks)
        has_search_bottleneck = any(b['name'] == 'arch_search' for b in bottlenecks if b['type'] == 'phase')

        if has_ab_bottleneck or has_awb_slowdown:
            recommendations.append(
                "A/B training is the primary bottleneck. The AWB forward pass computes "
                "A @ W @ B.T inside the gradient computation, which cannot be cached. "
                "Consider: (1) Reducing ab_training_epochs, (2) Using Condition 3 (skip_transfer), "
                "(3) Larger batch sizes to amortize overhead."
            )

        if has_search_bottleneck:
            recommendations.append(
                "Architecture search is expensive. Consider: (1) Reducing search space, "
                "(2) Using validation subsets, (3) Early stopping in search."
            )

        if not recommendations:
            recommendations.append("No major bottlenecks identified. Profile at finer granularity.")

        return recommendations

    def get_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        total_time = time.time() - self._start_time if self._start_time else 0

        return {
            'total_time_sec': total_time,
            'total_tasks': self._total_tasks,
            'phases': self.get_phase_summary(),
            'hamiltonian': self.get_hamiltonian_analysis(),
            'bottlenecks': self.get_bottleneck_analysis(),
            'jit_compiles': len(self._jit_compiles),
            'jit_details': self._jit_compiles[:20],  # First 20
        }

    def print_analysis(self):
        """Print detailed analysis to console."""
        report = self.get_report()

        print("\n" + "="*70)
        print("AWB PIPELINE PROFILING ANALYSIS")
        print("="*70)

        print(f"\nTotal time: {report['total_time_sec']:.1f}s")
        print(f"Tasks profiled: {report['total_tasks']}")

        # Phase breakdown
        print("\n" + "-"*70)
        print("PHASE BREAKDOWN")
        print("-"*70)
        print(f"{'Phase':<20} {'Count':<8} {'Total(ms)':<12} {'Mean(ms)':<12} {'Ham.Calls':<10} {'GPU%':<8}")
        print("-"*70)

        total_ms = sum(p['total_ms'] for p in report['phases'].values())
        for phase, stats in sorted(report['phases'].items(), key=lambda x: -x[1]['total_ms']):
            pct = (stats['total_ms'] / total_ms * 100) if total_ms > 0 else 0
            gpu = f"{stats.get('gpu_mean', 0):.0f}" if 'gpu_mean' in stats else "N/A"
            print(f"{phase:<20} {stats['count']:<8} {stats['total_ms']:<12.0f} "
                  f"{stats['mean_ms']:<12.1f} {stats['hamiltonian_calls']:<10} {gpu:<8} ({pct:.1f}%)")

        # Hamiltonian analysis
        ham = report['hamiltonian']
        print("\n" + "-"*70)
        print("HAMILTONIAN GRADIENT ANALYSIS")
        print("-"*70)

        if ham['standard_mode']:
            std = ham['standard_mode']
            print(f"Standard mode: {std['calls']} calls, {std['mean_ms']:.2f}ms mean, {std['total_ms']:.0f}ms total")

        if ham['awb_mode']:
            awb = ham['awb_mode']
            print(f"AWB mode:      {awb['calls']} calls, {awb['mean_ms']:.2f}ms mean, {awb['total_ms']:.0f}ms total")

        if 'awb_slowdown_factor' in ham:
            print(f"\nAWB SLOWDOWN: {ham['awb_slowdown_factor']:.1f}x slower than standard mode")

        # Bottlenecks
        bottlenecks = report['bottlenecks']
        if bottlenecks['bottlenecks']:
            print("\n" + "-"*70)
            print("IDENTIFIED BOTTLENECKS")
            print("-"*70)
            for b in bottlenecks['bottlenecks']:
                if b['type'] == 'phase':
                    print(f"  [{b['name']}] {b['percent']:.1f}% of time - {b['suggestion']}")
                elif b['type'] == 'awb_mode':
                    print(f"  [AWB Mode] {b['slowdown']:.1f}x slowdown - {b['suggestion']}")
                elif b['type'] == 'jit':
                    print(f"  [JIT] {b['count']} recompilations - {b['suggestion']}")

        # Recommendations
        if bottlenecks['recommendations']:
            print("\n" + "-"*70)
            print("RECOMMENDATIONS")
            print("-"*70)
            for i, rec in enumerate(bottlenecks['recommendations'], 1):
                print(f"\n{i}. {rec}")

        print("\n" + "="*70)


def create_awb_wrapper(train_fn):
    """Create a wrapper that instruments an AWB training function.

    This is a decorator/wrapper approach that doesn't require modifying
    the original training code.

    Usage:
        from jax_profiler.hooks.awb import create_awb_wrapper, AWBPipelineHooks

        hooks = AWBPipelineHooks()
        wrapped_train = create_awb_wrapper(original_train_fn)

        # Training now automatically instrumented
        wrapped_train(config, hooks=hooks)
    """
    def wrapper(*args, hooks: Optional[AWBPipelineHooks] = None, **kwargs):
        if hooks is None:
            return train_fn(*args, **kwargs)

        hooks.start()
        try:
            result = train_fn(*args, **kwargs)
        finally:
            hooks.print_analysis()

        return result

    return wrapper
