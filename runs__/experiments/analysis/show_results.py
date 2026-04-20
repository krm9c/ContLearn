#!/usr/bin/env python
"""
Show all results from a records pickle: CL metrics, per-task performance matrix,
per-task loss curves, and compute timing.

Usage:
    python show_results.py <records.pkl>
    python show_results.py <checkpoint_dir>          # auto-picks latest
    python show_results.py <records.pkl> --csv out.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from compute_metrics import (
    compute_acc,
    compute_bwt,
    compute_forgetting,
    compute_fwt,
    extract_performance_matrix,
)


def load_record_dict(path: Path) -> dict:
    if path.is_dir():
        pkls = sorted(path.glob("*_records.pkl"), key=lambda p: p.stat().st_mtime)
        if not pkls:
            sys.exit(f"No *_records.pkl found in {path}")
        path = pkls[-1]
        print(f"Using: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "record_dict" in data:
        return data["record_dict"]
    return data


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def show_metadata(rd):
    meta = rd.get("metadata", {})
    if not meta:
        return
    print_section("EXPERIMENT METADATA")
    for k in ["dataset", "network", "problem", "n_tasks", "epochs_per_task",
              "awb_enabled", "lr", "batch_size", "grad_weights", "run_id"]:
        if k in meta:
            print(f"  {k:20s}: {meta[k]}")


def show_performance_matrix(rd):
    matrix = extract_performance_matrix(rd)
    if matrix is None:
        print_section("PERFORMANCE MATRIX")
        print("  Not available (per_task_eval_enabled was off)")
        return None

    T = matrix.shape[0]
    print_section(f"PERFORMANCE MATRIX  A[j][i]  ({T} tasks)")
    print(f"  A[j][i] = accuracy on task i after training task j")
    print()

    # Header
    header = "  After\\On " + "".join(f"  Task {i:2d}" for i in range(T))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j in range(T):
        row = f"  Task {j:2d}   "
        for i in range(T):
            v = matrix[j, i]
            if not np.isnan(v):
                row += f"  {v:6.3f}"
            else:
                row += "       -"
        valid = matrix[j, :j+1]
        valid = valid[~np.isnan(valid)]
        avg = np.mean(valid) if len(valid) > 0 else float('nan')
        row += f"   avg={avg:.3f}" if not np.isnan(avg) else "   avg=  n/a"
        print(row)

    return matrix


def show_cl_metrics(matrix):
    if matrix is None:
        return
    print_section("CONTINUAL LEARNING METRICS")

    # Check if enough data for metrics (need at least the last row filled)
    T = matrix.shape[0]
    last_row = matrix[T-1, :]
    if np.all(np.isnan(last_row)):
        print("  Not enough data yet (run still in progress)")
        return

    # Replace NaN with 0 for metric computation on partial matrices
    m = np.nan_to_num(matrix, nan=0.0)
    acc = compute_acc(m)
    bwt = compute_bwt(m)
    fwt = compute_fwt(m)
    fgt = compute_forgetting(m)

    print(f"  ACC (Avg Accuracy)     : {acc:.4f}")
    print(f"  BWT (Backward Transfer): {bwt:+.4f}  {'(forgetting)' if bwt < 0 else '(positive transfer)'}")
    print(f"  FWT (Forward Transfer) : {fwt:+.4f}")
    print(f"  Forgetting             : {fgt:.4f}")


def show_per_task_summary(rd):
    tasks = rd.get("tasks", {})
    if not tasks:
        return
    print_section("PER-TASK TRAINING SUMMARY")
    print(f"  {'Task':>4s}  {'Arch Changed':>12s}  {'Reason':>25s}  {'Final V Loss':>12s}  {'Test Curr':>10s}  {'Test Exp':>10s}")
    print("  " + "-" * 80)

    for tid in sorted(tasks.keys()):
        info = tasks[tid]
        changed = info.get("architecture_changed", False)
        reason = info.get("change_reason", "")
        mt = info.get("main_training", {})

        v_loss = mt["V"][-1] if mt.get("V") else None
        test_curr = mt["test_current"][-1] if mt.get("test_current") else None
        test_exp = mt["test_experience"][-1] if mt.get("test_experience") else None

        v_str = f"{v_loss:.6f}" if v_loss is not None else "n/a"
        tc_str = f"{test_curr:.4f}" if test_curr is not None else "n/a"
        te_str = f"{test_exp:.4f}" if test_exp is not None else "n/a"

        print(f"  {tid:4d}  {str(changed):>12s}  {reason:>25s}  {v_str:>12s}  {tc_str:>10s}  {te_str:>10s}")


def show_architecture_history(rd):
    hist = rd.get("architecture_history", {})
    if not hist:
        return
    print_section("ARCHITECTURE HISTORY")
    for tid in sorted(hist.keys()):
        h = hist[tid]
        changed = h.get("arch_changed", False)
        orig = h.get("original_arch")
        final = h.get("final_arch")
        ratio = h.get("loss_ratio")

        print(f"  Task {tid}: changed={changed}", end="")
        if ratio is not None:
            print(f"  loss_ratio={ratio:.4f}", end="")
        if changed and orig != final:
            print(f"\n         {orig} -> {final}", end="")
        print()


def show_compute_timing(rd):
    tasks = rd.get("tasks", {})
    has_timing = any(tasks[t].get("compute") for t in tasks)
    if not has_timing:
        return

    print_section("COMPUTE TIMING (per task, cumulative across restarts)")

    all_steps = set()
    for tid in tasks:
        for k, v in tasks[tid].get("compute", {}).items():
            if isinstance(v, dict) and "time_s" in v:
                all_steps.add(k)
    all_steps.discard("total")
    step_order = ["preliminary", "arch_search", "ab_training", "compute_v",
                  "warmup", "main", "standard_training"]
    steps = [s for s in step_order if s in all_steps]
    steps += sorted(all_steps - set(step_order))

    header = f"  {'Task':>4s}"
    for s in steps:
        header += f"  {s:>14s}"
    header += f"  {'TOTAL':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grand_total = 0.0
    for tid in sorted(tasks.keys()):
        compute = tasks[tid].get("compute", {})
        if not compute:
            continue
        row = f"  {tid:4d}"
        for s in steps:
            t = compute.get(s, {}).get("time_s")
            row += f"  {t:14.1f}" if t is not None else f"  {'-':>14s}"
        total = compute.get("total", {}).get("time_s")
        row += f"  {total:10.1f}" if total is not None else f"  {'-':>10s}"
        if total:
            grand_total += total
        print(row)

    print(f"\n  Grand total: {grand_total:.1f}s ({grand_total/3600:.2f}h)")


def export_csv(rd, out_path):
    matrix = extract_performance_matrix(rd)
    if matrix is None:
        print(f"Cannot export CSV: no performance matrix")
        return

    T = matrix.shape[0]
    lines = ["metric,value"]
    lines.append(f"ACC,{compute_acc(matrix):.6f}")
    lines.append(f"BWT,{compute_bwt(matrix):.6f}")
    lines.append(f"FWT,{compute_fwt(matrix):.6f}")
    lines.append(f"Forgetting,{compute_forgetting(matrix):.6f}")
    lines.append("")
    lines.append("after_task," + ",".join(f"task_{i}" for i in range(T)))
    for j in range(T):
        row = f"{j}," + ",".join(f"{matrix[j,i]:.6f}" if i <= j else "" for i in range(T))
        lines.append(row)

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\nExported to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Show results from a records pickle")
    parser.add_argument("path", help="Path to *_records.pkl or checkpoint directory")
    parser.add_argument("--csv", help="Export metrics + matrix to CSV")
    args = parser.parse_args()

    rd = load_record_dict(Path(args.path))

    show_metadata(rd)
    matrix = show_performance_matrix(rd)
    show_cl_metrics(matrix)
    show_per_task_summary(rd)
    show_architecture_history(rd)
    show_compute_timing(rd)

    if args.csv:
        export_csv(rd, args.csv)


if __name__ == "__main__":
    main()
