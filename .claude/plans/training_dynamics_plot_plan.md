# Plan: Comprehensive Training Dynamics Visualization

## Objective
Create a large multi-panel plot comparing Baseline vs AWB training dynamics, showing loss and accuracy progression with architecture change markers.

## Plot Structure (2x3 grid, figsize ~18x12)

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   BASELINE LOSS     │   AWB LOSS          │   LOSS COMPARISON   │
│   vs Epochs         │   vs Update Steps   │   (Overlay)         │
│   (per task color)  │   + Arch markers    │                     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│   BASELINE ACC      │   AWB ACC           │   FINAL COMPARISON  │
│   Train & Test      │   Train & Test      │   Bar chart         │
│   vs Epochs         │   + Arch markers    │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

## Data Sources
- Baseline: `outputs/synthetic_graph_minimal_baseline_run0/classification_synthetic_taskshift_gcn_run0_records.pkl`
- AWB: `outputs/synthetic_graph_minimal_awb_run0/classification_synthetic_taskshift_gcn_awb_run0_records.pkl`

## Architecture Markers (AWB only)
- Task 0→1: Mark where architecture changed from [10,32,32]/[32,32,16,5] → [10,72,72]/[72,172,196,5]
- Task 1→2: Mark where architecture changed to [10,142,142]/[142,332,316,5]
- Use vertical lines with annotations showing param count

## Execution Environment
- Environment: `jax__diffusion` conda environment
- Backend: JAX Metal plugin (Apple GPU)
- Verification: Print JAX devices to confirm GPU usage

## Implementation Steps

1. **Environment Setup**
   - Activate `jax__diffusion` environment
   - Verify JAX Metal backend is active
   - Print device info

2. **Data Loading**
   - Load both pickle files
   - Extract iteration-level data (steps, loss, train_acc, test_cur, test_exp, task_id)
   - Extract architecture history from AWB records

3. **Plot Creation**
   - Panel 1 (top-left): Baseline loss vs epochs, colored by task
   - Panel 2 (top-middle): AWB loss vs steps, with vertical lines at architecture changes
   - Panel 3 (top-right): Overlay comparison of both methods
   - Panel 4 (bottom-left): Baseline train/test accuracy vs epochs
   - Panel 5 (bottom-middle): AWB train/test accuracy vs steps, with arch markers
   - Panel 6 (bottom-right): Final accuracy bar comparison

4. **Architecture Annotations**
   - Add vertical dashed lines at architecture change points
   - Add text annotations with architecture specs and param counts
   - Use different line styles for different architecture transitions

5. **Styling**
   - Task colors: Task 0 = green, Task 1 = blue, Task 2 = purple
   - Method distinction: Baseline = solid lines, AWB = dashed lines
   - Architecture markers: Red vertical lines with annotations
   - Legend placement: Outside plot or in least cluttered area

## Output
- Save to: `outputs/training_dynamics_comparison.png`
- Resolution: 300 DPI
- Format: PNG with tight bounding box

## Verification
- Confirm plot shows all 3 tasks for both methods
- Verify architecture markers appear at correct steps
- Check that loss decreases and accuracy increases over training
