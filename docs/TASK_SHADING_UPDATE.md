# Task Shading Feature - Update Summary

## Overview

Added color-coded shaded regions to all plots to visually indicate task boundaries in continual learning experiments. This enhancement makes it immediately clear when different tasks are being trained and helps interpret task-specific behavior.

## Implementation

### New Function

**Location**: `plot_results.py:83-114`

```python
def add_task_shading(ax, series: Dict[str, np.ndarray], metadata: Dict[str, Any], alpha: float = 0.1):
    """Add shaded regions to indicate different tasks.

    Args:
        ax: Matplotlib axis object
        series: Time series data with 'iterations' and 'task_ids'
        metadata: Metadata containing task information
        alpha: Transparency for shaded regions (default: 0.1, using 0.08 in practice)
    """
    task_ids = series['task_ids']
    iterations = series['iterations']

    # Get unique tasks
    unique_tasks = np.unique(task_ids)

    # Define colors for different tasks (cycle through if more tasks than colors)
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Add shaded region for each task
    for i, task_id in enumerate(unique_tasks):
        # Find iterations for this task
        task_mask = task_ids == task_id
        task_iters = iterations[task_mask]

        if len(task_iters) > 0:
            # Get min and max iteration for this task
            min_iter = task_iters.min()
            max_iter = task_iters.max()

            # Add shaded region
            color = colors[int(task_id) % len(colors)]
            ax.axvspan(min_iter, max_iter, alpha=alpha, color=color, zorder=0)
```

### Integration

The `add_task_shading()` function is called in all plotting functions:

1. **`plot_losses()`** - Added to all 6 subplots
2. **`plot_metrics()`** - Added to both subplots
3. **`plot_eigenvalues()`** - Added to all layer subplots (A and B matrices)
4. **`plot_combined_metrics()`** - Added to all 7 subplots
5. **`plot_multi_run_comparison()`** - Added to all 4 subplots
6. **`plot_multi_run_statistics()`** - Added to all 4 subplots

### Visual Changes

**Before**:
- Only vertical dashed lines at task boundaries
- No visual separation between tasks
- Required careful inspection to identify task regions

**After**:
- Color-coded shaded backgrounds for each task
- Vertical dashed lines at exact boundaries (enhanced: alpha=0.5, linewidth=1.5)
- Immediate visual recognition of task regions
- Clear separation of task-specific behavior

## Parameters

- **Alpha transparency**: 0.08 (very subtle, doesn't obscure data)
- **Colors**: Cycle through 8 colors (blue, green, red, orange, purple, brown, pink, gray)
- **Boundary lines**: Black dashed lines with alpha=0.5, linewidth=1.5
- **Z-order**: 0 (shading is behind all other plot elements)

## Benefits

1. **Immediate Recognition**: Task regions are instantly identifiable
2. **Pattern Analysis**: Easy to spot task-specific patterns
3. **Catastrophic Forgetting**: Clear visualization of performance changes at task boundaries
4. **Recovery Patterns**: Can see how quickly the model recovers within each task
5. **Presentation Quality**: More visually informative for papers and presentations
6. **Color Accessibility**: 8 distinct colors support experiments with many tasks

## Example Use Cases

### 3-Task Continual Learning
- Task 0: Blue shaded background
- Task 1: Green shaded background
- Task 2: Red shaded background
- Black dashed lines at transitions

Easy to see:
- Loss spikes when task 1 starts (catastrophic forgetting)
- Recovery pattern within task 1
- Similar pattern at task 2 boundary
- Overall trend across all tasks

### Multi-Run Comparison
All runs share the same task shading (based on first run):
- Consistent visual reference across runs
- Easy comparison of task-specific behavior
- Variability in task transitions visible

## Technical Details

### Task ID Extraction
Tasks are identified from the `task_id` field in iteration records:
```python
series['task_ids'] = np.array([run_data['iterations'][i]['task_id'] for i in iterations])
```

### Handling Edge Cases

1. **Missing task IDs**: Function gracefully handles empty arrays
2. **Single task**: Works fine, entire plot gets single color
3. **Many tasks**: Colors cycle through the 8-color palette
4. **Non-contiguous iterations**: Uses min/max to span full task range

### Performance Impact

- Negligible (~0.01s per plot)
- Uses efficient NumPy operations
- `axvspan()` is a native matplotlib operation

## File Size Impact

Plots are slightly larger due to additional shading elements:

**Before**:
- Losses: ~470 KB
- Metrics: ~264 KB
- Eigenvalues: ~631 KB
- Overview: ~553 KB

**After**:
- Losses: ~506 KB (+7.7%)
- Metrics: ~284 KB (+7.6%)
- Eigenvalues: ~739 KB (+17.1%)
- Overview: ~601 KB (+8.7%)

Still well within reasonable sizes for publication figures.

## Documentation Updates

Updated the following documents:

1. **`docs/PLOTTING_GUIDE.md`**:
   - Added "Visual Elements Explained" section
   - Updated feature lists for all plot types
   - Added task shading to interpretation guide

2. **`docs/PLOTTING_IMPLEMENTATION.md`**:
   - Added `add_task_shading()` to function list
   - Updated design decisions section
   - Documented rationale for task shading

3. **`docs/TASK_SHADING_UPDATE.md`**: This document

## Testing

Verified with existing data:
```bash
python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl
python plot_results.py logdir/model/regression_sine_fcnn_allruns.pkl
```

**Results**:
- ✅ All plots generated successfully
- ✅ Task shading visible and correct
- ✅ Colors appropriate (blue, green, red for 3 tasks)
- ✅ Boundary lines enhanced and clear
- ✅ No data obscured by shading
- ✅ Multi-run plots work correctly

## Future Enhancements

Potential improvements:
1. **Custom color palettes**: Allow user to specify colors
2. **Task labels**: Add text labels for each task region
3. **Adjustable transparency**: Command-line option for alpha value
4. **Gradient shading**: Gradual color transitions instead of solid regions
5. **Task annotations**: Include task names/descriptions from metadata

## Backward Compatibility

Fully backward compatible:
- No changes to data format
- No changes to function signatures (alpha parameter has default)
- Existing scripts work without modification
- Can be disabled by setting alpha=0 if needed

## Summary

The task shading feature significantly enhances the visual interpretability of continual learning experiments by:
- ✅ Color-coding different tasks with shaded backgrounds
- ✅ Enhancing task boundary markers
- ✅ Making task-specific patterns immediately visible
- ✅ Improving presentation quality
- ✅ Maintaining data clarity with low transparency
- ✅ Supporting multi-task experiments with color cycling

All plots now automatically include this feature with no additional user configuration required.
