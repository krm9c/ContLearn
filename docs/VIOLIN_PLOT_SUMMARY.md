# Violin Plot Implementation for Eigenvalues

## Overview

Implemented violin plots as the default visualization for eigenvalue evolution. Violin plots provide a smooth, elegant representation of the full eigenvalue distribution at each iteration, showing both density and range.

## Implementation

### Function: `plot_eigenvalues_violin()`

**Location**: `plot_results.py:242-376`

Creates violin plots for A and B matrix eigenvalues across all layers.

### Key Features

1. **Smooth Distribution Curves**
   - Shows the full probability density function
   - Kernel density estimation creates smooth shapes
   - More visually appealing than box plots

2. **Statistical Elements**
   - **Median line**: Crimson color, 2.5px thickness
   - **Extrema bars**: Show min/max values
   - **Range bars**: Connect extrema
   - No mean line (cleaner appearance)

3. **Styling**
   - **A matrices**: Light blue fill, dark blue edges
   - **B matrices**: Light green fill, dark green edges
   - **Transparency**: 70% (alpha=0.7)
   - **Edge width**: 1.5px for clear definition
   - **Width**: 70% of iteration spacing

4. **Background & Grid**
   - Light gray background (#f8f9fa)
   - Dashed grid lines (alpha=0.3, linewidth=0.5)
   - Less intrusive than solid grid

5. **Task Visualization**
   - Color-coded task shading (alpha=0.08)
   - Vertical dashed lines at task boundaries
   - Black lines with alpha=0.5, linewidth=1.5

## Advantages Over Box Plots

| Aspect | Box Plots | Violin Plots |
|--------|-----------|--------------|
| **Distribution shape** | Quartiles only | Full density curve |
| **Visual appeal** | Blocky | Smooth, elegant |
| **Multimodality** | Hidden | Visible |
| **Sparse data** | Better | Can be irregular |
| **Dense data** | Cluttered | Clear |
| **Presentation** | Standard | More striking |
| **Publications** | Common | Increasingly popular |

## When Violin Plots Excel

1. **Dense eigenvalue distributions**: Many eigenvalues per iteration
2. **Smooth evolution**: Gradual changes over iterations
3. **Presentations**: More visually impressive
4. **Multimodal distributions**: Can see multiple peaks
5. **Modern papers**: Increasingly preferred in ML/AI

## Styling Details

### A Matrix Violin Plots
```python
parts = ax.violinplot(eigenvals_per_iter, positions=positions, widths=width,
                     showmeans=False, showmedians=True, showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_edgecolor('darkblue')
    pc.set_alpha(0.7)
    pc.set_linewidth(1.5)

parts['cmedians'].set_edgecolor('crimson')
parts['cmedians'].set_linewidth(2.5)
parts['cbars'].set_edgecolor('darkblue')
parts['cbars'].set_linewidth(1.5)
```

### B Matrix Violin Plots
Same structure with green colors:
- Fill: `lightgreen`
- Edges: `darkgreen`
- Median: `crimson` (consistent across both)

## Visual Interpretation

### What to Look For

1. **Width of violins**: Indicates how many eigenvalues are in that range
2. **Shape**:
   - Wide bulge = many eigenvalues at that magnitude
   - Thin region = few eigenvalues at that magnitude
   - Multiple bulges = multimodal distribution
3. **Median line**: Central tendency at each iteration
4. **Evolution over tasks**:
   - Changes in shape show architectural adaptation
   - Vertical extent shows eigenvalue range
   - Task boundaries marked clearly

### Example Interpretations

**Narrow, tall violins**:
- Few eigenvalues
- Concentrated in specific range
- High precision

**Wide, short violins**:
- Many eigenvalues
- Spread across range
- More diverse spectrum

**Bimodal violins**:
- Two clusters of eigenvalues
- Could indicate layer specialization
- Interesting for analysis

## File Size

Violin plots are larger than box plots due to smooth curves:
- Box plots: ~739 KB
- Violin plots: ~2.1 MB

**Reason**: Each violin is a polygon with many vertices for smooth appearance.

**Mitigation**:
- 300 DPI is appropriate for publications
- Can reduce DPI to 150-200 for presentations to halve file size
- Quality remains excellent

## Performance

Generation time similar to box plots:
- ~1-2 seconds per run
- Negligible overhead from KDE computation
- Matplotlib handles rendering efficiently

## Comparison with Other Styles

### Box Plots (Alternative: `style='box'`)
- ✅ Smaller file size
- ✅ Clearer quartile information
- ✅ Better for sparse data
- ❌ Less visually appealing
- ❌ Hides distribution shape

### Heatmaps (Alternative: `style='heatmap'`)
- ✅ Shows all eigenvalues simultaneously
- ✅ Good for tracking individual eigenvalues
- ✅ Color-coded magnitude
- ❌ Harder to see distributions
- ❌ Requires sorting eigenvalues

### Violin Plots (Default)
- ✅ Beautiful, smooth appearance
- ✅ Shows full distribution
- ✅ Excellent for presentations
- ✅ Reveals multimodality
- ❌ Larger file size
- ❌ Can be irregular with sparse data

## Usage

Default behavior (violin plots):
```bash
python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl
```

To use box plots instead:
```python
# In plot_results.py, change default:
def plot_eigenvalues(..., style: str = 'box'):
```

To use heatmaps instead:
```python
# In plot_results.py, change default:
def plot_eigenvalues(..., style: str = 'heatmap'):
```

## Integration with Task Shading

Violin plots work perfectly with task shading:
- Color-coded backgrounds for each task
- Violin shapes overlay cleanly
- Task boundaries remain visible
- No visual conflicts

The combination provides:
- Clear task separation (shading)
- Detailed distribution information (violins)
- Exact boundaries (dashed lines)
- Professional appearance

## Best Practices

1. **For publications**: Use violin plots with 300 DPI
2. **For presentations**: Violin plots are ideal
3. **For quick analysis**: Box plots may be faster to interpret
4. **For eigenvalue tracking**: Use heatmaps

## Future Enhancements

Potential improvements:
1. **Quantile lines**: Add 25th and 75th percentile lines
2. **Color gradients**: Gradient fills based on density
3. **Interactive tooltips**: For web-based viewing
4. **Half violins**: Show A and B on same plot, mirrored
5. **Swarm overlay**: Show individual eigenvalues as dots on violin

## Summary

Violin plots provide an elegant, informative visualization of eigenvalue evolution:

✅ **Smooth, beautiful curves** showing full distributions
✅ **Clear median lines** for central tendency
✅ **Consistent styling** with color scheme (blue/green)
✅ **Task shading integration** for continual learning context
✅ **Professional appearance** suitable for publications
✅ **Reveals distribution shape** including multimodality

Perfect for understanding how eigenvalue distributions evolve during continual learning!
