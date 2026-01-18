# AWB Task 1 Instability Issue

## Problem
In the synthetic graph CL experiments, AWB shows **training instability at the end of Task 1** after architecture expansion from 3K → 53K params.

## Evidence
From `outputs/loss_across_tasks_correct.png`:
- AWB loss at Task 1: 0.29 → 0.61 → **1.26** (final epochs)
- Baseline stays stable: ~0.38 throughout Task 1
- AWB recovers in Task 2 (drops back to 0.30)

## Hypothesis
The expanded 53K param architecture may need:
1. More A/B training epochs (currently 50)
2. Lower learning rate after expansion
3. Longer warmup period

## Config Reference
- File: `runs__/configs/synthetic_graph_minimal_awb.json`
- Current settings: `awb_ab_training_epochs: 50`, `awb_ab_lr: 0.001`

## Related Files
- Results: `outputs/synthetic_graph_minimal_awb_run0/`
- Plot: `outputs/loss_across_tasks_correct.png`
- Full docs: `.claude/synthetic_graph_experiments.md`
