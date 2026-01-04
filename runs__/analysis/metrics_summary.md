# Continual Learning Metrics Summary

## SINE Dataset

| Condition | ACC | BWT | FWT | Forgetting |
|-----------|-----|-----|-----|------------|
| baseline | 0.026736 | -0.001873 | -0.029019 | 0.001873 |
| heuristics | 0.026322 | -0.001477 | -0.029018 | 0.001477 |
| arch_search | 0.026062 | -0.001403 | -0.029016 | 0.001403 |
| awb_full | 0.028967 | 0.000308 | -0.029027 | 0.000098 |

## MNIST Dataset

| Condition | ACC | BWT | FWT | Forgetting |
|-----------|-----|-----|-----|------------|
| baseline | 0.908035 | -0.019649 | -0.836695 | 0.021804 |
| heuristics | 0.907322 | -0.019708 | -0.836592 | 0.021122 |
| arch_search | 0.862799 | -0.063674 | -0.839164 | 0.063674 |
| awb_full | 0.872435 | -0.060550 | -0.840544 | 0.062585 |


## Metric Interpretations

- **ACC (Average Accuracy/MSE)**: Mean performance across all tasks after training.
- **BWT (Backward Transfer)**: Negative values indicate catastrophic forgetting.
- **FWT (Forward Transfer)**: Positive values indicate beneficial knowledge transfer.
- **Forgetting**: Average maximum performance drop per task. Lower is better.
