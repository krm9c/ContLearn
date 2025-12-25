# JLSE Experiment Runs

This directory contains logs and results from sequential experiment runs.

## Directory Structure

```
jlse/
├── logs/          # Execution logs for each condition
└── results/       # (Reserved for future use)
```

## Running Experiments

From the `kkt_run/` directory, run one of the following scripts:

```bash
# Run all 4 conditions for SINE
./run_sine.sh

# Run all 4 conditions for MNIST
./run_mnist.sh

# Run all 4 conditions for CIFAR-10
./run_cifar10.sh

# Run all 4 conditions for CIFAR-100
./run_cifar100.sh
```

Each script runs 4 conditions sequentially:
1. **Condition 1**: Baseline (no AWB)
2. **Condition 2**: Heuristics
3. **Condition 3**: Architecture search without transfer
4. **Condition 4**: AWB Full (complete pipeline)

## Logs

Logs are saved with timestamps:
```
jlse/logs/{dataset}_{condition}_{timestamp}.log
```

Example:
```
jlse/logs/sine_condition1_baseline_20231225_143022.log
jlse/logs/mnist_condition4_awb_full_20231225_150145.log
```

## Output Data

Training outputs (pickle files, figures) are saved to the main `outputs/` directory at the repository root.
