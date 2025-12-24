# AWB (Adaptive Weight Basis) Testing Suite

This directory contains scripts for testing all steps of the AWB pipeline to ensure mathematical correctness and measure performance.

## AWB 5-Step Algorithm Overview

```
Task 0: Standard CL training (no AWB)

Tasks 1+:
    STEP 1: Preliminary training on new task
    STEP 2: Decide if architecture change needed (loss comparison)
    If change needed:
        STEP 3a: Architecture search for optimal dimensions
        STEP 3b: Train A/B matrices with W frozen
        STEP 4: Compute V = A @ W @ B.T (weight transformation)
        STEP 5: Train V with A/B frozen
    Else:
        Continue standard training
```

## Directory Structure

```
awb_tests/
├── README.md                    # This file
├── run_all_tests.sh            # Run all AWB tests
├── test_step1_preliminary.py   # STEP 1: Preliminary training test
├── test_step2_decision.py      # STEP 2: Architecture change decision test
├── test_step3a_arch_search.py  # STEP 3a: Architecture search test
├── test_step3b_ab_training.py  # STEP 3b: A/B matrix training test
├── test_step4_v_transform.py   # STEP 4: V transformation test
├── test_step5_v_training.py    # STEP 5: V training test
├── test_full_pipeline.py       # Full AWB pipeline integration test
├── test_mathematical_correctness.py  # Mathematical property verification
├── benchmark_performance.py    # Performance benchmarking
├── configs/                    # Test configurations
│   ├── awb_test_mlp.json      # MLP AWB test config
│   ├── awb_test_cnn.json      # CNN AWB test config
│   └── awb_test_gcn.json      # GCN AWB test config
├── logs/                       # Test logs
└── results/                    # Test results and reports
```

## Usage

### Run All Tests
```bash
./awb_tests/run_all_tests.sh
```

### Run Individual Step Tests
```bash
# Test specific step
python awb_tests/test_step1_preliminary.py
python awb_tests/test_step3a_arch_search.py

# Run with verbose output
python awb_tests/test_step4_v_transform.py --verbose
```

### Run Mathematical Correctness Tests
```bash
python awb_tests/test_mathematical_correctness.py
```

### Run Performance Benchmarks
```bash
python awb_tests/benchmark_performance.py --output results/benchmark_report.json
```

## Test Descriptions

### Step Tests

| Test | Description | Checks |
|------|-------------|--------|
| `test_step1_preliminary.py` | Preliminary training | Loss decreases, gradients flow |
| `test_step2_decision.py` | Architecture decision | Threshold logic, loss comparison |
| `test_step3a_arch_search.py` | Architecture search | Candidate generation, search convergence |
| `test_step3b_ab_training.py` | A/B matrix training | W frozen, A/B update, loss converges |
| `test_step4_v_transform.py` | V = A @ W @ B.T | Shape correctness, forward pass equivalence |
| `test_step5_v_training.py` | V training | A/B frozen, V updates, loss converges |

### Mathematical Correctness Tests

- **V transformation correctness**: `V = A @ W @ B.T` produces correct output
- **Partition correctness**: A/B correctly frozen/unfrozen
- **Gradient flow**: Gradients only update trainable parameters
- **Shape consistency**: Shapes match after architecture change
- **Output equivalence**: `model(x)` == `model_with_V(x)` after transformation

### Performance Benchmarks

- **JIT compilation time** per step
- **Training throughput** (samples/sec)
- **Memory usage** per step
- **GPU utilization** during training
- **Architecture search overhead**

## Expected Results

### Mathematical Properties

1. **V Transformation (Step 4)**
   - `output_awb = A @ W @ B.T @ x` should match `output_v = V @ x`
   - Error should be < 1e-5 (numerical precision)

2. **Partition Correctness**
   - During A/B training: W gradients = 0
   - During V training: A, B gradients = 0

3. **Loss Convergence**
   - Each step should show decreasing loss
   - Final loss < initial loss

### Performance Baselines (MLP on GPU)

| Step | Expected Time | Notes |
|------|---------------|-------|
| Step 1 (100 epochs) | ~10s | Preliminary training |
| Step 3a (5 trials) | ~5s per trial | Bayesian search |
| Step 3b (50 epochs) | ~5s | A/B training |
| Step 4 | <100ms | V computation |
| Step 5 (100 epochs) | ~10s | V training |

## Troubleshooting

### Test Failures

1. **Shape mismatch**: Check architecture sizes after transformation
2. **Gradient issues**: Verify partition masks are applied correctly
3. **Loss not decreasing**: Check learning rate and optimizer settings

### Performance Issues

1. **Slow JIT**: Expected on first call, should be cached after
2. **High memory**: Reduce batch size or model size
3. **Low GPU utilization**: Increase batch size
