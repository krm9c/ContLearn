# Test Quick Reference Card

## 🚀 Quick Commands

| Command | Description |
|---------|-------------|
| `./run_tests.sh` | Run all tests (default) |
| `./run_tests.sh --fast` | Skip slow integration tests |
| `./run_tests.sh --cov` | Run with coverage report |
| `python run_tests.py --all` | Run all tests (Python) |
| `pytest tests/` | Run all tests (pytest directly) |

## 📁 Test Files

| File | Tests |
|------|-------|
| `test_models.py` | MLP, CNN, CNN3D, Linear layers |
| `test_graph_models.py` | GCN, GAT, myNN, pooling |
| `test_data.py` | Data loading, experience replay |
| `test_trainer.py` | Training, loss functions, metrics |
| `test_checkpoint.py` | Model initialization |
| `test_runners.py` | Training pipelines (integration) |
| `test_utils.py` | Utilities, preprocessing |
| `test_config.py` | Configuration management |

## 🎯 Run Specific Tests

```bash
# By category
./run_tests.sh --models
./run_tests.sh --data
./run_tests.sh --trainer
./run_tests.sh --graph

# By file
pytest tests/test_models.py
pytest tests/test_data.py

# By class
pytest tests/test_models.py::TestMLP

# By function
pytest tests/test_models.py::TestMLP::test_mlp_forward_pass

# By pattern
pytest tests/ -k regression
pytest tests/ -k "graph and not slow"
```

## 🔍 Common Options

| Option | Flag | Description |
|--------|------|-------------|
| Verbose | `-v` | Detailed test output |
| Show prints | `-s` | Display print statements |
| Coverage | `--cov` | Coverage report |
| HTML coverage | `--cov-html` | HTML coverage report |
| Parallel | `--parallel` | Run tests in parallel |
| Stop on fail | `-x` | Stop at first failure |
| Pattern | `-k PATTERN` | Run matching tests |

## 📊 Coverage

```bash
# Terminal coverage
./run_tests.sh --cov

# HTML coverage (better visualization)
./run_tests.sh --cov-html
open htmlcov/index.html

# With pytest
pytest tests/ --cov=utils --cov=training --cov=config --cov=data
```

## 🐛 Debugging

```bash
# Show prints and stop on first failure
pytest tests/test_models.py -s -x

# Show local variables on failure
pytest tests/test_models.py -l

# Single test with verbose output
pytest tests/test_models.py::TestMLP::test_mlp_forward_pass -v -s
```

## ⚡ Fast Workflow

```bash
# During development (fast feedback)
./run_tests.sh --fast

# Before commit (full validation)
./run_tests.sh --all --cov

# Check specific change
pytest tests/test_models.py -v
```

## 📚 Documentation

- **Detailed Guide**: [TESTING.md](TESTING.md)
- **Test Documentation**: [tests/README.md](tests/README.md)
- **Help**: `./run_tests.sh --help`

## 💡 Tips

1. Use `--fast` for quick iteration during development
2. Use `--cov-html` to identify untested code
3. Use `-k` pattern matching to run related tests
4. Use `--parallel` for faster execution on multi-core systems
5. Integration tests (in `test_runners.py`) are slower but thorough

## 🔧 Requirements

```bash
pip install pytest pytest-cov pytest-xdist
```
