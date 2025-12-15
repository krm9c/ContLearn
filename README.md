# ContLearn 
This is jax repository of universal CL code. The present code work with CNN/FNN and GNN.
--
## Dependencies
jax
equinox
optax
diffrax
numpy 
pandas
matplotlib
tmux

--
## Continual Learning without Architecture change
 The main branch has code that 
performs continual learning without modifying architecture.  

### Code Execution
sh script.sh

--
## Continual Learning with Architecture change
For code regarding the paper, the effect of architecture to mitigate forgetting in continual learning which seeks to 
change the architecture on the fly while training continually.  

### Code Execution

Please Checkout branch AWBT_code, which can be done by 

git checkout AWBT_code

and execute programs by 

sh script.sh

## Modify parameters
To modify parameters use the json files in the json folder.

- **JAX/Equinox Implementation**: Modern, functional approach with JIT compilation
- **Hamiltonian Gradient Computation**: Physics-inspired regularization for continual learning
- **AWB Architecture Morphing**: Adaptive Weight Basis for dynamic network expansion during training
- **Multiple Datasets**: Sine regression, MNIST, CIFAR-10/100, synthetic graphs
- **Flexible Architectures**: MLP, CNN, CNN3D, GCN with easy extensibility
- **Automatic Plotting**: Loss curves, metrics, eigenvalue analysis
- **Experience Replay**: Built-in replay buffer management
- **Configurable Everything**: JSON-based configuration with smart defaults

## Installation

### Prerequisites
- Python 3.9+
- CUDA (optional, for GPU support)

### Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml
conda activate jaxss

# Install in editable mode
pip install -e ".[all]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[all]"

# Or install minimal dependencies
pip install -e .

# Optional: install specific extras
pip install -e ".[vision]"     # For MNIST/CIFAR
pip install -e ".[graph]"       # For graph datasets
pip install -e ".[plotting]"    # For visualization
pip install -e ".[dev]"         # For development/testing
```

## Quick Start

### Basic Training

```bash
# Sine wave regression
python scripts/run.py config/sine.json

# MNIST classification
python scripts/run.py config/mnist.json

# CIFAR-10 classification
python scripts/run.py config/cifar10.json
```

### With AWB Architecture Morphing

```bash
# Enable AWB for adaptive architecture changes
python scripts/run.py config/sine_awb.json
python scripts/run.py config/mnist_awb.json
```

### Multiple Runs with Plotting

```bash
# Run 3 times and generate comparison plots
python scripts/run.py config/sine.json --runs 3

# Skip automatic plotting
python scripts/run.py config/sine.json --no-plots

# Custom output directory
python scripts/run.py config/sine.json --figures-dir outputs/my_figures
```

## Supported Datasets

| Dataset | Type | Auto-Download | Tasks |
|---------|------|---------------|-------|
| Sine | Regression | ✅ | 5 |
| MNIST | Classification | ✅ | 10 classes |
| Permuted MNIST | Classification | ✅ | Permuted pixels |
| CIFAR-10 | Classification | ✅ | 10 classes |
| CIFAR-100 | Classification | ✅ | 100 classes |
| Synthetic Graphs | Graph Classification | ✅ | Configurable |

**Note**: All datasets are automatically downloaded on first use via torchvision/built-in loaders.

## Supported Models

- **MLP (FCNN)**: Fully connected neural networks for regression/classification
- **CNN**: 2D convolutional networks for MNIST
- **CNN3D**: 3D convolutional networks for CIFAR
- **GCN**: Graph convolutional networks for graph classification

## Configuration

All experiments are controlled via JSON config files. See [`config/TEMPLATE.md`](config/TEMPLATE.md) for comprehensive documentation of 80+ available parameters.

### Minimal Config Example

```json
{
    "data": "mnist",
    "n_task": 5,
    "epochs_per_task": 100
}
```

The framework automatically selects appropriate settings based on the dataset.

### AWB Config Example

```json
{
    "data": "sine",
    "n_task": 5,
    "epochs_per_task": 1000,
    "awb_enabled": true,
    "awb_preliminary_epochs": 100,
    "awb_ab_training_epochs": 50,
    "arch_search_epochs": 1000
}
```

## Project Structure

```
ContLearn/
├── config/              # Training configurations
│   ├── TEMPLATE.md      # Configuration documentation
│   ├── sine.json        # Sine regression config
│   ├── mnist.json       # MNIST config
│   └── *.json           # More configs
├── src/cl/              # Core framework code
│   ├── config/          # Configuration management
│   ├── core/            # Training loops, Hamiltonian, AWB
│   ├── datasets/        # Dataset implementations
│   ├── models/          # Neural network architectures
│   └── runners/         # Problem-specific orchestration
├── scripts/             # Executable scripts
│   ├── run.py           # Main training script
│   └── plot_results.py  # Plotting utility
├── tests/               # Unit and integration tests
├── docs/                # Internal documentation
└── outputs/             # Generated outputs and plots
```

## Testing

```bash
# Run all tests
./run_tests.sh --all

# Run specific test suites
./run_tests.sh --models      # Model tests
./run_tests.sh --datasets    # Dataset tests
./run_tests.sh --integration # Integration tests
./run_tests.sh --fast        # Skip slow tests

# With coverage
./run_tests.sh --all --cov

# Using pytest directly
pytest                       # All tests
pytest tests/test_models.py  # Specific file
pytest -v --tb=short         # Verbose output
```

## Advanced Usage

### Custom Datasets

See [`docs/LAYER_AWB_ARCHITECTURE.md`](docs/LAYER_AWB_ARCHITECTURE.md) for guidance on:
- Adding new datasets
- Creating custom models
- Extending the framework

### Architecture Search

Enable AWB for automatic architecture adaptation:

```json
{
    "awb_enabled": true,
    "arch_search_epochs": 1000,
    "arch_search_lr": 0.001
}
```

**Note**: Set `arch_search_epochs >= epochs_per_task` to ensure fair comparison between architectures.

### Hamiltonian Gradient Tuning

Control the gradient weighting via `grad_weights: [alpha, beta, gamma]`:

```json
{
    "grad_weights": [0.01, 0.98, 0.1]
}
```

- `alpha`: Current task loss gradient weight
- `beta`: Experience replay gradient weight
- `gamma`: Hamiltonian regularization weight

### Learning Rate Schedules

```json
{
    "lr_schedule": "exponential",
    "lr_decay_factor": 0.9,
    "lr_min": 1e-6
}
```

Supported schedules: `constant`, `step`, `exponential`, `cosine`, `linear`

## Plotting

Results are automatically plotted after training:

1. **Losses** (`*_losses.png`): All loss components (H, V, dV, gradient norms)
2. **Metrics** (`*_metrics.png`): Train and test performance
3. **Eigenvalues** (`*_eigenvalues.png`): Weight matrix eigenvalues (or A/B matrices in AWB mode)
4. **Overview** (`*_overview.png`): Combined visualization

Manual plotting:

```bash
python scripts/plot_results.py outputs/my_results.pkl --output-dir figures
```

## Documentation

- **[Configuration Reference](config/TEMPLATE.md)**: Complete parameter documentation
- **[Architecture Guide](docs/LAYER_AWB_ARCHITECTURE.md)**: Extending the framework
- **[AWB Implementation](docs/AWB_REFACTORING_SUMMARY.md)**: Technical details on AWB
- **[Developer Guide](CLAUDE.md)**: Development workflow and patterns

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- [JAX](https://github.com/google/jax): High-performance numerical computing
- [Equinox](https://github.com/patrick-kidger/equinox): JAX neural networks
- [Optax](https://github.com/deepmind/optax): Gradient processing and optimization

## Contributing

Contributions are welcome! See [`docs/LAYER_AWB_ARCHITECTURE.md`](docs/LAYER_AWB_ARCHITECTURE.md) for guidelines on adding new:
- Datasets
- Models
- Loss functions
- Metrics

## Contact

For questions or issues, please open an issue on GitHub.
