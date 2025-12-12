# ContLearn Configuration Parameters Guide

This document describes all configuration parameters used in ContLearn JSON config files, including required parameters and optional overrides with their default values.

## Core Parameters (Required)

### Basic Settings
- **`delta`**: Small perturbation value for gradual task drift (float, e.g., 1e-03)
- **`tensorfile`**: TensorBoard log directory path (string)
- **`file`**: Results dictionary output path (string)
- **`model_path`**: Model checkpoint save path (string)
- **`flag`**: Regularization coefficients [lambda1, lambda2] (array of 2 floats)

### Task Configuration
- **`n_task`**: Number of continual learning tasks (integer)
- **`epochs_per_task`**: Training epochs per task (integer)
- **`save_iter`**: Frequency of saving metrics to TensorBoard (integer)

### Dataset Settings
- **`data`**: Dataset name
  - Regression: `"sine"`
  - Classification: `"mnist"`, `"cifar10"`, `"cifar100"`, `"permuted_mnist"`
  - Graph: `"ENZYMES"`, `"MUTAG"`, `"PROTEINS"`, `"synthetic"`

### Problem Type
- **`loss`**: Loss function type
  - `"mse"` for regression
  - `"class"` for classification
- **`metric`**: Evaluation metric (same as loss)
- **`problem`**: Data structure type
  - `"vectors"` for image/vector data
  - `"graph"` for graph data
- **`prob`**: Problem category
  - `"regression"`, `"classification"`, `"graphclassification"`

### Training Settings
- **`lr`**: Learning rate for optimizer (float, e.g., 1e-4)
- **`batch_size`**: Batch size for training (integer)
  - For graphs, use `"batch"` instead
- **`network`**: Network architecture type
  - `"fcnn"` (fully connected/MLP)
  - `"cnn"` (convolutional)
  - `"gnn"` (graph neural network)

### Model Architecture
- **`n_layers`**: Number of layers in network (integer)
- **`hln`**: Hidden layer neurons (integer)
  - Used to construct `mlp_hidden_layers` if not explicitly provided

### Classification-Specific
- **`n_class`**: Number of output classes (integer)

### Graph-Specific
- **`class_per_task`**: Number of classes per task for graph classification (integer)
- **`batch`**: Batch size for graph data (integer)

---

## Optional Parameters (Override Defaults)

### General Optional Parameters

#### Random Seeds
- **`seed`**: Random seed (default: 5678)
- **`graph_seed`**: Random seed for graph datasets (default: 10)

#### Batch Sizes (Fine-grained Control)
- **`vector_batch_size`**: Batch size for vector/image data (default: 64)
  - Falls back to `batch_size` if not provided
- **`graph_batch_size`**: Batch size for graph data (default: 20)
  - Falls back to `batch` if not provided

#### Experience Replay Buffer Sizes
- **`vector_replay_size`**: Experience replay buffer size for vectors (default: 20000)
- **`graph_replay_size`**: Experience replay buffer size for graphs (default: 200000)
- **`class_replay_size`**: Experience replay buffer size for classification (default: 200000)

### MLP-Specific Optional Parameters
- **`mlp_hidden_layers`**: Explicit MLP architecture (array of integers)
  - Default: `[hln] * n_layers`
  - Example: `[129, 129, 129]`

### CNN-Specific Optional Parameters (MNIST/Omniglot)
- **`filter_size`**: Convolutional filter size (default: 3)
- **`input_size`**: Input image size (default: 28 for MNIST)
- **`channel_in`**: Number of input channels (default: 1 for grayscale)
- **`channel_out`**: CNN output channels (default: 3)
- **`cnn_feed_sizes`**: CNN feed-forward layer sizes (default: [1875, 512, 64, 10])
- **`awb_arch`**: AWB architecture (default: [1875, 700, 100, 10])
- **`awb_filter_size`**: AWB filter size (default: filter_size + 2)
- **`padding`**: Convolution padding (default: 0)
- **`stride`**: Convolution stride (default: 1)

### CNN3D-Specific Optional Parameters (CIFAR)
- **`filter_size`**: Convolutional filter size (default: 3)
- **`input_size`**: Input image size (default: 32 for CIFAR)
- **`channel_in`**: Number of input channels (default: 3 for RGB)
- **`channel_out`**: First conv layer output channels (default: 32)
- **`cnn3d_feed_sizes`**: CNN3D feed-forward layer sizes (default: [2304, 512, 256, num_classes])
  - First value is calculated from conv output
- **`awb_filter_increment`**: AWB filter size increment (default: 2)
- **`awb_hidden_layers`**: AWB architecture hidden layers (default: [512, 256])

### GCN-Specific Optional Parameters
- **`gcn_sizes`**: GCN layer sizes (default: [input_size, 128])
- **`gcn_mlp_sizes`**: GCN MLP layer sizes (default: [128, 128, 128, num_classes])
- **`awb_fnn_arch`**: AWB FNN architecture (default: [100, 140, 140, out_size])
- **`awb_gcn_arch`**: AWB GCN architecture (default: [in_size, 100])

### Data Augmentation Parameters
- **`rotation_range`**: Rotation range in degrees for augmentation (default: 180)
- **`scaling_range`**: Scaling range for augmentation (default: [1, 2])
- **`train_test_split`**: Train/test data split ratio (default: 0.8)

### Permuted MNIST Specific
- **`image_size`**: MNIST image size (default: 28)
- **`permutation_seed_multiplier`**: Seed multiplier for task-specific permutations (default: 1000)

### Omniglot Specific
- **`omni_num_classes`**: Total number of Omniglot classes to sample from (default: 10)
- **`omni_num_select`**: Number of Omniglot classes to select per task (default: 3)

### Architecture Search Parameters
- **`arch_search`**: Enable architecture search (default: false)
- **`arch_start_task`**: Task number to start architecture search (default: 999, i.e., disabled)
- **`arch_search_epochs`**: Epochs for architecture search (default: 100)
- **`arch_search_threshold`**: Loss threshold for search termination (default: 0.6)
- **`arch_search_max_iter`**: Maximum search iterations (default: 10)
- **`arch_search_range`**: Search grid range (default: 5)
- **`arch_search_mlp_increment`**: MLP size increment step (default: 15)
- **`arch_search_large_increment`**: Large architecture jump size (default: 250)

### Synthetic Graph Dataset Parameters
- **`synthetic_num_graphs`**: Number of synthetic graphs (default: 1000)
- **`synthetic_num_channels`**: Number of graph channels (default: 5)
- **`synthetic_avg_num_nodes`**: Average number of nodes per graph (default: 2)
- **`synthetic_num_classes`**: Number of graph classes (default: 10)

---

## Configuration Examples

### Minimal Regression Config (Sine Wave)
```json
{
    "delta": 1e-03,
    "tensorfile": "logdir/tensorboard/sine",
    "file": "logdir/dicts/sine",
    "model_path": "logdir/model/sine",
    "flag": [1, 0],
    "n_task": 20,
    "epochs_per_task": 500,
    "save_iter": 1,
    "data": "sine",
    "loss": "mse",
    "metric": "mse",
    "problem": "vectors",
    "lr": 1e-4,
    "batch_size": 64,
    "n_layers": 4,
    "hln": 129,
    "network": "fcnn",
    "prob": "regression"
}
```

### CIFAR-10 with Custom CNN3D Architecture
```json
{
    "data": "cifar10",
    "prob": "classification",
    "network": "cnn",
    "n_class": 10,
    "batch_size": 128,
    "lr": 1e-4,
    "epochs_per_task": 1000,
    "cnn3d_feed_sizes": [2304, 512, 256, 10],
    "filter_size": 3,
    "channel_out": 32
}
```

### Graph Classification with Custom GCN
```json
{
    "data": "ENZYMES",
    "prob": "graphclassification",
    "network": "gnn",
    "n_class": 6,
    "class_per_task": 3,
    "batch": 64,
    "lr": 1e-4,
    "epochs_per_task": 500,
    "gcn_sizes": [5, 128],
    "gcn_mlp_sizes": [128, 128, 128, 6]
}
```

---

## Notes

1. **Parameter Precedence**: Specific parameters (e.g., `vector_batch_size`) override general ones (e.g., `batch_size`), which override defaults from `config/constants.py`.

2. **Architecture Inference**: If architecture parameters are not provided, they are constructed from `n_layers` and `hln`, or from defaults in `constants.py`.

3. **Automatic Size Calculation**: Some sizes (e.g., CNN input layer size) are calculated automatically based on conv/pool operations and don't need to be specified.

4. **JSON Format**: All config files are pure JSON without comment fields. Comments in this guide are for documentation only.

5. **Validation**: The framework validates that required parameters exist and uses sensible defaults for optional ones.
