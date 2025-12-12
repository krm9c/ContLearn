# Modularity Improvement TODOs

This document tracks planned refactoring tasks to improve code modularity in the ContLearn codebase.

---

## Priority 1: High Impact, Low Effort

### 1.1 Consolidate Linear Layer Classes
- **File:** `utils/model.py`
- **Status:** [ ] Not Started
- **Description:** Merge 4 nearly identical Linear classes (Linear, Linear1, Linear2, Linear3) into a single parameterized class
- **Current Issues:**
  - `Linear` (lines 421-436): `x @ weight.T`, bias shape `(1, out_size)`
  - `Linear1` (lines 102-116): `weight @ x`, bias shape `(out_size, 1)`
  - `Linear2` (lines 438-453): `weight @ x`, bias shape `(out_size, 1)`, squeeze bias
  - `Linear3` (lines 455-470): `x @ weight.T`, bias shape `(1, out_size)`
- **Solution:** Create single class with parameters:
  ```python
  class Linear(eqx.Module):
      def __init__(self, in_size, out_size, key, transpose_weight=True, bias_shape='row'):
          # transpose_weight: True for x @ W.T, False for W @ x
          # bias_shape: 'row' for (1, out), 'col' for (out, 1)
  ```
- **Files to Update After:** `utils/model.py` (MLP, CNN, CNN3D, myNN classes)

---

### 1.2 Extract AWB Matrix Initialization Utility
- **File:** `utils/model.py`
- **Status:** [ ] Not Started
- **Description:** Create shared utility function for AWB (A/B transformation matrix) initialization
- **Current Duplication:**
  - CNN (lines 291-298)
  - CNN3D (lines 389-398)
  - myNN (lines 590-596)
- **Solution:**
  ```python
  def initialize_awb_matrices(layer_sizes, target_arch, key_seed=5):
      """Initialize A and B matrices for AWB transformations."""
      initializer = jax.nn.initializers.glorot_uniform()
      A = [initializer(jax.random.PRNGKey(key_seed), (y, x))
           for x, y in zip(layer_sizes[:-1], target_arch[:-1])]
      B = [initializer(jax.random.PRNGKey(key_seed), (y, x))
           for x, y in zip(layer_sizes[1:], target_arch[1:])]
      return A, B
  ```

---

### 1.3 Create Data Augmentation Pipeline
- **File:** `utils/data.py`
- **Status:** [ ] Not Started
- **Description:** Extract repeated augmentation code into reusable functions
- **Current Duplication:** Identical code in `mnist()`, `cifar10()`, `cifar100()` methods
  ```python
  rot_angle = np.random.random()*180
  scaling = np.random.random()+1
  X = torchvision.transforms.functional.affine(X, rot_angle,
      translate=(scaling, scaling), scale=1, shear=rot_angle)
  ```
- **Solution:**
  ```python
  def apply_random_affine_augmentation(X):
      """Apply random rotation, translation, and shear augmentation."""
      rot_angle = np.random.random() * 180
      scaling = np.random.random() + 1
      return torchvision.transforms.functional.affine(
          X, rot_angle, translate=(scaling, scaling), scale=1, shear=rot_angle)
  ```

---

### 1.4 Create Train/Test Split Utility
- **File:** `utils/data.py`
- **Status:** [ ] Not Started
- **Description:** Extract repeated train/test split logic into utility function
- **Current Duplication:** Same code in `mnist()`, `cifar10()`, `cifar100()`, `permuted_mnist()`, `omni()`
  ```python
  index = np.random.randint(0, X.shape[0], int(0.8*X.shape[0]))
  self.X_train = X[index]
  self.y_train = y[index]
  index = np.random.randint(0, X.shape[0], int(0.2*X.shape[0]))
  self.X_test = X[index]
  self.y_test = y[index]
  ```
- **Solution:**
  ```python
  def random_train_test_split(X, y, train_ratio=0.8):
      """Split data into train/test sets with random sampling."""
      n_samples = X.shape[0]
      train_idx = np.random.randint(0, n_samples, int(train_ratio * n_samples))
      test_idx = np.random.randint(0, n_samples, int((1 - train_ratio) * n_samples))
      return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
  ```

---

### 1.5 Create Model Factory Function
- **File:** `run.py`
- **Status:** [ ] Not Started
- **Description:** Extract duplicated model initialization into factory function
- **Current Duplication:** Same model selection logic in lines 305-322 and 351-368
- **Solution:**
  ```python
  def create_model(config, input_shape, output_shape=None, seed=5678):
      """Factory function to create model based on configuration."""
      key = jax.random.PRNGKey(seed)
      key, subkey = jax.random.split(key, 2)

      if config['prob'] == 'regression':
          return MLP(sizes=[input_shape, config['hln'], config['hln'], output_shape])
      elif config['prob'] == 'classification':
          if config['data'] in ['cifar10', 'cifar100']:
              num_classes = config.get('n_class', 10)
              return CNN3D(subkey, filter_size=3, feed_sizes=[2304, 512, 256, num_classes],
                          channel_in=3, channel_out=32, num_classes=num_classes)
          else:
              return CNN(subkey, 3, [1875, 512, 64, 10])
      elif config['problem'] == 'graph':
          return myNN(in_size=input_shape, feed_sizes=[128, 128, 128, 10],
                     gcn_sizes=[5, 128], node_num=input_shape,
                     out_size=config['n_class'])
  ```

---

## Priority 2: Medium Impact, Medium Effort

### 2.1 Create Configuration Constants Module
- **File:** `utils/constants.py` (new file)
- **Status:** [ ] Not Started
- **Description:** Centralize magic numbers and default values
- **Constants to Extract:**
  - `TRAIN_TEST_SPLIT = 0.8`
  - `DEFAULT_SEED = 5678`
  - `DEFAULT_BATCH_SIZE = 64`
  - `DEFAULT_LR = 1e-4`
  - `DEFAULT_EXP_REPLAY_LEN = 20000`
- **Architecture Defaults:**
  ```python
  CNN_ARCH = {
      'filter_size': 3,
      'channel_out': 3,
      'feed_sizes': [1875, 512, 64, 10]
  }
  CNN3D_ARCH = {
      'filter_size': 3,
      'channel_in': 3,
      'channel_out': 32,
      'feed_sizes': [2304, 512, 256]
  }
  ```

---

### 2.2 Create Trainer Configuration Builder
- **File:** `run.py`
- **Status:** [ ] Not Started
- **Description:** Extract repeated trainer config dict construction
- **Current Duplication:** Config dicts built in `train_model_reg()`, `train_model_class()`, `load_checkpoint()`
- **Solution:**
  ```python
  def build_trainer_config(config, overrides=None):
      """Build configuration dict for trainer methods."""
      trainer_config = {
          'batch_size': config.get('batch_size', 64),
          'opt': config.get('opt', 'Nash'),
          'problem': config['problem'],
          'data_id': config['data'],
          'flag': config.get('flag', [0, 0]),
          'len_exp_replay': config.get('len_exp_replay', 20000),
          'network': config.get('network', 'fcnn')
      }
      if overrides:
          trainer_config.update(overrides)
      return trainer_config
  ```

---

### 2.3 Create Optimizer Factory
- **File:** `run.py`
- **Status:** [ ] Not Started
- **Description:** Standardize optimizer creation (currently uses adamw vs adam inconsistently)
- **Solution:**
  ```python
  def create_optimizer(config):
      """Create optimizer based on configuration."""
      optimizer_type = config.get('optimizer', 'adam')
      lr = config.get('lr', 1e-4)

      optimizers = {
          'adam': optax.adam,
          'adamw': optax.adamw,
          'sgd': optax.sgd
      }
      return optimizers[optimizer_type](lr)
  ```

---

### 2.4 Extract Weight/Bias Initialization for Architecture Search
- **File:** `run.py`
- **Status:** [ ] Not Started
- **Description:** Remove duplicated weight/bias list generation in `arch_search_GCN()` and `arch_search_MLP()`
- **Solution:**
  ```python
  def initialize_layer_params(sizes, key_seed=5):
      """Initialize weight and bias lists for given layer sizes."""
      initializer = jax.nn.initializers.glorot_uniform()
      weights = [initializer(jax.random.PRNGKey(key_seed), (y, x))
                for x, y in zip(sizes[:-1], sizes[1:])]
      biases = [initializer(jax.random.PRNGKey(key_seed), (1, y))
               for y in sizes[1:]]
      return weights, biases
  ```

---

## Priority 3: Lower Priority, Technical Debt

### 3.1 Implement Dataset Registry Pattern
- **File:** `utils/data.py`
- **Status:** [ ] Not Started
- **Description:** Replace if-elif chain in `generate_dataset()` with registry pattern
- **Current Issue:** Long if-elif chain (lines 315-328) for dataset method dispatch
- **Solution:**
  ```python
  DATASET_LOADERS = {
      'omni': lambda self, tid: self.omni(tid),
      'mnist': lambda self, tid: self.mnist(tid),
      'cifar10': lambda self, tid: self.cifar10(tid),
      'cifar100': lambda self, tid: self.cifar100(tid),
      'permuted_mnist': lambda self, tid: self.permuted_mnist(tid),
      'sine': lambda self, tid: self.sine(tid),
      'wind': lambda self, tid: self.wind(tid),
  }

  def generate_dataset(self, task_id, batch_size, phase):
      if phase == 'training':
          loader = DATASET_LOADERS.get(self.dataset_id)
          if loader:
              loader(self, task_id)
      # ... rest of method
  ```

---

### 3.2 Create Abstract Base Model Class
- **File:** `utils/model.py`
- **Status:** [ ] Not Started
- **Description:** Create base class for models with AWB support
- **Solution:**
  ```python
  class AWBModelBase(eqx.Module):
      """Base class for models supporting AWB transformations."""

      @abstractmethod
      def __call__(self, x):
          """Standard forward pass."""
          pass

      @abstractmethod
      def get_AWBT(self, x):
          """Forward pass with AWB transformation."""
          pass

      def _apply_awb_linear(self, x, A, W, B, bias=None):
          """Apply AWB transformation: A @ W @ B.T @ x + bias"""
          x = A @ W @ jnp.transpose(B) @ x
          if bias is not None:
              x = x + bias
          return x
  ```

---

### 3.3 Remove Duplicate train__CL__graph Method
- **File:** `utils/trainer.py`
- **Status:** [ ] Not Started
- **Description:** There appear to be two implementations of `train__CL__graph` (lines 415-606 and commented version 608-800+)
- **Action:** Review and remove duplicate/legacy code

---

### 3.4 Standardize Dataset Info Printing
- **File:** `run.py`
- **Status:** [ ] Not Started
- **Description:** Extract repeated dataset info printing into utility
- **Current Duplication:** Same print pattern in `load_graph_data()` for each dataset type
- **Solution:**
  ```python
  def print_dataset_info(name, dataset, train_count=None):
      """Print standardized dataset information."""
      print(f'Dataset: {name}')
      print('=' * 22)
      if train_count:
          print(f'Number of graphs: {train_count}')
      print(f'Number of features: {dataset.num_features}')
      print(f'Number of classes: {dataset.num_classes}')
  ```

---

## Progress Tracking

| Task | Priority | Status | Completed Date |
|------|----------|--------|----------------|
| 1.1 Consolidate Linear Classes | P1 | [ ] | - |
| 1.2 AWB Initialization Utility | P1 | [ ] | - |
| 1.3 Data Augmentation Pipeline | P1 | [ ] | - |
| 1.4 Train/Test Split Utility | P1 | [ ] | - |
| 1.5 Model Factory Function | P1 | [ ] | - |
| 2.1 Constants Module | P2 | [ ] | - |
| 2.2 Trainer Config Builder | P2 | [ ] | - |
| 2.3 Optimizer Factory | P2 | [ ] | - |
| 2.4 Weight/Bias Init Utility | P2 | [ ] | - |
| 3.1 Dataset Registry Pattern | P3 | [ ] | - |
| 3.2 Abstract Base Model | P3 | [ ] | - |
| 3.3 Remove Duplicate Trainer | P3 | [ ] | - |
| 3.4 Dataset Info Printing | P3 | [ ] | - |

---

## Estimated Impact

- **Lines of duplicated code to remove:** ~200+
- **New utility functions:** 8-10
- **New modules:** 1 (`utils/constants.py`)
- **Maintainability improvement:** High
- **Risk:** Low (mostly refactoring, no behavior changes)

---

*Last Updated: 2025-12-11*
