"""
Default constants and hyperparameters for continual learning framework.

All tunable parameters should have defaults defined here and can be overridden
via JSON config files. This ensures consistency and reduces redundancy.

Added by Claude: Comprehensive defaults for layer-level AWB refactor.
"""

# ============================================================================
# RANDOM SEEDS
# ============================================================================
DEFAULT_SEED = 5678
DEFAULT_GRAPH_SEED = 10

# ============================================================================
# DATASET-DRIVEN CONFIGURATION MAPPING
# ============================================================================
# Each dataset automatically selects: prob, problem, network, loss, metric
# Users only need to specify 'data' - everything else is auto-selected

DATASET_CONFIG_MAP = {
    "sine": {
        "prob": "regression",
        "problem": "vectors",
        "network": "fcnn",
        "loss": "mse",
        "metric": "mse",
    },
    "mnist": {
        "prob": "classification",
        "problem": "vectors",
        "network": "cnn",
        "loss": "class",
        "metric": "class",
    },
    "permuted_mnist": {
        "prob": "classification",
        "problem": "vectors",
        "network": "cnn",
        "loss": "class",
        "metric": "class",
    },
    "cifar10": {
        "prob": "classification",
        "problem": "vectors",
        "network": "cnn3d",
        "loss": "class",
        "metric": "class",
    },
    "cifar100": {
        "prob": "classification",
        "problem": "vectors",
        "network": "cnn3d",
        "loss": "class",
        "metric": "class",
    },
    "synthetic": {
        "prob": "classification",
        "problem": "graph",
        "network": "gcn",
        "loss": "class",
        "metric": "class",
    },
}

# Fallback defaults if dataset not in map
DEFAULT_PROB = "regression"
DEFAULT_PROBLEM = "vectors"
DEFAULT_DATA = "sine"
DEFAULT_NETWORK = "fcnn"
DEFAULT_LOSS = "mse"
DEFAULT_METRIC = "mse"

# ============================================================================
# TRAINING LOOP DEFAULTS
# ============================================================================
DEFAULT_N_TASK = 5
DEFAULT_EPOCHS_PER_TASK = 100
DEFAULT_BATCH_SIZE = 64  # General default, overridden by problem type
DEFAULT_BATCH_SIZE_REGRESSION = 64
DEFAULT_BATCH_SIZE_CLASSIFICATION = 128
DEFAULT_BATCH_SIZE_GRAPH = 20
DEFAULT_BATCH_SIZE_VECTOR = DEFAULT_BATCH_SIZE_REGRESSION  # Backward compatibility alias
DEFAULT_BATCH_SIZE_CLASS = DEFAULT_BATCH_SIZE_CLASSIFICATION  # Backward compatibility alias
DEFAULT_SAVE_ITER = 10  # Save metrics every N epochs
DEFAULT_MODEL_PATH = "outputs/model"

# ============================================================================
# EXPERIENCE REPLAY DEFAULTS
# ============================================================================
DEFAULT_LEN_EXP_REPLAY = 20000  # General default
DEFAULT_REPLAY_BUFFER_VECTOR = 20000
DEFAULT_REPLAY_BUFFER_GRAPH = 200000
DEFAULT_TRAIN_TEST_SPLIT = 0.8

# ============================================================================
# OPTIMIZER DEFAULTS
# ============================================================================
DEFAULT_OPTIMIZER = "adam"  # Options: "adam", "adamw", "sgd", "rmsprop"
DEFAULT_LR = 1e-4  # General default learning rate
DEFAULT_LR_REGRESSION = 1e-4
DEFAULT_LR_CLASSIFICATION = 1e-3
DEFAULT_LR_GRAPH = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM = 0.9  # For SGD/RMSprop

# ============================================================================
# LEARNING RATE SCHEDULE DEFAULTS
# ============================================================================
DEFAULT_LR_SCHEDULE = "constant"  # Options: "constant", "step", "exponential", "cosine", "linear"
DEFAULT_LR_DECAY_FACTOR = 0.9  # For step/exponential schedules
DEFAULT_LR_DECAY_STEPS = 1  # Steps between LR decay
DEFAULT_LR_MIN = 1e-6  # Minimum learning rate

# ============================================================================
# HAMILTONIAN GRADIENT COMPUTATION DEFAULTS
# ============================================================================
DEFAULT_FLAG = [1.0, 1.0]  # Regularization weights for dV/dx and dV/dtheta
DEFAULT_GRAD_WEIGHTS = [0.01, 0.98, 0.1]  # [alpha, beta, gamma] for [current_task, experience, hamiltonian]

# ============================================================================
# DEBUG MODE DEFAULTS
# ============================================================================
DEFAULT_DEBUG_MODE = False
DEFAULT_DEBUG_LIMIT = 100  # Number of samples when debug_mode=True

# ============================================================================
# MLP (Fully Connected Neural Network) DEFAULTS
# ============================================================================
DEFAULT_N_LAYERS = 4
DEFAULT_HLN = 256  # Hidden layer size

# ============================================================================
# CNN (Convolutional Neural Network) DEFAULTS
# ============================================================================
DEFAULT_FILTER_SIZE = 4
DEFAULT_CHANNEL_OUT_CNN = 3  # For MNIST-like single channel input
DEFAULT_CHANNEL_OUT_CNN3D = 32  # For CIFAR-like multi-channel input
DEFAULT_CHANNEL_IN_MNIST = 1
DEFAULT_CHANNEL_IN_CIFAR = 3
DEFAULT_INPUT_SIZE_MNIST = 28
DEFAULT_INPUT_SIZE_CIFAR = 32
DEFAULT_PADDING = 0
DEFAULT_STRIDE = 1
DEFAULT_POOL_SIZE = 2
DEFAULT_POOL_STRIDE = 2

# CNN Feed-forward layer defaults (architecture-specific)
DEFAULT_CNN_MNIST_FEED = [1875, 512, 64, 10]  # MNIST with channel_out=3, filter_size=4
DEFAULT_CNN_CIFAR_FEED = [2304, 512, 256, 10]  # CIFAR with channel_out=32, filter_size=3
DEFAULT_CNN3D_CIFAR_ARCH = DEFAULT_CNN_CIFAR_FEED  # Backward compatibility alias

# ============================================================================
# GCN (Graph Convolutional Network) DEFAULTS
# ============================================================================
DEFAULT_GCN_SIZES = [None, 128]  # First element set to input feature size
DEFAULT_GCN_FEED_SIZES = [128, 128, 128, 10]  # MLP after GCN layers
DEFAULT_GCN_MLP_SIZES = DEFAULT_GCN_FEED_SIZES  # Backward compatibility alias

# ============================================================================
# CLASSIFICATION DEFAULTS
# ============================================================================
DEFAULT_NUM_CLASSES = 10
DEFAULT_CLASS_PER_TASK = 2  # For incremental class learning

# ============================================================================
# DATASET-SPECIFIC DEFAULTS
# ============================================================================

# Sine Wave Regression
DEFAULT_SINE_DELTA = 0.001  # Perturbation magnitude
DEFAULT_SINE_TIME_STEP = 0.1  # Time step for sine wave generation
DEFAULT_SINE_DATA_PATH = "data/Incremental_Sine1e^4.p"
DEFAULT_SINE_TEST_SIZE = 0.2

# MNIST/CIFAR Data Augmentation
DEFAULT_ROTATION_RANGE = 180  # Degrees
DEFAULT_SCALING_RANGE = (1, 2)  # Min, max scaling factors
DEFAULT_PERMUTATION_SEED_MULTIPLIER = 1000  # For permuted MNIST

# Synthetic Graph Dataset
DEFAULT_SYNTHETIC_NUM_GRAPHS = 1000
DEFAULT_SYNTHETIC_NUM_CHANNELS = 5  # Node feature channels
DEFAULT_SYNTHETIC_AVG_NUM_NODES = 2
DEFAULT_SYNTHETIC_NUM_CLASSES = 10

# ============================================================================
# AWB (Adaptive Weight Basis) DEFAULTS
# ============================================================================

# Master switch
DEFAULT_AWB_ENABLED = False

# AWB 5-Step Pipeline
DEFAULT_AWB_PRELIMINARY_EPOCHS = 10  # STEP 1: Preliminary training epochs
DEFAULT_AWB_AB_TRAINING_EPOCHS = 50  # STEP 3b: A/B matrix training epochs
DEFAULT_AWB_AB_WARMUP_EPOCHS = 2  # STEP 5: Warmup epochs after V computation
DEFAULT_AWB_AB_MAX_ITERATIONS = 8  # Max iterations for A/B training loop
DEFAULT_AWB_AVERAGING_WINDOW = 10  # Epochs to average for loss computation

# AWB Decision Thresholds
DEFAULT_AWB_CHANGE_THRESHOLD_HIGH = 0.1  # Loss ratio threshold to trigger arch change
DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA = 0.1  # Minimum loss increase to trigger change
DEFAULT_AWB_AB_THRESHOLD_BASE = 0.1  # Base threshold for AB training convergence

# AWB Architecture Defaults (target architectures)
DEFAULT_AWB_FILTER_INCREMENT = 2  # Increment for conv filter expansion
DEFAULT_AWB_CNN_ARCH = [1875, 700, 100, 10]  # For MNIST/Omniglot CNN
DEFAULT_AWB_CNN3D_HIDDEN = [512, 256]  # For CIFAR CNN3D hidden layers
DEFAULT_AWB_GCN_ARCH = [100]  # For GCN part
DEFAULT_AWB_FNN_ARCH = [100, 140, 140]  # For GCN FNN part

# ============================================================================
# ARCHITECTURE SEARCH DEFAULTS
# ============================================================================

# General Architecture Search
DEFAULT_ARCH_SEARCH_ENABLED = False
DEFAULT_ARCH_SEARCH_START_TASK = 999  # 999 = never
DEFAULT_ARCH_SEARCH_EPOCHS = 10  # General default
DEFAULT_ARCH_SEARCH_LR = 1e-3
DEFAULT_ARCH_SEARCH_BATCH_SIZE = 20
DEFAULT_ARCH_SEARCH_EXP_REPLAY = 20000
DEFAULT_ARCH_SEARCH_MAX_ITER = 5
DEFAULT_ARCH_SEARCH_THRESHOLD = 0.95

# CNN Architecture Search
DEFAULT_CNN_ARCH_SEARCH_EPOCHS = 2  # Per search iteration for CNN
DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS = 2  # Per search iteration for CNN3D
DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD = 0.6  # Loss ratio threshold for arch change
DEFAULT_ARCH_SEARCH_HIDDEN_RANGE = 3  # Range for hidden layer search (0 to N-1)
DEFAULT_ARCH_SEARCH_FILTER_MIN = 2  # Minimum filter size
DEFAULT_ARCH_SEARCH_FILTER_MAX = 5  # Maximum filter size (exclusive)
DEFAULT_ARCH_SEARCH_FILTER_RANGE = (2, 5)  # Min, max filter sizes

# MLP/GCN Architecture Search
DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP = 2  # Step size for MLP layer search
DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN = 2  # Step size for GCN layer search
DEFAULT_ARCH_SEARCH_RANGE = 5  # Range for layer size search
DEFAULT_ARCH_SEARCH_MLP_INCREMENT = 15  # Increment for MLP layer expansion
DEFAULT_ARCH_SEARCH_LARGE_INCREMENT = 250  # Large increment for layer expansion

# Architecture Search Loss Averaging
DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT = 1  # Initial loss averaging window
DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL = 1  # Poll loss averaging window
DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW = 1  # General averaging window
DEFAULT_ARCH_SEARCH_ITER_INCREMENT = 3  # Search iteration increment

# ============================================================================
# RANDOM KEY OFFSETS (for deterministic initialization)
# ============================================================================
DEFAULT_RANDOM_KEY_OFFSET_CONV2  = 100  # Offset for second conv layer keys
DEFAULT_RANDOM_KEY_OFFSET_ACONV2 = 200  # Offset for A_conv2 keys
DEFAULT_RANDOM_KEY_OFFSET_BCONV2 = 300  # Offset for B_conv2 keys
