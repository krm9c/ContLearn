"""
Default constants and hyperparameters for continual learning framework.

These can be overridden via JSON config files.
"""

# ========== Random Seeds ==========
DEFAULT_SEED = 5678
DEFAULT_GRAPH_SEED = 10

# ========== Model Architecture Defaults ==========
DEFAULT_CHANNEL_OUT_CNN = 3
DEFAULT_CHANNEL_OUT_CNN3D = 32
DEFAULT_CHANNEL_IN_MNIST = 1
DEFAULT_CHANNEL_IN_CIFAR = 3
DEFAULT_INPUT_SIZE_MNIST = 28
DEFAULT_INPUT_SIZE_CIFAR = 32

# ========== Training Defaults ==========
DEFAULT_BATCH_SIZE_GRAPH = 20
DEFAULT_BATCH_SIZE_VECTOR = 64
DEFAULT_BATCH_SIZE_CLASS = 128
DEFAULT_REPLAY_BUFFER_GRAPH = 200000
DEFAULT_REPLAY_BUFFER_VECTOR = 20000
DEFAULT_TRAIN_TEST_SPLIT = 0.8

# ========== Data Augmentation Defaults ==========
DEFAULT_ROTATION_RANGE = 180
DEFAULT_SCALING_RANGE = (1, 2)
DEFAULT_PERMUTATION_SEED_MULTIPLIER = 1000
DEFAULT_NUM_OMNI_CLASSES = 10
DEFAULT_NUM_OMNI_SELECT = 3

# ========== Convolution Defaults ==========
DEFAULT_PADDING = 0
DEFAULT_STRIDE = 1
DEFAULT_POOL_SIZE = 2
DEFAULT_POOL_STRIDE = 2

# ========== Default Architectures by Problem Type ==========
DEFAULT_MLP_REGRESSION_ARCH = None  # Inferred from input/output
DEFAULT_CNN_MNIST_ARCH = [1875, 512, 64, 10]
DEFAULT_CNN3D_CIFAR_ARCH = [2304, 512, 256, 10]
DEFAULT_GCN_SIZES = [None, 128]  # First element set to input size
DEFAULT_GCN_MLP_SIZES = [128, 128, 128, 10]

# ========== Sine Dataset Defaults ==========
# Added by Claude: time step for sine wave generation
DEFAULT_SINE_TIME_STEP = 0.1  # Results in 100 time points (np.arange(0, 1, 0.01))

# ========== Synthetic Graph Dataset Defaults ==========
DEFAULT_SYNTHETIC_NUM_GRAPHS = 1000
DEFAULT_SYNTHETIC_NUM_CHANNELS = 5
DEFAULT_SYNTHETIC_AVG_NUM_NODES = 2
DEFAULT_SYNTHETIC_NUM_CLASSES = 10

# ========== AWB (Adaptive Weight Basis) Defaults ==========
# Master switch
DEFAULT_AWB_ENABLED = False

# AWB Architecture Defaults
DEFAULT_AWB_FILTER_INCREMENT = 2
DEFAULT_AWB_CNN_ARCH = [1875, 700, 100, 10]  # For MNIST/Omniglot CNN
DEFAULT_AWB_CNN3D_HIDDEN = [512, 256]  # For CIFAR CNN3D
DEFAULT_AWB_FNN_ARCH = [100, 140, 140]  # For GCN FNN part
DEFAULT_AWB_GCN_ARCH = [100]  # For GCN part

# AWB Training Pipeline (5-Step Algorithm)
DEFAULT_AWB_PRELIMINARY_EPOCHS = 2     # STEP 1: Epochs before checking arch change
DEFAULT_AWB_AB_TRAINING_EPOCHS = 2    # STEP 3b: Epochs to train A/B matrices
DEFAULT_AWB_AB_WARMUP_EPOCHS = 2       # STEP 5: Warmup epochs after V = AWB^T
DEFAULT_AWB_CHANGE_THRESHOLD_HIGH = 0.7  # Ratio threshold to trigger arch change
DEFAULT_AWB_CHANGE_THRESHOLD_MIN_DELTA = 0.1  # Min loss increase to trigger change
DEFAULT_AWB_AB_THRESHOLD_BASE = 0.6      # Base threshold for AB training convergence
DEFAULT_AWB_AB_MAX_ITERATIONS = 8        # Max iterations for AB training loop
DEFAULT_AWB_AVERAGING_WINDOW = 10        # Epochs to average for loss computation

# ========== Architecture Search Defaults ==========
DEFAULT_ARCH_SEARCH_ENABLED = False
DEFAULT_ARCH_SEARCH_START_TASK = 999  # 999 = never
DEFAULT_ARCH_SEARCH_EPOCHS = 100
DEFAULT_ARCH_SEARCH_THRESHOLD = 0.9
DEFAULT_ARCH_SEARCH_MAX_ITER = 10
DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP = 10
DEFAULT_ARCH_SEARCH_STEP_SIZE_GCN = 10
DEFAULT_ARCH_SEARCH_RANGE = 5
DEFAULT_ARCH_SEARCH_FILTER_RANGE = (2, 5)
DEFAULT_ARCH_SEARCH_AVERAGING_WINDOW = 15
DEFAULT_ARCH_SEARCH_MLP_INCREMENT = 15
DEFAULT_ARCH_SEARCH_LARGE_INCREMENT = 250

# CNN Architecture Search Defaults
DEFAULT_CNN_ARCH_SEARCH_EPOCHS = 5          # Epochs per search iteration (CNN)
DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS = 100      # Epochs per search iteration (CNN3D)
DEFAULT_ARCH_SEARCH_LR = 1e-3               # Learning rate for arch search
DEFAULT_ARCH_SEARCH_BATCH_SIZE = 20         # Batch size for arch search
DEFAULT_ARCH_SEARCH_EXP_REPLAY = 20000      # Experience replay buffer for arch search
DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD = 0.6    # Loss ratio threshold for arch change
DEFAULT_ARCH_SEARCH_HIDDEN_RANGE = 3        # Range for hidden layer search (0 to N-1)
DEFAULT_ARCH_SEARCH_FILTER_MIN = 2          # Minimum filter size
DEFAULT_ARCH_SEARCH_FILTER_MAX = 5          # Maximum filter size (exclusive)
DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT = 15   # Initial loss averaging window
DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL = 10   # Poll loss averaging window
DEFAULT_ARCH_SEARCH_ITER_INCREMENT = 3      # Search iteration increment
DEFAULT_NUM_CLASSES = 10                    # Default number of output classes

# Random key offsets for deterministic weight initialization
DEFAULT_RANDOM_KEY_OFFSET_CONV2 = 100       # Offset for second conv layer keys
DEFAULT_RANDOM_KEY_OFFSET_ACONV2 = 200      # Offset for A_conv2 keys
DEFAULT_RANDOM_KEY_OFFSET_BCONV2 = 300      # Offset for B_conv2 keys
