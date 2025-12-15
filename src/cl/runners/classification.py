"""
Training runner for classification tasks.

Orchestrates the training loop for classification problems (e.g., MNIST, CIFAR)
with optional AWB (Adaptive Weight Basis) pipeline for architecture morphing.

The AWB 5-step algorithm:
    Task 0: Standard CL training
    Tasks 1+:
        STEP 1: Preliminary training on new task
        STEP 2: Decide if architecture change needed
        If change needed:
            STEP 3a: Architecture search
            STEP 3b: Train A/B with W frozen
            STEP 4: Compute V = A @ W @ B.T
            STEP 5: Train V with A/B frozen
        Else:
            Continue standard training
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Dict, Any

from ..core.trainer import Trainer
from ..core.awb import (
    compute_avg_loss,
    should_change_arch,
    compute_ab_threshold,
    partition_for_AB_training_cnn,
    partition_for_standard_training_cnn,
    partition_for_AB_training_cnn3d,
    partition_for_standard_training_cnn3d,
)
from ..models.cnn import CNN, CNN3D
from ..models.layers import Linear2
from ..datasets.mnist import MNISTDataset, PermutedMNISTDataset
from ..datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
from ..arch_search.cnn_search import arch_search_CNN_fresh, prepABs, prepABs_CNN3D
from ..config.constants import (
    DEFAULT_BATCH_SIZE_CLASS,
    DEFAULT_REPLAY_BUFFER_GRAPH,
    DEFAULT_AWB_ENABLED,
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
    DEFAULT_CHANNEL_OUT_CNN,
    DEFAULT_CHANNEL_OUT_CNN3D,
    DEFAULT_INPUT_SIZE_MNIST,
    DEFAULT_INPUT_SIZE_CIFAR,
    DEFAULT_AWB_CNN_ARCH,
    DEFAULT_CNN3D_CIFAR_ARCH,
)


def create_classification_optimizer(config: Dict[str, Any]) -> optax.GradientTransformationExtraArgs:
    """Create optimizer from configuration for classification tasks.

    Args:
        config: Configuration dictionary with:
            - optimizer: Optimizer type (default: 'adam')
            - lr: Learning rate (default: 1e-3)

    Returns:
        Configured optimizer
    """
    lr = config.get('lr', 1e-3)
    optimizer_name = config.get('optimizer', 'adam').lower()

    if optimizer_name == 'adam':
        return optax.adam(lr)
    elif optimizer_name == 'adamw':
        weight_decay = config.get('weight_decay', 1e-4)
        return optax.adamw(lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optax.sgd(lr, momentum=momentum)
    else:
        return optax.adam(lr)


def load_classification_checkpoint(config: Dict[str, Any]):
    """Load or create model, trainer, optimizer, and dataset for classification.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (trainer, optimizer, dataset, model)
    """
    # Create dataset based on data type
    data_type = config.get('data', 'mnist')

    dataset_config = {
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_CLASS),
        'len_exp_replay': config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_GRAPH),
        'debug_mode': config.get('debug_mode', False),
        'debug_limit': config.get('debug_limit', 100),
        'n_task': config.get('n_task', 5),
        'problem': config.get('problem', 'classification'),
        'network': config.get('network', 'cnn'),
    }

    # Added by Claude: Handle different dataset types
    if data_type == 'permuted_mnist':
        dataset = PermutedMNISTDataset(dataset_config)
    elif data_type == 'cifar10':
        dataset = CIFAR10Dataset(dataset_config)
    elif data_type == 'cifar100':
        dataset = CIFAR100Dataset(dataset_config)
    else:  # Default to MNIST
        dataset = MNISTDataset(dataset_config)

    # Get model configuration
    filter_size = config.get('filter_size', 4)
    num_classes = config.get('n_class', 10)

    # Added by Claude: Create appropriate model based on dataset type
    if data_type in ['cifar10', 'cifar100']:
        # CIFAR uses CNN3D (3-channel input, two conv layers)
        channel_out = config.get('channel_out', DEFAULT_CHANNEL_OUT_CNN3D)
        channel_in = config.get('channel_in', 3)
        input_size = config.get('input_size', DEFAULT_INPUT_SIZE_CIFAR)

        # Calculate feed layer input size for CNN3D (two conv+pool layers)
        # After conv1: (input_size - filter_size + 1)
        # After pool1 (stride=2): conv1_out // 2
        # After conv2: (pool1_out - filter_size + 1)
        # After pool2 (stride=2): conv2_out // 2
        conv1_out = input_size - filter_size + 1
        pool1_out = conv1_out // 2
        conv2_out = pool1_out - filter_size + 1
        pool2_out = conv2_out // 2
        flatten_size = pool2_out * pool2_out * channel_out * 2  # channel_out * 2 after second conv

        # Default feed sizes for CIFAR
        feed_sizes = config.get('feed_sizes', [flatten_size, 512, 256, num_classes])
        feed_sizes[0] = flatten_size  # Always use calculated flatten size

        awb_arch = config.get('awb_arch', DEFAULT_CNN3D_CIFAR_ARCH.copy())

        model = CNN3D(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_in=channel_in,
            channel_out=channel_out,
            num_classes=num_classes,
        )
    else:
        # MNIST uses CNN (1-channel input, single conv layer)
        channel_out = config.get('channel_out', DEFAULT_CHANNEL_OUT_CNN)
        input_size = config.get('input_size', DEFAULT_INPUT_SIZE_MNIST)

        # Calculate feed layer input size based on conv output
        # After conv: (input_size - filter_size + 1)
        # After MaxPool2d(kernel_size=2, stride=2): conv_out // 2
        conv_output = (input_size - filter_size + 1)
        pool_output = conv_output // 2
        flatten_size = channel_out * pool_output * pool_output

        # Default feed sizes: [flatten_size, 512, 64, 10]
        feed_sizes = config.get('feed_sizes', [flatten_size, 512, 64, num_classes])
        feed_sizes[0] = flatten_size  # Always use calculated flatten size

        awb_arch = config.get('awb_arch', DEFAULT_AWB_CNN_ARCH.copy())

        model = CNN(
            key=jax.random.PRNGKey(0),
            filter_size=filter_size,
            feed_sizes=feed_sizes,
            input_size=input_size,
            channel_in=1,  # MNIST is grayscale
            channel_out=channel_out,
            awb_arch=awb_arch,
        )

    # Create trainer for classification
    trainer = Trainer(
        loss=config.get('loss', 'class'),
        metric=config.get('metric', 'class'),
        problem=config.get('problem', 'vectors'),
    )

    # Create optimizer
    optimizer = create_classification_optimizer(config)

    return trainer, optimizer, dataset, model


def save_cnn_layer_weights(model):
    """Save current CNN layer weights and biases before architecture search.

    Args:
        model: Equinox CNN model

    Returns:
        Tuple of (conv_weights, feed_weights, feed_biases)
    """
    conv_weights = [model.conv_layers[j].weight for j in range(len(model.conv_layers))]
    feed_weights = [model.feed_layers[j].weight for j in range(len(model.feed_layers))]
    feed_biases = [model.feed_layers[j].bias for j in range(len(model.feed_layers))]
    return conv_weights, feed_weights, feed_biases


def restore_cnn_layer_weights(model, conv_weights, feed_weights, feed_biases):
    """Restore saved CNN layer weights and biases to model.

    Args:
        model: Equinox CNN model
        conv_weights: List of conv weight matrices
        feed_weights: List of feed weight matrices
        feed_biases: List of feed bias vectors

    Returns:
        Model with restored weights
    """
    for j in range(len(conv_weights)):
        model = eqx.tree_at(lambda x: x.conv_layers[j].weight, model, conv_weights[j])
    for j in range(len(feed_weights)):
        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, feed_weights[j])
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, feed_biases[j])
    return model


def compute_V_from_AWB_cnn(model):
    """Compute new weights V = A @ W @ B.T for CNN model.

    This is STEP 4 of the AWB algorithm for CNN: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Args:
        model: Equinox CNN model with trained A_conv, B_conv, A_feed, B_feed matrices

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    # Transform conv layer weights
    # Conv weights shape: [channel_out, channel_in, H, W]
    # A_conv and B_conv are lists of [new_filter_size, old_filter_size] matrices
    new_conv_weights = []
    for i in range(model.channel_out):
        # For single input channel (MNIST): weight[i][0] is [H, W]
        transformed = model.A_conv[i] @ model.conv_layers[0].weight[i][0] @ jnp.transpose(model.B_conv[i])
        new_conv_weights.append([transformed])

    model = eqx.tree_at(lambda x: x.conv_layers[0].weight, model, jnp.array(new_conv_weights))

    # Transform feed layer weights
    for j in range(len(model.feed_sizes) - 1):
        # Compute transformed weight: V = A @ W @ B.T
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        # Compute transformed bias: Vb = A @ bias
        Vb = model.A_feed[j] @ model.feed_layers[j].bias

        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)

    return model


# Added by Claude: CNN3D version for two-conv-layer models (CIFAR)
def compute_V_from_AWB_cnn3d(model):
    """Compute new weights V = A @ W @ B.T for CNN3D model.

    This is STEP 4 of the AWB algorithm for CNN3D: after training A and B matrices,
    we compute the effective weights V and update the model to use them.

    Args:
        model: Equinox CNN3D model with trained A_conv1, B_conv1, A_conv2, B_conv2, A_feed, B_feed

    Returns:
        Updated model with weights set to V = A @ W @ B.T
    """
    # Transform conv layer 1 weights (channel_in -> channel_out)
    # A_conv1[i][c] transforms filter for output channel i, input channel c
    new_conv1_weights = []
    for i in range(model.channel_out):
        channel_weights = []
        for c in range(model.channel_in):
            transformed = model.A_conv1[i][c] @ model.conv_layers[0].weight[i][c] @ jnp.transpose(model.B_conv1[i][c])
            channel_weights.append(transformed)
        new_conv1_weights.append(channel_weights)

    model = eqx.tree_at(lambda x: x.conv_layers[0].weight, model, jnp.array(new_conv1_weights))

    # Transform conv layer 2 weights (channel_out -> channel_out * 2)
    new_conv2_weights = []
    for i in range(model.channel_out * 2):
        channel_weights = []
        for c in range(model.channel_out):
            transformed = model.A_conv2[i][c] @ model.conv_layers[1].weight[i][c] @ jnp.transpose(model.B_conv2[i][c])
            channel_weights.append(transformed)
        new_conv2_weights.append(channel_weights)

    model = eqx.tree_at(lambda x: x.conv_layers[1].weight, model, jnp.array(new_conv2_weights))

    # Transform feed layer weights
    for j in range(len(model.feed_sizes) - 1):
        # Compute transformed weight: V = A @ W @ B.T
        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
        # Compute transformed bias: Vb = A @ bias
        Vb = model.A_feed[j] @ model.feed_layers[j].bias

        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)

    return model


def set_new_AB_matrices_cnn(model, original_feed_sizes, new_feed_sizes, original_filter, new_filter):
    """Set new A/B matrices for CNN (single conv layer) architecture transition.

    Args:
        model: CNN model
        original_feed_sizes: Original feed layer sizes
        new_feed_sizes: New feed layer sizes
        original_filter: Original filter size
        new_filter: New filter size

    Returns:
        Model with updated A/B matrices
    """
    A_feed, B_feed, A_conv, B_conv = prepABs(model, original_feed_sizes, original_filter)

    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.A_conv, model, A_conv)
    model = eqx.tree_at(lambda x: x.B_conv, model, B_conv)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed_sizes)
    model = eqx.tree_at(lambda x: x.filter_size, model, new_filter)

    return model


# Added by Claude: CNN3D version for two-conv-layer models (CIFAR)
def set_new_AB_matrices_cnn3d(model, original_feed_sizes, new_feed_sizes, original_filter, new_filter):
    """Set new A/B matrices for CNN3D (two conv layers) architecture transition.

    Args:
        model: CNN3D model
        original_feed_sizes: Original feed layer sizes
        new_feed_sizes: New feed layer sizes
        original_filter: Original filter size
        new_filter: New filter size

    Returns:
        Model with updated A/B matrices
    """
    A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2 = prepABs_CNN3D(model, original_feed_sizes, original_filter)

    model = eqx.tree_at(lambda x: x.A_feed, model, A_feed)
    model = eqx.tree_at(lambda x: x.B_feed, model, B_feed)
    model = eqx.tree_at(lambda x: x.A_conv1, model, A_conv1)
    model = eqx.tree_at(lambda x: x.B_conv1, model, B_conv1)
    model = eqx.tree_at(lambda x: x.A_conv2, model, A_conv2)
    model = eqx.tree_at(lambda x: x.B_conv2, model, B_conv2)
    model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed_sizes)
    model = eqx.tree_at(lambda x: x.filter_size, model, new_filter)

    return model


def train_model_class(config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
    """Train model for classification task using unified training loop.

    When AWB is enabled (config['awb_enabled'] = True), uses the 5-step algorithm.
    When AWB is disabled (default), uses standard CL training for all tasks.

    Args:
        config: Configuration dictionary containing:
            - n_task: Number of tasks
            - epochs_per_task: Training epochs per task
            - batch_size: Batch size
            - lr: Learning rate
            - awb_enabled: Whether to use AWB pipeline
            - save_iter: Save metrics every N epochs
            - model_path: Path to save model
            - flag: Regularization flags [current_weight, experience_weight]
        run_id: Run identifier for logging

    Returns:
        record_dict: Dictionary containing training records
    """
    trainer, optim, data, model = load_classification_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = trainer.initialize_record_dict(config, run_id=run_id)

    # Added by Claude: Determine if using CNN3D (CIFAR) or CNN (MNIST)
    data_type = config.get('data', 'mnist')
    is_cnn3d = data_type in ['cifar10', 'cifar100']

    # Move AWB matrices to static (frozen)
    if is_cnn3d:
        # CNN3D has A_conv1, B_conv1, A_conv2, B_conv2 for two conv layers
        static = eqx.tree_at(
            lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
            static,
            replace=(model.A_conv1, model.B_conv1, model.A_conv2, model.B_conv2, model.A_feed, model.B_feed)
        )
        params = eqx.tree_at(
            lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
            params,
            replace=(None, None, None, None, None, None)
        )
    else:
        # CNN has A_conv, B_conv for single conv layer
        static = eqx.tree_at(
            lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
            static,
            replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed)
        )
        params = eqx.tree_at(
            lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
            params,
            replace=(None, None, None, None)
        )

    # Initialize optimizer state
    opt_state = optim.init(params)

    # Check if AWB pipeline is enabled
    awb_enabled = config.get('awb_enabled', DEFAULT_AWB_ENABLED)

    # AWB configuration parameters
    preliminary_epochs = config.get('awb_preliminary_epochs', DEFAULT_AWB_PRELIMINARY_EPOCHS)
    ab_training_epochs = config.get('awb_ab_training_epochs', DEFAULT_AWB_AB_TRAINING_EPOCHS)
    ab_warmup_epochs = config.get('awb_ab_warmup_epochs', DEFAULT_AWB_AB_WARMUP_EPOCHS)
    ab_max_iterations = config.get('awb_ab_max_iterations', DEFAULT_AWB_AB_MAX_ITERATIONS)
    averaging_window = config.get('awb_averaging_window', DEFAULT_AWB_AVERAGING_WINDOW)

    train_config = {
        'batch_size': config.get('batch_size', DEFAULT_BATCH_SIZE_CLASS),
        'problem': config.get('prob', 'classification'),
        'data_id': config.get('data', 'mnist'),
        'len_exp_replay': config.get('len_exp_replay', DEFAULT_REPLAY_BUFFER_GRAPH),
        'flag': config.get('flag', [1.0, 1.0]),
        'network': config.get('network', 'cnn'),
        'grad_weights': config.get('grad_weights', None),
    }

    # Track baseline losses for AWB decision logic
    end_last0 = None
    end_last = None
    arch_history = []

    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE_CLASS)

    for i in range(config['n_task']):
        print(f"\n{'='*50}")
        print(f"Task {i}")
        print(f"{'='*50}")

        # Generate dataloaders
        dataloader_curr, dataloader_exp = data.generate_dataset(
            task_id=i, batch_size=batch_size, phase='training'
        )
        test_loader_curr, test_loader_exp = data.generate_dataset(
            task_id=i, batch_size=batch_size, phase='testing'
        )

        # Append to experience BEFORE training
        data.append_to_experience(i)

        train_data = (dataloader_curr, dataloader_exp,
                      (test_loader_curr, test_loader_exp), (test_loader_curr, test_loader_exp))

        if i == 0:
            # Task 0: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='classification'
            )
            end_last0 = compute_avg_loss(record_dict.get('iterations', {}), i,
                                         config['epochs_per_task'], averaging_window)
            end_last = end_last0
            print(f"Task 0 baseline loss: {end_last0:.6f}")

        elif awb_enabled:
            # AWB PIPELINE FOR TASKS 1+
            print(f"AWB Pipeline enabled for task {i}")

            # STEP 1: Preliminary training
            print(f"STEP 1: Preliminary training ({preliminary_epochs} epochs)")
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=preliminary_epochs, save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='classification'
            )

            model = eqx.combine(params, static)
            trainWLoss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                          preliminary_epochs, averaging_window)

            # STEP 2: Decide if architecture change is needed
            print(f"STEP 2: Checking architecture change (loss={trainWLoss:.6f})")
            change_arch = should_change_arch(trainWLoss, end_last0, end_last)
            original_feed_sizes = list(model.feed_sizes)
            original_filter = model.filter_size
            conv_weights, feed_weights, feed_biases = save_cnn_layer_weights(model)

            if True:
                print("ARCHITECTURE CHANGE TRIGGERED!")

                # STEP 3a: Architecture search (creates fresh models for each candidate)
                opt_mlp, opt_filter = arch_search_CNN_fresh(
                    filter_size=original_filter,
                    feed_sizes=original_feed_sizes,
                    task=i,
                    trainW_loss=trainWLoss,
                    og_epochs=preliminary_epochs,
                    config=config,
                    dataloader_curr=dataloader_curr,
                    dataloader_exp=dataloader_exp,
                    test_loader_curr=test_loader_curr,
                    test_loader_exp=test_loader_exp,
                    trainer=trainer
                )
                print(f"Optimal Architecture: feed_sizes={opt_mlp}, filter={opt_filter}")
                arch_history.append({'feed_sizes': opt_mlp, 'filter': opt_filter})

                # Restore weights after search
                model = restore_cnn_layer_weights(model, conv_weights, feed_weights, feed_biases)

                if opt_mlp != original_feed_sizes or opt_filter != original_filter:
                    # Set new A/B matrices for the transition
                    # Added by Claude: Use correct function based on model type
                    if is_cnn3d:
                        model = set_new_AB_matrices_cnn3d(model, original_feed_sizes, opt_mlp,
                                                         original_filter, opt_filter)
                    else:
                        model = set_new_AB_matrices_cnn(model, original_feed_sizes, opt_mlp,
                                                        original_filter, opt_filter)

                    # STEP 3b: Train A/B with W frozen
                    print(f"STEP 3b: Training A/B matrices with W frozen")
                    # Added by Claude: Use correct partition function for model type
                    if is_cnn3d:
                        diff_model, static_model = partition_for_AB_training_cnn3d(model)
                    else:
                        diff_model, static_model = partition_for_AB_training_cnn(model)
                    optim2 = optax.adam(1e-4)
                    opt_state2 = optim2.init(diff_model)
                    ab_threshold = compute_ab_threshold(trainWLoss, end_last)

                    diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                        train_data, diff_model, static_model, opt_state2, optim2,
                        n_iter=ab_training_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        notABTrain=False, problem_type='vectors', loss_type='classification'
                    )

                    AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                               ab_training_epochs, averaging_window)
                    ab_iter = 1
                    while (trainWLoss * ab_threshold < AB_loss) and (ab_iter < ab_max_iterations):
                        diff_model, static_model, opt_state2, record_dict = trainer.train__CL(
                            train_data, diff_model, static_model, opt_state2, optim2,
                            n_iter=ab_training_epochs, save_iter=config['save_iter'],
                            task_id=i, config=train_config, record_dict=record_dict,
                            notABTrain=False, problem_type='vectors', loss_type='classification'
                        )
                        AB_loss = compute_avg_loss(record_dict.get('iterations', {}), i,
                                                   ab_training_epochs, averaging_window)
                        ab_iter += 1

                    model = eqx.combine(diff_model, static_model)

                    # STEP 4: Compute V = A @ W @ B.T
                    print("STEP 4: Computing V = A @ W @ B.T")
                    # Added by Claude: Use correct compute_V function for model type
                    if is_cnn3d:
                        model = compute_V_from_AWB_cnn3d(model)
                        params, static = partition_for_standard_training_cnn3d(model)
                    else:
                        model = compute_V_from_AWB_cnn(model)
                        params, static = partition_for_standard_training_cnn(model)

                    # STEP 5: Train V with A/B frozen
                    print(f"STEP 5: Training with new weights V")
                    optim = optax.adam(1e-3)
                    opt_state = optim.init(params)

                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=ab_warmup_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='classification'
                    )
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='classification'
                    )
                else:
                    print("Architecture search found same architecture, continuing normal training")
                    arch_history.append({'feed_sizes': original_feed_sizes, 'filter': original_filter})
                    params, static, opt_state, record_dict = trainer.train__CL(
                        train_data, params, static, opt_state, optim,
                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict,
                        problem_type='vectors', loss_type='classification'
                    )
            else:
                print("Architecture did NOT change - continuing standard training")
                arch_history.append({'feed_sizes': original_feed_sizes, 'filter': original_filter})
                params, static, opt_state, record_dict = trainer.train__CL(
                    train_data, params, static, opt_state, optim,
                    n_iter=ab_warmup_epochs + config['epochs_per_task'],
                    save_iter=config['save_iter'], task_id=i,
                    config=train_config, record_dict=record_dict,
                    problem_type='vectors', loss_type='classification'
                )

            end_last = compute_avg_loss(record_dict.get('iterations', {}), i,
                                        config['epochs_per_task'], averaging_window)

            # Re-partition model for next iteration
            model = eqx.combine(params, static)
            params, static = eqx.partition(model, eqx.is_array)            # Added by Claude: Handle CNN3D vs CNN partitioning
            if is_cnn3d:
                static = eqx.tree_at(
                    lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                    static,
                    replace=(model.A_conv1, model.B_conv1, model.A_conv2, model.B_conv2, model.A_feed, model.B_feed)
                )
                params = eqx.tree_at(
                    lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                    params,
                    replace=(None, None, None, None, None, None)
                )
            else:
                static = eqx.tree_at(
                    lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
                    static,
                    replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed)
                )
                params = eqx.tree_at(
                    lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
                    params,
                    replace=(None, None, None, None)
                )

        else:
            # AWB DISABLED: Standard Training
            params, static, opt_state, record_dict = trainer.train__CL(
                train_data, params, static, opt_state, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config=train_config, record_dict=record_dict,
                problem_type='vectors', loss_type='classification'
            )

    # Print architecture history if AWB was used
    if awb_enabled and arch_history:
        print("\nArchitecture history:")
        for task_idx, arch in enumerate(arch_history):
            print(f"  Task {task_idx + 1}: feed_sizes={arch['feed_sizes']}, filter={arch['filter']}")

    # Save model and records
    model = eqx.combine(params, static)
    model_path = config.get('model_path', 'outputs/mnist_model')

    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    eqx.tree_serialise_leaves(model_path + '.eqx', model)
    trainer.save_record_dict(record_dict, model_path)

    print(f"\nModel saved to: {model_path}.eqx")

    del model, params, static
    return record_dict
