"""
Architecture search for CNN models.

Contains search functions for:
- CNN (single-channel images like MNIST)
- CNN3D (multi-channel images like CIFAR)
- Preparation functions for A/B transformation matrices
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from ..models.cnn import CNN, CNN3D
from ..models.layers import Linear2
from ..config.constants import (
    DEFAULT_CNN_ARCH_SEARCH_EPOCHS,
    DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_LR,
    DEFAULT_ARCH_SEARCH_BATCH_SIZE,
    DEFAULT_ARCH_SEARCH_EXP_REPLAY,
    DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD,
    DEFAULT_ARCH_SEARCH_HIDDEN_RANGE,
    DEFAULT_ARCH_SEARCH_FILTER_MIN,
    DEFAULT_ARCH_SEARCH_FILTER_MAX,
    DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT,
    DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL,
    DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_ITER_INCREMENT,
    DEFAULT_NUM_CLASSES,
    DEFAULT_INPUT_SIZE_CIFAR,
    DEFAULT_RANDOM_KEY_OFFSET_CONV2,
    DEFAULT_RANDOM_KEY_OFFSET_ACONV2,
    DEFAULT_RANDOM_KEY_OFFSET_BCONV2,
)


def arch_search_CNN_fresh(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                          dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp,
                          trainer=None):
    """
    Architecture search for CNN that creates FRESH models for each candidate architecture.

    This avoids the Equinox limitation where eqx.tree_at cannot change array shapes.
    Instead of updating existing model layers, we create new CNN instances with the
    correct architecture for each candidate.

    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size
    """
    from ..core.trainer import Trainer

    if trainer is None:
        trainer = Trainer(loss='class', metric='class', problem='vectors')

    # Get model configuration from config
    channel_out = config.get('channel_out', 3)
    input_size = config.get('input_size', 28)
    channel_in = config.get('channel_in', 1)

    # Added by Claude: Get arch search parameters from config with fallback to defaults
    search_epochs = config.get('arch_search_epochs', DEFAULT_CNN_ARCH_SEARCH_EPOCHS)
    search_lr = config.get('arch_search_lr', DEFAULT_ARCH_SEARCH_LR)
    search_batch_size = config.get('arch_search_batch_size', DEFAULT_ARCH_SEARCH_BATCH_SIZE)
    search_exp_replay = config.get('arch_search_exp_replay', DEFAULT_ARCH_SEARCH_EXP_REPLAY)
    search_loss_threshold = config.get('arch_search_loss_threshold', DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD)
    search_hidden_range = config.get('arch_search_hidden_range', DEFAULT_ARCH_SEARCH_HIDDEN_RANGE)
    search_filter_min = config.get('arch_search_filter_min', DEFAULT_ARCH_SEARCH_FILTER_MIN)
    search_filter_max = config.get('arch_search_filter_max', DEFAULT_ARCH_SEARCH_FILTER_MAX)
    search_loss_window_init = config.get('arch_search_loss_window_init', DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT)
    search_loss_window_poll = config.get('arch_search_loss_window_poll', DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL)
    search_step_mlp = config.get('arch_search_step_size_mlp', DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP)
    search_max_iter = config.get('arch_search_max_iter', DEFAULT_ARCH_SEARCH_MAX_ITER)
    search_iter_increment = config.get('arch_search_iter_increment', DEFAULT_ARCH_SEARCH_ITER_INCREMENT)
    num_classes = config.get('n_class', DEFAULT_NUM_CLASSES)

    i = task
    original_arch = list(feed_sizes)
    x = original_arch[1]  # first hidden layer size
    y = original_arch[2]  # second hidden layer size
    og_epochs = search_epochs

    # Calculate initial architecture input size
    def calc_feed_input_size(fil_size, img_size, ch_out):
        """Calculate feed layer input size for given filter."""
        conv_output = img_size - fil_size + 1
        pool_output = conv_output - 2 + 1  # MaxPool2d(kernel_size=2) with default stride=1
        return ch_out * pool_output * pool_output

    # Train initial model with original architecture
    initial_feed_sizes = list(original_arch)
    initial_feed_sizes[0] = calc_feed_input_size(filter_size, input_size, channel_out)

    # Create fresh model for initial architecture
    initial_model = CNN(
        key=jax.random.PRNGKey(0),
        filter_size=filter_size,
        feed_sizes=initial_feed_sizes,
        input_size=input_size,
        channel_in=channel_in,
        channel_out=channel_out,
    )

    arch_params, arch_static = eqx.partition(initial_model, eqx.is_array)
    arch_static = eqx.tree_at(lambda m: m.A_conv, arch_static, replace=initial_model.A_conv)
    arch_static = eqx.tree_at(lambda m: m.B_conv, arch_static, replace=initial_model.B_conv)
    arch_static = eqx.tree_at(lambda m: m.A_feed, arch_static, replace=initial_model.A_feed)
    arch_static = eqx.tree_at(lambda m: m.B_feed, arch_static, replace=initial_model.B_feed)
    arch_params = eqx.tree_at(
        lambda m: (m.A_conv, m.B_conv, m.A_feed, m.B_feed),
        arch_params,
        replace=(None, None, None, None)
    )

    poll_dict = trainer.initialize_record_dict(config, run_id=0)
    optim = optax.adam(search_lr)
    opt_state = optim.init(arch_params)

    train_config = {
        'batch_size': search_batch_size,
        'problem': config.get('prob', 'classification'),
        'data_id': config.get('data', 'mnist'),
        'len_exp_replay': search_exp_replay,
        'flag': config.get('flag', [1.0, 1.0]),
        'network': config.get('network', 'cnn'),
    }

    # Train initial architecture
    arch_params, arch_static, opt_state, poll_dict = trainer.train__CL(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)), arch_params, arch_static, opt_state, optim,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=i, config=train_config, record_dict=poll_dict,
        problem_type='vectors', loss_type='classification'
    )

    # Get loss from initial training
    iterations = poll_dict.get('iterations', {})
    if iterations:
        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-search_loss_window_init:]]
        loss_orig = np.mean(recent_losses) if recent_losses else trainW_loss
    else:
        loss_orig = trainW_loss

    threshold = search_loss_threshold
    opt_loss = loss_orig
    opt_mlp = list(initial_feed_sizes)
    opt_filter = filter_size
    current_iter = 1
    step_mlp = search_step_mlp

    # Architecture search loop
    while (opt_loss >= loss_orig * threshold) and (current_iter < search_max_iter):
        for p in range(search_filter_min, search_filter_max):
            for n in range(0, search_hidden_range):
                for j in range(0, search_hidden_range):
                    curr_filter = p
                    # Calculate new architecture
                    curr_mlp = [
                        calc_feed_input_size(curr_filter, input_size, channel_out),
                        x + current_iter * (j + 1) * step_mlp,
                        y + current_iter * (n + 1) * step_mlp,
                        num_classes
                    ]

                    # Create FRESH model with new architecture
                    candidate_model = CNN(
                        key=jax.random.PRNGKey(current_iter * 100 + p * 10 + n + j),
                        filter_size=curr_filter,
                        feed_sizes=curr_mlp,
                        input_size=input_size,
                        channel_in=channel_in,
                        channel_out=channel_out,
                    )

                    # Partition for training
                    cand_params, cand_static = eqx.partition(candidate_model, eqx.is_array)
                    cand_static = eqx.tree_at(lambda m: m.A_conv, cand_static, replace=candidate_model.A_conv)
                    cand_static = eqx.tree_at(lambda m: m.B_conv, cand_static, replace=candidate_model.B_conv)
                    cand_static = eqx.tree_at(lambda m: m.A_feed, cand_static, replace=candidate_model.A_feed)
                    cand_static = eqx.tree_at(lambda m: m.B_feed, cand_static, replace=candidate_model.B_feed)
                    cand_params = eqx.tree_at(
                        lambda m: (m.A_conv, m.B_conv, m.A_feed, m.B_feed),
                        cand_params,
                        replace=(None, None, None, None)
                    )

                    record_dict_arch = trainer.initialize_record_dict(config, run_id=0)
                    optim2 = optax.adam(search_lr)
                    opt_state2 = optim2.init(cand_params)

                    # Train candidate architecture
                    cand_params, cand_static, opt_state2, record_dict_arch = trainer.train__CL(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)), cand_params, cand_static, opt_state2, optim2,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict_arch,
                        problem_type='vectors', loss_type='classification'
                    )

                    # Get loss from candidate training
                    iterations = record_dict_arch.get('iterations', {})
                    if iterations:
                        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-search_loss_window_poll:]]
                        poll_loss = np.mean(recent_losses) if recent_losses else opt_loss
                    else:
                        poll_loss = opt_loss

                    print(f"curr_mlp: {curr_mlp}, curr_filter: {curr_filter}, curr_loss: {poll_loss:.4f}, opt_loss: {opt_loss:.4f}")

                    if poll_loss < opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp[:]
                        opt_filter = curr_filter
                    print(f"opt_mlp: {opt_mlp}, opt_filter: {opt_filter}")

        current_iter += search_iter_increment

    return opt_mlp, opt_filter


def arch_search_CNN(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp,
                    current_model=None, trainer=None):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN
    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size
    """
    from ..core.trainer import Trainer
    from ..runners.classification import load_classification_checkpoint

    if trainer is None:
        trainer = Trainer(loss='class', metric='class', problem='vectors')

    if current_model is None:
        _, _, _, current_model = load_classification_checkpoint(config)

    arch_model = current_model
    i = task
    original_arch = list(feed_sizes)
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = DEFAULT_CNN_ARCH_SEARCH_EPOCHS
    conv_output_size = arch_model.calc_output_size(filter_size)
    maxpool_output_size = arch_model.pool_output_size(2, conv_output_size)
    # Set MLP input layer to correct size corresponding to new filter size output for Convnet layer
    original_arch[0] = maxpool_output_size * maxpool_output_size * arch_model.channel_out
    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, original_arch)
    arch_model = eqx.tree_at(lambda x: x.filter_size, arch_model, filter_size)

    initializer = jax.nn.initializers.glorot_uniform()
    feed_wlist = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(feed_sizes[:], feed_sizes[1:])]
    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in feed_sizes[1:]]
    conv_wlist = [
        [jax.random.normal(jax.random.PRNGKey(j), shape=(arch_model.filter_size, arch_model.filter_size))]
        for j in range(0, arch_model.channel_out)
    ]

    for j in range(len(arch_model.feed_sizes) - 1):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, feed_wlist[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, feed_blist[j])

    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace=jnp.array(conv_wlist))

    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(lambda x: x.A_conv, arch_static, replace=arch_model.A_conv)
    arch_static = eqx.tree_at(lambda x: x.B_conv, arch_static, replace=arch_model.B_conv)
    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace=arch_model.A_feed)
    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace=arch_model.B_feed)
    arch_params = eqx.tree_at(
        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
        arch_params,
        replace=(None, None, None, None)
    )

    poll_dict = trainer.initialize_record_dict(config, run_id=0)
    optim = optax.adam(DEFAULT_ARCH_SEARCH_LR)
    opt_state = optim.init(arch_params)

    train_config = {
        'batch_size': DEFAULT_ARCH_SEARCH_BATCH_SIZE,
        'problem': config.get('prob', 'classification'),
        'data_id': config.get('data', 'mnist'),
        'len_exp_replay': DEFAULT_ARCH_SEARCH_EXP_REPLAY,
        'flag': config.get('flag', [1.0, 1.0]),
        'network': config.get('network', 'cnn'),
    }

    arch_params, arch_static, opt_state, poll_dict = trainer.train__CL(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)), arch_params, arch_static, opt_state, optim,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=i, config=train_config, record_dict=poll_dict,
        problem_type='vectors', loss_type='classification'
    )

    arch_model = eqx.combine(arch_params, arch_static)

    # Architecture search loop - get loss from initial training
    iterations = poll_dict.get('iterations', {})
    if iterations:
        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT:]]
        loss_orig = np.mean(recent_losses) if recent_losses else trainW_loss
    else:
        loss_orig = trainW_loss

    threshold = DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD
    opt_loss = loss_orig
    opt_mlp = list(arch_model.feed_sizes)
    opt_filter = arch_model.filter_size
    curr_mlp = opt_mlp[:]
    curr_filter = opt_filter
    search_iter = 1
    step_mlp = DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP

    while (opt_loss >= loss_orig * threshold) and (search_iter < DEFAULT_ARCH_SEARCH_MAX_ITER):
        for p in range(DEFAULT_ARCH_SEARCH_FILTER_MIN, DEFAULT_ARCH_SEARCH_FILTER_MAX):  # for filter
            for n in range(0, DEFAULT_ARCH_SEARCH_HIDDEN_RANGE):
                for j in range(0, DEFAULT_ARCH_SEARCH_HIDDEN_RANGE):
                    curr_filter = p
                    curr_mlp = [3, x + search_iter * (j + 1) * step_mlp, y + search_iter * (n + 1) * step_mlp, DEFAULT_NUM_CLASSES]
                    conv_output_size = arch_model.calc_output_size(curr_filter)
                    maxpool_output_size = arch_model.pool_output_size(2, conv_output_size)
                    # Set MLP input layer to correct size
                    curr_mlp[0] = maxpool_output_size * maxpool_output_size * arch_model.channel_out

                    arch_model = eqx.tree_at(
                        lambda x: (x.feed_sizes, x.filter_size),
                        arch_model,
                        replace=(curr_mlp, curr_filter)
                    )
                    initializer = jax.nn.initializers.glorot_uniform()
                    feed_wlist = [
                        initializer(jax.random.PRNGKey(5), (y_size, x_size))
                        for x_size, y_size in zip(arch_model.feed_sizes[:], arch_model.feed_sizes[1:])
                    ]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y_size, 1)) for y_size in arch_model.feed_sizes[1:]]
                    conv_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(c), shape=(arch_model.filter_size, arch_model.filter_size))]
                        for c in range(0, arch_model.channel_out)
                    ]

                    for r in range(len(arch_model.feed_sizes) - 1):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].weight, arch_model, feed_wlist[r])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].bias, arch_model, feed_blist[r])

                    weights_list = jnp.array(conv_wlist)
                    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace=weights_list)

                    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                    arch_static = eqx.tree_at(lambda x: x.A_conv, arch_static, replace=arch_model.A_conv)
                    arch_static = eqx.tree_at(lambda x: x.B_conv, arch_static, replace=arch_model.B_conv)
                    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace=arch_model.A_feed)
                    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace=arch_model.B_feed)
                    arch_params = eqx.tree_at(
                        lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed),
                        arch_params,
                        replace=(None, None, None, None)
                    )

                    record_dict_arch = trainer.initialize_record_dict(config, run_id=0)
                    optim2 = optax.adam(DEFAULT_ARCH_SEARCH_LR)
                    opt_state2 = optim2.init(arch_params)

                    arch_params, arch_static, opt_state2, record_dict_arch = trainer.train__CL(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)), arch_params, arch_static, opt_state2, optim2,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict_arch,
                        problem_type='vectors', loss_type='classification'
                    )

                    arch_model = eqx.combine(arch_params, arch_static)
                    iterations = record_dict_arch.get('iterations', {})
                    if iterations:
                        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL:]]
                        poll_loss = np.mean(recent_losses) if recent_losses else opt_loss
                    else:
                        poll_loss = opt_loss

                    print(f"previous -- curr_mlp: {curr_mlp}, curr_filter: {curr_filter}, curr_loss: {poll_loss:.4f}, opt_loss: {opt_loss:.4f}")
                    if poll_loss < opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp[:]
                        opt_filter = curr_filter
                    print(f"next -- opt_mlp: {opt_mlp}, opt_filter: {opt_filter}")
                    arch_model = eqx.combine(arch_params, arch_static)
        search_iter += DEFAULT_ARCH_SEARCH_ITER_INCREMENT

    return opt_mlp, opt_filter


def arch_search_CNN3D(filter_size, feed_sizes, task, trainW_loss, og_epochs, config, dataloader_curr,
                      dataloader_exp, test_loader_curr, test_loader_exp, current_model=None, trainer=None):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN3D (3-channel images)
    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size
    """
    from ..core.trainer import Trainer
    from ..runners.classification import load_classification_checkpoint

    if trainer is None:
        trainer = Trainer(loss='class', metric='class', problem='vectors')

    if current_model is None:
        _, _, _, current_model = load_classification_checkpoint(config)

    arch_model = current_model
    i = task
    original_arch = list(feed_sizes)
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = DEFAULT_CNN3D_ARCH_SEARCH_EPOCHS

    # Calculate output size after two conv+pool layers for CNN3D
    input_size = DEFAULT_INPUT_SIZE_CIFAR
    after_conv1 = arch_model.calc_output_size(input_size, filter_size)
    after_conv2 = arch_model.calc_output_size(after_conv1, filter_size)
    # Output channels after second conv is channel_out * 2
    original_arch[0] = after_conv2 * after_conv2 * arch_model.channel_out * 2

    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, original_arch)
    arch_model = eqx.tree_at(lambda x: x.filter_size, arch_model, filter_size)

    initializer = jax.nn.initializers.glorot_uniform()
    feed_wlist = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(feed_sizes[:], feed_sizes[1:])]
    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in feed_sizes[1:]]

    # Conv weights for CNN3D: [channel_out, channel_in, H, W] for first layer
    conv1_wlist = [
        [[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_in + c),
                            shape=(arch_model.filter_size, arch_model.filter_size))
          for c in range(arch_model.channel_in)] for j in range(arch_model.channel_out)]
    ]
    # Second conv layer: [channel_out*2, channel_out, H, W]
    conv2_wlist = [
        [[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_CONV2),
                            shape=(arch_model.filter_size, arch_model.filter_size))
          for c in range(arch_model.channel_out)] for j in range(arch_model.channel_out * 2)]
    ]

    for j in range(len(arch_model.feed_sizes) - 1):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, feed_wlist[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, feed_blist[j])

    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace=jnp.array(conv1_wlist[0]))
    arch_model = eqx.tree_at(lambda x: x.conv_layers[1].weight, arch_model, replace=jnp.array(conv2_wlist[0]))

    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(lambda x: x.A_conv1, arch_static, replace=arch_model.A_conv1)
    arch_static = eqx.tree_at(lambda x: x.B_conv1, arch_static, replace=arch_model.B_conv1)
    arch_static = eqx.tree_at(lambda x: x.A_conv2, arch_static, replace=arch_model.A_conv2)
    arch_static = eqx.tree_at(lambda x: x.B_conv2, arch_static, replace=arch_model.B_conv2)
    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace=arch_model.A_feed)
    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace=arch_model.B_feed)
    arch_params = eqx.tree_at(
        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
        arch_params,
        replace=(None, None, None, None, None, None)
    )

    poll_dict = trainer.initialize_record_dict(config, run_id=0)
    optim = optax.adam(DEFAULT_ARCH_SEARCH_LR)
    opt_state = optim.init(arch_params)

    train_config = {
        'batch_size': DEFAULT_ARCH_SEARCH_BATCH_SIZE,
        'problem': config.get('prob', 'classification'),
        'data_id': config.get('data', 'cifar10'),
        'len_exp_replay': DEFAULT_ARCH_SEARCH_EXP_REPLAY,
        'flag': config.get('flag', [1.0, 1.0]),
        'network': config.get('network', 'cnn'),
    }

    arch_params, arch_static, opt_state, poll_dict = trainer.train__CL(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)), arch_params, arch_static, opt_state, optim,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=i, config=train_config, record_dict=poll_dict,
        problem_type='vectors', loss_type='classification'
    )

    arch_model = eqx.combine(arch_params, arch_static)

    # Architecture search loop - get loss from initial training
    iterations = poll_dict.get('iterations', {})
    if iterations:
        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-DEFAULT_ARCH_SEARCH_LOSS_WINDOW_INIT:]]
        loss_orig = np.mean(recent_losses) if recent_losses else trainW_loss
    else:
        loss_orig = trainW_loss

    threshold = DEFAULT_ARCH_SEARCH_LOSS_THRESHOLD
    opt_loss = loss_orig
    opt_mlp = list(arch_model.feed_sizes)
    opt_filter = arch_model.filter_size
    curr_mlp = opt_mlp[:]
    curr_filter = opt_filter
    search_iter = 1
    step_mlp = DEFAULT_ARCH_SEARCH_STEP_SIZE_MLP

    while (opt_loss >= loss_orig * threshold) and (search_iter < DEFAULT_ARCH_SEARCH_MAX_ITER):
        for p in range(DEFAULT_ARCH_SEARCH_FILTER_MIN, DEFAULT_ARCH_SEARCH_FILTER_MAX):  # filter size search
            for n in range(0, DEFAULT_ARCH_SEARCH_HIDDEN_RANGE):
                for layer_j in range(0, DEFAULT_ARCH_SEARCH_HIDDEN_RANGE):
                    curr_filter = p
                    curr_mlp = [0, x + search_iter * (layer_j + 1) * step_mlp, y + search_iter * (n + 1) * step_mlp, config.get('n_class', DEFAULT_NUM_CLASSES)]

                    # Calculate new input size for feed layers based on filter size
                    after_conv1 = arch_model.calc_output_size(DEFAULT_INPUT_SIZE_CIFAR, curr_filter)
                    after_conv2 = arch_model.calc_output_size(after_conv1, curr_filter)
                    curr_mlp[0] = after_conv2 * after_conv2 * arch_model.channel_out * 2

                    arch_model = eqx.tree_at(
                        lambda x: (x.feed_sizes, x.filter_size),
                        arch_model,
                        replace=(curr_mlp, curr_filter)
                    )

                    initializer = jax.nn.initializers.glorot_uniform()
                    feed_wlist = [
                        initializer(jax.random.PRNGKey(5), (y_size, x_size))
                        for x_size, y_size in zip(arch_model.feed_sizes[:], arch_model.feed_sizes[1:])
                    ]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y_size, 1)) for y_size in arch_model.feed_sizes[1:]]

                    # Reinitialize conv weights with new filter size
                    conv1_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(c_out * arch_model.channel_in + c),
                                           shape=(arch_model.filter_size, arch_model.filter_size))
                         for c in range(arch_model.channel_in)] for c_out in range(arch_model.channel_out)
                    ]
                    conv2_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(c_out * arch_model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_CONV2),
                                           shape=(arch_model.filter_size, arch_model.filter_size))
                         for c in range(arch_model.channel_out)] for c_out in range(arch_model.channel_out * 2)
                    ]

                    for r in range(len(arch_model.feed_sizes) - 1):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].weight, arch_model, feed_wlist[r])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].bias, arch_model, feed_blist[r])

                    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace=jnp.array(conv1_wlist))
                    arch_model = eqx.tree_at(lambda x: x.conv_layers[1].weight, arch_model, replace=jnp.array(conv2_wlist))

                    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                    arch_static = eqx.tree_at(lambda x: x.A_conv1, arch_static, replace=arch_model.A_conv1)
                    arch_static = eqx.tree_at(lambda x: x.B_conv1, arch_static, replace=arch_model.B_conv1)
                    arch_static = eqx.tree_at(lambda x: x.A_conv2, arch_static, replace=arch_model.A_conv2)
                    arch_static = eqx.tree_at(lambda x: x.B_conv2, arch_static, replace=arch_model.B_conv2)
                    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace=arch_model.A_feed)
                    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace=arch_model.B_feed)
                    arch_params = eqx.tree_at(
                        lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                        arch_params,
                        replace=(None, None, None, None, None, None)
                    )

                    record_dict_arch = trainer.initialize_record_dict(config, run_id=0)
                    optim2 = optax.adam(DEFAULT_ARCH_SEARCH_LR)
                    opt_state2 = optim2.init(arch_params)

                    arch_params, arch_static, opt_state2, record_dict_arch = trainer.train__CL(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)), arch_params, arch_static, opt_state2, optim2,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=i, config=train_config, record_dict=record_dict_arch,
                        problem_type='vectors', loss_type='classification'
                    )

                    arch_model = eqx.combine(arch_params, arch_static)
                    iterations = record_dict_arch.get('iterations', {})
                    if iterations:
                        recent_losses = [iterations[iter_key]['losses'].get('V', 0) for iter_key in sorted(iterations.keys())[-DEFAULT_ARCH_SEARCH_LOSS_WINDOW_POLL:]]
                        poll_loss = np.mean(recent_losses) if recent_losses else opt_loss
                    else:
                        poll_loss = opt_loss

                    print(f"curr_mlp: {curr_mlp}, curr_filter: {curr_filter}, curr_loss: {poll_loss:.4f}, opt_loss: {opt_loss:.4f}")

                    if poll_loss < opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp[:]
                        opt_filter = curr_filter
                    print(f"opt_mlp: {opt_mlp}, opt_filter: {opt_filter}")
                    arch_model = eqx.combine(arch_params, arch_static)
        search_iter += DEFAULT_ARCH_SEARCH_ITER_INCREMENT

    return opt_mlp, opt_filter


def prepABs(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN architecture search.

    When filter size changes, the flattened conv output size (feed_sizes[0]) also changes.
    This function handles all three cases:
    1. Both feed hidden layers AND conv filter change
    2. Only feed hidden layers change (filter stays same)
    3. Only conv filter changes (feed hidden layers stay same, but feed_sizes[0] changes)
    """
    opt_MLParch = list(model.feed_sizes)  # Added by Claude: Convert to list for comparison
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    # Added by Claude: Check if hidden layers changed (indices 1:3), not including feed_sizes[0]
    # which changes when filter size changes
    # Convert to lists to ensure proper comparison (in case model.feed_sizes is JAX array)
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(opt_MLParch[1:3]))
    filter_changed = (opt_filter != prev_filter_size)

    if hidden_changed and filter_changed:
        print("New feed AND conv!!!------------------")
        # Both changed: need transformation matrices for all feed layers
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        B_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
        A_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
    elif hidden_changed and not filter_changed:
        print("New FEED ONLY!!!------------------")
        # Only hidden layers changed: need transformation for feed layers, identity for conv
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        B_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        A_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
    else:
        # Added by Claude: Only filter changed - hidden layers stay same but feed_sizes[0] changes
        # because the flattened conv output size depends on filter size
        print("New CONV ONLY!!!------------------")
        B_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
        A_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
        # Added by Claude: For feed layers, hidden layers are unchanged so use identity,
        # BUT B_feed[0] maps from prev_feed_sizes[0] (old flattened size) to opt_MLParch[0] (new flattened size)
        # A_feed maps output dimensions, B_feed maps input dimensions
        # B_feed[i] shape: (new_input_size, old_input_size) for layer i
        # A_feed[i] shape: (new_output_size, old_output_size) for layer i
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]  # Output dims unchanged
        # B_feed[0] needs to transform from prev_feed_sizes[0] to opt_MLParch[0]
        B_feed = []
        B_feed.append(initializer(jax.random.PRNGKey(10), (opt_MLParch[0], prev_feed_sizes[0])))
        # Rest of B_feed are identity (hidden layer input dims unchanged)
        for x in prev_feed_sizes[1:-1]:
            B_feed.append(jnp.eye(x, x))

    return A_feed, B_feed, A_conv, B_conv


def prepABs_CNN3D(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN3D architecture search.
    Handles multi-channel conv layers (A_conv1, B_conv1, A_conv2, B_conv2).

    When filter size changes, the flattened conv output size (feed_sizes[0]) also changes.
    This function handles all three cases:
    1. Both feed hidden layers AND conv filter change
    2. Only feed hidden layers change (filter stays same)
    3. Only conv filter changes (feed hidden layers stay same, but feed_sizes[0] changes)
    """
    opt_MLParch = list(model.feed_sizes)  # Added by Claude: Convert to list for comparison
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    # Added by Claude: Check if hidden layers changed (indices 1:3), not including feed_sizes[0]
    # Convert to lists to ensure proper comparison (in case model.feed_sizes is JAX array)
    hidden_changed = (list(prev_feed_sizes[1:3]) != list(opt_MLParch[1:3]))
    filter_changed = (opt_filter != prev_filter_size)

    if hidden_changed and filter_changed:
        print("CNN3D: New feed AND conv!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Conv1 AWB matrices (channel_in input channels)
        A_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        B_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + DEFAULT_RANDOM_KEY_OFFSET_CONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        # Conv2 AWB matrices (channel_out input channels)
        A_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_ACONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        B_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_BCONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]

    elif hidden_changed and not filter_changed:
        print("CNN3D: New FEED ONLY!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        A_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]

    else:
        # Added by Claude: Only filter changed - hidden layers stay same but feed_sizes[0] changes
        print("CNN3D: New CONV ONLY!!!------------------")
        # Conv AWB matrices
        A_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        B_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + DEFAULT_RANDOM_KEY_OFFSET_CONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        A_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_ACONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        B_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + DEFAULT_RANDOM_KEY_OFFSET_BCONV2), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        # Added by Claude: For feed layers, hidden layers are unchanged so use identity,
        # BUT B_feed[0] maps from prev_feed_sizes[0] (old flattened size) to opt_MLParch[0] (new flattened size)
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]  # Output dims unchanged
        # B_feed[0] needs to transform from prev_feed_sizes[0] to opt_MLParch[0]
        B_feed = []
        B_feed.append(initializer(jax.random.PRNGKey(10), (opt_MLParch[0], prev_feed_sizes[0])))
        # Rest of B_feed are identity (hidden layer input dims unchanged)
        for x in prev_feed_sizes[1:-1]:
            B_feed.append(jnp.eye(x, x))

    return A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2
