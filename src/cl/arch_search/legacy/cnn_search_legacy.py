"""
Legacy architecture search functions for CNN models.

DEPRECATED: These functions use hardcoded constants instead of reading from config files.
They have been replaced by arch_search_CNN_fresh which properly reads configuration.

These functions are kept here for historical reference only and should NOT be used.
The runners (classification.py, generic_runner.py) now use arch_search_CNN_fresh instead.

Moved to legacy on 2025-12-14 to clean up codebase and avoid confusion about
which arch_search functions actually respect config file parameters.
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
)


def arch_search_CNN(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp,
                    current_model=None, trainer=None):
    """
    DEPRECATED: Use arch_search_CNN_fresh instead.

    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN
    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size

    PROBLEM: Uses hardcoded DEFAULT_* constants instead of reading from config.
    """
    from ...core.trainer import Trainer
    from ...runners.classification import load_classification_checkpoint

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
    DEPRECATED: Use arch_search_CNN_fresh instead (works for both CNN and CNN3D).

    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN3D (3-channel images)
    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size

    PROBLEM: Uses hardcoded DEFAULT_* constants instead of reading from config.
    """
    from ...core.trainer import Trainer
    from ...runners.classification import load_classification_checkpoint

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
