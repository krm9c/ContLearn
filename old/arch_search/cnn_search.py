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

from training.checkpoint import load_checkpoint


def arch_search_CNN(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN
    RETURNS:
        opt_arch: (list) contains the best MLP architecture for the current (and prev) tasks
    """
    trainer1, optim, __, arch_model = load_checkpoint(config)
    i = task
    original_arch = feed_sizes
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = 100

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

    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__class(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)), arch_params, arch_static, optim,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=i, config={
            'batch_size': 20,
            'opt': 'Nash',
            'problem': config['prob'],
            'data_id': config['data'],
            'len_exp_replay': 20000,
            "flag": config['flag'],
            'network': config['network'],
        }, dictum=poll_dict
    )

    arch_model = eqx.combine(arch_params, arch_static)

    # Architecture search loop
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 15)])
    threshold = .6
    loss = loss_orig
    step = 1
    x = original_arch[1]
    y = original_arch[2]
    opt_loss = loss_orig
    opt_mlp = arch_model.feed_sizes
    opt_filter = arch_model.filter_size
    curr_mlp = opt_mlp
    curr_filter = opt_filter
    k = 1
    m = 1
    step_mlp = 10

    while (opt_loss >= loss_orig * threshold) and (k < 10):
        for p in range(2, 5):  # for filter
            for n in range(0, 3):
                for j in range(0, 3):
                    curr_filter = p
                    curr_mlp = [3, x + k * (j + 1) * step_mlp, y + k * (n + 1) * step_mlp, 10]
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
                        initializer(jax.random.PRNGKey(5), (y, x))
                        for x, y in zip(arch_model.feed_sizes[:], arch_model.feed_sizes[1:])
                    ]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in arch_model.feed_sizes[1:]]
                    conv_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(j), shape=(arch_model.filter_size, arch_model.filter_size))]
                        for j in range(0, arch_model.channel_out)
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

                    record_dict_arch = {}
                    optim2 = optax.adam(1e-3)

                    arch_params, arch_static, optim2, record_dict_arch[str(i)] = trainer1.train__CL__class(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)), arch_params, arch_static, optim2,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=i, config={
                            'batch_size': 20,
                            'opt': 'Nash',
                            'problem': config['prob'],
                            'data_id': config['data'],
                            'len_exp_replay': 20000,
                            "flag": config['flag'],
                            'network': config['network'],
                        }, dictum=record_dict_arch
                    )

                    arch_model = eqx.combine(arch_params, arch_static)
                    arch_dict = record_dict_arch[str(i)]
                    poll_loss = np.mean([arch_dict["train" + str((i + 1) * og_epochs - r)][0] for r in range(1, 10)])
                    print("curr_mlp for round: ", curr_mlp, "---- opt filter for round: ", curr_filter,
                          "---- curr_loss:", poll_loss, "----- opt loss: ", opt_loss)

                    if poll_loss < opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp
                        opt_filter = curr_filter
                    print("opt mlp for round: ", opt_mlp, "---- opt filter for round: ", opt_filter)
                    arch_model = eqx.combine(arch_params, arch_static)
        k += 3

    return opt_mlp, opt_filter


def arch_search_CNN3D(filter_size, feed_sizes, task, trainW_loss, og_epochs, config, dataloader_curr,
                      dataloader_exp, test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN3D (3-channel images)
    RETURNS:
        opt_mlp: (list) contains the best MLP architecture for the current (and prev) tasks
        opt_filter: (int) optimal filter size
    """
    trainer1, optim, __, arch_model = load_checkpoint(config)
    i = task
    original_arch = feed_sizes.copy()
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = 100

    # Calculate output size after two conv+pool layers for CNN3D
    # CIFAR: 32x32 input -> conv1 -> pool -> conv2 -> pool -> flatten
    input_size = 32  # CIFAR image size
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
        [[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_out + c + 100),
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

    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__class(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)), arch_params, arch_static, optim,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=i, config={
            'batch_size': 20,
            'opt': 'Nash',
            'problem': config['prob'],
            'data_id': config['data'],
            'len_exp_replay': 20000,
            "flag": config['flag'],
            'network': config['network'],
        }, dictum=poll_dict
    )

    arch_model = eqx.combine(arch_params, arch_static)

    # Architecture search loop
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 15)])
    threshold = .6
    loss = loss_orig
    opt_loss = loss_orig
    opt_mlp = arch_model.feed_sizes.copy()
    opt_filter = arch_model.filter_size
    curr_mlp = opt_mlp.copy()
    curr_filter = opt_filter
    k = 1
    step_mlp = 10

    while (opt_loss >= loss_orig * threshold) and (k < 10):
        for p in range(2, 5):  # filter size search
            for n in range(0, 3):
                for j in range(0, 3):
                    curr_filter = p
                    curr_mlp = [0, x + k * (j + 1) * step_mlp, y + k * (n + 1) * step_mlp, config.get('n_class', 10)]

                    # Calculate new input size for feed layers based on filter size
                    after_conv1 = arch_model.calc_output_size(32, curr_filter)
                    after_conv2 = arch_model.calc_output_size(after_conv1, curr_filter)
                    curr_mlp[0] = after_conv2 * after_conv2 * arch_model.channel_out * 2

                    arch_model = eqx.tree_at(
                        lambda x: (x.feed_sizes, x.filter_size),
                        arch_model,
                        replace=(curr_mlp, curr_filter)
                    )

                    initializer = jax.nn.initializers.glorot_uniform()
                    feed_wlist = [
                        initializer(jax.random.PRNGKey(5), (y, x))
                        for x, y in zip(arch_model.feed_sizes[:], arch_model.feed_sizes[1:])
                    ]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in arch_model.feed_sizes[1:]]

                    # Reinitialize conv weights with new filter size
                    conv1_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_in + c),
                                           shape=(arch_model.filter_size, arch_model.filter_size))
                         for c in range(arch_model.channel_in)] for j in range(arch_model.channel_out)
                    ]
                    conv2_wlist = [
                        [jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_out + c + 100),
                                           shape=(arch_model.filter_size, arch_model.filter_size))
                         for c in range(arch_model.channel_out)] for j in range(arch_model.channel_out * 2)
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

                    record_dict_arch = {}
                    optim2 = optax.adam(1e-3)

                    arch_params, arch_static, optim2, record_dict_arch[str(i)] = trainer1.train__CL__class(
                        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                         (test_loader_curr, test_loader_exp)), arch_params, arch_static, optim2,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=i, config={
                            'batch_size': 20,
                            'opt': 'Nash',
                            'problem': config['prob'],
                            'data_id': config['data'],
                            'len_exp_replay': 20000,
                            "flag": config['flag'],
                            'network': config['network'],
                        }, dictum=record_dict_arch
                    )

                    arch_model = eqx.combine(arch_params, arch_static)
                    arch_dict = record_dict_arch[str(i)]
                    poll_loss = np.mean([arch_dict["train" + str((i + 1) * og_epochs - r)][0] for r in range(1, 10)])
                    print("curr_mlp for round: ", curr_mlp, "---- curr filter: ", curr_filter,
                          "---- curr_loss:", poll_loss, "----- opt loss: ", opt_loss)

                    if poll_loss < opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp.copy()
                        opt_filter = curr_filter
                    print("opt mlp for round: ", opt_mlp, "---- opt filter for round: ", opt_filter)
                    arch_model = eqx.combine(arch_params, arch_static)
        k += 3

    return opt_mlp, opt_filter


def prepABs(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN architecture search.
    """
    opt_MLParch = model.feed_sizes
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter != prev_filter_size):
        print("New feed AND conv!!!------------------")
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
    elif (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter == prev_filter_size):
        print("New FEED ONLY!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        B_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
        A_conv = [jnp.eye(opt_filter, opt_filter) for j in range(0, model.channel_out)]
    else:
        print("New CONV ONLY!!!------------------")
        B_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
        A_conv = [
            jax.random.normal(jax.random.PRNGKey(j), shape=(opt_filter, prev_filter_size))
            for j in range(0, model.channel_out)
        ]
        # Set feed A's B's to identity to keep them
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]
        B_feed = [jnp.eye(x, x) for x in prev_feed_sizes[:-1]]

    return A_feed, B_feed, A_conv, B_conv


def prepABs_CNN3D(model, prev_feed_sizes, prev_filter_size):
    """
    Prepare A and B transformation matrices for CNN3D architecture search.
    Handles multi-channel conv layers (A_conv1, B_conv1, A_conv2, B_conv2).
    """
    opt_MLParch = model.feed_sizes
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()

    if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter != prev_filter_size):
        print("CNN3D: New feed AND conv!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Conv1 AWB matrices (channel_in input channels)
        A_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        B_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + 100), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        # Conv2 AWB matrices (channel_out input channels)
        A_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 200), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        B_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 300), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]

    elif (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter == prev_filter_size):
        print("CNN3D: New FEED ONLY!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[1:], opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x, y in zip(prev_feed_sizes[:-1], opt_MLParch[:-1])]
        # Set conv A's B's to identity to keep them
        A_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jnp.eye(opt_filter, opt_filter) for c in range(model.channel_out)] for j in range(model.channel_out * 2)]

    else:
        print("CNN3D: New CONV ONLY!!!------------------")
        # Conv AWB matrices
        A_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        B_conv1 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + 100), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_in)] for j in range(model.channel_out)
        ]
        A_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 200), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        B_conv2 = [
            [jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 300), shape=(opt_filter, prev_filter_size))
             for c in range(model.channel_out)] for j in range(model.channel_out * 2)
        ]
        # Set feed A's B's to identity to keep them
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]
        B_feed = [jnp.eye(x, x) for x in prev_feed_sizes[:-1]]

    return A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2
