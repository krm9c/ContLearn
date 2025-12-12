"""
Architecture search for MLP models.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from training.checkpoint import load_checkpoint


def arch_search_MLP(original_arch, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp):
    """
    Architecture search for MLP (regression)
    """
    trainer1, optim, __, arch_model = load_checkpoint(config)
    i = task
    og_epochs = 500

    arch_model = eqx.tree_at(lambda x: x.sizes, arch_model, original_arch)
    initializer = jax.nn.initializers.glorot_uniform()
    weight_list = [
        initializer(jax.random.PRNGKey(i), (y, x))
        for x, y, i in zip(arch_model.sizes[:-1], arch_model.sizes[1:], range(1, len(arch_model.sizes)))
    ]
    bias_list = [
        initializer(jax.random.PRNGKey(i), (1, y))
        for y, i in zip(arch_model.sizes[1:], range(1, len(arch_model.sizes)))
    ]

    for j in range(len(arch_model.sizes) - 1):
        arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
        arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])

    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(
        lambda x: (x.A, x.B),
        arch_static,
        replace=(arch_model.A, arch_model.B)
    )
    arch_params = eqx.tree_at(lambda x: (x.A, x.B), arch_params, replace=(None, None))

    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__reg(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)),
        arch_params, arch_static, optim,
        n_iter=og_epochs, save_iter=config['save_iter'], task_id=i,
        config={
            'batch_size': 64,
            'opt': 'Nash',
            'problem': config['problem'],
            'data_id': config['data'],
            'flag': config['flag'],
            'len_exp_replay': 20000,
            'network': config['network']
        },
        dictum=poll_dict
    )

    arch_model = eqx.combine(arch_params, arch_static)
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 26)])

    threshold = 0.6
    x = original_arch[1]
    y = original_arch[2]
    opt_loss = loss_orig
    opt_arch = arch_model.sizes
    k = 0

    while (opt_loss >= loss_orig * threshold) and (k < 2):
        for n in range(5):
            for j in range(5):
                curr_arch = [3, x + 15 * n, y + 15 * j, 10]
                arch_model = eqx.tree_at(lambda x: x.sizes, arch_model, original_arch)

                initializer = jax.nn.initializers.glorot_uniform()
                weight_list = [
                    initializer(jax.random.PRNGKey(l), (y, x))
                    for x, y, l in zip(arch_model.sizes[:-1], arch_model.sizes[1:], range(1, len(arch_model.sizes)))
                ]
                bias_list = [
                    initializer(jax.random.PRNGKey(l), (1, y))
                    for y, l in zip(arch_model.sizes[1:], range(1, len(arch_model.sizes)))
                ]

                for j in range(len(arch_model.sizes) - 1):
                    arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
                    arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])

                arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                arch_static = eqx.tree_at(
                    lambda x: (x.A, x.B),
                    arch_static,
                    replace=(arch_model.A, arch_model.B)
                )
                arch_params = eqx.tree_at(lambda x: (x.A, x.B), arch_params, replace=(None, None))

                poll_dict = {}
                optim = optax.adam(1e-3)

                arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__reg(
                    (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                     (test_loader_curr, test_loader_exp)),
                    arch_params, arch_static, optim,
                    n_iter=og_epochs, save_iter=config['save_iter'], task_id=i,
                    config={
                        'batch_size': 64,
                        'opt': 'Nash',
                        'problem': config['problem'],
                        'data_id': config['data'],
                        'flag': config['flag'],
                        'len_exp_replay': 20000,
                        'network': config['network']
                    },
                    dictum=poll_dict
                )

                poll_dict1 = poll_dict[str(i)]
                poll_loss = np.mean([poll_dict1["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 51)])

                if poll_loss < opt_loss:
                    opt_loss = poll_loss
                    opt_arch = curr_arch

        if opt_arch[1] == original_arch[1] and opt_arch[2] == original_arch[2]:
            x = x + 250
            y = y + 250
        else:
            x = opt_arch[1]
            y = opt_arch[2]
        k += 1

    return opt_arch
