"""
Architecture search for GCN models.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from training.checkpoint import load_checkpoint


def arch_search_GCN(original_gcn, original_mlp, task, trainW_loss, og_epochs, config,
                    train_loader, mem_train_loader, test):
    """
    Architecture search for GCN and MLP layers (graph classification)
    Searches a local neighborhood to find optimal architecture dimensions.
    """

    trainer1, optim5, __, arch_model = load_checkpoint(config)
    i = task

    opt_gcn = original_gcn
    opt_mlp = original_mlp
    arch_model = eqx.tree_at(
        lambda x: (x.gcn_sizes, x.feed_sizes),
        arch_model,
        replace=(original_gcn, original_mlp)
    )
    initializer = jax.nn.initializers.glorot_uniform()
    weightsMLP_list = [
        initializer(jax.random.PRNGKey(5), (y, x))
        for x, y in zip(arch_model.feed_sizes[:-1], arch_model.feed_sizes[1:])
    ]
    biasMLP_list = [
        initializer(jax.random.PRNGKey(5), (1, y))
        for y in arch_model.feed_sizes[1:]
    ]
    weightsGCN_list = [
        initializer(jax.random.PRNGKey(5), (x, y))
        for x, y in zip(arch_model.gcn_sizes[:-1], arch_model.gcn_sizes[1:])
    ]
    biasGCN_list = [
        initializer(jax.random.PRNGKey(5), (1, y))
        for y in arch_model.gcn_sizes[1:]
    ]

    for k in range(len(arch_model.gcn_layers)):
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])

    for j in range(len(arch_model.feed_layers)):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])

    record_dict_arch = {}
    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        arch_static,
        replace=(arch_model.A_gcn, arch_model.B_gcn, arch_model.A_feed, arch_model.B_feed)
    )
    arch_params = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        arch_params,
        replace=(None, None, None, None)
    )

    arch_params, arch_static, optim5, record_dict_arch[str(i)] = trainer1.train__CL__graph(
        (mem_train_loader, test, train_loader), arch_params, arch_static, optim5,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=task, config={'batch_size': config['batch']},
        dictum=record_dict_arch
    )

    arch_model = eqx.combine(arch_params, arch_static)
    arch_dict = record_dict_arch[str(i)]
    loss_orig = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 10)])

    loss_opt = loss_orig
    z2 = original_gcn[1]
    x1 = original_mlp[1]
    x2 = original_mlp[2]
    step_gcn = 10
    step_mlp = 10
    n = 1

    while (n < 3) or (loss_opt < 0.8 * loss_orig):
        for j in range(3):
            curr_gcn = [original_gcn[0], z2 + n * (j + 1) * step_gcn]
            arch_model = eqx.tree_at(lambda x: x.gcn_sizes, arch_model, curr_gcn)

            initializer = jax.nn.initializers.glorot_uniform()
            weightsGCN_list = [
                initializer(jax.random.PRNGKey(5), (x, y))
                for x, y in zip(arch_model.gcn_sizes[:-1], arch_model.gcn_sizes[1:])
            ]
            biasGCN_list = [
                initializer(jax.random.PRNGKey(5), (1, y))
                for y in arch_model.gcn_sizes[1:]
            ]

            for k in range(len(arch_model.gcn_layers)):
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])

            for k in range(3):
                for r in range(3):
                    curr_mlp = [curr_gcn[-1], x1 + n * (k + 1) * step_mlp, x2 + n * (r + 1) * step_mlp, 10]
                    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, curr_mlp)

                    weightsMLP_list = [
                        initializer(jax.random.PRNGKey(5), (y, x))
                        for x, y in zip(curr_mlp[:-1], curr_mlp[1:])
                    ]
                    biasMLP_list = [
                        initializer(jax.random.PRNGKey(5), (1, y))
                        for y in curr_mlp[1:]
                    ]

                    for j in range(len(arch_model.feed_layers)):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])

                    record_dict_arch = {}
                    optim6 = optax.adamw(1e-4)
                    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                    arch_static = eqx.tree_at(
                        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
                        arch_static,
                        replace=(arch_model.A_gcn, arch_model.B_gcn, arch_model.A_feed, arch_model.B_feed)
                    )
                    arch_params = eqx.tree_at(
                        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
                        arch_params,
                        replace=(None, None, None, None)
                    )

                    arch_params, arch_static, optim6, record_dict_arch[str(i)] = trainer1.train__CL__graph(
                        (mem_train_loader, test, train_loader), arch_params, arch_static, optim6,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=task, config={'batch_size': config['batch']},
                        dictum=record_dict_arch
                    )

                    arch_model = eqx.combine(arch_params, arch_static)
                    arch_dict = record_dict_arch[str(i)]
                    loss_poll = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 10)])

                    if loss_poll < loss_opt:
                        opt_gcn = curr_gcn
                        opt_mlp = curr_mlp
                        loss_opt = loss_poll
        n += 3

    return opt_gcn, opt_mlp
