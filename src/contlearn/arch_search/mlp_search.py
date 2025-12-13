"""
Architecture search for MLP models.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from contlearn.training.checkpoint import load_checkpoint
from contlearn.config.constants import (
    DEFAULT_ARCH_SEARCH_EPOCHS,
    DEFAULT_ARCH_SEARCH_THRESHOLD,
    DEFAULT_ARCH_SEARCH_MAX_ITER,
    DEFAULT_ARCH_SEARCH_MLP_INCREMENT,
    DEFAULT_ARCH_SEARCH_LARGE_INCREMENT,
    DEFAULT_BATCH_SIZE_VECTOR,
)


            # #CL training regression problem
            # #print("The params after CL train: ", params)
            # #print("The statics after CL train: ", static)
            # #print("WEIGHTS BEFORE Arch Search: ", model.layers[0].weight)
            # model = eqx.combine(params, static)
            # arch_dict = record_dict_preAB[str(i)]
            # trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
            # #-------------------------------------STEP 2: Get new architecture------------------------------------------#
            # print("STEP 2: Search for Best architecture for the data in task " , i)
            # original_arch = model.sizes
            # opt_arch = arch_search(original_arch,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
            # print("NEW Architecture: ", opt_arch)
            # #print("WEIGHTS AFTER SEARCH BUT BEFORE AB TRAIN: ", model.layers[0].weight)

            # if opt_arch != original_arch: #if arch search found a new architecture, then...
            #     #----------STEP 3a: Set New Arch and Set/Prep A and B to proper sizes-----------------#
            #     og_epochs = 350
            #     print("STEP 3a: Set new Architecture and set/prep A and B to proper sizes")
            #     s = original_arch
            #     #opt_arch = [3,385+75*i,385+50*i,10]
            #     model = eqx.tree_at(lambda x: x.sizes, model, opt_arch)
            #     initializer = jax.nn.initializers.glorot_uniform()
            #     A_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[1:],model.sizes[1:])]
            #     B_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[:-1],model.sizes[:-1])]
            #     model = eqx.tree_at(lambda x: x.A, model, A_list)
            #     model = eqx.tree_at(lambda x: x.B, model, B_list)
            #     #print("A BEFORE TRAIN: ", model.A[0])
            #     #print("WEIGHTS AFTER SETTING A,B: ", model.layers[0].weight)
            #     #print("model after set A,B:", model)

            #     #-------------STEP 3b: Freeze W train only on A,B-----------#
            #     og_epochs = 2000
            #     print("STEP 3b: Train A and B fix W for ", og_epochs, " epochs")
            #     model1 = model
            #     filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
            #     filter_spec = eqx.tree_at(lambda x: (x.A,x.B), filter_spec, replace=(True,True),)
            #     #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)
            #     diff_model, static_model = eqx.partition(model, filter_spec)
            #     #print("MAKE AB Params diff_model: ", diff_model)
            #     #print("MAKE Weights Static static_model: ", static_model)
            #     import optax
            #     optim2 = optax.adam(1e-4)
            #     diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
            #                                                                         (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
            #                                                                         n_iter=og_epochs, save_iter=config['save_iter'],\
            #                                                                         task_id=i, config={
            #                                                                             'batch_size': 64,
            #                                                                             'opt': 'Nash',
            #                                                                             'problem': config['problem'],
            #                                                                             'data_id': config['data'],
            #                                                                             "flag": config['flag'],
            #                                                                             'len_exp_replay': 20000,
            #                                                                             'network': config['network'],
            #                                                                             }, dictum=record_dict_AB, notABTrain = False) #CL training regression problem
                
            #     AB_dict = record_dict_AB[str(i)]
            #     trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
            #     a = 1
            #     threshold = 0.6
            #     print("AB Loss after first AB training: ", trainABLoss)
            #     if trainABLoss<=0.10:
            #         threshold = .75
            #     while(trainABLoss> threshold*trainWLoss):
            #         #og_epochs = 1000
            #         diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
            #                                                                         (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
            #                                                                         n_iter=og_epochs, save_iter=config['save_iter'],\
            #                                                                         task_id=i, config={
            #                                                                             'batch_size': 64,
            #                                                                             'opt': 'Nash',
            #                                                                             'problem': config['problem'],
            #                                                                             'data_id': config['data'],
            #                                                                             "flag": config['flag'],
            #                                                                             'len_exp_replay': 20000,
            #                                                                             'network': config['network'],
            #                                                                             }, dictum=record_dict_AB, notABTrain = False) #CL training regression problem
            #         AB_dict = record_dict_AB[str(i)]
            #         prevABLoss = trainABLoss
            #         trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
            #         a +=1
            #         print("AB Loss after AB training iteration ", a-1, ": ", trainABLoss)
            #         if prevABLoss< trainABLoss:
            #             print("AB Loss is increasing, breaking out of AB training loop")
            #             break
            #         if a==8:
            #             print("too many AB training iterations, breaking out of AB training loop")
            #             break
            #     model = eqx.combine(diff_model,static_model)
            #     #print("WEIGHTS AFTER AB TRAIN: ", model.layers[0].weight)
            #     #print("A AFTER TRAIN: ", model.A[0])

            #     #-----------------------STEP 4: Set new V = AWB^T-----------------------------#
            #     print("STEP 4: Set the new weights V = AWB^T")
            #     #print("-------------------------------------")
            #     for j in range(len(model.sizes)-1):
            #         Vw = model.A[j] @ model.layers[j].weight @ jnp.transpose(model.B[j])
            #         #print("shape of bias:", model.layers[i].bias.shape)
            #         #print("shape of A: ", model.A[i].shape)
            #         Vb = model.layers[j].bias@ model.A[j].T
            #         model = eqx.tree_at(lambda x: x.layers[j].weight, model, Vw)
            #         model = eqx.tree_at(lambda x: x.layers[j].bias, model, Vb)
            #     params, static = eqx.partition(model, eqx.is_array)
            #     static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
            #     static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
            #     params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
            #     print("PARAMS AFTER V SET: ", params)
            #     print("STATIC AFTER V SET: ", static)
            #     #print("weights size after setting V: ", jnp.shape(model.layers[0].weight))
            #     #print("WEIGHTS AFTER SETTING V: ", model.layers[0].weight)
            #     #print("A BEFORE TRAIN V: ", model.A[0])

            #     #-----------STEP 5: Train with weights V for full epochs & record------------#
            #     print("STEP 5: Train the model with weights V for full amount of epochs")
            #     import optax
            #     optim3 = optax.adam(1e-3)
            #     record_dict_dummy = {}
                
            #     params, static, optim3, record_dict_dummy[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
            #                                                                         (test_loader_curr, test_loader_exp)),params, static, optim3, \
            #                                                                         n_iter=50, save_iter=config['save_iter'],\
            #                                                                         task_id=i, config={
            #                                                                             'batch_size': 64,
            #                                                                             'opt': 'Nash',
            #                                                                             'problem': config['problem'],
            #                                                                             'data_id': config['data'],
            #                                                                             "flag": config['flag'],
            #                                                                             'len_exp_replay': 20000,
            #                                                                             'network': config['network'],
            #                                                                             }, dictum=record_dict_dummy)
                
            #     params, static, optim3, record_dict[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
            #                                                                       (test_loader_curr, test_loader_exp)),params, static, optim3, \
            #                                                                      n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
            #                                                                      task_id=i, config={
            #                                                                         'batch_size': 64,
            #                                                                         'opt': 'Nash',
            #                                                                         'problem': config['problem'],
            #                                                                         'data_id': config['data'],
            #                                                                         "flag": config['flag'],
            #                                                                         'len_exp_replay': 20000,
            #                                                                         'network': config['network'],
            #                                                                         }, dictum=record_dict) #CL training regression problem
            #     params, static = eqx.partition(model, eqx.is_array)
            #     static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
            #     static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
            #     params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
            #     #print("After re-split params: ", params)
            #     #print("After re-split statics: ", static)
            #     same_arch = False
            #     optim1 = optim3
            # else: #if arch search did not find a new architecture, then...
            #     #---------------STEP 3: Train with original weights and architecture---------#
            #     params, static = eqx.partition(model, eqx.is_array)
            #     static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
            #     static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
            #     params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
            #     params, static, optim1, record_dict[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
            #                                                                       (test_loader_curr, test_loader_exp)),params, static, optim1, \
            #                                                                      n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
            #                                                                      task_id=i, config={
            #                                                                         'batch_size': 64,
            #                                                                         'opt': 'Nash',
            #                                                                         'problem': config['problem'],
            #                                                                         'data_id': config['data'],
            #                                                                         "flag": config['flag'],
            #                                                                         'len_exp_replay': 20000,
            #                                                                         'network': config['network'],
            #                                                                         }, dictum=record_dict) #CL training regression problem
            #     params, static = eqx.partition(model, eqx.is_array)
            #     static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
            #     static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
            #     params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
            #     #print("After re-split params: ", params)
            #     #print("After re-split statics: ", static)
            #     same_arch = True
            





def arch_search_MLP(original_arch, task, trainW_loss, og_epochs, config,
                    dataloader_curr, dataloader_exp, test_loader_curr, test_loader_exp):
    """
    Architecture search for MLP (regression)
    """
    trainer1, optim, __, arch_model = load_checkpoint(config)
    i = task
    og_epochs = config.get('arch_search_epochs', DEFAULT_ARCH_SEARCH_EPOCHS)

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
            'batch_size': config.get('vector_batch_size', DEFAULT_BATCH_SIZE_VECTOR),
            'opt': 'Nash',
            'problem': config['problem'],
            'data_id': config['data'],
            'flag': config['flag'],
            'len_exp_replay': config.get('vector_replay_size', 20000),
            'network': config['network']
        },
        dictum=poll_dict
    )

    arch_model = eqx.combine(arch_params, arch_static)
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train" + str((i + 1) * og_epochs - j)][0] for j in range(1, 26)])

    threshold = config.get('arch_search_threshold', DEFAULT_ARCH_SEARCH_THRESHOLD)
    x = original_arch[1]
    y = original_arch[2]
    opt_loss = loss_orig
    opt_arch = arch_model.sizes
    k = 0
    search_range = config.get('arch_search_range', 5)
    mlp_increment = config.get('arch_search_mlp_increment', DEFAULT_ARCH_SEARCH_MLP_INCREMENT)

    while (opt_loss >= loss_orig * threshold) and (k < config.get('arch_search_max_iter', DEFAULT_ARCH_SEARCH_MAX_ITER)):
        for n in range(search_range):
            for j in range(search_range):
                curr_arch = [3, x + mlp_increment * n, y + mlp_increment * j, 10]
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
            large_increment = config.get('arch_search_large_increment', DEFAULT_ARCH_SEARCH_LARGE_INCREMENT)
            x = x + large_increment
            y = y + large_increment
        else:
            x = opt_arch[1]
            y = opt_arch[2]
        k += 1

    return opt_arch
