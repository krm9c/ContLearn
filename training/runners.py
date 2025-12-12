"""
Training runners for different problem types.

Contains the main training loop functions for:
- Graph classification
- Regression
- Classification
"""

import equinox as eqx

from .checkpoint import load_checkpoint
from data.loaders import continuum_Graph_classification


def train_model_graph(config):
    """Train model for graph classification task"""
    trainer, optim, data, test, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    memory_train = []
    record_dict_preAB = {}
    record_dict_AB = {}

    static = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        static,
        replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed)
    )
    params = eqx.tree_at(
        lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
        params,
        replace=(None, None, None, None)
    )

    for i in range(config['n_task']):
        train_loader, mem_train_loader, memory_train = continuum_Graph_classification(
            data, memory_train,
            n_class=config['n_class'],
            select=config['class_per_task']
        )

        if i == 0:
            og_epochs = config['epochs_per_task']
            params, static, optim, record_dict[str(i)] = trainer.train__CL__graph(
                (mem_train_loader, test, train_loader), params, static, optim,
                n_iter=og_epochs, save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                dictum=record_dict
            )
        else:
            # Architecture search and adaptive training (simplified for merged version)
            params, static, optim, record_dict[str(i)] = trainer.train__CL__graph(
                (mem_train_loader, test, train_loader), params, static, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                dictum=record_dict
            )

        model = eqx.combine(params, static)
        params, static = eqx.partition(model, eqx.is_array)
        static = eqx.tree_at(
            lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
            static,
            replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed)
        )
        params = eqx.tree_at(
            lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),
            params,
            replace=(None, None, None, None)
        )

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)
    del model, params, static

    return record_dict_preAB, record_dict_AB, record_dict


def train_model_reg(config):
    """Train model for regression task"""
    trainer, optim, data, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}

    static = eqx.tree_at(lambda x: (x.A, x.B), static, replace=(model.A, model.B))
    params = eqx.tree_at(lambda x: (x.A, x.B), params, replace=(None, None))

    for i in range(config['n_task']):
        dataloader_curr, dataloader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training'
        )
        test_loader_curr, test_loader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing'
        )

        params, static, optim, record_dict[str(i)] = trainer.train__CL__reg(
            (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
             (test_loader_curr, test_loader_exp)),
            params, static, optim, n_iter=config['epochs_per_task'],
            save_iter=config['save_iter'], task_id=i,
            config={
                'batch_size': 64,
                'opt': 'Nash',
                'problem': config['problem'],
                'data_id': config['data'],
                'flag': config['flag'],
                'len_exp_replay': 20000,
                'network': config['network']
            },
            dictum=record_dict
        )

        data.append_to_experience(i)


        if config['arch_search'] and i >= config['arch_start_task']:
            # # Architecture search and adaptive training (simplified for merged version)
            # import numpy as np 
            # model = eqx.combine(params, static)
            # arch_dict = record_dict_preAB[str(i)]
            # trainWLoss = np.mean([arch_dict["train"+str((i+1)*config['epoch_task']-j)][0] for j in range(1,51)])
            # #-------------------------------------STEP 2: Get new architecture------------------------------------------#
            # print("STEP 2: Search for Best architecture for the data in task " , i)
            # original_arch = model.sizes
            # opt_arch = arch_search(original_arch,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
            # print("NEW Architecture: ", opt_arch)
            pass


    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)
    del model, params, static

    return record_dict_preAB, record_dict_AB, record_dict


def train_model_class(config):
    """Train model for classification task"""
    trainer, optim, data, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}

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

    for i in range(config['n_task']):
        print("task--", i)

        dataloader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training'
        )
        test_loader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing'
        )

        data.append_to_experience(i)

        params, static, optim, record_dict[str(i)] = trainer.train__CL__class(
            (dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),
             (test_loader_curr, test_loader_curr)),
            params, static, optim, n_iter=config['epochs_per_task'],
            save_iter=config['save_iter'], task_id=i,
            config={
                'batch_size': config['batch_size'],
                'opt': 'Nash',
                'problem': config['prob'],
                'data_id': config['data'],
                'len_exp_replay': 200000,
                'flag': config['flag'],
                'network': config['network']
            },
            dictum=record_dict
        )

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'], model)
    del model, params, static

    return record_dict_preAB, record_dict_AB, record_dict
