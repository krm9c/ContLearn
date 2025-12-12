"""
Merged training script combining run_AWBTallFunctions.py and run.py

This unified script supports:
- Graph Classification
- Regression Problems
- Classification Problems
with architecture search and adaptive learning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
import json
import os
import signal
import optax
import sys
import itertools
import functools
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple
import jax.tree_util as tree
import jax.tree_util as jtu
from jaxopt import OptaxSolver
from jax import lax
import diffrax
import equinox as eqx
from torch.utils.tensorboard import SummaryWriter
import random


# =========================================================
# Local imports - uses unified utilities with AWBT support
from utils.model import *
from utils.utils import *
from utils.trainer import *
from utils.data import *

# =========================================================
# Additional imports for graph data handling
from torch_geometric.loader import DataLoader

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================
class Params:
    """Class that loads hyperparameters from a json file."""
    
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters back to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Update parameters from JSON file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Dict-like access to Params instance"""
        return self.__dict__





# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def continuum_Graph_classification(dataset, memory_train, n_class=6, select=2, batch_size=32):
    """
    Construct experience/memory/replay data for continual learning
    
    Args:
        dataset: Benchmark dataset loaded via load_graph_data()
        memory_train: List to accumulate memory data
        n_class: Number of classes
        select: Number of classes to select per task
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader: DataLoader for current task
        mem_train_loader: DataLoader for memory/replay buffer
        memory_train: Updated memory list
    """
    tasks = np.random.randint(0, n_class, select)
    stack = [(dataset[j].y.numpy() in tasks) for j in range(len(dataset))]
    datas = [dataset[k] for k, val in enumerate(stack) if val == True]
    
    for k in range(len(datas)):
        datas[k].n_nodes = datas[k].num_nodes
    memory_train += datas
    
    
    train_loader = DataLoader(datas, batch_size=batch_size, shuffle=False)
    mem_train_loader = DataLoader(memory_train, batch_size=batch_size, shuffle=False)
    # print("I am picking the classes", tasks, len(memory_train), len(datas), len(train_loader), len(mem_train_loader))
    return train_loader, mem_train_loader, memory_train


def load_graph_data(data_label):
    """
    Load benchmark graph datasets
    
    Args:
        data_label: Dataset name ('MUTAG', 'ENZYMES', 'PROTEINS', 'MNIST', etc.)
    
    Returns:
        Dataset or (train_set, test_set) tuple
    """
    import torch
    
    def transform(data):
        data.n_nodes = data.num_nodes
        return data
    
    if data_label in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        from torch_geometric.datasets import TUDataset
        torch_geometric.seed.seed_everything(10)
        dataset = TUDataset(root='data/TUDataset', name=data_label, transform=transform).shuffle()
        length = len(dataset)
        train__ = dataset[:int(0.80*length)]
        test__ = dataset[int(0.80*length):]
        
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(train__)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return train__, test__
        
    elif data_label == 'MNIST':
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='data/GNNBench', name='MNIST').shuffle()
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label in ['cora', 'PubMed', 'CiteSeer', 'cora_ML']:
        from torch_geometric.datasets import CitationFull
        from torch_geometric.transforms import NormalizeFeatures
        dataset = CitationFull(root='data/CitationFull', name=data_label)
        return dataset
        
    elif data_label == 'Reddit':
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root='data/Reddit')
        return dataset

    elif data_label == 'tox21':
        from torch_geometric.datasets import MoleculeNet
        dataset = MoleculeNet(root='data/tox21', name="tox21")
        return dataset
        
    elif data_label == 'synthetic':
        from torch_geometric.datasets import FakeDataset
        torch_geometric.seed.seed_everything(10)
        dataset = FakeDataset(num_graphs=1000, num_channels=5,
                            avg_num_nodes=2, num_classes=10, transform=transform)
        length = len(dataset)
        train__ = dataset[:int(0.80*length)]
        test__ = dataset[int(0.80*length):]
        return train__, test__


def generate_sine(delta):
    """
    Generate sine data for continual learning task
    
    Args:
        delta: Small perturbation value for gradual task drift
    """
    list_x = []
    list_y = []
    time = np.arange(0, 1, 0.1)
    length_trajectory = time.shape[0]
    data = {}
    total_samples = 40
    
    np.random.seed(1)
    frequency = (np.random.random([total_samples, 1]) * 60) * np.ones([total_samples, 1])
    amplitude = (np.random.random() * 1) * np.ones([total_samples, 1])
    phase = (np.random.random() * 90) * np.ones([total_samples, 1])
    
    for i in range(40):
        y = (amplitude) * np.sin(2 * np.pi * frequency * time + phase)
        frequency = frequency + delta
        amplitude = amplitude + delta
        data['task'+str(i)] = (y, time, phase, amplitude, frequency)
    
    with open('Incremental_Sine1e^4.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_return_dataset(config):
    """
    Load or generate dataset based on configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dataset or data_return object
    """
    if config['data_id'] == 'sine':
        generate_sine(config['delta'])
        return data_return(config)
    elif config['problem'] == 'graphclassification':
        return load_graph_data(config['data_id'])
    else:
        return data_return(config)
    

# ============================================================================
# MODEL CHECKPOINT & INITIALIZATION
# ============================================================================

def load_checkpoint(config):
    """
    Load checkpoint: Initialize dataset, model, optimizer, and trainer
    
    Args:
        config: Configuration dictionary
    
    Returns:
        trainer, optimizer, dataset, model (and test set if graph classification)
    """
    SEED = 5678
    
    if config['prob'] == 'graphclassification':
        dataset, test = load_return_dataset(
        {
            'batch_size': 20,
            'opt': 'Nash',
            'problem': config['prob'],
            'data_id': config['data'],
            'len_exp_replay': 200000,
            'network': config['network'],
            'delta': config['delta']
        })
        
        memory_train = []
        _, _, memory_train = continuum_Graph_classification(
            dataset, memory_train,
            n_class=config['n_class'],
            select=config['class_per_task']
        )
        
        x = memory_train[0].x
        
        from torch_geometric.loader import DataLoader
        test = DataLoader(test, batch_size=config['batch'], shuffle=True)
        
        # Model definition
        if config['prob'] == 'regression':
            model = MLP(sizes=[x.shape[1], config['hln'], config['hln'], y.shape[1]])
        elif config['prob'] == 'classification':
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            # Use CNN3D for CIFAR (3-channel 32x32), CNN for MNIST/Omni (1-channel 28x28)
            if config['data'] in ['cifar10', 'cifar100']:
                num_classes = config.get('n_class', 10)
                # CIFAR: 3x32x32 -> conv1(32) -> pool -> conv2(64) -> pool -> flatten
                # Output size after 2 conv+pool with filter_size=3: 6x6x64 = 2304
                model = CNN3D(subkey, filter_size=3, feed_sizes=[2304, 512, 256, num_classes], channel_in=3, channel_out=32, num_classes=num_classes)
            else:
                model = CNN(subkey, 3, [1875, 512, 64, 10])
        elif config['problem'] == 'graph':
            model = myNN(in_size=x.shape[1], feed_sizes=[128, 128, 128, 10],
                        gcn_sizes=[5, 128], node_num=x.shape[0],
                        out_size=config['n_class'])

        optim = optax.adamw(config['lr'])
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                         problem=config['problem'],
                         logdir=str(config['tensorfile']))
        
        return trainer, optim, dataset, test, model
    
    else:
        dataset = load_return_dataset({
            'batch_size': 20,
            'opt': 'Nash',
            'problem': config['prob'],
            'data_id': config['data'],
            'len_exp_replay': 20000,
            'network': config['network'],
            'delta': config['delta']
        })
        
        dataloader_curr, dataloader_exp = dataset.generate_dataset(
            task_id=0, batch_size=config['batch_size'], phase='training')
        test_loader_curr, test_loader_exp = dataset.generate_dataset(
            task_id=0, batch_size=config['batch_size'], phase='testing')
        
        x, y = next(iter(dataloader_curr))
        y = y.numpy().astype(np.float64)

        # Model definition
        if config['prob'] == 'regression':
            model = MLP(sizes=[x.shape[1], config['hln'], config['hln'], y.shape[1]])
        elif config['prob'] == 'classification':
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            # Use CNN3D for CIFAR (3-channel 32x32), CNN for MNIST/Omni (1-channel 28x28)
            if config['data'] in ['cifar10', 'cifar100']:
                num_classes = config.get('n_class', 10)
                # CIFAR: 3x32x32 -> conv1(32) -> pool -> conv2(64) -> pool -> flatten
                # Output size after 2 conv+pool with filter_size=3: 6x6x64 = 2304
                model = CNN3D(subkey, filter_size=3, feed_sizes=[2304, 512, 256, num_classes],
                             channel_in=3, channel_out=32, num_classes=num_classes)
            else:
                model = CNN(subkey, 3, [1875, 512, 64, 10])
        elif config['problem'] == 'graph':
            model = myNN(in_size=x.shape[1], feed_sizes=[128, 128, 128, 10],
                        gcn_sizes=[5, 128], node_num=x.shape[0],
                        out_size=config['n_class'])

        optim = optax.adam(config['lr'])
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                         problem=config['problem'],
                         logdir=str(config['tensorfile']))
        
        return trainer, optim, dataset, model

# ============================================================================
# ARCHITECTURE SEARCH FUNCTIONS
# ============================================================================
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
    arch_model = eqx.tree_at(lambda x: (x.gcn_sizes, x.feed_sizes), arch_model,
                             replace=(original_gcn, original_mlp))
    initializer = jax.nn.initializers.glorot_uniform()
    weightsMLP_list = [initializer(jax.random.PRNGKey(5), (y, x))
                      for x, y in zip(arch_model.feed_sizes[:-1], arch_model.feed_sizes[1:])]
    biasMLP_list = [initializer(jax.random.PRNGKey(5), (1, y))
                   for y in arch_model.feed_sizes[1:]]
    weightsGCN_list = [initializer(jax.random.PRNGKey(5), (x, y))
                      for x, y in zip(arch_model.gcn_sizes[:-1], arch_model.gcn_sizes[1:])]
    biasGCN_list = [initializer(jax.random.PRNGKey(5), (1, y))
                   for y in arch_model.gcn_sizes[1:]]
    
    for k in range(len(arch_model.gcn_layers)):
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])
    

    for j in range(len(arch_model.feed_layers)):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])
    

    record_dict_arch = {}
    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), arch_static,replace=(arch_model.A_gcn, arch_model.B_gcn,
                                     arch_model.A_feed, arch_model.B_feed))
    arch_params = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed),arch_params, replace=(None, None, None, None))
    
    arch_params, arch_static, optim5, record_dict_arch[str(i)] = trainer1.train__CL__graph(
        (mem_train_loader, test, train_loader), arch_params, arch_static, optim5,
        n_iter=og_epochs, save_iter=config['save_iter'],
        task_id=task, config={'batch_size': config['batch']},
        dictum=record_dict_arch)
    
    arch_model = eqx.combine(arch_params, arch_static)
    arch_dict = record_dict_arch[str(i)]
    loss_orig = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1, 10)])
    
    loss_opt = loss_orig
    z2 = original_gcn[1]
    x1 = original_mlp[1]
    x2 = original_mlp[2]
    step_gcn = 10
    step_mlp = 10
    n = 1
    
    while (n < 3) or (loss_opt < 0.8 * loss_orig):
        for j in range(3):
            curr_gcn = [original_gcn[0], z2 + n * (j+1) * step_gcn]
            arch_model = eqx.tree_at(lambda x: x.gcn_sizes, arch_model, curr_gcn)
            
            initializer = jax.nn.initializers.glorot_uniform()
            weightsGCN_list = [initializer(jax.random.PRNGKey(5), (x, y))
                             for x, y in zip(arch_model.gcn_sizes[:-1], arch_model.gcn_sizes[1:])]
            biasGCN_list = [initializer(jax.random.PRNGKey(5), (1, y))
                          for y in arch_model.gcn_sizes[1:]]
            
            for k in range(len(arch_model.gcn_layers)):
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])
            
            for k in range(3):
                for r in range(3):
                    curr_mlp = [curr_gcn[-1], x1 + n * (k+1) * step_mlp, x2 + n * (r+1) * step_mlp, 10]
                    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, curr_mlp)
                    
                    weightsMLP_list = [initializer(jax.random.PRNGKey(5), (y, x))
                                     for x, y in zip(curr_mlp[:-1], curr_mlp[1:])]
                    biasMLP_list = [initializer(jax.random.PRNGKey(5), (1, y))
                                  for y in curr_mlp[1:]]
                    
                    for j in range(len(arch_model.feed_layers)):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])
                    
                    record_dict_arch = {}
                    optim6 = optax.adamw(1e-4)
                    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                    arch_static = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), arch_static,
                                             replace=(arch_model.A_gcn, arch_model.B_gcn,
                                                     arch_model.A_feed, arch_model.B_feed))
                    arch_params = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), arch_params,
                                             replace=(None, None, None, None))
                    
                    arch_params, arch_static, optim6, record_dict_arch[str(i)] = trainer1.train__CL__graph(
                        (mem_train_loader, test, train_loader), arch_params, arch_static, optim6,
                        n_iter=og_epochs, save_iter=config['save_iter'],
                        task_id=task, config={'batch_size': config['batch']},
                        dictum=record_dict_arch)
                    
                    arch_model = eqx.combine(arch_params, arch_static)
                    arch_dict = record_dict_arch[str(i)]
                    loss_poll = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1, 10)])
                    
                    if loss_poll < loss_opt:
                        opt_gcn = curr_gcn
                        opt_mlp = curr_mlp
                        loss_opt = loss_poll
        n += 3
    
    return opt_gcn, opt_mlp


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
    weight_list = [initializer(jax.random.PRNGKey(i), (y, x))
                  for x, y, i in zip(arch_model.sizes[:-1], arch_model.sizes[1:], range(1, len(arch_model.sizes)))]
    bias_list = [initializer(jax.random.PRNGKey(i), (1, y))
                for y, i in zip(arch_model.sizes[1:], range(1, len(arch_model.sizes)))]
    
    for j in range(len(arch_model.sizes) - 1):
        arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
        arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])
    
    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    arch_static = eqx.tree_at(lambda x: (x.A, x.B), arch_static,
                             replace=(arch_model.A, arch_model.B))
    arch_params = eqx.tree_at(lambda x: (x.A, x.B), arch_params, replace=(None, None))
    
    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__reg(
        (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
         (test_loader_curr, test_loader_exp)),
        arch_params, arch_static, optim,
        n_iter=og_epochs, save_iter=config['save_iter'], task_id=i,
        config={'batch_size': 64, 'opt': 'Nash', 'problem': config['problem'],
               'data_id': config['data'], 'flag': config['flag'],
               'len_exp_replay': 20000, 'network': config['network']},
        dictum=poll_dict)
    
    arch_model = eqx.combine(arch_params, arch_static)
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1, 26)])
    
    threshold = 0.6
    x = original_arch[1]
    y = original_arch[2]
    opt_loss = loss_orig
    opt_arch = arch_model.sizes
    k = 0
    
    while (opt_loss >= loss_orig * threshold) and (k < 2):
        for n in range(5):
            for j in range(5):
                curr_arch = [3, x + 15*n, y + 15*j, 10]
                arch_model = eqx.tree_at(lambda x: x.sizes, arch_model, original_arch)
                
                initializer = jax.nn.initializers.glorot_uniform()
                weight_list = [initializer(jax.random.PRNGKey(l), (y, x))
                             for x, y, l in zip(arch_model.sizes[:-1], arch_model.sizes[1:], range(1, len(arch_model.sizes)))]
                bias_list = [initializer(jax.random.PRNGKey(l), (1, y))
                           for y, l in zip(arch_model.sizes[1:], range(1, len(arch_model.sizes)))]
                
                for j in range(len(arch_model.sizes) - 1):
                    arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
                    arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])
                
                arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                arch_static = eqx.tree_at(lambda x: (x.A, x.B), arch_static,
                                         replace=(arch_model.A, arch_model.B))
                arch_params = eqx.tree_at(lambda x: (x.A, x.B), arch_params, replace=(None, None))
                
                poll_dict = {}
                optim = optax.adam(1e-3)
                
                arch_params, arch_static, optim, poll_dict[str(i)] = trainer1.train__CL__reg(
                    (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
                     (test_loader_curr, test_loader_exp)),
                    arch_params, arch_static, optim,
                    n_iter=og_epochs, save_iter=config['save_iter'], task_id=i,
                    config={'batch_size': 64, 'opt': 'Nash', 'problem': config['problem'],
                           'data_id': config['data'], 'flag': config['flag'],
                           'len_exp_replay': 20000, 'network': config['network']},
                    dictum=poll_dict)
                
                poll_dict1 = poll_dict[str(i)]
                poll_loss = np.mean([poll_dict1["train"+str((i+1)*og_epochs-j)][0] for j in range(1, 51)])
                
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



# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_model_graph(config):
    """Train model for graph classification task"""
    trainer, optim, data, test, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    memory_train = []
    record_dict_preAB = {}
    record_dict_AB = {}
    
    static = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), static, replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed))
    params = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), params, replace=(None, None, None, None))
    
    for i in range(config['n_task']):
        

        train_loader, mem_train_loader, memory_train = continuum_Graph_classification(
            data, memory_train,\
            n_class=config['n_class'],
            select=config['class_per_task'])
        

        if i == 0:
            og_epochs = config['epochs_per_task']
            params, static, optim, record_dict[str(i)] = trainer.train__CL__graph(
                (mem_train_loader, test, train_loader), params, static, optim,
                n_iter=og_epochs, save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                dictum=record_dict)
        else:
            # Architecture search and adaptive training (simplified for merged version)
            params, static, optim, record_dict[str(i)] = trainer.train__CL__graph(
                (mem_train_loader, test, train_loader), params, static, optim,
                n_iter=config['epochs_per_task'], save_iter=config['save_iter'],
                task_id=i, config={'batch_size': config['batch']},
                dictum=record_dict)
        
        model = eqx.combine(params, static)
        params, static = eqx.partition(model, eqx.is_array)
        static = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), static,
                            replace=(model.A_gcn, model.B_gcn, model.A_feed, model.B_feed))
        params = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn, x.A_feed, x.B_feed), params,
                            replace=(None, None, None, None))
    
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
            task_id=i, batch_size=config['batch_size'], phase='training')
        test_loader_curr, test_loader_exp = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing')
        
        params, static, optim, record_dict[str(i)] = trainer.train__CL__reg(
            (dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),
             (test_loader_curr, test_loader_exp)),
            params, static, optim, n_iter=config['epochs_per_task'],
            save_iter=config['save_iter'], task_id=i,
            config={'batch_size': 64, 'opt': 'Nash', 'problem': config['problem'],
                   'data_id': config['data'], 'flag': config['flag'],
                   'len_exp_replay': 20000, 'network': config['network']},
            dictum=record_dict)
        
        data.append_to_experience(i)
    
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'] + '.eqx', model)
    del model, params, static
    
    return record_dict_preAB, record_dict_AB, record_dict




#===============Arch Search Function for CNN Architecture=======================#
def arch_search_CNN(filter_size, feed_sizes, task, trainW_loss, og_epochs, config,dataloader_curr,\
                 dataloader_exp,test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN
    ARGUMENTS:
    RETURNS: 
        opt_arch: (list) contains the best MLP architecture for the current (and prev) tasks
    """
    trainer1, optim, __, arch_model  = load_checkpoint(config)
    i = task
    original_arch = feed_sizes
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = 100
    #print("model before setting new size: ", arch_model)s
    conv_output_size = arch_model.calc_output_size(filter_size)
    maxpool_output_size = arch_model.pool_output_size(2,conv_output_size)
    #set MLP input layer to correct size correspongding to new filter size output for Convnet layer
    original_arch[0] = maxpool_output_size*maxpool_output_size*arch_model.channel_out
    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, original_arch)
    arch_model = eqx.tree_at(lambda x: x.filter_size, arch_model, filter_size)
    initializer = jax.nn.initializers.glorot_uniform()
    feed_wlist = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[:],feed_sizes[1:])]
    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in feed_sizes[1:]]
    conv_wlist = [[jax.random.normal(jax.random.PRNGKey(j),shape = (arch_model.filter_size,arch_model.filter_size))] for j in range(0,arch_model.channel_out)]
    #print("conv weights list: ", jnp.array(conv_wlist).shape)
    #print(('current model: ', arch_model))
    for j in range(len(arch_model.feed_sizes)-1):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, feed_wlist[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, feed_blist[j])
    #print("current filter weights: ", arch_model.conv_layers[0].weight[0][0].shape)
    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace= jnp.array(conv_wlist))
    #print("model after setting: ", arch_model)

    arch_params, arch_static = eqx.partition(arch_model,eqx.is_array)
    arch_static = eqx.tree_at(lambda x: x.A_conv, arch_static, replace= arch_model.A_conv)
    arch_static = eqx.tree_at(lambda x: x.B_conv, arch_static, replace= arch_model.B_conv)
    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace= arch_model.A_feed)
    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace= arch_model.B_feed)
    arch_params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), arch_params, replace= (None,None,None,None))
    #print("model after resetting sizes and weights: ", arch_model)
    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)]= trainer1.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                           (test_loader_curr, test_loader_exp)),arch_params, arch_static, optim, \
                                                                          n_iter=og_epochs, save_iter=config['save_iter'], \
                                                                          task_id=i,config={
                                                                            'batch_size': 20,
                                                                            'opt': 'Nash',
                                                                            'problem': config['prob'],
                                                                            'data_id': config['data'],
                                                                            'len_exp_replay': 20000,
                                                                            "flag": config['flag'],
                                                                            'network': config['network'],
                                                                            }, dictum = poll_dict)

    arch_model = eqx.combine(arch_params, arch_static)
    #more search------------------------
    arch_dict = poll_dict[str(i)]
    loss_orig = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,15)])
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
    m=1
    step_mlp = 10
    while(opt_loss>=loss_orig*threshold) and (k<10):
        for p in range(2,5): #for filter
            for n in range(0,3):
                for j in range(0,3):
                    #curr_arch = [3,x+15*n,y+15*j,10]
                    curr_filter = p
                    #print("curr filter: ", curr_filter)
                    curr_mlp = [3, x+k*(j+1)*step_mlp, y+k*(n+1)*step_mlp,10]
                    #print("curr mlp: ", curr_mlp)
                    conv_output_size = arch_model.calc_output_size(curr_filter)
                    maxpool_output_size = arch_model.pool_output_size(2,conv_output_size)
                    #set MLP input layer to correct size correspongding to new filter size output for Convnet layer
                    curr_mlp[0] = maxpool_output_size*maxpool_output_size*arch_model.channel_out

                    arch_model = eqx.tree_at(lambda x: (x.feed_sizes, x.filter_size), arch_model, replace = (curr_mlp, curr_filter))
                    initializer = jax.nn.initializers.glorot_uniform()
                    feed_wlist = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(arch_model.feed_sizes[:],arch_model.feed_sizes[1:])]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in arch_model.feed_sizes[1:]]
                    conv_wlist = [[jax.random.normal(jax.random.PRNGKey(j),shape = (arch_model.filter_size,arch_model.filter_size))] for j in range(0,arch_model.channel_out)]
                    for r in range(len(arch_model.feed_sizes)-1):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].weight, arch_model, feed_wlist[r])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[r].bias, arch_model, feed_blist[r])
                    #weights_list = [[(model.conv_layers[0].weight[i][0])] for i in range(0,model.channel_out)]
                    weights_list = jnp.array(conv_wlist)
                    arch_model = eqx.tree_at(lambda x: x.conv_layers[0].weight, arch_model, replace = weights_list)

                    arch_params, arch_static = eqx.partition(arch_model,eqx.is_array)
                    arch_static = eqx.tree_at(lambda x: x.A_conv, arch_static, replace= arch_model.A_conv)
                    arch_static = eqx.tree_at(lambda x: x.B_conv, arch_static, replace= arch_model.B_conv)
                    arch_static = eqx.tree_at(lambda x: x.A_feed, arch_static, replace= arch_model.A_feed)
                    arch_static = eqx.tree_at(lambda x: x.B_feed, arch_static, replace= arch_model.B_feed)
                    arch_params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), arch_params, replace= (None,None,None,None))
                    #print("==========================")
                    #print("model after setting: ", arch_model)
                    record_dict_arch = {}
                    optim2 = optax.adam(1e-3)
               
                    arch_params, arch_static, optim2, record_dict_arch[str(i)]= trainer1.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                           (test_loader_curr, test_loader_exp)),arch_params, arch_static, optim2, \
                                                                          n_iter=og_epochs, save_iter=config['save_iter'], \
                                                                          task_id=i,config={
                                                                            'batch_size': 20,
                                                                            'opt': 'Nash',
                                                                            'problem': config['prob'],
                                                                            'data_id': config['data'],
                                                                            'len_exp_replay': 20000,
                                                                            "flag": config['flag'],
                                                                            'network': config['network'],
                                                                            }, dictum = record_dict_arch)
                    arch_model = eqx.combine(arch_params,arch_static) #recombine the model
                    #determine whehter curr_arch is opt_arch for each
                    arch_dict = record_dict_arch[str(i)]
                    poll_loss = np.mean([arch_dict["train"+str((i+1)*og_epochs-r)][0] for r in range(1,10)])
                    print("curr_mlp for round: ", curr_mlp, "---- opt filter for round: ", curr_filter, "---- curr_loss:", poll_loss, "----- opt loss: ", opt_loss)
                    # if loss_poll<opt_loss:
                    #     opt_gcn = curr_filter
                    #     opt_mlp = curr_mlp
                    #     opt_loss = loss_poll
                    # m+=1
                    # #print("ROUND ",m ,": opt_gcn: ", opt_gcn, "---- opt_mlp: ", opt_mlp)
        

                    # poll_dict1 = poll_dict[str(i)]
                    #poll_loss = np.mean([poll_dict1["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                    # print("curr arch: ", curr_mlp, "--------- curr loss: ", poll_loss, "--------- opt loss: ", opt_loss)
                    if poll_loss<opt_loss:
                        opt_loss = poll_loss
                        opt_mlp = curr_mlp
                        opt_filter = curr_filter
                    print("opt mlp for round: ", opt_mlp, "---- opt filter for round: ", opt_filter)
                    arch_model = eqx.combine(arch_params,arch_static) #recombine the model
        k+=3
    return opt_mlp, opt_filter


#===============Arch Search Function for CNN3D Architecture (CIFAR-10/100)=======================#
def arch_search_CNN3D(filter_size, feed_sizes, task, trainW_loss, og_epochs, config, dataloader_curr,
                      dataloader_exp, test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for CNN3D (3-channel images)
    ARGUMENTS:
        filter_size: initial filter size for conv layers
        feed_sizes: initial feed forward layer sizes
        task: current task id
        trainW_loss: training loss
        og_epochs: number of epochs for architecture search
        config: configuration dictionary
        dataloader_curr, dataloader_exp: training dataloaders
        test_loader_curr, test_loader_exp: test dataloaders
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
    conv1_wlist = [[[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_in + c),
                     shape=(arch_model.filter_size, arch_model.filter_size))
                     for c in range(arch_model.channel_in)] for j in range(arch_model.channel_out)]]
    # Second conv layer: [channel_out*2, channel_out, H, W]
    conv2_wlist = [[[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_out + c + 100),
                     shape=(arch_model.filter_size, arch_model.filter_size))
                     for c in range(arch_model.channel_out)] for j in range(arch_model.channel_out * 2)]]

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
    arch_params = eqx.tree_at(lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                              arch_params, replace=(None, None, None, None, None, None))

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
        }, dictum=poll_dict)

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

                    arch_model = eqx.tree_at(lambda x: (x.feed_sizes, x.filter_size), arch_model,
                                             replace=(curr_mlp, curr_filter))

                    initializer = jax.nn.initializers.glorot_uniform()
                    feed_wlist = [initializer(jax.random.PRNGKey(5), (y, x))
                                  for x, y in zip(arch_model.feed_sizes[:], arch_model.feed_sizes[1:])]
                    feed_blist = [initializer(jax.random.PRNGKey(5), (y, 1)) for y in arch_model.feed_sizes[1:]]

                    # Reinitialize conv weights with new filter size
                    conv1_wlist = [[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_in + c),
                                    shape=(arch_model.filter_size, arch_model.filter_size))
                                    for c in range(arch_model.channel_in)] for j in range(arch_model.channel_out)]
                    conv2_wlist = [[jax.random.normal(jax.random.PRNGKey(j * arch_model.channel_out + c + 100),
                                    shape=(arch_model.filter_size, arch_model.filter_size))
                                    for c in range(arch_model.channel_out)] for j in range(arch_model.channel_out * 2)]

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
                    arch_params = eqx.tree_at(lambda x: (x.A_conv1, x.B_conv1, x.A_conv2, x.B_conv2, x.A_feed, x.B_feed),
                                              arch_params, replace=(None, None, None, None, None, None))

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
                        }, dictum=record_dict_arch)

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


#===============Prep ABs Function for CNN3D=======================#
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
        A_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + 100), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_in)] for j in range(model.channel_out)]
        # Conv2 AWB matrices (channel_out input channels)
        A_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 200), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 300), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_out)] for j in range(model.channel_out * 2)]

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
        A_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_in)] for j in range(model.channel_out)]
        B_conv1 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_in + c + 100), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_in)] for j in range(model.channel_out)]
        A_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 200), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        B_conv2 = [[jax.random.normal(jax.random.PRNGKey(j * model.channel_out + c + 300), shape=(opt_filter, prev_filter_size))
                    for c in range(model.channel_out)] for j in range(model.channel_out * 2)]
        # Set feed A's B's to identity to keep them
        A_feed = [jnp.eye(x, x) for x in prev_feed_sizes[1:]]
        B_feed = [jnp.eye(x, x) for x in prev_feed_sizes[:-1]]

    return A_feed, B_feed, A_conv1, B_conv1, A_conv2, B_conv2


def prepABs(model,prev_feed_sizes,prev_filter_size):
    opt_MLParch = model.feed_sizes
    opt_filter = model.filter_size
    initializer = jax.nn.initializers.glorot_uniform()
    if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter !=prev_filter_size):
        print("New feed AND conv!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[1:],opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[:-1],opt_MLParch[:-1])]
        B_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (opt_filter,prev_filter_size)) for j in range(0,model.channel_out)]
        A_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (opt_filter,prev_filter_size)) for j in range(0,model.channel_out)]
    elif(prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter ==prev_filter_size):
        print("New FEED ONLY!!!------------------")
        A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[1:],opt_MLParch[1:])]
        B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[:-1],opt_MLParch[:-1])]
        #set conv A's B's to identity to keep them
        B_conv = [jnp.eye(opt_filter,opt_filter) for j in range(0,model.channel_out)]
        A_conv = [jnp.eye(opt_filter,opt_filter) for j in range(0,model.channel_out)]
    else:
        print("New CONV ONLY!!!------------------")
        B_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (opt_filter,prev_filter_size)) for j in range(0,model.channel_out)]
        A_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (opt_filter,prev_filter_size)) for j in range(0,model.channel_out)]
        #set feed A's B's to identity to keep them
        A_feed = [jnp.eye(x,x) for x in prev_feed_sizes[1:]]
        B_feed = [jnp.eye(x,x) for x in prev_feed_sizes[:-1]]
    return A_feed, B_feed, A_conv, B_conv




def train_model_class(config):
    """Train model for classification task"""
    trainer, optim, data, model = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}
    
    static = eqx.tree_at(lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed), static,
                        replace=(model.A_conv, model.B_conv, model.A_feed, model.B_feed))
    params = eqx.tree_at(lambda x: (x.A_conv, x.B_conv, x.A_feed, x.B_feed), params,
                        replace=(None, None, None, None))
    
    for i in range(config['n_task']):
        print("task--", i)
        
        dataloader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='training')
        test_loader_curr, _ = data.generate_dataset(
            task_id=i, batch_size=config['batch_size'], phase='testing')
        

        data.append_to_experience(i)


        params, static, optim, record_dict[str(i)] = trainer.train__CL__class(
            (dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),
             (test_loader_curr, test_loader_curr)),
            params, static, optim, n_iter=config['epochs_per_task'],
            save_iter=config['save_iter'], task_id=i,
            config={'batch_size': config['batch_size'], 'opt': 'Nash',
                   'problem': config['prob'], 'data_id': config['data'],
                   'len_exp_replay': 200000, 'flag': config['flag'],
                   'network': config['network']},
            dictum=record_dict)
        
        
    
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'], model)
    del model, params, static
    
    return record_dict_preAB, record_dict_AB, record_dict


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_merged.py",
        description="Unified training script for continual learning"
    )
    subparsers = parser.add_subparsers(help='', dest='command')
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("runs", default=1, help="the number of total runs")
    train_parser.add_argument("json", default=None, help="directory with configurations")
    
    basic_path = 'jsons/'
    args = parser.parse_args()
    json_path = os.path.join(basic_path + str(args.json))
    
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    
    params = Params(json_path).dict
    
    if args.runs is not None:
        params['runs'] = int(args.runs)
    else:
        params['runs'] = 5
    
    if args.command == 'train':
        record_dict = {}
        record_dict_preAB = {}
        record_dict_AB = {}
        
        for j in range(params['runs']):
            print(f"runs {j}, problem: {params['problem']}")
            
            if params['prob'] == 'regression':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_reg(params)
            elif params['prob'] == 'classification':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_class(params)
            elif params['problem'] == 'graph':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] = train_model_graph(params)
        
        # Save results
        if 'file' in params:
            with open(str(params['file']) + '.pkl', 'wb') as f:
                pickle.dump(record_dict, f)
            print(f"Saved results to {params['file']}.pkl")
