"""
Data loading functions for continual learning experiments.

This module handles loading and preparing datasets for:
- Graph classification
- Regression problems
- Classification problems
"""

import numpy as np
import pickle
import torch_geometric
from torch_geometric.loader import DataLoader

from utils.data import data_return
from config.constants import (
    DEFAULT_GRAPH_SEED,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_SYNTHETIC_NUM_GRAPHS,
    DEFAULT_SYNTHETIC_NUM_CHANNELS,
    DEFAULT_SYNTHETIC_AVG_NUM_NODES,
    DEFAULT_SYNTHETIC_NUM_CLASSES,
)


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
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)
        dataset = TUDataset(root='data/TUDataset', name=data_label, transform=transform).shuffle()
        length = len(dataset)
        train_split = DEFAULT_TRAIN_TEST_SPLIT
        train__ = dataset[:int(train_split * length)]
        test__ = dataset[int(train_split * length):]

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
        torch_geometric.seed.seed_everything(DEFAULT_GRAPH_SEED)
        dataset = FakeDataset(
            num_graphs=DEFAULT_SYNTHETIC_NUM_GRAPHS,
            num_channels=DEFAULT_SYNTHETIC_NUM_CHANNELS,
            avg_num_nodes=DEFAULT_SYNTHETIC_AVG_NUM_NODES,
            num_classes=DEFAULT_SYNTHETIC_NUM_CLASSES,
            transform=transform
        )
        length = len(dataset)
        train_split = DEFAULT_TRAIN_TEST_SPLIT
        train__ = dataset[:int(train_split * length)]
        test__ = dataset[int(train_split * length):]
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
        data['task' + str(i)] = (y, time, phase, amplitude, frequency)

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
