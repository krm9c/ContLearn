"""
Model checkpoint and initialization utilities.
"""

import jax
import optax
from torch_geometric.loader import DataLoader

from data.loaders import load_return_dataset, continuum_Graph_classification
from utils.model import MLP, CNN, CNN3D, myNN
from utils.trainer import Trainer


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
                model = CNN3D(subkey, filter_size=3, feed_sizes=[2304, 512, 256, num_classes],
                              channel_in=3, channel_out=32, num_classes=num_classes)
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
        y = y.numpy().astype(float)

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
