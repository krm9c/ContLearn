"""
Model checkpoint and initialization utilities.
"""

import jax
import optax
from torch_geometric.loader import DataLoader

from data.loaders import load_return_dataset, continuum_Graph_classification
from utils.model import MLP, CNN, CNN3D, myNN
from utils.trainer import Trainer
from config.constants import (
    DEFAULT_SEED,
    DEFAULT_BATCH_SIZE_GRAPH,
    DEFAULT_BATCH_SIZE_VECTOR,
    DEFAULT_REPLAY_BUFFER_GRAPH,
    DEFAULT_REPLAY_BUFFER_VECTOR,
    DEFAULT_CNN_MNIST_ARCH,
    DEFAULT_CNN3D_CIFAR_ARCH,
    DEFAULT_GCN_SIZES,
    DEFAULT_GCN_MLP_SIZES,
)


def load_checkpoint(config):
    """
    Load checkpoint: Initialize dataset, model, optimizer, and trainer

    Args:
        config: Configuration dictionary

    Returns:
        trainer, optimizer, dataset, model (and test set if graph classification)
    """
    SEED = config.get('seed', DEFAULT_SEED)

    if config['prob'] == 'graphclassification':
        # Use graph_batch_size if provided, otherwise fall back to batch/batch_size, then default
        batch_size = config.get('graph_batch_size', config.get('batch', config.get('batch_size', DEFAULT_BATCH_SIZE_GRAPH)))
        dataset, test = load_return_dataset(
            {
                'batch_size': batch_size,
                'opt': 'Nash',
                'problem': config['prob'],
                'data_id': config['data'],
                'len_exp_replay': config.get('graph_replay_size', DEFAULT_REPLAY_BUFFER_GRAPH),
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
            # Build MLP architecture from config
            mlp_hidden = config.get('mlp_hidden_layers', [config['hln']] * config.get('n_layers', 2))
            model = MLP(sizes=[x.shape[1]] + mlp_hidden + [y.shape[1]])
        elif config['prob'] == 'classification':
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            # Use CNN3D for CIFAR (3-channel 32x32), CNN for MNIST/Omni (1-channel 28x28)
            if config['data'] in ['cifar10', 'cifar100']:
                num_classes = config.get('n_class', 10)
                # Get architecture from config or use defaults
                cnn3d_arch = config.get('cnn3d_feed_sizes', DEFAULT_CNN3D_CIFAR_ARCH.copy())
                # Ensure last layer matches num_classes
                cnn3d_arch[-1] = num_classes
                model = CNN3D(subkey,
                              filter_size=config.get('filter_size', 3),
                              feed_sizes=cnn3d_arch,
                              num_classes=num_classes)
            else:
                # MNIST/Omniglot
                cnn_arch = config.get('cnn_feed_sizes', DEFAULT_CNN_MNIST_ARCH.copy())
                model = CNN(subkey,
                           filter_size=config.get('filter_size', 3),
                           feed_sizes=cnn_arch)
        elif config['problem'] == 'graph':
            # Get GCN architectures from config or use defaults
            gcn_sizes = config.get('gcn_sizes', DEFAULT_GCN_SIZES.copy())
            gcn_sizes[0] = x.shape[1]  # Set input size
            gcn_mlp_sizes = config.get('gcn_mlp_sizes', DEFAULT_GCN_MLP_SIZES.copy())
            gcn_mlp_sizes[-1] = config['n_class']  # Set output size

            model = myNN(in_size=x.shape[1],
                        feed_sizes=gcn_mlp_sizes,
                        gcn_sizes=gcn_sizes,
                        node_num=x.shape[0],
                        out_size=config['n_class'])

        optim = optax.adamw(config['lr'])
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                          problem=config['problem'])

        return trainer, optim, dataset, test, model

    else:
        # Use vector_batch_size if provided, otherwise fall back to batch_size, then default
        batch_size = config.get('vector_batch_size', config.get('batch_size', DEFAULT_BATCH_SIZE_VECTOR))
        dataset = load_return_dataset({
            'batch_size': batch_size,
            'opt': 'Nash',
            'problem': config['prob'],
            'data_id': config['data'],
            'len_exp_replay': config.get('vector_replay_size', DEFAULT_REPLAY_BUFFER_VECTOR),
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
            # Build MLP architecture from config
            mlp_hidden = config.get('mlp_hidden_layers', [config['hln']] * config.get('n_layers', 2))
            model = MLP(sizes=[x.shape[1]] + mlp_hidden + [y.shape[1]])
        elif config['prob'] == 'classification':
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            # Use CNN3D for CIFAR (3-channel 32x32), CNN for MNIST/Omni (1-channel 28x28)
            if config['data'] in ['cifar10', 'cifar100']:
                num_classes = config.get('n_class', 10)
                # Get architecture from config or use defaults
                cnn3d_arch = config.get('cnn3d_feed_sizes', DEFAULT_CNN3D_CIFAR_ARCH.copy())
                # Ensure last layer matches num_classes
                cnn3d_arch[-1] = num_classes
                model = CNN3D(subkey,
                              filter_size=config.get('filter_size', 3),
                              feed_sizes=cnn3d_arch,
                              num_classes=num_classes)
            else:
                # MNIST/Omniglot
                cnn_arch = config.get('cnn_feed_sizes', DEFAULT_CNN_MNIST_ARCH.copy())
                model = CNN(subkey,
                           filter_size=config.get('filter_size', 3),
                           feed_sizes=cnn_arch)
        elif config['problem'] == 'graph':
            # Get GCN architectures from config or use defaults
            gcn_sizes = config.get('gcn_sizes', DEFAULT_GCN_SIZES.copy())
            gcn_sizes[0] = x.shape[1]  # Set input size
            gcn_mlp_sizes = config.get('gcn_mlp_sizes', DEFAULT_GCN_MLP_SIZES.copy())
            gcn_mlp_sizes[-1] = config['n_class']  # Set output size

            model = myNN(in_size=x.shape[1],
                        feed_sizes=gcn_mlp_sizes,
                        gcn_sizes=gcn_sizes,
                        node_num=x.shape[0],
                        out_size=config['n_class'])

        optim = optax.adam(config['lr'])
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                          problem=config['problem'])

        return trainer, optim, dataset, model
