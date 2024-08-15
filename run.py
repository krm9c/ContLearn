
import argparse
import json
import os
import signal
import sys
import itertools

# packages imports
import functools
from functools import partial
import pickle
import numpy as np_
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jaxopt import OptaxSolver
from jax import lax
import diffrax
import equinox as eqx


from torch.utils.tensorboard import SummaryWriter
# local imports
from utils.utils import *
from utils.model import *
from utils.trainer import *
from utils.data import *




class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
          `params.dict['learning_rate']"""
        return self.__dict__



import random
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
def continuum_Graph_classification(dataset, memory_train, memory_test, batch_size, task_id):
        
        # print("new task", task_id)
        stack = [(dataset[j].y==task_id).item() for j in range(len(dataset))]
        datas = [ dataset[k] for k,val in enumerate(stack) if val== True] 
        lengtha=len(datas)
        
        # random.shuffle(datas)
        train_dataset = datas[:int(0.80*lengtha)]
        test_dataset = datas[int(0.80*lengtha):]
        memory_train+=train_dataset
        memory_test+=test_dataset
        
        # print(dataset.to_dense)
        
        # data = datas[0]
        # print(data.adj.shape, data.x.shape)
        
        # print(transform(data))
        # print(transform(data).adj.shape, transform(data).x.shape)
        # print(transform(data).adj.to_dense().shape)
        
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')    
        print(f'Memory:  Number of training graphs: {len(memory_train)}')
        print(f'Memory:  Number of test graphs: {len(memory_test)}')

        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        mem_train_loader = DataLoader(memory_train, batch_size=batch_size, shuffle=True)
        mem_test_loader = DataLoader(memory_test, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, mem_train_loader, mem_test_loader, memory_train, memory_test 


def load_graph_data(data_label):   
    import torch    
    if data_label == 'MUTAG' or data_label == 'ENZYMES' or data_label=='PROTEINS':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='data/TUDataset', name=data_label).shuffle()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='MNIST':
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='data/GNNBench', name='MNIST').shuffle()
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='cora' or data_label=='PubMed'\
        or data_label =='CiteSeer' or data_label=='cora_ML':
        from torch_geometric.datasets import CitationFull
        from torch_geometric.transforms import NormalizeFeatures
        dataset = CitationFull(root='data/CitationFull', name=data_label)
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='Reddit':
        print(data_label)
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root='data/Reddit')
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='tox21':
        print(data_label)
        from torch_geometric.datasets import MoleculeNet
        dataset = MoleculeNet(root='data/tox21', name="tox21")
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset


def generate_sine(delta):
    import pickle
    list_x = []
    list_y = []
    data = {}
    a = 10
    # x_0 = np.random.normal(loc=0.0, scale=1.0, size=(10000, 1))
    time = np.arange(0, 1, 0.1)
    length_trajectory = time.shape[0]
    data = {}
    total_samples= 40
    frequency= (np.random.random([total_samples,1])*60)*np.ones([total_samples, 1])
    amplitude= (np.random.random()*1)*np.ones([total_samples, 1])
    phase = (np.random.random()*90)*np.ones([total_samples, 1])
    # print(frequency, amplitude, phase)
    for i in range(40):
        # y = np.zeros([total_samples, length_trajectory])
        # for j in range(total_samples):
        y = (amplitude)*np.sin(2*np.pi*frequency*time+phase)
        # print(y.shape)
        # print("j",j, "y", y[j,:]) 
        # print(y.shape, amplitude.shape, frequency.shape, phase.shape, time.shape)'
        frequency = frequency + delta
        amplitude = amplitude + delta
        data['task'+str(i)] = (y, time, phase, amplitude, frequency)

    print("Pickling  samples...")
    with open('Incremental_Sine1e^3.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished Pickling")
        
        
def load_return_dataset(config):
    # --------------------------------
    # if the data is sine, generate data.
    if config['data_id']=='sine':
        generate_sine(config['delta'])
        return data_return(config) 
    elif config['data_id']=='ENZYMES':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='data/TUDataset', name=config['data_id'])
        # --------------------------------
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')    
        # --------------------------------
        return dataset
    else:
        return data_return(config) 

def load_checkpoint(config):
    trainer = Trainer(Loss=config['loss'], metric=config['metric'],
            problem=config['problem'], logdir=str(config['tensorfile']) )
    data = load_return_dataset({
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['prob'],
                  'data_id': config['data'],
                  'len_exp_replay': 20000,
                  'network': config['network'],
                  'delta': config['delta']
                  })
    
    
    if config['prob']=='graphclassification':
      memory_train=[]
      memory_test=[]
      dataloader_curr, test_loader_curr, dataloader_exp, test_loader_exp, memory_train, memory_test =\
        continuum_Graph_classification(dataset, memory_train, memory_test, batch_size=config['batch_size'], task_id=i)
      
      batch = next(iter(dataloader_curr))
      batch = batch.ToDense()
      x = batch.x
      adj = batch.adj
      y = batch.y
      print(x.shape, adj.shape, y.shape)
      
    
      
    else:
      dataloader_curr, dataloader_exp = data.generate_dataset(task_id=0,
                          batch_size=config['batch_size'], phase='training')
      test_loader_curr, test_loader_exp = data.generate_dataset(task_id=0,
                      batch_size=config['batch_size'], phase='testing')
      x, y = next(iter(dataloader_curr))
      y = y.numpy().astype(np_.float64)
    
    
    if config['prob'] == 'regression':
        model = MLP(key=0, input_dim=x.shape[1],
                out_dim=y.shape[1], n_layers=config['n_layers'], hln=config['hln'])
    elif config['prob'] == 'classification':
        SEED = 5678
        key = jax.random.PRNGKey(SEED)
        key, subkey = jax.random.split(key, 2)
        model = CNN(subkey)
    elif config['prob'] == 'graphclassification':
        key = jax.random.PRNGKey(SEED) 
        model = myNN(in_size=x.shape[1], hid_size=1000, node_num=dat.shape[0], out_size=y.shape[1], graph=False)
      
      
            
    # model = eqx.tree_deserialise_leaves(
    # "../CL__jax/model/__MLP__task__9.eqx", model)
    params, static = eqx.partition(model, eqx.is_array)
    # initialize the loss function
    func = trainer.return_loss_grad
    # initialize the optimizer
    optim = optax.adamw(config['lr'])
    return trainer, optim, data, model 
        

 
def train_model_graph(config):
    trainer, optim, data, model  = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    for i in range(config['n_task']):
        print("task--", i)       
        if i==0:
            dataloader_curr, _=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, _=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim, record_dict[str(i)] =  trainer.train__CL__reg(dataloader_curr,\
            dataloader_curr, (test_loader_curr, test_loader_curr), (test_loader_curr, test_loader_curr),\
            params, static, optim,  n_iter=config['epochs_per_task'],\
            save_iter=config['save_iter'], task_id=i, config={
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['problem'],
                  'data_id': config['data'],
                  "flag": config['flag'],
                  'len_exp_replay': 20000,
                  'network': config['network'],
                  }, dictum=record_dict)
        else:
            dataloader_curr, dataloader_exp=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, test_loader_exp=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim, record_dict[str(i)] =\
            trainer.train__CL__reg(dataloader_curr,dataloader_curr,\
            (test_loader_curr, test_loader_curr), (test_loader_curr, test_loader_curr),\
            params, static, optim,  n_iter=config['epochs_per_task'],\
            save_iter=config['save_iter'], task_id=i, config={
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['prob'],
                  'data_id': config['data'],
                  "flag": config['flag'],
                  'len_exp_replay': 20000,
                  'network': config['network'],
                  }, dictum=record_dict)
            
        data.append_to_experience(i)
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path']+'.eqx', model)
    del model
    del params
    del static
    return record_dict
  
  
def train_model_reg(config):
    trainer, optim, data, model  = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    for i in range(config['n_task']):
        print("task--", i)       
        if i==0:
            dataloader_curr, _=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, _=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim, record_dict[str(i)] =  trainer.train__CL__reg(dataloader_curr,\
            dataloader_curr, (test_loader_curr, test_loader_curr), (test_loader_curr, test_loader_curr),\
            params, static, optim,  n_iter=config['epochs_per_task'],\
            save_iter=config['save_iter'], task_id=i, config={
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['problem'],
                  'data_id': config['data'],
                  "flag": config['flag'],
                  'len_exp_replay': 20000,
                  'network': config['network'],
                  }, dictum=record_dict)
        else:
            dataloader_curr, dataloader_exp=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, test_loader_exp=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim, record_dict[str(i)] =\
            trainer.train__CL__reg(dataloader_curr,dataloader_curr,\
            (test_loader_curr, test_loader_curr), (test_loader_curr, test_loader_curr),\
            params, static, optim,  n_iter=config['epochs_per_task'],\
            save_iter=config['save_iter'], task_id=i, config={
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['prob'],
                  'data_id': config['data'],
                  "flag": config['flag'],
                  'len_exp_replay': 20000,
                  'network': config['network'],
                  }, dictum=record_dict)
            
        data.append_to_experience(i)
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path']+'.eqx', model)
    del model
    del params
    del static
    return record_dict
    
def train_model_class(config):
    trainer, optim, data, model  = load_checkpoint(config)
    params, static = eqx.partition(model, eqx.is_array)
    
    dict = {}
    for i in range(config['n_task']):
        print("task--", i)       
        if i==0:
            dataloader_curr, _=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, _=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim, dict[str(i)] =\
            trainer.train__CL__class(dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),\
            (test_loader_curr, test_loader_curr), params, static, optim, \
            n_iter=config['epochs_per_task'], save_iter=config['save_iter'], task_id=i,config={
                  'batch_size': config['batch_size'],
                  'opt': 'Nash',
                  'problem': config['prob'],
                  'data_id': config['data'],
                  'len_exp_replay': 20000,
                  "flag": config['flag'],
                  'network': config['network'],
                  })
        else:
            dataloader_curr, dataloader_exp=\
            data.generate_dataset(task_id=i,
            batch_size=config['batch_size'], phase='training')
            test_loader_curr, test_loader_exp=\
                data.generate_dataset(task_id=i,
                batch_size=config['batch_size'],\
                phase='testing')
            params, static, optim_outer, dict[str(i)]=\
            trainer.train__CL__class(dataloader_curr, dataloader_exp,\
            (test_loader_curr, test_loader_exp), (test_loader_curr, test_loader_exp),params, static, optim, \
            n_iter=config['epochs_per_task'], save_iter=config['save_iter'], task_id=i,config={
                  'batch_size': 20,
                  'opt': 'Nash',
                  'problem': config['prob'],
                  'data_id': config['data'],
                  'len_exp_replay': 20000,
                  "flag": config['flag'],
                  'network': config['network'],
                  })
        data.append_to_experience(i)
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(config['model_path'], model)
    del model
    del params
    del static
    return dict
  
# def test_model(stuff):
#   params, static, args.epoch, save_path, models_path, testloader, trainer, df_test, adj= stuff
#   pred = []
#   actual_y = []
#   L = []
#   for batch in testloader:
#       (x, y) = batch
#       x = x.numpy().astype(np_.float32)
#       y = y.numpy().astype(np_.float32)
#       loss, pred__, _= trainer.evaluate__graph(0, (x, y, adj), params, static)
#       pred.append(pred__)  
#       actual_y.append(y)
#       L.append(loss.item())
#   pred = np.concatenate(pred).reshape(-1,1)
#   actual_y = np.concatenate(actual_y)
#   print("The average test MSE is", sum(L)/len(L))
#   df_test['pred_ac'] =pred  
#   plot_metrics(L, pred, actual_y, filename=save_path+"metrics.png")
#   time_arr = df_test['time'].values
#   plot_error_graphics(time_arr[0], df_test,filename=save_path+"error_graphics.png")
    

########################################################################
## The main run loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Test different datasets and different models",
    )
    subparsers = parser.add_subparsers(help='', dest='command')
    
    
    
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("runs", default=1, help="the number of total runs")
    train_parser.add_argument("json", default=None, help="directory with configurations")
    # train_parser.add_argument("save_iter", default=1, type=int, help="saving frequency")
    # train_parser.add_argument("epochs", default=10, type=int, help="number of epochs to train for")
    basic_path='jsons/'
    args = parser.parse_args()
    json_path = os.path.join(basic_path+str(args.json))
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict      
    if args.runs is not None:
      params['runs'] = int(args.runs)
    else:
      params['runs'] = 5
    print("The configuration is", params)
    if args.command == 'train':
      record_dict ={}
      
      
      for j in range(params['runs']):
        print("runs", j)
        
        if params['problem']=='regression':
          record_dict[str(j)] =train_model_reg(params)
        elif params['problem']=='classification':
          record_dict[str(j)] =train_model_class(params)
        else:
          record_dict[str(j)] =train_model_graph(params)
      
      
      import pickle 
      with open(str(params['file'])+'.pkl', 'wb') as f:
          pickle.dump(record_dict, f)
        
      
      # train_model(config)  
      # stuff = params, static, args.epochs, save_path, model_path, testloader, trainer, df_test, adj     
      # test_model(stuff)
      

