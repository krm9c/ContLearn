import jax
import jax.numpy as jnp
import numpy as np
import argparse #GOAL: allows parsing arguments in the command line easily
import json #GOAL: funcs which all for converting code between JSON and Python. (i.e. json.loads() can take json string turn into Python dictionary, json.dumps() can turn python dict into JSON string) 
import os
import signal
import optax
import sys
import itertools

# -----packages imports
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
import jax.tree_util as jtu
import diffrax
import equinox as eqx


from torch.utils.tensorboard import SummaryWriter
# local imports
from utilsAWBTallfunc.utilsAWBT import * #GOAL: provide various easier operation. #CONTAINS: funcs for matrix operations (i.e. special situtation matrix multiplication, normalization) and two graphing funcs. for visualization
from utilsAWBTallfunc.modelAWBT import * #GOAL: class from which we can construct types of NN. #CONTAINS: MLP, CNN, GCN, Linear (uses equinox)
from utilsAWBTallfunc.trainerAWBT import * #GOAL: CL training constructed NN on data. #CONTAINS: loss funcs (i.e. mse, cross-entropy loss), an accuracy of predictions func, loss and pred/accuracy graph constructing func, and CL training functions
from utilsAWBTallfunc.dataAWBT import * #GOAL: take in dataset and prepare for learning #CONTAINS: preparing and batching funcs (uses torch and torchvision)


#=================CLASS TO SET UP PARAMETERS FROM JSON FILE===========================#
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
        with open(json_path) as f: #"with open()" reads from a given file
            params = json.load(f) #json.loads() takes JSON file and converts it to python and stores in params. (KR json params folder is files of JSON strings which would turn to python dicts)
            self.__dict__.update(params) #allows us to modify the dictionary. We set the dictionary as the params
            '''__dict__ is a special attribute every python object has that's a dictionary which stores the object's instance varibales and values.
            we can modify the attribute __dict__ using funcs like .update()'''  

    
    def save(self, json_path): #GOAL: save changes to parameters by converting the dictionary back to JSON file
        with open(json_path, 'w') as f: #open the JSON file and write to it ('w' is write file pointer)
            json.dump(self.__dict__, f, indent=4) #takes the python dict and wrtie it to a JSON file. ("indent" improves the readability of the json file)

    def update(self, json_path): #GOAL: allows us to easily update/modify the parameters in the JSON file
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
          `params.dict['learning_rate']"""
        return self.__dict__ 
#=========================================================#


import random
def continuum_Graph_classification(dataset, memory_train, n_class=6, select=2, batch_size=32):
    """
    GOAL: construct the experience/memory/replay data
    
    ARGUMENTS:
        dataset: Some Benchmark dataset, which was already loaded in using 'load_graph_data()' method.
        memory_train: (list) the func 'load_checkpoint(),' which calls this func sends an empty list for memory_train
        n_class: (int) number of classes
        select: (int) number of choices to select from total classes
        batch_size: (int) number of batches to make
    RETURN:
        train_loader: an object created using 'DataLoader()' class in torch_geometric.loader
        mem_train_loader: an object created using 'DataLoader()' class in torch_geometric.loader 
        memory_train: (list) list of "randomly" selected data
    """
    
    tasks =  np.random.randint(0,n_class,select) #produces array of size 1x2 (ex. [3,5]) of randomly chosen integers between 0 (inlcusive) and 6 (exclusive)
    #data.y: Target to train against
    stack= [(dataset[j].y.numpy() in tasks) for j in range(len(dataset))] #constructs a boolean list of T/F based on if 'dataset[j].y.numpy()' is a number 
    #contaned in the list of numbers produced by task. EX: tasks = [1,4], dataset[0].y.numpy() = [1], dataset[1].y.numpy() = [0], dataset[2].y.numpy() = [4],
    # then stack = [True, False, True,...] the list would continue for the length of the datatset. 'MUTAG' for example is 150 long.
    """
    Rewriting the list comprehension:
    stack = []
    for j in range(len(dataset)):
        if dataset[j].y.numpy() in tasks:
            stack.append(True)
        else:
            stack.append(False)
    """
    datas = [dataset[k] for k,val in enumerate(stack) if val== True]
    #enumerate(): a built-in function that allows you to iterate over an iterable (like a list, tuple, or string) while keeping track of the index.
    #of the current item. 'k' is the index and 'val' is the corresponding 'T/F'. Thus, 'datas' is only those dataset arrays for which stack has "True."
    #EX: using previous ex where we had stack = [True, False, True,...], then datas = [Data(edge_index=[2, 34], x=[15, 7], edge_attr=[34, 4], y=[1]), Data(edge_index=[2, 50], x=[23, 7], edge_attr=[50, 4], y=[1]),...]
    #where the first in the list corresponds to the first "True" in stack and the second in the list corresponds to the second "True"
    for k in range(len(datas)):
        datas[k].n_nodes = datas[k].num_nodes #resetting the number of nodes i.e. 'datas[k].n_nodes' may not contain anything, so set with 'datas[k].num_nodes'
    memory_train+=datas #Since memory train = [], this is a copy of the datas list.
    #append datas to list-- '+=' operator to append objects to a list EX. list = [1,2] then list += [3,4] would be [1,2,3,4].
    from torch_geometric.loader import DataLoader #A data loader which merges data objects from a torch_geometric.data.Dataset to mini-batches.
    train_loader = DataLoader(datas, batch_size=batch_size, shuffle=False) #'dataset' (Dataset) – The dataset from which to load the data. 'batch_size' (int) – How many samples per batch to load. (default: 1)
    mem_train_loader = DataLoader(memory_train, batch_size=batch_size, shuffle=False) 
    print("I am picking the classes", tasks, len(memory_train), len(datas), len(train_loader), len(mem_train_loader))
    return train_loader, mem_train_loader, memory_train


#----------------LOAD GRAPH DATA--------------------#
def load_graph_data(data_label): 
    """
    GOAL: Load the benchmark graph datasets. All can be obtained using torch_geometric.datasets
    Args:
        data_label: string (In practice, this is config['data_id'] == params['data'] would be 'sine', 'synthetic', 'MUTAG', 'ENZYME',...)
    Returns: 1 or 2
        train__: the training set of the data  (for MUTAG, ENZYMES and PROTEINS)
        test__: the test set of the data
        
        **or**
        
        dataset: the data set (for MNIST, cora, CiteSeer, PubMed, cora_ML). This is a class i.e. 'Reddit()', 'CiteSeerFull()"
    """
    import torch
    
    #SMALL FUNCTION FIRST
    def transform(data):
            data.n_nodes = data.num_nodes
            return data
    #----#
    
    if data_label == 'MUTAG' or data_label == 'ENZYMES' or data_label=='PROTEINS':
        from torch_geometric.datasets import TUDataset
        #TUDataset: A collection of benchmark datasets for graph classification and regression.
        torch_geometric.seed.seed_everything(10) 
        dataset = TUDataset(root='data/TUDataset', name=data_label, transform=transform).shuffle() #obtain the desired data and shuffle/randomize it
        length = len(dataset)
        train__ = dataset[:int(0.80*length)] #make the training set (first 80% of data) Ex. 'MUTAG(150)'
        test__ = dataset[int(0.80*length):] #make the test set (last 20% of data) Ex. 'MUTAG(38)'
        #format print a table describing the data (i.e. length of train set, # of features of graph, # of classes of graphs)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(train__)}')
        print(f'Number of features: {dataset.num_features}') #'.num_features' is an attribute of the instance 'dataset' that we created from the TUDataset
        print(f'Number of classes: {dataset.num_classes}') #'.num_classes' is an attribute of the instance 'dataset' that we created from the TUDataset   
        return train__, test__  #return the two data sets
    elif data_label=='MNIST':
        from torch_geometric.datasets import GNNBenchmarkDataset
        #GNNBenchmarkDataset: A variety of artificially and semi-artificially generated graph datasets from the “Benchmarking Graph Neural Networks” paper.
        dataset = GNNBenchmarkDataset(root='data/GNNBench', name='MNIST').shuffle() #obtain the desired data and shuffle/randomize it
        print()
        #format print a table describing the data (i.e. length of train set, # of features of graph, # of classes of graphs)
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}') #'.num_features' is an attribute of the instance 'dataset' that we created from the GNNDataBench
        print(f'Number of classes: {dataset.num_classes}') #'.num_classes' is an attribute of the instance 'dataset' that we created from the GNNDataBench
        return dataset #return the data set

    elif data_label=='cora' or data_label=='PubMed'\
        or data_label =='CiteSeer' or data_label=='cora_ML':
        from torch_geometric.datasets import CitationFull
        #CitationFull: The full citation network datasets from the “Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking” paper. Nodes represent 
        #documents and edges represent citation links. Datasets include "Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed".
        from torch_geometric.transforms import NormalizeFeatures #Row-normalizes the attributes given in attrs to sum-up to one
        dataset = CitationFull(root='data/CitationFull', name=data_label) #obtain desired benchmark set from CitationFull. 'CitationFull()' is a class use 'root' and 'name' to retrieve correct dataset.
        data= dataset[0] #gives 'Data(x=[4230, 602], edge_index=[2, 10674], y=[4230])'
        print(data)
        print("from the load dataset", data.x) #'data.x': Node feature matrix with shape [num_nodes, num_node_features]. Ex: "torch.Size([4230, 602])" for size of data.x
        #format print a table describing the data (i.e. length of train set, # of features of graph, # of classes of graphs)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')#'.num_features' is an attribute of the instance 'dataset' that we created from the CitationFull. Describes # of features
        print(f'Number of classes: {dataset.num_classes}') #'.num_classes' is an attribute of the instance 'dataset' that we created from the CitationFull. Describes # of classes
        return dataset
    elif data_label=='Reddit':
        #Reddit: The Reddit dataset from the “Inductive Representation Learning on Large Graphs” paper, containing Reddit posts belonging to different communities.
        print(data_label)
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root='data/Reddit')
        data= dataset[0]
        print(data) #'Data(x=[232965, 602], edge_index=[2, 114615892], y=[232965], train_mask=[232965], val_mask=[232965], test_mask=[232965])'
        print("from the load dataset", data.x) #'data.x': Node feature matrix with shape [num_nodes, num_node_features].
        #format print a table describing the data (i.e. length of train set, # of features of graph, # of classes of graphs)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='tox21':
        print(data_label)
        from torch_geometric.datasets import MoleculeNet
        #A Benchmark collection from "Molecular Machine Learning” paper, containing datasets from physical chemistry, biophysics and physiology. 
        #Datasets include "ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox."
        dataset = MoleculeNet(root='data/tox21', name="tox21")
        print("from the load dataset", data.x)
        #format print a table describing the data (i.e. length of train set, # of features of graph, # of classes of graphs)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='synthetic':
        from torch_geometric.datasets import FakeDataset
        #A fake dataset that returns randomly generated Data objects.
        torch_geometric.seed.seed_everything(10) 
        #dataset = FakeDataset(num_graphs= 1000, num_channels=5,\
            #avg_num_nodes=2, num_classes = 10, transform=transform).shuffle() #shuffle the fakedata after importing
        dataset = FakeDataset(num_graphs= 1000, num_channels=5,\
            avg_num_nodes=2, num_classes = 10, transform=transform) #shuffle the fakedata after importing
        length = len(dataset)
        train__ = dataset[:int(0.80*length)] #create training set 80%
        test__ = dataset[int(0.80*length):] #create test set 20%
        # print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(train__)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return train__, test__  
        

#----------GENERATE SINE DATA-----------------#
def generate_sine(delta):
    """
    GOAL: generate sine data
    Args:
        delta: float (very small--this is 'delta' from parameters dictionary)
    Returns:
        NONE (the function writes a dictionary of sine data to a file via pickle (serialization))
    """
    import pickle
    list_x = []
    list_y = []
    data = {}
    a = 10
    time = np.arange(0, 1, 0.1) #array: [0 .1 .2 .3 .4 .5 .6 .7 .8 .9]
    length_trajectory = time.shape[0] #shape of time = 10
    data = {}
    total_samples= 40
    np.random.seed(1) #set the seed for the random number generator to 3. This ensures that the random numbers generated are reproducible.
    #Need freq, amp, phase shift for "amp*2pi*freq*time+phase" to generate the sine data
    frequency= (np.random.random([total_samples,1])*60)*np.ones([total_samples, 1])  #40x1 array of floats randomly chosen
    amplitude= (np.random.random()*1)*np.ones([total_samples, 1]) #randomly choose a # between 0 and 1 then make a 40x1 where every entry is that number
    phase = (np.random.random()*90)*np.ones([total_samples, 1]) #randomly choose a # between 0 and 1 then make a 40x1 where every entry is that number *90
    for i in range(40):
        y = (amplitude)*np.sin(2*np.pi*frequency*time+phase)
        frequency = frequency + delta #This perturbs the frequency a little more with each j
        amplitude = amplitude + delta #This perturbs the amplitude a little more with each j
        data['task'+str(i)] = (y, time, phase, amplitude, frequency) #produces {'task0': (y_0,time_0,phase_0,amp_0,freq_0), 'task1': (y_1,time_1,phase_1,amp_1,freq_1),...}
        
    print("Pickling  samples...")
    with open('Incremental_Sine1e^4.p', 'wb') as fp: #create a file called 'Incremental_Sine1e^4.p'
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)# write to the file by serializing the dictionary 'data' we created in the last loop
    print("Finished Pickling")
        

#----------------LOAD NECESSARY DATA------------------------#
def load_return_dataset(config):
    """
    GOAL: Load/generate all necessary data
    Args:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)
    Returns:
        data_retun(config): This is an instance of a class which can be found in dataset_utils.py. This has many attributes but here are two important ones
                ATTRIBUTES:
                    dataset: contains the actual data set
                    dataset_id: This is the value of config['data_id'] (or rather params['data'])          
    """
    if config['data_id']=='sine': # if the data is sine, then generate data
        generate_sine(config['delta']) #go to generate_sine() func in this .py file. This generates sine data and writes it to a file via pickle
        return data_return(config)  #'data_return()' is a class imported from dataset_utils.py file.
        #For 'sine', 'data_retun(config).dataset' attribute contains the data we pickled in the "generate_sine" method. It unpickled the data in the file we wrote to in the generate_sine method.
    elif config['problem']=='graphclassification': # the data is graph classification then...
        return load_graph_data(config['data_id']) #go to "load_graph_data()" func in this .py file.
    else:
        return data_return(config) #'data_return()' is a class in 'utils.data.py.' Based on 'config[data_id]' we retrieve data for the cases where we don't have 'sine' or graphclassification.
        #it uses torch and pickle to load the data. Options: 'omni', 'sine', 'mnist', 'synthetic'. However, 'omni' is only one in example JSON files

#--------------LOAD CHECKPOINT---------------------#
def load_checkpoint(config):
    """
    GOAL: Load and set all necessary information. Loads necessary datasets, constructs model, sets the optimizer, and constructs the trainer.
    Args: 
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)
    Returns:
        trainer: an object constructed using trainer class in "utils.trainer.py"
        optim: optimizer (using optax)
        dataset: dataset generated in "load_return_dataset()." This is an object
        test: An object contructed using torch.dataloader
        model: An object, constructed using "utils.model.py"
    """
    SEED = 5678
    if config['prob']=='graphclassification': #if it is a graphclassification problem, then...
        dataset, test = load_return_dataset({
                    'batch_size': 20,
                    'opt': 'Nash',
                    'problem': config['prob'],
                    'data_id': config['data'],
                    'len_exp_replay': 20000,
                    'network': config['network'],
                    'delta': config['delta']
                    }) #go to load_return_dataset() in this .py file. "config[key]" is keys from parameters dict. 
                       #'prob' is type of problem (class, reg, graphclass), 
                       #'data' is since, synthetic, enayme, mutag, omni.
                       # 'network' is type of NN (gnn, fcnn, cnn)
        memory_train=[]
        _,_, memory_train =\
        continuum_Graph_classification(dataset, memory_train,\
        n_class=config['n_class'],\
        select=config['class_per_task'])
        x= memory_train[0].x #RECALL: data.x  = Node feature matrix with shape [num_nodes, num_node_features]
        #Format print basic information to the user about the datasets
        print(f'Number of training graphs: {len(dataset)}')
        print(f'Number of test graphs: {len(dataset)}')    
        print(f'Memory:  Number of training graphs: {len(memory_train)}')
        print(f'Memory:  Number of test graphs: {len(test)}')
        from torch_geometric.loader import DataLoader
        test = DataLoader(test, batch_size=config['batch'], shuffle=True) #batch the test data
        
        # Model definition--Construct the model based on the type of problem
        #Each of these classes can be found in 'utils.model.py'. He uses Equinox to construct each model. MyNN is GCN custom
        if config['prob'] == 'regression': #if regression use MLP
            #model = MLP(key=0, input_dim=x.shape[1],\
                    #out_dim=y.shape[1], n_layers=config['n_layers'],\
                    #hln=config['hln'])
            
            model = MLP(sizes = [x.shape[1],config['hln'],config['hln'],y.shape[1]])
            #model = MLP(sizes = [x.shape[1],75,75,y.shape[1]])
            #print("MODEL IN LOAD_CHECK:", model)
        elif config['prob'] == 'classification': #if classification use CNN
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            model = CNN(subkey,3,[1875,512,64,10])
        elif config['problem'] == 'graph': #if graph use myNN. This is a GCN.
            model1= myNNorig(in_size=x.shape[1], hid_size=config["hln"],\
                node_num=x.shape[0], out_size=config['n_class'])
            #print("HIS GCN MODEL: ", model1)
            print("------------------")
            model = myNN(in_size=x.shape[1], feed_sizes = [128,128,128,10], gcn_sizes = [5,128],\
                node_num=x.shape[0], out_size=config['n_class'])
            #print("here is gcn model: ", model) #print the model 
        optim = optax.adamw(config['lr']) #set adamw as optimizer (Adam with weight decay regularization), 'lr' is the learning rate
        trainer = Trainer(Loss=config['loss'], metric=config['metric'], 
                problem=config['problem'], logdir=str(config['tensorfile'])) #This is a class in "utils.trainer.py".  Creates NN Trainer
        
        return trainer, optim, dataset, test, model 

    else: #if it's not a graphclassification problem, then...
        dataset = load_return_dataset({
                    'batch_size': 20,
                    'opt': 'Nash',
                    'problem': config['prob'],
                    'data_id': config['data'],
                    'len_exp_replay': 20000,
                    'network': config['network'],
                    'delta': config['delta']
                    }) #get the appropriate dataset (this is reg. or class, seems data will not come from "load_graph_data()")

        dataloader_curr, dataloader_exp = dataset.generate_dataset(task_id=0,batch_size=config['batch_size'], phase='training') 
        #RECALL: 'dataset' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'dataloader_curr' and 'dataloader_exp' are DataLoader() objects containing data for task 0. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.
        
        test_loader_curr, test_loader_exp = dataset.generate_dataset(task_id=0, batch_size=config['batch_size'], phase='testing')
        #RECALL: 'dataset' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'test_loader_curr' and 'test_loader_exp' are DataLoader() objects containing data for task 0. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.

        x, y = next(iter(dataloader_curr)) #iterate through the batches of data in 'dataloader_curr" using iterator and calling 'next'
        y = y.numpy().astype(np_.float64) #astype(): This method is called on the array and converts its data type to 64-bit floating point
             
        # Model definition--Construct the model based on the type of problem
        #Each of these classes can be found in 'utils.model.py'. He uses Equinox to construct each model. MyNN is GCN custom
        if config['prob'] == 'regression': #If regression, use MLP
            #model = MLP(key=0, input_dim=x.shape[1],\
                    #out_dim=y.shape[1], n_layers=config['n_layers'],\
                    #hln=config['hln'])
            model = MLP(sizes = [x.shape[1],config['hln'],config['hln'],y.shape[1]])
            #model = MLP(sizes = [x.shape[1],75,75,y.shape[1]])
            print("MODEL IN LOAD_CHECK:", model)
        elif config['prob'] == 'classification': #if classification, use CNN
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            model = CNN(subkey,3,[1875,512,64,10])
        elif config['problem'] == 'graph': #if graph, use myNN which is GCN
            model1 = myNNorig(in_size=x.shape[1], hid_size=config["hln"],\
                node_num=x.shape[0], out_size=config['n_class'])
            #print("HIS GCN MODEL: ", model1)
            #print("------------------")
            model = myNN(in_size=x.shape[1], feed_sizes = [128,128,128,10], gcn_sizes = [5,128],\
                node_num=x.shape[0], out_size=config['n_class'])
            #print("here is gcn model: ", model) 
        optim = optax.adam(config['lr']) #adam optimizer, 'lr' is learning rate
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                problem=config['problem'], logdir=str(config['tensorfile'])) #This is a class in "utils.trainer.py".  Creates NN Trainer

        return trainer, optim, dataset, model

#-----------------------architecture search for GRAPH CLASSIFICATION PROBLEM-----------------#
def arch_search_GCN(original_gcn, original_mlp, task, trainW_loss, og_epochs, config,train_loader, mem_train_loader,test):
    #first obtain loss value on current arch
    og_epochs = 50
    i = task
    #original_gcn = [5,100]
    #original_mlp = [100,140,150,10]
    opt_gcn = original_gcn
    opt_mlp = original_mlp
    trainer1, optim5, __, __, arch_model  = load_checkpoint(config) #Get trainer, optim, model
    #set current architecture
    arch_model = eqx.tree_at(lambda x: (x.gcn_sizes, x.feed_sizes), arch_model, replace = (original_gcn, original_mlp))
    #set weights for all randomly initially
    initializer = jax.nn.initializers.glorot_uniform()
    weightsMLP_list =[initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(arch_model.feed_sizes[:],arch_model.feed_sizes[1:])]
    biasMLP_list =[initializer(jax.random.PRNGKey(5), (1, y)) for y in arch_model.feed_sizes[1:]]
    weightsGCN_list = [initializer(jax.random.PRNGKey(5), (x, y)) for x,y in zip(arch_model.gcn_sizes[:],arch_model.gcn_sizes[1:])]
    biasGCN_list = [initializer(jax.random.PRNGKey(5), (1, y)) for y in arch_model.gcn_sizes[1:]]
    for k in range(len(arch_model.gcn_layers)):
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
        arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])
    #print("model after setting new gcn weights: ", arch_model)
    for j in range(0,len(arch_model.feed_layers)):
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])
    #print("model after setting new MLP weights: ", arch_model)
    record_dict_arch = {}
    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
    #reset AB's to static
    arch_static = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), arch_static, replace= (arch_model.A_gcn, arch_model.B_gcn,arch_model.A_feed,arch_model.B_feed))
    arch_params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), arch_params, replace= (None,None,None,None))
    #train
    arch_params, arch_static, optim5, record_dict_arch[str(i)] = trainer1.train__CL__graph((mem_train_loader, test, train_loader), arch_params,arch_static,optim5, \
                                                                n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                    task_id=task, config={'batch_size': config['batch']},\
                                                                    dictum=record_dict_arch) #complete the CL training
    arch_model = eqx.combine(arch_params,arch_static) #recombine the model
    arch_dict = record_dict_arch[str(i)]

    loss_orig = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,10)])
    print("trainWLoss: ", trainW_loss, "-- short loss: ", loss_orig)
    loss_opt = loss_orig
    
    #--actual search
    ## for right now assume arch_gcn = [5,z]
    ## for right now assume arch_mlp = [z,_,_,10]
    z2 = original_gcn[1]
    x1 = original_mlp[1]
    x2 = original_mlp[2]
    step_gcn = 10
    step_mlp = 10
    n = 1 # controls out spread
    m=1
    while (n<3) or (loss_opt< .8*loss_orig): #search in two bigger neighborhoods
        for j in range(3): #create outer for loop going through gcn arch
            curr_gcn = [original_gcn[0],z2+n*(j+1)*step_gcn]
            arch_model = eqx.tree_at(lambda x: x.gcn_sizes, arch_model, curr_gcn) #set current gcn
            #create and set new weights for gcn randomly
            initializer = jax.nn.initializers.glorot_uniform()
            weightsGCN_list = [initializer(jax.random.PRNGKey(5), (x, y)) for x,y in zip(arch_model.gcn_sizes[:],arch_model.gcn_sizes[1:])]
            biasGCN_list = [initializer(jax.random.PRNGKey(5), (1, y)) for y in arch_model.gcn_sizes[1:]]
            for k in range(len(arch_model.gcn_layers)):
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, arch_model, weightsGCN_list[k])
                arch_model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, arch_model, biasGCN_list[k])
            #print("model after setting new gcn weights: ", arch_model)
            for k in range(3): #create inner loop going through MLP arch   #3
                for r in range(3): #3
                    curr_mlp = [curr_gcn[-1], x1+n*(k+1)*step_mlp, x2+n*(r+1)*step_mlp,10]
                    arch_model = eqx.tree_at(lambda x: x.feed_sizes, arch_model, curr_mlp) #set current arch
                    print("========= curr_gcn: ", curr_gcn, "========== curr_mlp: ", curr_mlp)
                    #create and set new weights for MLP randomly
                    weightsMLP_list =[initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(curr_mlp[:],curr_mlp[1:])]
                    biasMLP_list =[initializer(jax.random.PRNGKey(5), (1, y)) for y in curr_mlp[1:]]
                    for j in range(0,len(arch_model.feed_layers)):
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].weight, arch_model, weightsMLP_list[j])
                        arch_model = eqx.tree_at(lambda x: x.feed_layers[j].bias, arch_model, biasMLP_list[j])
                    #train for some epochs and compare loss value to previous
                    record_dict_arch = {}
                    optim6 = optax.adamw(1e-4)
                    arch_params, arch_static = eqx.partition(arch_model, eqx.is_array)
                    #reset AB's to static
                    arch_static = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), arch_static, replace= (arch_model.A_gcn, arch_model.B_gcn,arch_model.A_feed,arch_model.B_feed))
                    arch_params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), arch_params, replace= (None,None,None,None))
                    #train
                    arch_params, arch_static, optim6, record_dict_arch[str(i)] = trainer1.train__CL__graph((mem_train_loader, test, train_loader), arch_params,arch_static,optim6, \
                                                                                n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                    task_id=task, config={'batch_size': config['batch']},\
                                                                                    dictum=record_dict_arch) #complete the CL training
                    arch_model = eqx.combine(arch_params,arch_static) #recombine the model
                    #determine whehter curr_arch is opt_arch for each
                    arch_dict = record_dict_arch[str(i)]
                    loss_poll = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,10)])
                    if loss_poll<loss_opt:
                        opt_gcn = curr_gcn
                        opt_mlp = curr_mlp
                        loss_opt = loss_poll
                    m+=1
                    print("ROUND ",m ,": opt_gcn: ", opt_gcn, "---- opt_mlp: ", opt_mlp, "---opt_loss: ", loss_opt)
        n+=3 #add three because we skip to next neighborhood
    return  opt_gcn, opt_mlp  
        

#------------------------TRAINING for GRAPH CLASSIFICATION PROBLEM---------------------------#
def train_model_graph(config):
    """
    GOAL: construct and CL train a model which is for a graph classification problem

    ARGUMENTS:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)

    RETURNS:
        record_dict: dictionary 
    """
    trainer, optim, data, test, model  = load_checkpoint(config) #Get trainer, optim, data, model
    #RECALL: 'trainer' is a 'Trainer()' object from 'utils.trainer.py.'
    #        'optim' is the optimizer (from optax)
    #        'data' is a 'data_return()' object from 'utils.data.py'
    #        'model' is a NN model (i.e. MLP, CNN, etc.) from 'utils.model.py'
    
    params, static = eqx.partition(model, eqx.is_array) #separate the params and static of the model
    record_dict = {}
    memory_train=[]
    record_dict_preAB = {}
    record_dict_AB = {}
    static = eqx.tree_at(lambda x: x.A_gcn, static, replace= model.A_gcn)
    static = eqx.tree_at(lambda x: x.B_gcn, static, replace= model.B_gcn)
    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
    params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
    prev_loss_step1 = []
    gcn_arch_list = []
    mlp_arch_list = []
    #print(len(data), test)
    for i in range(config['n_task']): #for loop running for total number of tasks prescribed
        print("task--", i)#tell current task #
        if i == 0:
            train_loader, mem_train_loader, memory_train  = continuum_Graph_classification(data, memory_train,n_class=config['n_class'],\
                                                                select=config['class_per_task']) #constructs training, memory train data sets, 
            #each with batches.
            """
            NOTES on train_CL_graph: See other file.
            """
            og_epochs = 25
            og_epochs = config['epochs_per_task']
            params, static, optim, record_dict[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim, \
                                                                                n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={'batch_size': config['batch']},\
                                                                                    dictum=record_dict) #complete the CL training
            optim1 = optim
            rec_dict = record_dict[str(i)]
            end_last = np.mean([rec_dict["train"+str((i+1)*125-j)][0] for j in range(1,10)])
        else:
            task = i
            train_loader, mem_train_loader, memory_train  = continuum_Graph_classification(data, memory_train,n_class=config['n_class'],\
                                                                select=config['class_per_task']) #constructs training, memory train data sets, 
            #each with batches.
            """
            NOTES on train_CL_graph: See other file.
            """
            #==================STEP 1: Train W for a little bit on new task========================#
            print("STEP 1: Train W for a little bit on new task-------------")
            og_epochs = 50
            params, static, optim1, record_dict_preAB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim1, \
                                                                                n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={'batch_size': config['batch']},\
                                                                                    dictum=record_dict_preAB) #complete the CL training
            arch_dict = record_dict_preAB[str(i)]
            trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,10)])
            prev_loss_step1.append(trainWLoss)
            print("original loss for preAB to compare: ", trainWLoss)
            #-----------------STEP 2: Search for new architecture (gcn size and MLP size)------------#
            print("STEP 2: Find new architecture (arch search in future)----------")
            #We will incorporate search later, set new arch for now
            prev_feed_sizes = model.feed_sizes
            prev_gcn_size = model.gcn_sizes
            task = i
            #opt_gcn, opt_MLParch = arch_search_GCN(prev_gcn_size, prev_feed_sizes, task, trainWLoss, og_epochs, config,train_loader, mem_train_loader,test)
            print("Here's end_last: ", end_last, "Here's trainWLoss: ", trainWLoss)
            gcn_weight_layer = []
            gcn_bias_layer = []
            mlp_weight_layer = []
            mlp_bias_layer = []
            for j in range(len(model.feed_layers)):
                mlp_weight_layer.append(model.feed_layers[j].weight)
                mlp_bias_layer.append(model.feed_layers[j].bias)
            for j in range(len(model.gcn_layers)):
                gcn_weight_layer.append(model.gcn_layers[j].weight)
                gcn_bias_layer.append(model.gcn_layers[j].bias)
            preAgcn= model.A_gcn[0]
            preAfeed=model.A_feed[0]
            preWgcn=model.gcn_layers[0].weight
            preWfeed=model.feed_layers[0].weight
            if (end_last+0.1<=trainWLoss):
                print("WE ARE CHANGING ARCHITECTURE!!!")
                opt_gcn, opt_MLParch = arch_search_GCN(prev_gcn_size, prev_feed_sizes, task, trainWLoss, og_epochs, config,train_loader, mem_train_loader,test)
                
                print("NEW  GNN Architecture: ", opt_gcn)
                print("NEW MLP Arch: ", opt_MLParch)
                gcn_arch_list.append(opt_gcn)
                mlp_arch_list.append(opt_MLParch)
                model = eqx.combine(params,static) #put back model
                for k in range(len(model.gcn_layers)):
                        aw = gcn_weight_layer[k]
                        ab = gcn_bias_layer[k]
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, model, aw)
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, model, ab)
                for j in range(0,len(model.feed_layers)):
                    cw = mlp_weight_layer[j]
                    cb = mlp_bias_layer[j]
                    model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, cw)
                    model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, cb)
                print("Did A gcn change (True): ", jnp.allclose(preAgcn, model.A_gcn[0]))
                print("Did A feed change (True): ", jnp.allclose(preAfeed, model.A_feed[0]))
                print("Did W gcn stay same (True): ",jnp.allclose(preWgcn, model.gcn_layers[0].weight))
                print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                print()
            else:
                opt_gcn = prev_gcn_size
                opt_MLParch = prev_feed_sizes
                gcn_arch_list.append(opt_gcn)
                mlp_arch_list.append(opt_MLParch)
                print("ARCHITECTURE Did NOT change")


        
            if (prev_feed_sizes != opt_MLParch) or (prev_gcn_size != opt_gcn):
                #-----------------STEP 3A: Set new architecture and ABs----------------#
                print("STEP 3A: Set new architecture and ABs-----------------")
                if (prev_feed_sizes != opt_MLParch) and (prev_gcn_size != opt_gcn):
                    #model = eqx.combine(params,static) #put back model
                    model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_MLParch) #set new arch
                    model = eqx.tree_at(lambda x: x.gcn_sizes, model, opt_gcn) #set new gcn
                    initializer = jax.nn.initializers.glorot_uniform()
                    B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[1:],model.feed_sizes[1:])]
                    A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[:-1],model.feed_sizes[:-1])]
                    B_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[1:],model.gcn_sizes[1:])]
                    A_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[:-1],model.gcn_sizes[:-1])]
                    model = eqx.tree_at(lambda x: (x.A_feed, x.B_feed, x.A_gcn, x.B_gcn), model, replace = (A_feed, B_feed, A_gcn, B_gcn))
                    print("===========================")
                    print("model after setting everything: ", model)
                    print("===========================")
                    print()
                    print("New feed AND gcn!!!------------------")
                    #---------------Step 3b: Train ONLY A's and B's---------------------#
                    print("STEP 3B: Train ONLY on the A's and B's; fix W----------------")
                    preAgcn = model.A_gcn[0]
                    preAfeed = model.A_feed[0]
                    preWgcn = model.gcn_layers[0].weight
                    preWfeed = model.feed_layers[0].weight
                    og_epochs = 500
                    model1 = model
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn, x.A_feed, x.B_feed), filter_spec, replace=(True,True,True, True),)
                    diff_model, static_model = eqx.partition(model, filter_spec) #makes weights static and A's, B's params
                    import optax
                    optim2 = optax.adamw(1e-4) #set new optimizer
                    #==============put metric here for training of AB based on what loss in preAB was======#
                    a = 0
                    b=0
                    AB_convList =[]
                    diff_model, static_model, optim2, record_dict_AB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), diff_model,static_model,optim2, \
                                                                                            n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                                task_id=i, config={'batch_size': config['batch']},\
                                                                                                dictum=record_dict_AB, notABTrain = False) #complete the CL training
                    AB_dict = record_dict_AB[str(i)]
                    AB_loss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,20)])
                    print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss)
             
                    #instead could do all based on end_last base threshold on ratio of trainWLoss and end_last
                    ratio = trainWLoss/end_last
                    if ratio >3.0:
                        threshold = 1/ratio
                        #threshold = jnp.max(.25, threshold)
                    elif 2.0<= ratio<3.0:
                        #threshold = .5
                        threshold = 1/ratio
                    elif 1.0<=ratio<2.0:
                        threshold = .75
                        threshold = 1/ratio
                    else:
                        threshold = .9

                    print("Here is ratio: ", ratio, " --------- and threshold: ", threshold, "------- Goal: ", trainWLoss*threshold)
                    print()
                    while (trainWLoss*threshold < AB_loss) and (a<15):
                        diff_model, static_model, optim2, record_dict_AB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), diff_model,static_model,optim2, \
                                                                                            n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                                task_id=i, config={'batch_size': config['batch']},\
                                                                                                dictum=record_dict_AB, notABTrain = False) #complete the CL training
                        AB_dict = record_dict_AB[str(i)]
                        AB_loss_curr = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,20)])
                        print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)
                        diff = AB_loss_curr - trainWLoss
                        AB_convList.append(AB_loss_curr)
                        print("AB_convList: ", AB_convList)

                        if AB_loss_curr>AB_loss: #check if loss is increasing
                            #print("AB_loss is increasing, decrease epochs")
                            print("AB_convList[-2]: ", AB_convList[-2], "AB_loss_curr: ", AB_loss_curr)
                            if (AB_convList[-2]<AB_loss_curr):
                                print("training is increasing...break")
                                break
                            #prev_ab = AB_loss_curr
                            og_epochs = 100
                            AB_loss = AB_loss_curr
                            a+=1
                        else:
                            AB_loss = AB_loss_curr
                            a+=1
                            og_epochs = 500
                        #check if converging
                        if a>7:
                            conv_Diff = np.mean([AB_convList[-1],AB_convList[-2],AB_convList[-3]])
                            if jnp.abs(conv_Diff-AB_convList[-1]) <= 0.01:
                                print("AB_loss converged, breaking")
                                break
                    model = eqx.combine(diff_model,static_model) #recombine params and statics
                    print("heres i after AB train:", i)
                    #print("A_gcn after AB train (CHANGE): ", model.A_gcn[0])
                    #print("A_feed after AB train (CHANGE): ", model.A_feed[0])
                    #print("gcn weights after AB train (NO CHANGE): ", model.gcn_layers[0].weight)
                    #print("feed weights after AB Train (NO CHANGE): ", model.feed_layers[0].weight)
                    print("=========Check if A changed and W froze=============")
                    print("Did A gcn change (False): ", jnp.allclose(preAgcn, model.A_gcn[0]))
                    print("Did A feed change (False): ", jnp.allclose(preAfeed, model.A_feed[0]))
                    print("Did W gcn stay same (True): ",jnp.allclose(preWgcn, model.gcn_layers[0].weight))
                    print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                    print()
                    #=====================STEP 4: Set V = AWB for gcn and mlp========================#
                    print("--------------STEP 4: Set V = AWB for gcn and mlp-----------------")
                    #print("here is model again: ", model)
                    #weights_list = [[(model.A_gcn[i]@(model.gcn_layers[i].weight)@jnp.transpose(model.B_gcn[i]))] for i in range(0,len(model.gcn_layers))]
                    #bias_list = [[(model.gcn_layers[i].bias@ model.B_gcn[i].T)] for i in range(0,len(model.gcn_layers))]
                    for k in range(len(model.gcn_layers)):
                        Vw = model.A_gcn[k]@(model.gcn_layers[k].weight)@jnp.transpose(model.B_gcn[k])
                        Vb = (model.gcn_layers[k].bias@ model.B_gcn[k].T)
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, model, Vw)
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, model, Vb)
                        #model = eqx.tree_at(lambda x: x.gcn_layers[i].weight, model, replace = weights_list[i])
                        #model = eqx.tree_at(lambda x: x.gcn_layers[i].bias, model, replace = bias_list[i])
                    #print("model after setting new gcn weights: ", model)
                    for j in range(0,len(model.feed_layers)):
                        #print("A: ", model.A_feed[j].shape)
                        # print("W: ", model.feed_layers[j].weight.shape)
                        # print("W.T: ", model.feed_layers[j].weight.T.shape)
                        # print("B.T: ", jnp.transpose(model.B_feed[j]).shape)
                        Vw = (model.A_feed[j] @ model.feed_layers[j].weight.T @ jnp.transpose(model.B_feed[j])).T
                        #print("Vw shape: ", Vw.shape)
                        Vb = model.feed_layers[j].bias@ model.B_feed[j].T
                        #print("Vb shape: ", Vb.shape)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)
                    print("======================")
                    print("Model after setting new V weights: ", model)
                    print("======================")
                    #reset params and static
                    params, static = eqx.partition(model, eqx.is_array)
                    #print("here is params: ", params)
                    #print("here is static", static)
                    #reset A's and B's as statics again
                    static = eqx.tree_at(lambda x: x.A_gcn, static, replace= model.A_gcn)
                    static = eqx.tree_at(lambda x: x.B_gcn, static, replace= model.B_gcn)
                    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                    params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                    # print("here is params AFTER: ", params)
                    # print("here is static AFTER:", static)
                    print("STEP 5: Train CNN on new weights and record---------------")
                    optim3 = optax.adamw(1e-4)
                    optim1 = optim3
                    params, static, optim1, record_dict[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim1, \
                                                                                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                            task_id=i, config={'batch_size': config['batch']},\
                                                                                            dictum=record_dict) #complete the CL training
                    #recombine to model and the resplit so the model and params/statics are updated for next task
                    rec_dict = record_dict[str(i)]
                    end_last = np.mean([rec_dict["train"+str((i+1)*125-j)][0] for j in range(1,10)])
                    
                elif (prev_feed_sizes != opt_MLParch) and (prev_gcn_size == opt_gcn):
                    model = eqx.combine(params,static) #put back model
                    model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_MLParch) #set new arch
                    initializer = jax.nn.initializers.glorot_uniform()
                    B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[1:],model.feed_sizes[1:])]
                    A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_feed_sizes[:-1],model.feed_sizes[:-1])]
                    #B_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[1:],model.gcn_sizes[1:])]
                    #A_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[:-1],model.gcn_sizes[:-1])]
                    model = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn), model, replace = (A_gcn, B_gcn))
                    print("===========================")
                    print("model after setting everything: ", model)
                    print("===========================")
                    print()
                    print("New feed ONLY!!!------------------")
                    #---------------Step 3b: Train ONLY A's and B's---------------------#
                    print("STEP 3B: Train ONLY on the A's and B's; fix W----------------")
                    preAgcn = model.A_gcn[0]
                    preAfeed = model.A_feed[0]
                    preWgcn = model.gcn_layers[0].weight
                    preWfeed = model.feed_layers[0].weight
                    og_epochs = 200
                    model1 = model
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn), filter_spec, replace=(True,True),)
                    diff_model, static_model = eqx.partition(model, filter_spec) #makes weights static and A's, B's params
                    import optax
                    optim2 = optax.adamw(1e-4) #set new optimizer
                    #==============put metric here for training of AB based on what loss in preAB was======#
                    a = 0
                    b=0
                    AB_convList =[]
                    diff_model, static_model, optim2, record_dict_AB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), diff_model,static_model,optim2, \
                                                                                            n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                                task_id=i, config={'batch_size': config['batch']},\
                                                                                                dictum=record_dict_AB, notABTrain = False) #complete the CL training
                    AB_dict = record_dict_AB[str(i)]
                    AB_loss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,10)])
                    print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)
                    AB_convList = []
                    prev_ab = AB_loss
                   #instead could do all based on end_last base threshold on ratio of trainWLoss and end_last
                    ratio = trainWLoss/end_last
                    if ratio >3.0:
                        threshold = 1/ratio
                    elif 2.0<= ratio<3.0:
                        #threshold = .5
                        threshold = 1/ratio
                    elif 1.0<=ratio<2.0:
                        threshold = .75
                        threshold = 1/ratio
                    else:
                        threshold = .9

                    print("Here is ratio: ", ratio, " --------- and threshold: ", threshold)

                    while (trainWLoss*threshold < AB_loss) and (a<15):
                        diff_model, static_model, optim2, record_dict_AB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), diff_model,static_model,optim2, \
                                                                                            n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                                task_id=i, config={'batch_size': config['batch']},\
                                                                                                dictum=record_dict_AB, notABTrain = False) #complete the CL training
                        AB_dict = record_dict_AB[str(i)]
                        AB_loss_curr = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,20)])
                        print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)
         
                        AB_convList.append(AB_loss_curr)
                       
                        if AB_loss_curr>AB_loss: #check if loss is increasing
                            #print("AB_loss is increasing, decrease epochs")
                            print("AB_convList[-2]: ", AB_convList[-2], "AB_loss_curr: ", AB_loss_curr)
                            if (AB_convList[-2]<AB_loss_curr):
                                print("training is increasing...break")
                                break
                            #prev_ab = AB_loss_curr
                            og_epochs = 100
                            AB_loss = AB_loss_curr
                            a+=1
                        else:
                            AB_loss = AB_loss_curr
                            a+=1
                            og_epochs = 200
                        #check if converging
                        if a>7:
                            conv_Diff = np.mean([AB_convList[-1],AB_convList[-2],AB_convList[-3]])
                            if jnp.abs(conv_Diff-AB_convList[-1]) <= 0.01:
                                print("AB_loss converged, breaking")
                                break

                    model = eqx.combine(diff_model,static_model) #recombine params and statics
                    print("heres i after AB train:", i)
                    #print("A_gcn after AB train (CHANGE): ", model.A_gcn[0])
                    #print("A_feed after AB train (CHANGE): ", model.A_feed[0])
                    #print("gcn weights after AB train (NO CHANGE): ", model.gcn_layers[0].weight)
                    #print("feed weights after AB Train (NO CHANGE): ", model.feed_layers[0].weight)
                    print("=========Check if A changed and W froze=============")
                    print("Did A gcn change (True): ", jnp.allclose(preAgcn, model.A_gcn[0]))
                    print("Did A feed change (False): ", jnp.allclose(preAfeed, model.A_feed[0]))
                    print("Did W gcn stay same (True): ",jnp.allclose(preWgcn, model.gcn_layers[0].weight))
                    print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                    print()
                    #=====================STEP 4: Set V = AWB for mlp========================#
                    print("--------------STEP 4: Set V = AWB for mlp-----------------")
                    for j in range(0,len(model.feed_layers)):
                        #print("A: ", model.A_feed[j].shape)
                        # print("W: ", model.feed_layers[j].weight.shape)
                        # print("W.T: ", model.feed_layers[j].weight.T.shape)
                        # print("B.T: ", jnp.transpose(model.B_feed[j]).shape)
                        Vw = (model.A_feed[j] @ model.feed_layers[j].weight.T @ jnp.transpose(model.B_feed[j])).T
                        #print("Vw shape: ", Vw.shape)
                        Vb = model.feed_layers[j].bias@ model.B_feed[j].T
                        #print("Vb shape: ", Vb.shape)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)
                    print("======================")
                    print("Model after setting new V weights: ", model)
                    print("======================")
                    #reset params and static
                    params, static = eqx.partition(model, eqx.is_array)
                    #print("here is params: ", params)
                    #print("here is static", static)
                    #reset A's and B's as statics again
                    static = eqx.tree_at(lambda x: x.A_gcn, static, replace= model.A_gcn)
                    static = eqx.tree_at(lambda x: x.B_gcn, static, replace= model.B_gcn)
                    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                    params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                    # print("here is params AFTER: ", params)
                    # print("here is static AFTER:", static)
                    print("STEP 5: Train CNN on new weights and record---------------")
                    optim3 = optax.adamw(1e-4)
                    optim1 = optim3
                    params, static, optim1, record_dict[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim1, \
                                                                                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                            task_id=i, config={'batch_size': config['batch']},\
                                                                                            dictum=record_dict) #complete the CL training
                    #recombine to model and the resplit so the model and params/statics are updated for next task
                    rec_dict = record_dict[str(i)]
                    end_last = np.mean([rec_dict["train"+str((i+1)*125-j)][0] for j in range(1,10)])
                elif (prev_feed_sizes == opt_MLParch) and (prev_gcn_size != opt_gcn):
                    model = eqx.combine(params,static) #put back model
                    model = eqx.tree_at(lambda x: x.gcn_sizes, model, opt_gcn) #set new gcn
                    initializer = jax.nn.initializers.glorot_uniform()
                    B_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[1:],model.gcn_sizes[1:])]
                    A_gcn = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(prev_gcn_size[:-1],model.gcn_sizes[:-1])]
                    model = eqx.tree_at(lambda x: (x.A_gcn, x.B_gcn), model, replace = (A_gcn, B_gcn))
                    print("===========================")
                    print("model after setting everything: ", model)
                    print("===========================")
                    print()
                    print("New gcn ONLY!!!------------------")
                    #---------------Step 3b: Train ONLY A's and B's---------------------#
                    print("STEP 3B: Train ONLY on the A's and B's; fix W----------------")
                    preAgcn = model.A_gcn[0]
                    preAfeed = model.A_feed[0]
                    preWgcn = model.gcn_layers[0].weight
                    preWfeed = model.feed_layers[0].weight
                    og_epochs = 200
                    model1 = model
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn), filter_spec, replace=(True,True),)
                    diff_model, static_model = eqx.partition(model, filter_spec) #makes weights static and A's, B's params
                    import optax
                    optim2 = optax.adamw(1e-4) #set new optimizer
                    #==============put metric here for training of AB based on what loss in preAB was======#
                    AB_loss = trainWLoss *1.5
                    a = 0
                    b=0
                    AB_convList =[]
                    while (trainWLoss*.9 < AB_loss) and (a<15):
                        diff_model, static_model, optim2, record_dict_AB[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), diff_model,static_model,optim2, \
                                                                                            n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                                task_id=i, config={'batch_size': config['batch']},\
                                                                                                dictum=record_dict_AB, notABTrain = False) #complete the CL training
                        AB_dict = record_dict_AB[str(i)]
                        AB_loss_curr = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,10)])
                        print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)
                        AB_convList.append(AB_loss_curr)
                       
                        if AB_loss_curr>AB_loss: #check if loss is increasing
                            #print("AB_loss is increasing, decrease epochs")
                            print("AB_convList[-2]: ", AB_convList[-2], "AB_loss_curr: ", AB_loss_curr)
                            if (AB_convList[-2]<AB_loss_curr):
                                print("training is increasing...break")
                                break
                            #prev_ab = AB_loss_curr
                            og_epochs = 100
                            AB_loss = AB_loss_curr
                            a+=1
                        else:
                            AB_loss = AB_loss_curr
                            a+=1
                            og_epochs = 200
                        #check if converging
                        if a>7:
                            conv_Diff = np.mean([AB_convList[-1],AB_convList[-2],AB_convList[-3]])
                            if jnp.abs(conv_Diff-AB_convList[-1]) <= 0.01:
                                print("AB_loss converged, breaking")
                                break
                    
                    model = eqx.combine(diff_model,static_model) #recombine params and statics
                    print("heres i after AB train:", i)
                    #print("A_gcn after AB train (CHANGE): ", model.A_gcn[0])
                    #print("A_feed after AB train (CHANGE): ", model.A_feed[0])
                    #print("gcn weights after AB train (NO CHANGE): ", model.gcn_layers[0].weight)
                    #print("feed weights after AB Train (NO CHANGE): ", model.feed_layers[0].weight)
                    print("=========Check if A changed and W froze=============")
                    print("Did A gcn change (False): ", jnp.allclose(preAgcn, model.A_gcn[0]))
                    print("Did A feed change (True): ", jnp.allclose(preAfeed, model.A_feed[0]))
                    print("Did W gcn stay same (True): ",jnp.allclose(preWgcn, model.gcn_layers[0].weight))
                    print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                    print()
                    #=====================STEP 4: Set V = AWB for gcn========================#
                    print("--------------STEP 4: Set V = AWB for gcn-----------------")
                    #print("here is model again: ", model)
                    #weights_list = [[(model.A_gcn[i]@(model.gcn_layers[i].weight)@jnp.transpose(model.B_gcn[i]))] for i in range(0,len(model.gcn_layers))]
                    #bias_list = [[(model.gcn_layers[i].bias@ model.B_gcn[i].T)] for i in range(0,len(model.gcn_layers))]
                    for k in range(len(model.gcn_layers)):
                        Vw = model.A_gcn[k]@(model.gcn_layers[k].weight)@jnp.transpose(model.B_gcn[k])
                        Vb = (model.gcn_layers[k].bias@ model.B_gcn[k].T)
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].weight, model, Vw)
                        model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, model, Vb)
                        #model = eqx.tree_at(lambda x: x.gcn_layers[i].weight, model, replace = weights_list[i])
                        #model = eqx.tree_at(lambda x: x.gcn_layers[i].bias, model, replace = bias_list[i])
                    #print("model after setting new gcn weights: ", model)
                    print("======================")
                    print("Model after setting new V weights: ", model)
                    print("======================")
                    #reset params and static
                    params, static = eqx.partition(model, eqx.is_array)
                    #print("here is params: ", params)
                    #print("here is static", static)
                    #reset A's and B's as statics again
                    static = eqx.tree_at(lambda x: x.A_gcn, static, replace= model.A_gcn)
                    static = eqx.tree_at(lambda x: x.B_gcn, static, replace= model.B_gcn)
                    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                    params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                    print("here is params AFTER: ", params)
                    print("here is static AFTER:", static)
                else:
                    #--------------Step 5: Train on V for all epochs---------------------------#
                    print("STEP 5: Train CNN on new weights and record---------------")
                    optim3 = optax.adamw(1e-4)
                    optim1 = optim3
                    params, static, optim1, record_dict[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim1, \
                                                                                        n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                            task_id=i, config={'batch_size': config['batch']},\
                                                                                            dictum=record_dict) #complete the CL training
                    #recombine to model and the resplit so the model and params/statics are updated for next task
                    rec_dict = record_dict[str(i)]
                    end_last = np.mean([rec_dict["train"+str((i+1)*125-j)][0] for j in range(1,10)])
            else:
            #--------------Step 5: Train on V for all epochs---------------------------#
                print("STEP 5: Train CNN on new weights and record---------------")
                #optim3 = optax.adamw(1e-4)
                #optim1 = optim3
                params, static, optim1, record_dict[str(i)] = trainer.train__CL__graph((mem_train_loader, test, train_loader), params,static,optim1, \
                                                                                    n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                        task_id=i, config={'batch_size': config['batch']},\
                                                                                        dictum=record_dict) #complete the CL training
                #recombine to model and the resplit so the model and params/statics are updated for next task
                rec_dict = record_dict[str(i)]
                end_last = np.mean([rec_dict["train"+str((i+1)*125-j)][0] for j in range(1,10)])
            model = eqx.combine(params, static)
            params, static = eqx.partition(model, eqx.is_array)
            #print("here is params: ", params)
            #print("here is static", static)
            static = eqx.tree_at(lambda x: x.A_gcn, static, replace= model.A_gcn)
            static = eqx.tree_at(lambda x: x.B_gcn, static, replace= model.B_gcn)
            static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
            static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
            params = eqx.tree_at(lambda x: (x.A_gcn,x.B_gcn,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
            #print("here is params AFTER: ", params)
            #print("here is static AFTER:", static)
    # print("Final GCN Architectures: ", gcn_arch_list)
    # print("Final MLP Architectures: ", mlp_arch_list)
    for w in range(0,len(gcn_arch_list)):
        print("TASK ", w+1, ": ", "-- GCN Arch: ", gcn_arch_list[w], " MLP Arch: ", mlp_arch_list[w])
    model = eqx.combine(params, static) #close/combine model back together
    eqx.tree_serialise_leaves(config['model_path']+'.eqx', model) #eqx.tree_seriealise_leaves saves the model to a given file.
    del model #delete model to free memory
    del params #delete params to free memory
    del static #delete static to free memory  
    return record_dict_preAB, record_dict_AB, record_dict #contains info from CL trainining


#===============Arch Search Function for MLP Architecture=======================#
def arch_search_MLP(original_arch, task, trainW_loss, og_epochs, config,dataloader_curr,\
                 dataloader_exp,test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for MLP
    ARGUMENTS:
    RETURNS: 
        opt_arch: (list) contains the best MLP architecture for the current (and prev) tasks
    """
    trainer1, optim, __, arch_model  = load_checkpoint(config)
    i = task
    x = original_arch[1]
    y = original_arch[2]
    og_epochs = 500
    #print("model before setting new size: ", arch_model)
    arch_model = eqx.tree_at(lambda x: x.sizes, arch_model, original_arch)
    initializer = jax.nn.initializers.glorot_uniform()
    weight_list = [initializer(jax.random.PRNGKey(i), (y, x)) for x,y,i in zip(arch_model.sizes[:-1],arch_model.sizes[1:], range(1,len(arch_model.sizes)))]
    bias_list = [initializer(jax.random.PRNGKey(i), (1, y)) for y,i in zip(arch_model.sizes[1:], range(1,len(arch_model.sizes)))]
    #bias_list = [jax.random.truncated_normal(jax.random.PRNGKey(1),-(np.sqrt(1.0/((y+1)/2))/.87962566103423978),(np.sqrt(1.0/((y+1)/2))/.87962566103423978),shape = (1,y)) for y in arch_model.sizes[1:]]
    #weight_list = [jax.random.truncated_normal(jax.random.PRNGKey(1),-(np.sqrt(1.0/((y+x)/2))/.87962566103423978),(np.sqrt(1.0/((y+x)/2))/.87962566103423978),shape = (y,x)) for x,y in zip(arch_model.sizes[:-1],arch_model.sizes[1:])]
    for j in range(len(arch_model.sizes)-1):
        arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
        arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])
    arch_params, arch_static = eqx.partition(arch_model,eqx.is_array)
    arch_static = eqx.tree_at(lambda x: x.A, arch_static, replace= arch_model.A)
    arch_static = eqx.tree_at(lambda x: x.B, arch_static, replace= arch_model.B)
    arch_params = eqx.tree_at(lambda x: (x.A,x.B), arch_params, replace= (None,None))
    #print("model after resetting sizes and weights: ", arch_model)
    s = arch_model.sizes
    original_arch = arch_model.sizes
    poll_dict = {}
    arch_params, arch_static, optim, poll_dict[str(i)] =  trainer1.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                    (test_loader_curr, test_loader_exp)),arch_params, arch_static, optim, \
                                                                    n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                    task_id=i, config={
                                                                        'batch_size': 64,
                                                                        'opt': 'Nash',
                                                                        'problem': config['problem'],
                                                                        'data_id': config['data'],
                                                                        "flag": config['flag'],
                                                                        'len_exp_replay': 20000,
                                                                        'network': config['network'],
                                                                        }, dictum=poll_dict)
    arch_model = eqx.combine(arch_params, arch_static)
    #more search------------------------
    arch_dict = poll_dict[str(i)]
    loss_orig2 = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,26)])
    
    smallest = [trainW_loss, loss_orig2]
    print("loss list: ", smallest)
    loss_orig = smallest[np.argmin(smallest)]

    loss_orig = loss_orig2
    threshold = .6
    loss = loss_orig
    step = 1
    x = original_arch[1]
    y = original_arch[2]
    opt_loss = loss_orig
    opt_arch = arch_model.sizes
    curr_arch = opt_arch
    k = 0
    while(opt_loss>=loss_orig*threshold) and (k<2):
        for n in range(0,5):
            for j in range(0,5):
                curr_arch = [3,x+15*n,y+15*j,10]
                arch_model = eqx.tree_at(lambda x: x.sizes, arch_model, original_arch)
                initializer = jax.nn.initializers.glorot_uniform()
                weight_list = [initializer(jax.random.PRNGKey(l), (y, x)) for x,y,l in zip(arch_model.sizes[:-1],arch_model.sizes[1:], range(1,len(arch_model.sizes)))]
                bias_list = [initializer(jax.random.PRNGKey(l), (1, y)) for y,l in zip(arch_model.sizes[1:], range(1,len(arch_model.sizes)))]
                for j in range(len(arch_model.sizes)-1):
                    arch_model = eqx.tree_at(lambda x: x.layers[j].weight, arch_model, weight_list[j])
                    arch_model = eqx.tree_at(lambda x: x.layers[j].bias, arch_model, bias_list[j])
                arch_params, arch_static = eqx.partition(arch_model,eqx.is_array)
                #print("params after setting weights: ", params)
                #print(static)
                #print()
                poll_dict = {}
                arch_static = eqx.tree_at(lambda x: x.A, arch_static, replace= arch_model.A)
                arch_static = eqx.tree_at(lambda x: x.B, arch_static, replace= arch_model.B)
                arch_params = eqx.tree_at(lambda x: (x.A,x.B), arch_params, replace= (None,None))
                optim = optax.adam(1e-3)
                og_epochs = 500
                arch_params, arch_static, optim, poll_dict[str(i)] =  trainer1.train__CL__reg((dataloader_curr,\
                                                    dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                    (test_loader_curr, test_loader_exp)),arch_params, arch_static, optim, \
                                                    n_iter=og_epochs, save_iter=config['save_iter'],\
                                                    task_id=i, config={
                                                    'batch_size': 64,
                                                    'opt': 'Nash',
                                                    'problem': config['problem'],
                                                    'data_id': config['data'],
                                                    "flag": config['flag'],
                                                    'len_exp_replay': 20000,
                                                    'network': config['network'],
                                                    }, dictum=poll_dict)
                poll_dict1 = poll_dict[str(i)]
                poll_loss = np.mean([poll_dict1["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                print("curr arch: ", curr_arch, "--------- curr loss: ", poll_loss, "--------- opt loss: ", opt_loss)
                if poll_loss<opt_loss:
                    opt_loss = poll_loss
                    opt_arch = curr_arch
                print("opt arch for j round: ", opt_arch)
        if (opt_arch[1] == original_arch[1] and opt_arch[2] ==original_arch[2]):
            x = x+250
            y = y+250
        else:
            x = opt_arch[1]
            y = opt_arch[2]
        k+=1
    return opt_arch
        


#-------------TRAINING REGRESSION PROBLEM------------------#
def train_model_reg(config):
    """
    GOAL: construct and CL train a model which is for a regression problem

    ARGUMENTS:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)

    RETURNS:
        record_dict: dictionary 
    """
    trainer, optim, data, model  = load_checkpoint(config) #Get trainer, optim, data, model
    #RECALL: 'trainer' is a 'Trainer()' object from 'utils.trainer.py.'
    #        'data' is a 'data_return()' object from 'utils.data.py'
    #        'model' is a NN model (i.e. MLP, CNN, etc.) from 'utils.model.py'
    params, static = eqx.partition(model, eqx.is_array) #separate the params and static of the model
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}
    static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
    static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
    params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
    same_arch = True
    mlp_arch_list = []

    for i in range(config['n_task']): #for loop running for total number of tasks prescribed
        print("task--", i) #print what task we are on      
        if i==0: #On the first task, do...
            dataloader_curr, dataloader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'dataloader_curr' and 'dataloader_exp' are DataLoader() objects containing data for task 'i'. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.
            
            test_loader_curr, test_loader_exp= data.generate_dataset(task_id=i,batch_size=config['batch_size'], phase='testing')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'test_loader_curr' and 'test_loader_exp' are DataLoader() objects containing data for task 'i'. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.

            params, static, optim, record_dict[str(i)] =  trainer.train__CL__reg((dataloader_curr,\
            dataloader_exp, (test_loader_curr, test_loader_exp), (test_loader_curr, test_loader_exp)),\
            params, static, optim,  n_iter=config['epochs_per_task'],\
            save_iter=config['save_iter'], task_id=i, config={
                    'batch_size': 64,
                    'opt': 'Nash',
                    'problem': config['problem'],
                    'data_id': config['data'],
                    "flag": config['flag'],
                    'len_exp_replay': 20000,
                    'network': config['network'],
                    }, dictum=record_dict)
            optim1 = optim
            rec_dict = record_dict[str(i)]
            rec_epochs = config['epochs_per_task']
            end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
            print("end_last after task 0: ", end_last)
            end_last0 = end_last
            
        else:
            dataloader_curr, dataloader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'dataloader_curr' and 'dataloader_exp' are DataLoader() objectscontaining data for task 'i'. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.
            
            test_loader_curr, test_loader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='testing')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'test_loader_curr' and 'test_loader_exp' are DataLoader() objects containing data for task 'i'. This means they are iterables where each 
            #iterate is a batch of the data in the corresponding datasets.

            #------------------------------------OG CL train on first task i = 0 for some sub number of epochs-----------------------------#
            og_epochs = 250
            print("STEP 1: We train for ", og_epochs, " epochs on the next task")
            params, static, optim1, record_dict_preAB[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                  (test_loader_curr, test_loader_exp)),params, static, optim1, \
                                                                                 n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                 task_id=i, config={
                                                                                    'batch_size': 64,
                                                                                    'opt': 'Nash',
                                                                                    'problem': config['problem'],
                                                                                    'data_id': config['data'],
                                                                                    "flag": config['flag'],
                                                                                    'len_exp_replay': 20000,
                                                                                    'network': config['network'],
                                                                                    }, dictum=record_dict_preAB) #CL training regression problem

            model = eqx.combine(params, static)
            arch_dict = record_dict_preAB[str(i)]
            trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,11)])
            #print("trainWLoss: ", trainWLoss)
            #-------------------------------------Get new architecture?------------------------------------------#
            print("STEP 2: Search for Best architecture for the data in task? " , i)
            ratio2 = trainWLoss/end_last0
            print()
            print("ratio2: ", ratio2)
            print("Here's end_last: ", end_last, "Here's trainWLoss: ", trainWLoss)
            print()
            original_arch = model.sizes
            mlp_weight_layer = []
            mlp_bias_layer = []
            for j in range(len(model.layers)):
                mlp_weight_layer.append(model.layers[j].weight)
                mlp_bias_layer.append(model.layers[j].bias)
            preAfeed=model.A[0]
            preWfeed=model.layers[0].weight

            #+++++++++++First see whether we need to change the architecture or not++++++++++++++#
            

            if (trainWLoss/end_last0>.45) and (end_last+.01<=trainWLoss):
                change_arch = True

            if (trainWLoss/end_last0>.45) and (end_last+.01>trainWLoss):
                change_arch = False

            if (trainWLoss/end_last0<=.45):
                change_arch = False

            if change_arch ==True:
                print("WE ARE CHANGING ARCHITECTURE!!!")
                opt_arch = arch_search_MLP(original_arch,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
                print("NEW Architecture: ", opt_arch)
                mlp_arch_list.append(opt_arch)
                model = eqx.combine(params,static) #put back model
                for j in range(0,len(model.layers)):
                    cw = mlp_weight_layer[j]
                    cb = mlp_bias_layer[j]
                    model = eqx.tree_at(lambda x: x.layers[j].weight, model, cw)
                    model = eqx.tree_at(lambda x: x.layers[j].bias, model, cb)
                print("Did A feed change (True): ", jnp.allclose(preAfeed, model.A[0]))
                print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.layers[0].weight))
                print()
            else:
                opt_arch = original_arch
                mlp_arch_list.append(opt_arch)
                print("ARCHITECTURE Did NOT change")

            #++++++++++++++++++++++++++++++If arch changed, then we set new A and B's and train them++++++++++++++++++++++++++++++#
            if opt_arch != original_arch:
                #---------------------------------------Set New Arch and Set/Prep A and B to proper sizes----------------------------------#
                print("STEP 3a: Set new Architecture and set/prep A and B to proper sizes")
                #original_arch = model.sizes
                s = original_arch
                #opt_arch = [3,385+75*i,385+50*i,10]
                model = eqx.tree_at(lambda x: x.sizes, model, opt_arch)
                initializer = jax.nn.initializers.glorot_uniform()
                #A_list = [initializer(jax.random.PRNGKey(i), (y, x)) for x,y,i in zip(s[1:],model.sizes[1:], range(1,len(model.sizes)))]
                #B_list = [initializer(jax.random.PRNGKey(i), (y, x)) for x,y,i in zip(s[:-1],model.sizes[:-1], range(1,len(model.sizes)))]
                A_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[1:],model.sizes[1:])]
                B_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[:-1],model.sizes[:-1])]
                model = eqx.tree_at(lambda x: x.A, model, A_list)
                model = eqx.tree_at(lambda x: x.B, model, B_list)
                

                #--------------------------------A,B Training Loop--------------------------------------------------------#

                og_epochs = 2000
                print("STEP 3b: Train A and B fix W------------------------------")
                model1 = model
                preAfeed = model.A[0]
                preWfeed = model.layers[0].weight
                filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                filter_spec = eqx.tree_at(lambda x: (x.A,x.B), filter_spec, replace=(True,True),)
                #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)
                diff_model, static_model = eqx.partition(model, filter_spec)
                import optax
                #optim = optax.adamw(config['lr'])
                #opt= optax.adamw(1e-4)
                #opt_state = opt.init(eqx.filter(model,eqx.is_array))
                optim2 = optax.adam(1e-4)
                diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                    (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
                                                                                    n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={
                                                                                        'batch_size': 64,
                                                                                        'opt': 'Nash',
                                                                                        'problem': config['problem'],
                                                                                        'data_id': config['data'],
                                                                                        "flag": config['flag'],
                                                                                        'len_exp_replay': 20000,
                                                                                        'network': config['network'],
                                                                                        }, dictum=record_dict_AB, notABTrain = False) #CL training regression problem
                AB_dict = record_dict_AB[str(i)]
                a=0
                AB_loss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss)
                prev_ab = AB_loss

                task_ratio = 1.0-(float(i)/float(config['n_task']))

                ratio = trainWLoss/end_last
                if ratio >3.0:
                    threshold = 1/ratio
                    threshold = jnp.max(jnp.array([threshold, .45]), axis = None)
                elif 2.0<= ratio<3.0:
                    #threshold = .5
                    threshold = 1/ratio
                    threshold = jnp.min(jnp.array([threshold, .6]),axis = None)
                elif 1.0<=ratio<2.0:
                    threshold = 1/ratio
                    threshold = jnp.min(jnp.array([threshold, .75]), axis = None)
                else:
                    threshold = .8
                #threshold = threshold*task_ratio

                print("Here is ratio: ", ratio, " --------- and threshold: ", threshold, "------- Goal: ", trainWLoss*threshold)
                print()
                AB_convList = []
                AB_convList.append(AB_loss)
                while (trainWLoss*threshold < AB_loss) and (a<8):
                    ratio1 = AB_loss/trainWLoss
                    if ratio1 >3.0:
                        og_epochs = 8000
                    if 2.0<= ratio1<3.0:
                        og_epochs = 4000
                    else:
                        og_epochs = 2000
                    diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                    (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
                                                                                    n_iter=og_epochs, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={
                                                                                        'batch_size': 64,
                                                                                        'opt': 'Nash',
                                                                                        'problem': config['problem'],
                                                                                        'data_id': config['data'],
                                                                                        "flag": config['flag'],
                                                                                        'len_exp_replay': 20000,
                                                                                        'network': config['network'],
                                                                                        }, dictum=record_dict_AB, notABTrain = False) #CL training regression problem
                    AB_dict = record_dict_AB[str(i)]
                   
                    AB_loss_curr = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,20)])
                    print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)

                    AB_convList.append(AB_loss_curr)
                    print("AB_convList: ", AB_convList)
                    AB_loss = AB_loss_curr
                    a+=1
                  
                #print()
                model = eqx.combine(diff_model,static_model)
                print("heres i after AB train:", i)
                print("=========Check if A changed and W froze=============")
                print("Did A feed change (False): ", jnp.allclose(preAfeed, model.A[0]))
                print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.layers[0].weight))
                print()

                #-----------------------Set new V = AWB^T-----------------------------#
                print("STEP 4: Set the new weights V = AWB^T")
                #print("-------------------------------------")
                for j in range(len(model.sizes)-1):
                    Vw = model.A[j] @ model.layers[j].weight @ jnp.transpose(model.B[j])
                    #Vb = model.A[i] @ model.layers[i].bias
                    #print("shape of bias:", model.layers[i].bias.shape)
                    #print("shape of A: ", model.A[i].shape)
                    Vb = model.layers[j].bias@ model.A[j].T
                    model = eqx.tree_at(lambda x: x.layers[j].weight, model, Vw)
                    model = eqx.tree_at(lambda x: x.layers[j].bias, model, Vb)
                #print()
                #print()
                #print("MODEL AFTER SETTING V: ", model)
                params, static = eqx.partition(model, eqx.is_array)
                #print("AFTER V set check params: ", params)
                #print("After V set check static: ", static)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                #print("----------------------")
                # print("PARAMS AFTER V SET: ", params)
                # print("STATIC AFTER V SET: ", static)
                #print("weights size after setting V: ", jnp.shape(model.layers[0].weight))
                #print("WEIGHTS AFTER SETTING V: ", model.layers[0].weight)
                #print("A BEFORE TRAIN V: ", model.A[0])

                print("STEP 5: Train the model with weights V for full amount of epochs")
                import optax
                optim3 = optax.adam(1e-3)
                #optim3 = optim1
                record_dict_dummy = {}
                
                params, static, optim3, record_dict_dummy[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                    (test_loader_curr, test_loader_exp)),params, static, optim3, \
                                                                                    n_iter=50, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={
                                                                                        'batch_size': 64,
                                                                                        'opt': 'Nash',
                                                                                        'problem': config['problem'],
                                                                                        'data_id': config['data'],
                                                                                        "flag": config['flag'],
                                                                                        'len_exp_replay': 20000,
                                                                                        'network': config['network'],
                                                                                        }, dictum=record_dict_dummy)
                
                params, static, optim3, record_dict[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                  (test_loader_curr, test_loader_exp)),params, static, optim3, \
                                                                                 n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                 task_id=i, config={
                                                                                    'batch_size': 64,
                                                                                    'opt': 'Nash',
                                                                                    'problem': config['problem'],
                                                                                    'data_id': config['data'],
                                                                                    "flag": config['flag'],
                                                                                    'len_exp_replay': 20000,
                                                                                    'network': config['network'],
                                                                                    }, dictum=record_dict) #CL training regression problem
                #print("END OF TASK ", i)
                #print("After training V params: ", params)
                #print("After training V statics: ", static)
                #model = eqx.combine(params,static)
                #print("WEIGHTS AFTER Training V: ", model.layers[0].weight)
                #print("A After Training V: ", model.A[0])
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                # print("After re-split params: ", params)
                # print("After re-split statics: ", static)
                same_arch = False
                optim1 = optim3
                rec_dict = record_dict[str(i)]
                end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
                # print("end_last after training V: ", end_last)
                
            else: #+++++++++++++++If arch did NOT change, then just train V normally++++++++++++++++++++++#
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                record_dict_dummy = {}
                
                params, static, optim1, record_dict_dummy[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                    (test_loader_curr, test_loader_exp)),params, static, optim1, \
                                                                                    n_iter=50, save_iter=config['save_iter'],\
                                                                                    task_id=i, config={
                                                                                        'batch_size': 64,
                                                                                        'opt': 'Nash',
                                                                                        'problem': config['problem'],
                                                                                        'data_id': config['data'],
                                                                                        "flag": config['flag'],
                                                                                        'len_exp_replay': 20000,
                                                                                        'network': config['network'],
                                                                                        }, dictum=record_dict_dummy)
                
                params, static, optim1, record_dict[str(i)] =  trainer.train__CL__reg((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                  (test_loader_curr, test_loader_exp)),params, static, optim1, \
                                                                                 n_iter=config['epochs_per_task'], save_iter=config['save_iter'],\
                                                                                 task_id=i, config={
                                                                                    'batch_size': 64,
                                                                                    'opt': 'Nash',
                                                                                    'problem': config['problem'],
                                                                                    'data_id': config['data'],
                                                                                    "flag": config['flag'],
                                                                                    'len_exp_replay': 20000,
                                                                                    'network': config['network'],
                                                                                    }, dictum=record_dict) #CL training regression problem
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                # print("After re-split params: ", params)
                # print("After re-split statics: ", static)
                rec_dict = record_dict[str(i)]
                end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
                # print("end_last after training V: ", end_last)
                # print("end_last NO Arch change: ", end_last)
                same_arch = True
            #print()
            
        data.append_to_experience(i) #this is a method from data.utils.py adds data to be later retrained on?
    # for w in range(0,len(mlp_arch_list)):
    #     print("TASK ", w+1, "-------- MLP Arch: ", mlp_arch_list[w])
    rec_dict = record_dict[str(i)]
    end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
    model = eqx.combine(params, static) #close/combine model back together
    eqx.tree_serialise_leaves(config['model_path']+'.eqx', model) #eqx.tree_seriealise_leaves saves the model to a given file.
    del model #delete model to free memory
    del params #delete params to free memory
    del static #delete static to free memory  
    return record_dict_preAB, record_dict_AB,record_dict #dictionary containing info from CL trainining. In particular, '(V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)' for each task and 'train', 'test', etc.

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



#-------------TRAINING CLASSIFICATION PROBLEM------------------#
def train_model_class(config):
    """
    GOAL: construct and CL train a model which is for a classification problem

    ARGUMENTS:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)

    RETURNS:
        dict: dictionary 
    """
    trainer, optim, data, model  = load_checkpoint(config)
    print("model before training: ", model)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}
    print(model.conv_layers[0].weight[0][0])
    #reintialize A's and B's to static (they are put into params when we use eqx.partition(eqx.is_array))
    static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
    static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
    params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
    filter_arch_list = []
    mlp_arch_list = []
    rec_epochs = config['epochs_per_task']

    for i in range(config['n_task']):
        print("task--", i)
        # complete standard training for task i=0, no arch search or transfering to new arch       
        if i==0:
            dataloader_curr, _= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'dataloader_curr' and '_' are DataLoader() objects containing data for task 'i'. This means they are iterables where each iterate
            #is a batch of the data in the corresponding datasets.
            
            test_loader_curr, _= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='testing')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'test_loader_curr' and '_' are DataLoader() objects containing data for task 'i'. This means they are iterables where each iterate 
            #is a batch of the data in the corresponding datasets.
            """
            NOTES on 'train__CL__class': Only difference between 'train__CL__class' and 'train__CL__reg' is the function in 'return_Hamiltonian_class'
            The function does a softmax_cross_entropy on predicted values rather than l2. Rest is the same. See other file for more extensive notes.
            """
            #og_epochs = 50
            og_epochs = config['epochs_per_task']
            params, static, optim, record_dict[str(i)] = trainer.train__CL__class( (dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),\
                                                    (test_loader_curr, test_loader_curr)), params, static, optim, n_iter=og_epochs, \
                                                     save_iter=config['save_iter'], task_id=i,config={
                                                        'batch_size': config['batch_size'],
                                                        'opt': 'Nash',
                                                        'problem': config['prob'],
                                                        'data_id': config['data'],
                                                        'len_exp_replay': 20000,
                                                        "flag": config['flag'],
                                                        'network': config['network'],
                                                        }, dictum = record_dict)
            optim1 = optim
            #print("this is the dict:" , record_dict[str(i)])
            rec_dict = record_dict[str(i)]
            end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
            end_last0 = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
            arch_round = 0

        # For tasks i = 1 and beyond, search for opt arch and then transfer to new architecture if found
        else:
            dataloader_curr, dataloader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'dataloader_curr' and '_' are DataLoader() objects containing data for task 'i'. This means they are iterables where each iterate is a 
            #batch of the data in the corresponding datasets.
            
            test_loader_curr, test_loader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='testing')
            #RECALL: 'data' is a 'data_return()' object from utils.data.py, and 'generate_dataset' is a method in that class. It returns
            #'test_loader_curr' and '_' are DataLoader() objects containing data for task 'i'. This means they are iterables where each iterate is a 
            #batch of the data in the corresponding datasets.

            
            #----------------------Step 1: train W for a little bit on new task----------------------#
            print("STEP 1: Train W for a little bit on new task-------------")
            import optax
            lr = .0001
            if i<3:
                lr = 1e-4
            elif 3<=i<=7:
                lr = lr - 1e-5*(i-3)#set learning rate
            else:
                lr = 7e-5
            optim1 = optax.adamw(1e-4)
            og_epochs = 75 # change this to 100 and run again
            params, static, optim1, record_dict_preAB[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                           (test_loader_curr, test_loader_exp)),params, static, optim1, \
                                                                          n_iter=og_epochs, save_iter=config['save_iter'], \
                                                                          task_id=i,config={
                                                                            'batch_size': 20,
                                                                            'opt': 'Nash',
                                                                            'problem': config['prob'],
                                                                            'data_id': config['data'],
                                                                            'len_exp_replay': 20000,
                                                                            "flag": config['flag'],
                                                                            'network': config['network'],
                                                                            }, dictum = record_dict_preAB)
            print()
            arch_dict = record_dict_preAB[str(i)]
            trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,11)])
            print("original loss for preAB to compare: ", trainWLoss)
            #-----------------Step 2: Search for new architecture (filter size and MLP size)------------#
            print("STEP 2: Find new architecture (arch search in future)----------")
            #We will incorporate search later, set new arch for now
            prev_feed_sizes = model.feed_sizes
            prev_filter_size = model.filter_size
            print("Here's end_last: ", end_last, "Here's trainWLoss: ", trainWLoss)
            filter_weight_layer = []
            filter_bias_layer = []
            mlp_weight_layer = []
            mlp_bias_layer = []
            for j in range(len(model.feed_layers)):
                mlp_weight_layer.append(model.feed_layers[j].weight)
                mlp_bias_layer.append(model.feed_layers[j].bias)
            for j in range(len(model.conv_layers)):
                filter_weight_layer.append(model.conv_layers[j].weight)
                #filter_bias_layer.append(model.gcn_layers[j].bias)
            preAconv= model.A_conv[0]
            preAfeed=model.A_feed[0]
            preWgcn=model.conv_layers[0].weight
            preWfeed=model.feed_layers[0].weight

            if i>1:
                #end_last = end_100last
                # print("end last: ", end_last)
                # print("end first 100: ", end_100last)
                scientific_notation = "{:e}".format(end_100last) 

                # Extract the exponent part of the scientific notation
                # For example, for 0.004, it would be '4.000000e-03' and we need '-03'
                exponent_part = scientific_notation.split('e')[-1]

                # Convert the exponent string to an integer
                exponent = int(exponent_part) 
                # print("exponent of end_100last: ", exponent)
                # print("10**exponent: ", 10**exponent, "actual end_100last: ", end_100last)
                # print("compare: ", end_100last+3*(10**(exponent)), "with trainWLoss: ", trainWLoss)
                if (end_100last+3*10*(10**(exponent))<=trainWLoss):
                #if (end_100last+10**exponent<=trainWLoss):
                    change_arch = True
                else:
                    change_arch = False

                scientific_notation = "{:e}".format(end_last) 
            
                exponent_part = scientific_notation.split('e')[-1]
                exponent = int(exponent_part)
                # print("exponent of end_last: ", exponent)
                # print("10**exponent: ", 10**exponent, "actual end_last: ", end_last)
                # if 10**exponent<= 10**-6:
                #     change_arch = False

            if change_arch ==True:
                print("WE ARE CHANGING ARCHITECTURE!!!")
              
                opt_MLParch, opt_filter = arch_search_CNN(prev_filter_size,prev_feed_sizes,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
                # print("NEW  FILTER Architecture: ", opt_filter)
                # print("NEW MLP Arch: ", opt_MLParch)

                filter_arch_list.append(opt_filter)
                mlp_arch_list.append(opt_MLParch)
                model = eqx.combine(params,static) #put back model
                for k in range(0,len(model.conv_layers)):
                        aw = filter_weight_layer[k]
                        #ab = gcn_bias_layer[k]
                        model = eqx.tree_at(lambda x: x.conv_layers[k].weight, model, aw)
                        #model = eqx.tree_at(lambda x: x.gcn_layers[k].bias, model, ab)
                for j in range(0,len(model.feed_layers)):
                    cw = mlp_weight_layer[j]
                    cb = mlp_bias_layer[j]
                    model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, cw)
                    model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, cb)
                print("Did A gcn change (True): ", jnp.allclose(preAconv, model.A_conv[0]))
                print("Did A feed change (True): ", jnp.allclose(preAfeed, model.A_feed[0]))
                print("Did W gcn stay same (True): ",jnp.allclose(preWgcn, model.conv_layers[0].weight))
                print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                print()
            else:
                opt_filter = prev_filter_size
                opt_MLParch = prev_feed_sizes
                filter_arch_list.append(opt_filter)
                mlp_arch_list.append(opt_MLParch)
                print("ARCHITECTURE Did NOT change")
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



            #if a different architecture was chosen then use our method to transfer to new arch
            if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) or (opt_filter !=prev_filter_size):
                #----------------Step 3a: Set New Architecture and Reinitialize A's and B's accordingly------#
                print("STEP 3A: Set new architecture and A's, B's----------------")
                conv_output_size = model.calc_output_size(opt_filter)
                maxpool_output_size = model.pool_output_size(2,conv_output_size)
                #set MLP input layer to correct size correspongding to new filter size output for Convnet layer
                opt_MLParch[0] = maxpool_output_size*maxpool_output_size*model.channel_out
                print("this is the new spot 1 after!!!:==============", opt_MLParch[0])
                model = eqx.combine(params,static) #put back model
                model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_MLParch) #set new arch
                model = eqx.tree_at(lambda x: x.filter_size, model, opt_filter) #set new filter
                #initialize and set A's and B's according to optimal architecture found
                initializer = jax.nn.initializers.glorot_uniform()
                rand_list = []
                for k in range(0,len(prev_feed_sizes)-1):
                    rand_list.append(k)
                if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter !=prev_filter_size):
                    print("New feed AND conv!!!------------------")
                    A_feed = [initializer(jax.random.PRNGKey(j), (y, x)) for x,y,j in zip(prev_feed_sizes[1:],opt_MLParch[1:],rand_list[:])]
                    B_feed = [initializer(jax.random.PRNGKey(j), (y, x)) for x,y,j in zip(prev_feed_sizes[:-1],opt_MLParch[:-1],rand_list[:])]
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
                model = eqx.tree_at(lambda x: (x.A_feed, x.B_feed, x.A_conv, x.B_conv), model, replace = (A_feed, B_feed, A_conv, B_conv))
                print("===========================")
                print("model after setting everything: ", model)
                print("===========================")
                print()
                #---------------Step 3b: Train ONLY A's and B's---------------------#
                print("STEP 3B: Train ONLY on the A's and B's; fix W----------------")
                
              
                og_epochs = 50
                model1 = model
                if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter !=prev_filter_size):
                    print("New feed AND conv!!!------------------")
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_conv,x.B_conv, x.A_feed, x.B_feed), filter_spec, replace=(True,True,True, True),)
                    #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)
                elif(prev_feed_sizes[1:3] != opt_MLParch[1:3]) and (opt_filter ==prev_filter_size):
                    print("New FEED ONLY!!!------------------")
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_feed, x.B_feed), filter_spec, replace=(True,True),)
                    #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)
                else:
                    print("New CONV ONLY!!!------------------")
                    filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                    filter_spec = eqx.tree_at(lambda x: (x.A_conv,x.B_conv), filter_spec, replace=(True,True),)
                    #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)

                preAfilter = model.A_conv[0]
                preAfeed = model.A_feed[0]
                preWfilter = model.conv_layers[0].weight
                preWfeed = model.feed_layers[0].weight
                #print("weights before", model.conv_layers[0].weight)
                #print("weights before:", model.feed_layers[0].weight)

                diff_model, static_model = eqx.partition(model, filter_spec) #makes weights static and A's, B's params
                #print("MAKE AB Params diff_model: ", diff_model)
                #print("MAKE Weights Static static_model: ", static_model)
                #print()
                import optax
                lr1 = .0001
                if i<3:
                    lr1 = 1e-4
                else:
                    #lr = lr - 1e-5*(i-3)#set learning rate
                    lr1 = 9e-5
                print("This is the learning rate: ", lr1)
                optim2 = optax.adamw(1e-4) #set new optimizer
              
                #Train A's and B's only; weights are frozen
                diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
                                                                                n_iter=og_epochs, save_iter=config['save_iter'], \
                                                                                task_id=i,config={
                                                                                    'batch_size': 20,
                                                                                    'opt': 'Nash',
                                                                                    'problem': config['prob'],
                                                                                    'data_id': config['data'],
                                                                                    'len_exp_replay': 20000,
                                                                                    "flag": config['flag'],
                                                                                    'network': config['network'],
                                                                                    },dictum = record_dict_AB, notABTrain=False)
                AB_dict = record_dict_AB[str(i)]
                AB_loss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,21)])
                prevABLoss = AB_loss
                a=1
               
                round_1_thresh = .15*AB_loss
                if AB_loss>1.5:
                    round_1_thresh = (.1*AB_loss)
                else:
                    round_1_thresh = (.15*AB_loss)
                if i>5:
                    round_1_thresh = (.05*AB_loss)
                threshold = jnp.max(jnp.array([trainWLoss,round_1_thresh]), axis = None)
                print("here is train Wloss: ", trainWLoss, "Here is round_1_thresh: ", round_1_thresh, "here is what we chose:", threshold)
                
                #print("Here is ratio: ", ratio, " --------- and threshold: ", threshold, "goal: ", trainWLoss*threshold)
              
                og_epochs = 100
                AB_convList = []
                AB_convList.append(AB_loss)
                while (threshold < AB_loss) and (a<6):
                    diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                                (test_loader_curr, test_loader_exp)),diff_model, static_model, optim2, \
                                                                                n_iter=og_epochs, save_iter=config['save_iter'], \
                                                                                task_id=i,config={
                                                                                    'batch_size': 20,
                                                                                    'opt': 'Nash',
                                                                                    'problem': config['prob'],
                                                                                    'data_id': config['data'],
                                                                                    'len_exp_replay': 20000,
                                                                                    "flag": config['flag'],
                                                                                    'network': config['network'],
                                                                                    },dictum = record_dict_AB, notABTrain=False)
                    AB_dict = record_dict_AB[str(i)]
                   
                    AB_loss_curr = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,20)])
                    print("AB ROUND", a+1,": This is trainWLoss: ", trainWLoss, "===== This is AB_loss after og_epochs: ", AB_loss_curr)
                    diff = AB_loss_curr - trainWLoss
                    AB_convList.append(AB_loss_curr)
                    print("AB_convList: ", AB_convList)
                    if AB_loss_curr<1.5*threshold:
                        og_epochs = 100
                    elif (AB_loss_curr>=1.5*threshold) and (i>1):
                        og_epochs = 800
                    else:
                        og_epochs = 200
                    if a>5:
                        if (AB_convList[-1] > AB_convList[-2]) and (AB_convList[-2] > AB_convList[-3]):
                            print("AB_loss is increasing, breaking")
                            break
                    AB_loss = AB_loss_curr
                    a+=1
                    
                model = eqx.combine(diff_model,static_model) #recombine params and statics
                print("weights after", model.conv_layers[0].weight)
                print("weights after:", model.feed_layers[0].weight)
                print("Did A conv change (False): ", jnp.allclose(preAfilter, model.A_conv[0]))
                print("Did A feed change (False): ", jnp.allclose(preAfeed, model.A_feed[0]))
                print("Did W conv stay same (True): ",jnp.allclose(preWfilter, model.conv_layers[0].weight))
                print("Did W feed stay same (True): ", jnp.allclose(preWfeed, model.feed_layers[0].weight))
                print()
                #--------------Step 4: Set V = AWB^T for ConvNet and MLP--------------------#
                print("STEP 4: Set new V = AWB^T weights------------------")
                #set V for convolutional layer
                if (opt_filter !=prev_filter_size):
                    weights_list = [[(model.A_conv[k]@(model.conv_layers[0].weight[i][0])@jnp.transpose(model.B_conv[k]))] for k in range(0,model.channel_out)]
                    weights_list = jnp.array(weights_list)
                    model = eqx.tree_at(lambda x: x.conv_layers[0].weight, model, replace = weights_list)
                #set MLP layers
                if (prev_feed_sizes[1:3] != opt_MLParch[1:3]):
                    for j in range(0,len(model.feed_layers)):
                        Vw = model.A_feed[j] @ model.feed_layers[j].weight @ jnp.transpose(model.B_feed[j])
                        #print("Vw shape: ", Vw.shape)
                        Vb = (model.A_feed[j]@model.feed_layers[j].bias)
                        #print("Vb shape: ", Vb.shape)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].weight, model, Vw)
                        model = eqx.tree_at(lambda x: x.feed_layers[j].bias, model, Vb)
                    #print("MODEL AFTER V: ", model)

                #reset params and static
                params, static = eqx.partition(model, eqx.is_array)
                #print("here is params: ", params)
                #print("here is static", static)
                #reset A's and B's as statics again
                static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
                static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
                static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                #print("here is params AFTER: ", params)
                #print("here is static AFTER:", static)
                #--------------Step 5: Train on V for all epochs---------------------------#
                print("STEP 5: Train CNN on new weights and record---------------")
                lr = .0001
                if i>0:
                    lr = 1e-4
                elif 3<=i<=7:
                    lr = lr - 1e-5*(i-3)#set learning rate
                else:
                    lr = 7e-5
                optim3 = optax.adamw(1e-4)
                #optim1 = optim3
                dummy_dict = {}
                params, static, optim3, dummy_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim3, \
                                                                            n_iter=75, save_iter=config['save_iter'], \
                                                                            task_id=i,config={
                                                                                'batch_size': 20,
                                                                                'opt': 'Nash',
                                                                                'problem': config['prob'],
                                                                                'data_id': config['data'],
                                                                                'len_exp_replay': 20000,
                                                                                "flag": config['flag'],
                                                                                'network': config['network'],
                                                                                }, dictum = dummy_dict)
                
                params, static, optim3, record_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim3, \
                                                                            n_iter=config['epochs_per_task'], save_iter=config['save_iter'], \
                                                                            task_id=i,config={
                                                                                'batch_size': 20,
                                                                                'opt': 'Nash',
                                                                                'problem': config['prob'],
                                                                                'data_id': config['data'],
                                                                                'len_exp_replay': 20000,
                                                                                "flag": config['flag'],
                                                                                'network': config['network'],
                                                                                }, dictum = record_dict)
                #recombine to model and the resplit so the model and params/statics are updated for next task
                model = eqx.combine(params, static)
                params, static = eqx.partition(model, eqx.is_array)
                rec_dict = record_dict[str(i)]
                end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
                #end_100last = np.mean([rec_dict["train"+str((i)*rec_epochs+j)][0] for j in range(40,51)])
                end_100last = np.mean([rec_dict["train"+str((i)*rec_epochs+j)][0] for j in range(90,101)])

                #print("here is params: ", params)
                #print("here is static", static)
                static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
                static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
                static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                #print("here is params AFTER: ", params)
                #print("here is static AFTER:", static)
                #optim1 = optim3
                arch_round = arch_round + 1


            #if better architecture was not found, then skip to step (i.e. complete standard training)
            else: 
                #--------------Step 5: Train on V for all epochs---------------------------#
                print("STEP 5: Train CNN on new weights and record---------------")
                lr = .0001
                if i<3:
                    lr = 1e-4
                elif 3<=i<=7:
                    lr = lr - 1e-5*(i-3)#set learning rate
                else:
                    lr = 7e-5
                optim4 = optax.adamw(7e-5)
                optim4 = optim1
                dummy_dict = {}
                params, static, optim4, dummy_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim4, \
                                                                            n_iter=100, save_iter=config['save_iter'], \
                                                                            task_id=i,config={
                                                                                'batch_size': 20,
                                                                                'opt': 'Nash',
                                                                                'problem': config['prob'],
                                                                                'data_id': config['data'],
                                                                                'len_exp_replay': 20000,
                                                                                "flag": config['flag'],
                                                                                'network': config['network'],
                                                                                }, dictum = dummy_dict)
                params, static, optim4, record_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim4, \
                                                                            n_iter=500, save_iter=config['save_iter'], \
                                                                            task_id=i,config={
                                                                                'batch_size': 20,
                                                                                'opt': 'Nash',
                                                                                'problem': config['prob'],
                                                                                'data_id': config['data'],
                                                                                'len_exp_replay': 20000,
                                                                                "flag": config['flag'],
                                                                                'network': config['network'],
                                                                                }, dictum = record_dict)
                
                rec_epochs = 500
                rec_dict = record_dict[str(i)]
                end_last = np.mean([rec_dict["train"+str((i+1)*rec_epochs-j)][0] for j in range(1,10)])
                #end_100last = np.mean([rec_dict["train"+str((i)*rec_epochs+j)][0] for j in range(40,51)])
                end_100last = np.mean([rec_dict["train"+str((i)*rec_epochs+j)][0] for j in range(90,101)])
                
                
                model = eqx.combine(params, static)
                params, static = eqx.partition(model, eqx.is_array)
                #print("here is params: ", params)
                #print("here is static", static)
                static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
                static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
                static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                #print("here is params AFTER: ", params)
                #print("here is static AFTER:", static)

        data.append_to_experience(i)
    # print("Final Filter Architectures: ", filter_arch_list)
    # print("Final MLP Architectures: ", mlp_arch_list)
    # for w in range(0,len(filter_arch_list)):
    #     print("TASK ", w+1, ": ", "-- Filter Arch: ", filter_arch_list[w], " MLP Arch: ", mlp_arch_list[w])
    model = eqx.combine(params, static) #reconstruct the model
    eqx.tree_serialise_leaves(config['model_path'], model) #eqx.tree_seriealise_leaves saves the model to a given file.
    del model #delete model to free memory
    del params #delete params to free memory
    del static #delete static to free memory  
    return record_dict_preAB, record_dict_AB, record_dict #dictionary containing info from CL trainining. In particular, '(V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)' for each task and 'train', 'test', etc.
    

########################################################################
## The main run loop
if __name__ == "__main__":
    """
    NOTES on Arg parser:
    The argparse module’s support for command-line interfaces is built around an instance of 
        parser = argparse.ArgumentParser(prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help'). 
    It is a container for argument specifications and has options that apply to the parser as whole:
    """
    
    parser = argparse.ArgumentParser(prog="testKcontAWBT.py", description="Test different datasets and different models",) #construct parser
    subparsers = parser.add_subparsers(help='', dest='command')

    train_parser = subparsers.add_parser("train") #add subparser
    #.add_argument allows us to add a user-inputed value to parser
    train_parser.add_argument("runs", default=1, help="the number of total runs") #allows user to add value for arument "runs". the help kywd tells user what the value is.
    train_parser.add_argument("json", default=None, help="directory with configurations") # allows to add JSON

    basic_path='/Users/allyhahn/Documents/code/AWBT code/jsons/' #first part of file path
    args = parser.parse_args() #the "parser.parse_args()" method runs the parser and places the extracted data in a argparse.Namespace object
    
    '''
    NOTES on os.path.join():
    A function in the os module that joins one or more path components intelligently. It constructs a full path 
    by concatenating various components while automatically inserting the appropriate path separator
    '''
    
    json_path = os.path.join(basic_path+str(args.json)) #construct the entire file path using the basic path and user-inputed file path
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path) # method checks if path is an existing file or not. It return booelan T/F. "assert" tells program to halt if F and continue if T
    params = Params(json_path).dict
    print("this is params: ", params)      
    if args.runs is not None:
        params['runs'] = int(args.runs)
    else:
        params['runs'] = 5
    print("The configuration is", params)
    if args.command == 'train':
        record_dict ={}
        record_dict_preAB = {}
        record_dict_AB = {}
        for j in range(params['runs']):
            print("runs", j, params['problem'])
            if params['prob']=='regression':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)]=train_model_reg(params)
            elif params['prob']=='classification':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] =train_model_class(params)
            elif params['problem']=='graph':
                record_dict_preAB[str(j)], record_dict_AB[str(j)], record_dict[str(j)] =train_model_graph(params)


        import pickle 
        with open(str("/Users/allyhahn/Documents/code/AWBT code/logdir/dicts/log_induced_preAB400")+'.pkl', 'wb') as f:
            pickle.dump(record_dict_preAB, f)
        with open(str("/Users/allyhahn/Documents/code/AWBT code/logdir/dicts/log_induced_AB400")+'.pkl', 'wb') as f:
            pickle.dump(record_dict_AB, f)
        with open(str(params['file'])+'.pkl', 'wb') as f:
            pickle.dump(record_dict, f)
      