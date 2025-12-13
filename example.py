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
from utilsAWBT.utilsAWBT import * #GOAL: provide various easier operation. #CONTAINS: funcs for matrix operations (i.e. special situtation matrix multiplication, normalization) and two graphing funcs. for visualization
from utilsAWBT.modelAWBT import * #GOAL: class from which we can construct types of NN. #CONTAINS: MLP, CNN, GCN, Linear (uses equinox)
from utilsAWBT.trainerAWBT import * #GOAL: CL training constructed NN on data. #CONTAINS: loss funcs (i.e. mse, cross-entropy loss), an accuracy of predictions func, loss and pred/accuracy graph constructing func, and CL training functions
from utilsAWBT.dataAWBT import * #GOAL: take in dataset and prepare for learning #CONTAINS: preparing and batching funcs (uses torch and torchvision)


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

