#below are the imports from the utilsAWBT folder
#from utilsAWBT.utils import * #GOAL: provide various easier operation. #CONTAINS: funcs for matrix operations (i.e. special situtation matrix multiplication, normalization) and two graphing funcs. for visualization
from utilsAWBT.modelAWBT import * #GOAL: class from which we can construct types of NN. #CONTAINS: MLP, CNN, GCN, Linear (uses equinox)
#from utilsAWBT.trainerAWBT import * #GOAL: CL training constructed NN on data. #CONTAINS: loss funcs (i.e. mse, cross-entropy loss), an accuracy of predictions func, loss and pred/accuracy graph constructing func, and CL training functions
#from utilsAWBT.dataAWBT import * #GOAL: take in dataset and prepare for learning #CONTAINS: preparing and batching funcs (uses torch and torchvision)

#===================LOAD CHECKPOINT====================#
#note: the changes below are due to calling "MLP_AWB" and "CNN_AWB"
#instead of "MLP" and "CNN" in the load_checkpoint() function
def load_checkpoint_AWB(config):
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
            model = MLP_AWB(sizes = [x.shape[1],config['hln'],config['hln'],y.shape[1]])
            print("MODEL IN LOAD_CHECK:", model)
        elif config['prob'] == 'classification': #if classification use CNN
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            model = CNN_AWB(subkey,3,[1875,512,64,10])
        elif config['problem'] == 'graph': #if graph use myNN. This is a GCN.
            model = myNN(in_size=x.shape[1], hid_size=config["hln"],\
                node_num=x.shape[0], out_size=config['n_class']) 
        optim = optax.adamw(config['lr']) #set adamw as optimizer (Adam with weight decay regularization), 'lr' is the learning rate
        trainer = Trainer(Loss=config['loss'], metric=config['metric'], 
                problem=config['problem'], logdir=str(config['tensorfile']) ) #This is a class in "utils.trainer.py".  Creates NN Trainer
        
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
            model = MLP_AWB(sizes = [x.shape[1],config['hln'],config['hln'],y.shape[1]])
            print("MODEL IN LOAD_CHECK:", model)
        elif config['prob'] == 'classification': #if classification, use CNN
            key = jax.random.PRNGKey(SEED)
            key, subkey = jax.random.split(key, 2)
            model = CNN_AWB(subkey,3,[1875,512,64,10])
        elif config['problem'] == 'graph': #if graph, use myNN which is GCN
            model = myNN(in_size=x.shape[1], hid_size=config["hln"],\
                node_num=x.shape[0], out_size=config['n_class']) 
        optim = optax.adam(config['lr']) #adam optimizer, 'lr' is learning rate
        trainer = Trainer(Loss=config['loss'], metric=config['metric'],
                problem=config['problem'], logdir=str(config['tensorfile'])) #This is a class in "utils.trainer.py".  Creates NN Trainer

        return trainer, optim, dataset, model

#============Architecture NDDS Search for Regression Prob===========#




#=================Train Regression Problem===================#
def train_model_reg_AWB(config):
    """
    GOAL: construct and CL train a model which is for a regression problem

    ARGUMENTS:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)

    RETURNS:
        record_dict: dictionary 
    """
    trainer, optim, data, model  = load_checkpoint_AWB(config) #Get trainer, optim, data, model
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

            #------------------------------------Standard CL train on task i = 0-----------------------------#
            params, static, optim, record_dict[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr,\
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
            
        else:
            dataloader_curr, dataloader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            test_loader_curr, test_loader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='testing')

            #---------------STEP 1: Standard CL train on task i for few epochs--------------#
            og_epochs = 100
            #og_epochs = config['epochs_per_task'] 
            print("STEP 1: We train for ", og_epochs, " epochs on the next task")
            params, static, optim1, record_dict_preAB[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
            #print("The params after CL train: ", params)
            #print("The statics after CL train: ", static)
            #print("WEIGHTS BEFORE Arch Search: ", model.layers[0].weight)
            model = eqx.combine(params, static)
            arch_dict = record_dict_preAB[str(i)]
            trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
            #-------------------------------------STEP 2: Get new architecture------------------------------------------#
            print("STEP 2: Search for Best architecture for the data in task " , i)
            original_arch = model.sizes
            opt_arch = arch_search(original_arch,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
            print("NEW Architecture: ", opt_arch)
            #print("WEIGHTS AFTER SEARCH BUT BEFORE AB TRAIN: ", model.layers[0].weight)

            if opt_arch != original_arch: #if arch search found a new architecture, then...
                #----------STEP 3a: Set New Arch and Set/Prep A and B to proper sizes-----------------#
                og_epochs = 350
                print("STEP 3a: Set new Architecture and set/prep A and B to proper sizes")
                s = original_arch
                #opt_arch = [3,385+75*i,385+50*i,10]
                model = eqx.tree_at(lambda x: x.sizes, model, opt_arch)
                initializer = jax.nn.initializers.glorot_uniform()
                A_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[1:],model.sizes[1:])]
                B_list = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(s[:-1],model.sizes[:-1])]
                model = eqx.tree_at(lambda x: x.A, model, A_list)
                model = eqx.tree_at(lambda x: x.B, model, B_list)
                #print("A BEFORE TRAIN: ", model.A[0])
                #print("WEIGHTS AFTER SETTING A,B: ", model.layers[0].weight)
                #print("model after set A,B:", model)

                #-------------STEP 3b: Freeze W train only on A,B-----------#
                og_epochs = 2000
                print("STEP 3b: Train A and B fix W for ", og_epochs, " epochs")
                model1 = model
                filter_spec = jtu.tree_map(lambda _: False, model1) #this is a copy of the model
                filter_spec = eqx.tree_at(lambda x: (x.A,x.B), filter_spec, replace=(True,True),)
                #filter_spec = eqx.tree_at(lambda x: x.layers, filter_spec, replace=True,)
                diff_model, static_model = eqx.partition(model, filter_spec)
                #print("MAKE AB Params diff_model: ", diff_model)
                #print("MAKE Weights Static static_model: ", static_model)
                import optax
                optim2 = optax.adam(1e-4)
                diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                a = 1
                threshold = 0.6
                print("AB Loss after first AB training: ", trainABLoss)
                if trainABLoss<=0.10:
                    threshold = .75
                while(trainABLoss> threshold*trainWLoss):
                    #og_epochs = 1000
                    diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                    prevABLoss = trainABLoss
                    trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                    a +=1
                    print("AB Loss after AB training iteration ", a-1, ": ", trainABLoss)
                    if prevABLoss< trainABLoss:
                        print("AB Loss is increasing, breaking out of AB training loop")
                        break
                    if a==8:
                        print("too many AB training iterations, breaking out of AB training loop")
                        break
                model = eqx.combine(diff_model,static_model)
                #print("WEIGHTS AFTER AB TRAIN: ", model.layers[0].weight)
                #print("A AFTER TRAIN: ", model.A[0])

                #-----------------------STEP 4: Set new V = AWB^T-----------------------------#
                print("STEP 4: Set the new weights V = AWB^T")
                #print("-------------------------------------")
                for j in range(len(model.sizes)-1):
                    Vw = model.A[j] @ model.layers[j].weight @ jnp.transpose(model.B[j])
                    #print("shape of bias:", model.layers[i].bias.shape)
                    #print("shape of A: ", model.A[i].shape)
                    Vb = model.layers[j].bias@ model.A[j].T
                    model = eqx.tree_at(lambda x: x.layers[j].weight, model, Vw)
                    model = eqx.tree_at(lambda x: x.layers[j].bias, model, Vb)
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                print("PARAMS AFTER V SET: ", params)
                print("STATIC AFTER V SET: ", static)
                #print("weights size after setting V: ", jnp.shape(model.layers[0].weight))
                #print("WEIGHTS AFTER SETTING V: ", model.layers[0].weight)
                #print("A BEFORE TRAIN V: ", model.A[0])

                #-----------STEP 5: Train with weights V for full epochs & record------------#
                print("STEP 5: Train the model with weights V for full amount of epochs")
                import optax
                optim3 = optax.adam(1e-3)
                record_dict_dummy = {}
                
                params, static, optim3, record_dict_dummy[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                
                params, static, optim3, record_dict[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                #print("After re-split params: ", params)
                #print("After re-split statics: ", static)
                same_arch = False
                optim1 = optim3
            else: #if arch search did not find a new architecture, then...
                #---------------STEP 3: Train with original weights and architecture---------#
                params, static = eqx.partition(model, eqx.is_array)
                static = eqx.tree_at(lambda x: x.A, static, replace= model.A)
                static = eqx.tree_at(lambda x: x.B, static, replace= model.B)
                params = eqx.tree_at(lambda x: (x.A,x.B), params, replace= (None,None))
                params, static, optim1, record_dict[str(i)] =  trainer.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                #print("After re-split params: ", params)
                #print("After re-split statics: ", static)
                same_arch = True
            
        data.append_to_experience(i) #this is a method from data.utils.py adds data to be later retrained on?
    model = eqx.combine(params, static) #close/combine model back together
    eqx.tree_serialise_leaves(config['model_path']+'.eqx', model) #eqx.tree_seriealise_leaves saves the model to a given file.
    del model #delete model to free memory
    del params #delete params to free memory
    del static #delete static to free memory  
    return record_dict_preAB, record_dict_AB,record_dict #dictionary containing info from CL trainining. In particular, '(V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)' for each task and 'train', 'test', etc.
