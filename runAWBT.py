#below are the imports from the utilsAWBT folder
#from utilsAWBT.utils import * #GOAL: provide various easier operation. #CONTAINS: funcs for matrix operations (i.e. special situtation matrix multiplication, normalization) and two graphing funcs. for visualization
from utilsAWBT.modelAWBT import * #GOAL: class from which we can construct types of NN. #CONTAINS: MLP, CNN, GCN, Linear (uses equinox)
from utilsAWBT.trainerAWBT import * #GOAL: CL training constructed NN on data. #CONTAINS: loss funcs (i.e. mse, cross-entropy loss), an accuracy of predictions func, loss and pred/accuracy graph constructing func, and CL training functions
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
def arch_search(original_arch, task, trainW_loss, og_epochs, config,dataloader_curr,\
                 dataloader_exp,test_loader_curr, test_loader_exp):
    """
    GOAL: Complete a local "neighborhood-style" search for ideal architecture for MLP
    ARGUMENTS:
    RETURNS: 
        opt_arch: (list) contains the best MLP architecture for the current (and prev) tasks
    """
    trainer1, optim, __, arch_model  = load_checkpoint_AWB(config)
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
    arch_params, arch_static, optim, poll_dict[str(i)] =  trainer1.train__CL__reg_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
    return record_dict #dictionary containing info from CL trainining. In particular, '(V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)' for each task and 'train', 'test', etc.


#=================Architecture Search for Classification Problem===================#
#note: I am still debugging this. Will be added by end of week. Once in place it impacts
#a single line in the 'train_model_classification_AWB' function. This will be very easy to add
#in future.
def arch_searchCNN(prev_filter_size,prev_feed_sizes,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp):
    print("ARCH SEARCH CNN")

#=================Prep A and B for Classification Problem===================#
#note: chose to outsource this bit because it is repeated multiple times in next function
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


#=================Train Classification Problem===================#
def train_model_class_AWB(config):
    """
    GOAL: construct and CL train a model which is for a classification problem

    ARGUMENTS:
        config: Python dictionary (i.e. parameters for NN and other information. This was retrieved via JSON file in main)

    RETURNS:
        dict: dictionary 
    """
    trainer, optim, data, model  = load_checkpoint_AWB(config)
    params, static = eqx.partition(model, eqx.is_array)
    record_dict = {}
    record_dict_preAB = {}
    record_dict_AB = {}
    #reintialize A's and B's to static (they are put into params when we use eqx.partition(eqx.is_array))
    static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
    static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
    static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
    static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
    params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))

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
            params, static, optim, record_dict[str(i)] = trainer.train__CL__class_AWB( (dataloader_curr, dataloader_curr, (test_loader_curr, test_loader_curr),\
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

        # For tasks i = 1 and beyond, search for opt arch and then transfer to new architecture if found
        else:
            dataloader_curr, dataloader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='training')
            test_loader_curr, test_loader_exp= data.generate_dataset(task_id=i, batch_size=config['batch_size'], phase='testing')
       
            #----------------------STEP 1: Train task i for some epochs----------------------#
            print("STEP 1: Train W for a little bit on new task-------------")
            og_epochs = 100
            params, static, optim1, record_dict_preAB[str(i)]= trainer.train__CL__class_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
            trainWLoss = np.mean([arch_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,21)])
            #-----------------STEP 2: Search for new architecture (filter size and MLP size)------------#
            print("STEP 2: Find new architecture (arch search in future)----------")
            #We will incorporate search later, set new arch for now
            prev_feed_sizes = model.feed_sizes
            prev_filter_size = model.filter_size
            #opt_filter, opt_MLParch = arch_searchCNN(prev_filter_size,prev_feed_sizes,i,trainWLoss,og_epochs,config,dataloader_curr, dataloader_exp,test_loader_curr,test_loader_exp)
            opt_MLParch = [1875,700,100,10]
            opt_filter = 4
            print("NEW  FILTER Architecture: ", opt_filter)
            print("NEW MLP Arch: ", opt_MLParch)
            print()
            #

            #if a different architecture was chosen then use our method to transfer to new arch
            if (prev_feed_sizes[1:3] != opt_MLParch[1:3]) or (opt_filter !=prev_filter_size):
                #----------------STEP 3a: Set New Architecture and Reinitialize A's and B's accordingly------#
                print("STEP 3A: Set new architecture and A's, B's----------------")
                conv_output_size = model.calc_output_size(opt_filter)
                maxpool_output_size = model.pool_output_size(2,conv_output_size)
                #set MLP input layer to correct size correspongding to new filter size output for Convnet layer
                opt_MLParch[0] = maxpool_output_size*maxpool_output_size*model.channel_out
                #print("this is the new spot 1 after!!!:==============", opt_MLParch[0])
                model = eqx.combine(params,static) #put back model
                model = eqx.tree_at(lambda x: x.feed_sizes, model, opt_MLParch) #set new arch
                model = eqx.tree_at(lambda x: x.filter_size, model, opt_filter) #set new filter
                #initialize and set A's and B's according to optimal architecture found
                A_feed, B_feed, A_conv, B_conv = prepABs(model, prev_feed_sizes, prev_filter_size)
                model = eqx.tree_at(lambda x: (x.A_feed, x.B_feed, x.A_conv, x.B_conv), model, replace = (A_feed, B_feed, A_conv, B_conv))
                #print("model after setting everything: ", model)
                #print("A_conv: ", model.A_conv[0])
                #print("A_feed: ", model.A_feed[0])
                #print("conv weights before: ", model.conv_layers[0].weight)
                #print("feed weights before: ", model.feed_layers[0].weight)
                og_epochs = 100
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

                #---------------STEP 3b: Train ONLY A's and B's---------------------#
                print("STEP 3B: Train ONLY on the A's and B's; fix W----------------")
                diff_model, static_model = eqx.partition(model, filter_spec) #makes weights static and A's, B's params
                #print("MAKE AB Params diff_model: ", diff_model)
                #print("MAKE Weights Static static_model: ", static_model)
                import optax
                optim2 = optax.adamw(1e-4) #set new optimizer
                #Train A's and B's only; weights are frozen
                diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__class_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,51)])
                prevABLoss = trainABLoss
                a=1
                # Below we incorporate a form of metric for A and B training. We continue to train for a
                #set number of epochs on A and B until the loss value on A and B is 80% of the loss value
                #for training on original architecture. This insures both improvement will occur and overfitting
                #will not occur.
                og_epochs = 100
                if (trainABLoss>.5):
                        og_epochs = 300
                while (trainABLoss>trainWLoss*.8):
                    diff_model, static_model, optim2, record_dict_AB[str(i)] =  trainer.train__CL__class_AWB((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
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
                    prevABLoss = trainABLoss
                    trainABLoss = np.mean([AB_dict["train"+str((i+1)*og_epochs-j)][0] for j in range(1,15)])
                    a +=1
                    print("AB Loss after AB training iteration ", a-1, ": ", trainABLoss)
                    if trainABLoss<.1:
                        og_epochs = 100
                    if trainABLoss>prevABLoss:
                        print("increasing break out")
                        break
                    if a==7:
                        print("too many AB training iterations, breaking out of AB training loop")
                        break
                model = eqx.combine(diff_model,static_model) #recombine params and statics
                print("A_conv: ", model.A_conv[0])
                print("A_feed: ", model.A_feed[0])
                print("conv weights before: ", model.conv_layers[0].weight)
                print("feed weights before: ", model.feed_layers[0].weight)
                print()
                #--------------Step 4: Set V = AWB^T for ConvNet and MLP--------------------#
                print("STEP 4: Set new V = AWB^T weights------------------")
                #set V for convolutional layer
                if (opt_filter !=prev_filter_size):
                    weights_list = [[(model.A_conv[i]@(model.conv_layers[0].weight[i][0])@jnp.transpose(model.B_conv[i]))] for i in range(0,model.channel_out)]
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
                optim3 = optax.adamw(1e-4)
                optim1 = optim3
                params, static, optim1, record_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim1, \
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
                #print("here is params: ", params)
                #print("here is static", static)
                static = eqx.tree_at(lambda x: x.A_conv, static, replace= model.A_conv)
                static = eqx.tree_at(lambda x: x.B_conv, static, replace= model.B_conv)
                static = eqx.tree_at(lambda x: x.A_feed, static, replace= model.A_feed)
                static = eqx.tree_at(lambda x: x.B_feed, static, replace= model.B_feed)
                params = eqx.tree_at(lambda x: (x.A_conv,x.B_conv,x.A_feed,x.B_feed), params, replace= (None,None,None,None))
                #print("here is params AFTER: ", params)
                #print("here is static AFTER:", static)


            #if better architecture was not found, then skip to step (i.e. complete standard training)
            else: 
                #--------------STEP 2: Train on original weights for all epochs and record-----------------#
                print("STEP 5: Train CNN on new weights and record---------------")
                params, static, optim1, record_dict[str(i)]= trainer.train__CL__class((dataloader_curr, dataloader_exp, (test_loader_curr, test_loader_exp),\
                                                                            (test_loader_curr, test_loader_exp)),params, static, optim1, \
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
                #print(record_dict[str(i)])
                #recombine to model and the resplit so the model and params/statics are updated for next task
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
    model = eqx.combine(params, static) #reconstruct the model
    eqx.tree_serialise_leaves(config['model_path'], model) #eqx.tree_seriealise_leaves saves the model to a given file.
    del model #delete model to free memory
    del params #delete params to free memory
    del static #delete static to free memory  
    return record_dict #dictionary containing info from CL trainining. In particular, '(V, dV, dVstar_dx, dVstar_dtheta, H, grad_norm, grad_norm)' for each task and 'train', 'test', etc.
    

#=================Train Graph Classification Problem ===================#
#note: this is a placeholder function. It will be implemented later.
#This will be written and working by end of this week or early next week depending on 
#potential difficulties.
def train_model_graph_AWB(config):
    print("train graph classification model")