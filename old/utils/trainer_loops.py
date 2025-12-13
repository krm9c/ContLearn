"""Training loop methods for continual learning."""
import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np_
import optax
import torch_geometric.transforms as T
from tqdm import tqdm


class TrainingLoopsMixin:
    """Mixin class containing training loop methods for continual learning."""

    def _compute_metrics_on_sampled_batches(self, params, static, loader,
                                            num_batches=10, problem_type='vectors',
                                            notABTrain=True, transforms=None):
        """Efficiently compute metrics on N sampled batches from a loader.

        Args:
            params: Model parameters
            static: Static model components
            loader: Data loader to sample from (for vectors should be tuple of (current_loader, exp_loader))
            num_batches: Number of batches to sample (default 10)
            problem_type: 'vectors' or 'graph'
            notABTrain: Whether using normal training (True) or AWB training (False)
            transforms: Transform pipeline for graph data

        Returns:
            Tuple of (current_task_metric, experience_metric) - mean metric values across sampled batches
        """
        current_metrics = []
        exp_metrics = []

        if problem_type == 'graph':
            # For graphs, loader should be a tuple (current_loader, exp_loader)
            if isinstance(loader, tuple):
                current_loader, exp_loader = loader
                current_iter = iter(current_loader)
                exp_iter = iter(exp_loader)

                for i in range(num_batches):
                    try:
                        current_batch = next(current_iter)
                        exp_batch = next(exp_iter)
                    except StopIteration:
                        break

                    current_batch = transforms(current_batch)
                    exp_batch = transforms(exp_batch)

                    # Current task metric
                    data_current = (current_batch, current_batch)
                    metric_current = self.return_metric(params, static, data=data_current, notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    # Experience metric
                    data_exp = (exp_batch, exp_batch)
                    metric_exp = self.return_metric(params, static, data=data_exp, notABTrain=notABTrain)
                    exp_metrics.append(metric_exp)
            else:
                # Single loader - use same data for both
                iterator = iter(loader)
                for i in range(num_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    batch = transforms(batch)
                    data = (batch, batch)
                    metric = self.return_metric(params, static, data=data, notABTrain=notABTrain)
                    current_metrics.append(metric)
                    exp_metrics.append(metric)
        else:
            # For vectors, loader should be tuple (current_loader, exp_loader)
            if isinstance(loader, tuple):
                current_loader, exp_loader = loader
                current_iter = iter(current_loader)
                exp_iter = iter(exp_loader)

                for i in range(num_batches):
                    try:
                        current_batch = next(current_iter)
                        exp_batch = next(exp_iter)
                    except StopIteration:
                        break

                    # Current task data
                    x_curr, y_curr = current_batch
                    x_curr = jnp.array(x_curr.numpy().astype(np_.float64))
                    y_curr = jnp.array(y_curr.numpy().astype(np_.float64))
                    y_curr = jnp.squeeze(y_curr)
                    if y_curr.ndim == 1:
                        y_curr = jnp.expand_dims(y_curr, axis=-1)

                    metric_current = self.return_metric(params, static, data=(x_curr, y_curr), notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    # Experience data
                    x_exp, y_exp = exp_batch
                    x_exp = jnp.array(x_exp.numpy().astype(np_.float64))
                    y_exp = jnp.array(y_exp.numpy().astype(np_.float64))
                    y_exp = jnp.squeeze(y_exp)
                    if y_exp.ndim == 1:
                        y_exp = jnp.expand_dims(y_exp, axis=-1)

                    metric_exp = self.return_metric(params, static, data=(x_exp, y_exp), notABTrain=notABTrain)
                    exp_metrics.append(metric_exp)
            else:
                # Single loader - use same data for both
                iterator = iter(loader)
                for i in range(num_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    x, y = batch
                    x = jnp.array(x.numpy().astype(np_.float64))
                    y = jnp.array(y.numpy().astype(np_.float64))
                    y = jnp.squeeze(y)
                    if y.ndim == 1:
                        y = jnp.expand_dims(y, axis=-1)
                    metric = self.return_metric(params, static, data=(x, y), notABTrain=notABTrain)
                    current_metrics.append(metric)
                    exp_metrics.append(metric)

        current_mean = np_.mean(current_metrics) if current_metrics else 0.0
        exp_mean = np_.mean(exp_metrics) if exp_metrics else 0.0
        return current_mean, exp_mean

    def train__CL__graph(self, train__, params, static,\
                        optim, n_iter=1000, save_iter=5, record_dict = {},
                        task_id=0, config={}, notABTrain = True):
        # Unified format: (train_current, train_exp, test_loaders, test_loaders)
        # where test_loaders = (test_current, test_exp)
        trainloader, exploader, valloader, testloader = train__
        trainiter = iter(trainloader)
        expiter = iter(exploader)
        opt_state = optim.init(params)
        from tqdm import tqdm
        pbar = tqdm(range(n_iter), dynamic_ncols=True)
        V_star_max=[]
        dVstar_dx=[]
        dVstar_dtheta=[]
        dVstar_dadj=[]
        H=[]
        metrics=[]
        train_metrics=[]
        import torch_geometric.transforms as T
        # Custom transform to remove edge_attr before GCNNorm/ToDense
        # This ensures adj is 2D (N x N) instead of 3D (N x N x edge_features)
        class RemoveEdgeAttr:
            def __call__(self, data):
                data.edge_attr = None
                return data
        # Note: We remove edge_attr first so that ToDense creates a 2D adjacency matrix
        # which is what the GCN layers expect for matmul operations
        transforms = T.Compose([RemoveEdgeAttr(), T.GCNNorm(), T.ToDense(), T.NormalizeFeatures()])
        
        var_adj=[]
        var_x =[]
        for batch, batch_ex in zip(trainiter, expiter):
            batch = transforms(batch)
            batch_ex=transforms(batch_ex)
            # ---------------------------------------------------------------------
            # How do you ensure that these distances are reliable or rather not flipping 
            # all over the place
            # First problem, a distance metric, that does not care about the size of the nodes.
            x = batch.x.numpy()
            exp_x = batch_ex.x.numpy()
            var_x.append(jnp.sqrt(jnp.linalg.norm(  (jnp.mean( x, axis =0)-jnp.mean(exp_x, axis = 0 ) )**2 ) ))
            #print(var__)
            var_adj.append(( jnp.sqrt( jnp.linalg.norm(batch.adj.numpy()**2) )- jnp.sqrt(jnp.linalg.norm(batch_ex.adj.numpy()**2) ) )**2)
            #print(var__)    
        var_x = sum(var_x)/len(var_x)
        var_adj = 1e-3*(sum(var_adj)/len(var_adj))
        
        
        for step in pbar:
            # print("step -- I am going into the batch", step)
            trainiter = iter(trainloader)
            expiter = iter(exploader)
            for batch, batch_ex in zip(trainiter, expiter):
                batch = transforms(batch)
                batch_ex=transforms(batch_ex)
                # ---------------------------------------------------------------------
                # How do you ensure that these distances are reliable or rather not flipping 
                # all over the place
                # First problem, a distance metric, that does not care about the size of the nodes.
                x = batch.x.numpy()
                exp_x = batch_ex.x.numpy()#--------------------------------------------------------------------------
                delta_x = np_.random.normal(0, var_x, exp_x.shape)       
                delta_adj = np_.random.normal(0, var_adj, batch_ex.adj.shape)   
                data = (static, (batch, batch_ex, delta_x, delta_adj) )
                grad, losses = self.return_Hamiltonian_graph(params, data, notABTrain) #this function has been changed accordingly        
                (h, V, dV, dv_dtheta, dv_dx, dv_dadj)     = losses
                updates, opt_state = optim.update(grad, opt_state, params)
                params =  optax.apply_updates(params, updates)
                # ------------------------------------------------------------------
                #Updated the parameters, now working on storing and viewing things.
                V_star_max.append(V)
                dVstar_dx.append(dv_dx)
                dVstar_dtheta.append(dv_dtheta)
                dVstar_dadj.append(dv_dadj)
                H.append(h)
                train_metric = self.return_metric(params, static, data = (batch, batch_ex),notABTrain = notABTrain)
                train_metrics.append(train_metric)

            if step % save_iter ==0 and step > 0:
                train_metric_avg = np_.mean(train_metrics)
                test_current_metric, test_exp_metric = self._compute_metrics_on_sampled_batches(
                    params, static, testloader, num_batches=10,
                    problem_type='graph', notABTrain=notABTrain, transforms=transforms
                )

                V_star_maxtr = np_.mean(V_star_max)
                dVstar_dxtr = np_.mean(dVstar_dx)
                dVstar_dthetatr = np_.mean(dVstar_dtheta)
                dVstar_dadjtr = np_.mean(dVstar_dadj)
                Htr = np_.mean(H)

                pbar.set_postfix_str(
                    f"H={Htr:.6e} V/CE={V_star_maxtr.item():.6e} dV_dx={dVstar_dxtr:.6e} "
                    f"dV_dθ={dVstar_dthetatr:.6e} dV_dadj={dVstar_dadjtr:.6e} | "
                    f"Tr/Acc={train_metric_avg:.4f} Te/Cur={test_current_metric:.4f} Te/Exp={test_exp_metric:.4f}"
                )

                # Record metrics using unified recording system
                iteration = step + task_id * n_iter
                model = eqx.combine(params, static)

                record_dict['iterations'][iteration] = self.record_metrics(
                    iteration=iteration,
                    step=step,
                    task_id=task_id,
                    losses={
                        'H': float(Htr),
                        'V': float(V_star_maxtr),
                        'dV_dx': float(dVstar_dxtr),
                        'dV_dtheta': float(dVstar_dthetatr),
                        'dV_dadj': float(dVstar_dadjtr),
                    },
                    gradients={},
                    metrics={
                        'train': float(train_metric_avg),
                        'test_current': float(test_current_metric),
                        'test_experience': float(test_exp_metric),
                    },
                    model=model
                )

                V_star_max = []
                dVstar_dx = []
                dVstar_dtheta = []
                dVstar_dadj = []
                H = []
                train_metrics = []

        return params, static, optim, record_dict 




# ---------------------------------------------------------------------


    def train__CL__reg(self, train__, params,\
                    static, optim, n_iter=1000,\
                    save_iter=10, task_id=0,\
                    config={}, record_dict = {}, notABTrain = True):

        trainloader, exploader, valloader, testloader=train__
        trainiter = iter(trainloader)
        expiter = iter(exploader)
        batch = next(trainiter)
        x, y = batch
        x = x.numpy().astype(np_.float64)
        y = y.numpy().astype(np_.float64)
        batch = (x, y)
        
        opt_state = optim.init(params)
        from tqdm import tqdm 
        pbar = tqdm(range(n_iter))
        if task_id>0:
            flag=config["flag"]
        else:
            flag=config["flag"]
        sum_delta_x =0.
        for step in pbar:
            
            try:
                batch = next(trainiter)
            except StopIteration:
                trainiter = iter(trainloader)
                batch = next(trainiter)
            try:
                batch_ex = next(expiter)
            except StopIteration:
                expiter = iter(exploader)
                batch_ex = next(expiter)

            (x, y) = batch
            (exp_x, exp_y) = batch_ex
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            
            # #print(exp_x.shape, x.shape)
            
            delta_x = jnp.abs(exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)            
            data = static, ( x, y, exp_x, exp_y, delta_x, flag)
            #print("THESE ARE PARAMS IN TRAINER MODULE: ", params)
            grad, losses =  self.return_Hamiltonian_mse(params, data, notABTrain)      
            (H, V, dV, dVstar_dtheta, dVstar_dx)  = losses         
            
            grad_leav = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
            updates, opt_state = optim.update(grad, opt_state, params)
            params =  optax.apply_updates(params, updates)

            if step % save_iter == 0 and step > 0:
                train_metric = V  # MSE is the metric
                test_current_metric, test_exp_metric = self._compute_metrics_on_sampled_batches(
                    params, static, testloader, num_batches=10,
                    problem_type='vectors', notABTrain=notABTrain
                )

                pbar.set_postfix_str(
                    f"MSE={V:.6e} dV_dx={dVstar_dx:.6e} dV_dθ={dVstar_dtheta:.6e} "
                    f"H={H:.6e} ||dH_dθ||={grad_norm:.6e} | "
                    f"Te/Cur={test_current_metric:.6e} Te/Exp={test_exp_metric:.6e}"
                )

                # Record metrics using unified recording system
                iteration = step + task_id * n_iter
                model = eqx.combine(params, static)

                record_dict['iterations'][iteration] = self.record_metrics(
                    iteration=iteration,
                    step=step,
                    task_id=task_id,
                    losses={
                        'H': float(H),
                        'V': float(V),
                        'dV': float(dV),
                        'dV_dx': float(dVstar_dx),
                        'dV_dtheta': float(dVstar_dtheta),
                    },
                    gradients={
                        'grad_norm': float(grad_norm),
                    },
                    metrics={
                        'train': float(train_metric),
                        'test_current': float(test_current_metric),
                        'test_experience': float(test_exp_metric),
                    },
                    model=model
                )

            # ## Validation Metric calculations on the total exp_replay
            # if step %100==0:
            #     sum_delta_x=0.
            #     V_star_max=[]
            #     dVstar_dx=[]
            #     dVstar_dtheta=[]
            #     H=[]
            #     dV=[]
            #     loader_1, loader_2= valloader
            #     for (batch_x, batch_ex) in zip(loader_1, loader_2):
            #         (x, y) = batch_x
            #         (exp_x, exp_y) = batch_ex
            #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         delta_x = jnp.abs(exp_x-x)
            #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            #         delta_x = (delta_x/sum_delta_x)
            #         data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
            #         _, losses = self.return_Hamiltonian_mse(params,data,notABTrain = True)         
            #         (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
            #         V_star_max.append(v)
            #         dVstar_dx.append(dvstar_dx)
            #         dVstar_dtheta.append(dvstar_dtheta)
            #         H.append(h)
            #         dV.append(dv)
            #     V_star_max=np_.mean(V_star_max)
            #     dVstar_dx= np_.mean(dVstar_dx)
            #     dVstar_dtheta=np_.mean(dVstar_dtheta)
            #     dV=np_.mean(dV)
            #     H=np_.mean(H)
            #     # #print(H,  dVstar_dx, dVstar_dtheta)
            #     # pbar.set_postfix({"Valid/MSE:": V_star_max,
            #     #               "Train/dVstar_dx:": dVstar_dx,
            #     #               "Train/dVstar_dtheta:": dVstar_dtheta,
            #     #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
            #     #               "Train/||dH_dtheta||:": grad_norm\
            #     #             })

                # #if notABTrain:
                # self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/Loss/dV', dV.item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                # self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                
                # dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                # V_star_max+dVstar_dx+dVstar_dtheta )
                
        # ## Test Metric calculations on the total exp_replay
        # sum_delta_x=0.
        # V_star_max=[]
        # dVstar_dx=[]
        # dVstar_dtheta=[]
        # H=[]
        # loader_1, loader_2= valloader
        # for (batch_x, batch_ex) in zip(loader_1, loader_2):
        #     (x, y) = batch_x
        #     (exp_x, exp_y) = batch_ex
        #     x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
        #     delta_x = (exp_x-x)
        #     sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
        #     delta_x = (delta_x/sum_delta_x)
        #     data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
        #     _, losses = self.return_Hamiltonian_mse(params,data,notABTrain)         
        #     (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
        #     V_star_max.append(v)
        #     dVstar_dx.append(dvstar_dx)
        #     dVstar_dtheta.append(dvstar_dtheta)
        #     H.append(h)
            
                
        # V_star_max=np_.mean(V_star_max)
        # dVstar_dx= np_.mean(dVstar_dx)
        # dVstar_dtheta=np_.mean(dVstar_dtheta)
        # H=np_.mean(H)
        # self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter)
        # self.writer.add_scalar('test/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
        # self.writer.add_scalar('test/gradient/dVstar_dx',
        #                     dVstar_dx.item(), step+task_id*n_iter)
        # self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
        # dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H)

        return params, static, optim, record_dict


    def train__CL__class(self, train__, params,\
                    static, optim, n_iter=1000,\
                    save_iter=10, task_id=0,\
                    config={}, record_dict = {},notABTrain = True):

        trainloader, exploader, valloader, testloader=train__
        trainiter = iter(trainloader)
        expiter = iter(exploader)
        # optim_inner_x, optim_inner_mod = optim_inner
        batch = next(trainiter)
        #print(batch)
        x, y = batch
        x = x.numpy().astype(np_.float64)
        y = y.numpy().astype(np_.float64)
        batch = (x, y)
        opt_state = optim.init(params)
        from tqdm import tqdm 
        pbar = tqdm(range(n_iter))
        if task_id>0:
            flag=config["flag"]
        else:
            flag=config["flag"]

        sum_delta_x =0.
        mm=0.
        train_metrics = []
        for step in pbar:
            
            try:
                batch = next(trainiter)
            except StopIteration:
                trainiter = iter(trainloader)
                batch = next(trainiter)
            try:
                batch_ex = next(expiter)
            except StopIteration:
                expiter = iter(exploader)
                batch_ex = next(expiter)

            (x, y) = batch
            (exp_x, exp_y) = batch_ex
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            # print(exp_x.shape, x.shape)
            delta_x = jnp.abs(exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)            
            data = static, ( x, y, exp_x, exp_y, delta_x, flag)
            grad, losses =  self.return_Hamiltonian_class(params, data, notABTrain)    
            (H, V, dV, dVstar_dtheta, dVstar_dx)  = losses         
            grad_leav = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )

            updates, opt_state = optim.update(grad, opt_state, params)
            params =  optax.apply_updates(params, updates)

            train_metric = self.return_metric(params, static, data=(x, y), notABTrain=notABTrain)
            train_metrics.append(train_metric)

            if step % save_iter == 0 and step > 0:
                train_metric_avg = np_.mean(train_metrics)
                test_current_metric, test_exp_metric = self._compute_metrics_on_sampled_batches(
                    params, static, testloader, num_batches=10,
                    problem_type='vectors', notABTrain=notABTrain
                )

                pbar.set_postfix_str(
                    f"CE={V:.6e} dV_dx={dVstar_dx:.6e} dV_dθ={dVstar_dtheta:.6e} "
                    f"H={H:.6e} ||dH_dθ||={grad_norm:.6e} | "
                    f"Tr/Acc={train_metric_avg:.4f} Te/Cur={test_current_metric:.4f} Te/Exp={test_exp_metric:.4f}"
                )

                # Record metrics using unified recording system
                iteration = step + task_id * n_iter
                model = eqx.combine(params, static)

                record_dict['iterations'][iteration] = self.record_metrics(
                    iteration=iteration,
                    step=step,
                    task_id=task_id,
                    losses={
                        'H': float(H),
                        'V': float(V),
                        'dV': float(dV),
                        'dV_dx': float(dVstar_dx),
                        'dV_dtheta': float(dVstar_dtheta),
                    },
                    gradients={
                        'grad_norm': float(grad_norm),
                    },
                    metrics={
                        'train': float(train_metric_avg),
                        'test_current': float(test_current_metric),
                        'test_experience': float(test_exp_metric),
                    },
                    model=model
                )

                train_metrics = []

            # print("I am supposed to be printing data now.")
            # ## Validation Metric calculations on the total exp_replay
            # if step % 1==0:
            #     sum_delta_x=0.
            #     V_star_max=[]
            #     dVstar_dx=[]
            #     dVstar_dtheta=[]
            #     H=[]
            #     dV=[]
            #     metrics =[]
            #     loader_1, loader_2= valloader
            #     for (batch_x, batch_ex) in zip(loader_1, loader_2):
            #         (x, y) = batch_x
            #         (exp_x, exp_y) = batch_ex
            #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            #         delta_x = jnp.abs(exp_x-x)
            #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            #         delta_x = (delta_x/sum_delta_x)
            #         data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
            #         _, losses = self.return_Hamiltonian_class(params,data, notABTrain)         
            #         (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
            #         V_star_max.append(v)
            #         dVstar_dx.append(dvstar_dx)
            #         dVstar_dtheta.append(dvstar_dtheta)
            #         H.append(h)
            #         dV.append(dv)
        
            #         metrics.append(self.return_metric(params, static, data=(exp_x, exp_y), notABTrain=notABTrain))
                                  
                # V_star_max=np_.mean(V_star_max)
                # dVstar_dx= np_.mean(dVstar_dx)
                # dVstar_dtheta=np_.mean(dVstar_dtheta)
                # dV=np_.mean(dV)
                # H=np_.mean(H)
                # mm=(sum(metrics)/len(metrics))
                # # print(len(metrics), sum(metrics), mm)
                # # #print(H,  dVstar_dx, dVstar_dtheta)
                # # pbar.set_postfix({"Valid/MSE:": V_star_max,
                # #               "Train/dVstar_dx:": dVstar_dx,
                # #               "Train/dVstar_dtheta:": dVstar_dtheta,
                # #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
                # #               "Train/||dH_dtheta||:": grad_norm\
                # #             })
                
                # self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/Loss/Cross', V_star_max.item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/Loss/dV', dV.item(), step+task_id*n_iter)
                # self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                # self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                # self.writer.add_scalar('valid/metrics/ACC', mm.item(), step+task_id*n_iter)
                
                # dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                # V_star_max+dVstar_dx+dVstar_dtheta, mm.item() )
                
        # ## Test Metric calculations on the total exp_replay
        # sum_delta_x=0.
        # V_star_max=[]
        # dVstar_dx=[]
        # dVstar_dtheta=[]
        # H=[]
        # metrics=[]
        # loader_1, loader_2= testloader
        # for (batch_x, batch_ex) in zip(loader_1, loader_2):
        #     (x, y) = batch_x
        #     (exp_x, exp_y) = batch_ex
        #     x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        #     exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
        #     delta_x = (exp_x-x)
        #     sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
        #     delta_x = (delta_x/sum_delta_x)
        #     data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
        #     _, losses = self.return_Hamiltonian_class(params,data, notABTrain)         
        #     (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
        #     V_star_max.append(v)
        #     dVstar_dx.append(dvstar_dx)
        #     dVstar_dtheta.append(dvstar_dtheta)
        #     H.append(h)
        #     metrics.append(self.return_metric(params, static, data = (exp_x, exp_y),notABTrain = notABTrain))
        #     pbar.set_postfix({"test/Cross:": v})
        # V_star_max=np_.mean(V_star_max[-50:])
        # dVstar_dx= np_.mean(dVstar_dx)
        # dVstar_dtheta=np_.mean(dVstar_dtheta)
        # H=np_.mean(H)
        # mm = sum(metrics)/len(metrics)
        # self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter)
        # self.writer.add_scalar('test/Loss/Cross', V_star_max.item(), step+task_id*n_iter)
        # self.writer.add_scalar('test/gradient/dVstar_dx',
        #                     dVstar_dx.item(), step+task_id*n_iter)
        # self.writer.add_scalar('test/metrics/ACC',
        #                     mm.item(), step+task_id*n_iter)
        # self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
        # dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H, mm)

        return params, static, optim, record_dict
    
    



#
