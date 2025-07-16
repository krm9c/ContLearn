#Trainer file changes for AWB

#===============return_metric_AWB function==================#
def return_metric_AWB(self, params, statics, data, notABTrain):
        model = eqx.combine(params, statics)
        if self.problem=='vectors':
            x, y = data
            if self.metric == 'class':
                y=y.astype(jnp.int64)
                if (notABTrain == True): 
                    pred_y = jnp.argmax(jax.nn.log_softmax(jax.vmap(model)(x)), 1)
                else:
                    pred_y = jnp.argmax(jax.nn.log_softmax(jax.vmap(model.get_AWBT)(x)), 1)
                # print(pred_y)
                # pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
                return jnp.mean(y == pred_y)
            elif self.metric=='mse':
                return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
        elif self.problem== 'graph':
                batch, batch_ex = data   
                x_tog   = [batch.x.numpy(), batch_ex.x.numpy()]
                y_tog   = [batch.y.numpy(), batch_ex.y.numpy()] 
                adj_tog = [batch.adj.numpy(), batch_ex.adj.numpy()]
                b_tog   = [batch.batch.numpy(), batch_ex.batch.numpy()]
                n_tog   = [batch.n_nodes.numpy(), batch_ex.n_nodes.numpy()]
             
                if self.loss == 'class':
                    yhat = self.accuracy_graphs(params, statics,\
                            x_tog, adj_tog, b_tog, n_tog)
                    pred_y = jnp.argmax(yhat, axis=1)
                    return jnp.mean(jnp.concatenate
                                    (y_tog)==pred_y)
                

# ===============return_Hamiltonian_mse_AWB function (i.e. with flag)=================#
def return_Hamiltonian_mse_AWB(self, params, data,notABTrain = True):
        statics, (x, y, exp_x, exp_y, deltax, flag)  = data
        extra=y
        #model = eqx.combine(params, statics)
        #pred_y = jax.vmap(model)(x)
        #print("shape of pred_y: ", pred_y.shape)
        #print("shape of y:", y.shape)
        #pred_y = pred_y.squeeze(1)
        #print("shape of pred_y after squeeze:", pred_y.shape)
        if (notABTrain == True):
            def return_V_star_vector_mse(params, x):
                y = extra
                model = eqx.combine(params, statics)
                pred_y = jax.vmap(model)(x)
                pred_y = pred_y.squeeze(1)
                #print(optax.l2_loss(y, pred_y))
                return jnp.mean(optax.l2_loss(y, pred_y))
            def norm_param(x):
                return (x*-1) # (-1*1e-0/jnp.sqrt(jnp.linalg.norm(x**2))))
            xdot = deltax
            zero_dx = jnp.zeros(xdot.shape)
            delta_theta= jax.grad(return_V_star_vector_mse,argnums=(0))(params, x) 
            wdot= jax.tree_util.tree_map(norm_param, delta_theta)
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
            extra = exp_y
            grad_V =  jax.grad(return_V_star_vector_mse,argnums=(0))(params, exp_x) 
            #jax. linearize = Produces a linear approximation to fun using jvp() and partial eval.
            #jax.jvp Computes a (forward-mode) Jacobian-vector product of fun
            V, f_jvp = jax.linearize(return_V_star_vector_mse, params, exp_x)
            grad_dV = jax.grad(f_jvp)(wdot, xdot)
            dV= f_jvp(wdot, xdot)        
            def combine_grad(x, y, z, factor=1):
                return (x+y)+factor*z
            grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)
            return grad,( (V+dV), V, dV, \
                f_jvp(wdot, zero_dx),  f_jvp(zero_dtheta, xdot))
        
        #-----------Train A,B only-------------------#
        else:
            #model = eqx.combine(params, statics)
            #pred_y = jax.vmap(model)(x)
            #print("shape of pred_y: ", pred_y.shape)
            #print("shape of y:", y.shape)
            #pred_y = pred_y.squeeze(1)
            #print("shape of pred_y after squeeze:", pred_y.shape)
            def return_V_star_vector_mse(params, x):
                y = extra
                model = eqx.combine(params, statics)
                #print("the shape of x in TRAINER: ", x.shape)
                pred_y = jax.vmap(model.getAWB)(x) #changed to give preds using AWB
                #print("the shape of pred_y in getAWB: ", pred_y.shape)
                #pred_y = pred_y.squeeze(1)
                #print("the shape of pred_y in getAWB: ", pred_y.shape)
                #print(optax.l2_loss(y, pred_y))
                return jnp.mean(optax.l2_loss(y, pred_y))
            def norm_param(x):
                return (x*-1) # (-1*1e-0/jnp.sqrt(jnp.linalg.norm(x**2))))
            xdot = deltax
            zero_dx = jnp.zeros(xdot.shape)
            delta_theta= jax.grad(return_V_star_vector_mse,argnums=(0))(params, x) 
            wdot= jax.tree_util.tree_map(norm_param, delta_theta)
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
            extra = exp_y
            grad_V =  jax.grad(return_V_star_vector_mse,argnums=(0))(params, exp_x) 
            V, f_jvp = jax.linearize(return_V_star_vector_mse, params, exp_x)
            grad_dV = jax.grad(f_jvp)(wdot, xdot)
            dV= f_jvp(wdot, xdot)        
            def combine_grad(x, y, z, factor=1):
                return (x+y)+factor*z
            grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)
            return grad,( (V+dV), V, dV, \
                f_jvp(wdot, zero_dx),  f_jvp(zero_dtheta, xdot))
        
#===============Hamiltonian_class_AWB function (i.e. with flag)=================#
def return_Hamiltonian_class_AWB(self, params, data, notABTrain=True):
        statics, (x, y, exp_x, exp_y, deltax, flag)  = data
        extra=y
        #if 'notABTrain == True' then run as usual ------------------------#
        if (notABTrain ==True):
            def return_V_star_class(params, x):
                y = extra
                model = eqx.combine(params, statics)
                y=y.astype(jnp.int64)
                #print("shape of y: ", y.shape)
                #pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                #print("shape of pred: ", pred_y.shape)
                #print("mean: ", jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y)))
                return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels( pred_y,y ))
                # pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)    
                # print(pred_y.shape)
                # return loss 
            def norm_param(x):
                return (-1*x) #*(-1*1e-04/jnp.sqrt(jnp.linalg.norm(x**2))))
            xdot = deltax
            zero_dx = jnp.zeros(xdot.shape)
            delta_theta= jax.grad(return_V_star_class,argnums=(0))(params, x) 
            wdot= jax.tree_util.tree_map(norm_param, delta_theta)
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
            extra = exp_y
            
            grad_V =  jax.grad(return_V_star_class,argnums=(0))(params, exp_x) 
            V, f_jvp = jax.linearize(return_V_star_class, params, exp_x)
            grad_dV = jax.grad(f_jvp)(wdot, xdot)
            dV= f_jvp(wdot, xdot)
            
            def combine_grad(x, y, z, factor=1):
                return (x+y)+factor*z
        

            grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)
            return grad,( (V+dV), V, dV, \
                        f_jvp(wdot, zero_dx),\
                        f_jvp(zero_dtheta, xdot))
        # if notABTrain == False and we want to train on AB's only-----------------#
        else:
            def return_V_star_classAWB(params, x):
                y = extra
                model = eqx.combine(params, statics)
                y=y.astype(jnp.int64)
                #print("shape of y: ", y.shape)
                #pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                pred_y = jax.vmap(model.get_AWBT)(x)
                #print(pred_y.dtype)
                pred_y = jax.nn.log_softmax(pred_y)
                #print("mean: ", jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y)))
                return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels( pred_y,y ))
                # pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)    
                # print(pred_y.shape)
                # return loss 
            def norm_param(x):
                return (-1*x) #*(-1*1e-04/jnp.sqrt(jnp.linalg.norm(x**2))))
            xdot = deltax
            zero_dx = jnp.zeros(xdot.shape)
            delta_theta= jax.grad(return_V_star_classAWB,argnums=(0))(params, x) 
            wdot= jax.tree_util.tree_map(norm_param, delta_theta)
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
            extra = exp_y
            
            grad_V =  jax.grad(return_V_star_classAWB,argnums=(0))(params, exp_x) 
            V, f_jvp = jax.linearize(return_V_star_classAWB, params, exp_x)
            grad_dV = jax.grad(f_jvp)(wdot, xdot)
            dV= f_jvp(wdot, xdot)
            
            def combine_grad(x, y, z, factor=1):
                return (x+y)+factor*z
        

            grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)
            return grad,( (V+dV), V, dV, \
                        f_jvp(wdot, zero_dx),\
                        f_jvp(zero_dtheta, xdot))
        
#===============CL Train Regression AWB Function (i.e. with flag)=================#  
def train__CL__reg_AWB(self, train__, params,\
                static, optim, n_iter=1000,\
                save_iter=10, task_id=0,\
                config={}, dictum = {}, notABTrain = True):

    trainloader, exploader, valloader, testloader=train__
    trainiter = iter(trainloader)
    expiter = iter(exploader)
    # optim_inner_x, optim_inner_mod = optim_inner
    batch = next(trainiter)
    #print("This is a batch: ", batch)
    #print("This is static", static)
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
        grad, losses =  self.return_Hamiltonian_mse_AWB(params, data, notABTrain)      
        (H, V, dV, dVstar_dtheta, dVstar_dx)  = losses         
        
        grad_leav = jax.tree_util.tree_leaves(grad)
        grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
        updates, opt_state = optim.update(grad, opt_state, params)
        params =  optax.apply_updates(params, updates)
        # print("the details", task_id, step, step+task_id*n_iter )
        pbar.set_postfix({"Train/MSE:": V,
                            "Train/dVstar_dx:": dVstar_dx,
                            "Train/dVstar_dtheta:": dVstar_dtheta,
                            "Train/H:":  H,
                            "Train/||dH_dtheta||:": grad_norm,
                            "Train/Metric:": V
                        })
        #if notABTrain:
        self.writer.add_scalar('train/Loss/H', H.item(), step+task_id*n_iter)
        self.writer.add_scalar('train/Loss/MSE', V.item(), step+task_id*n_iter)
        self.writer.add_scalar('train/Loss/MSE', dV.item(), step+task_id*n_iter)
        self.writer.add_scalar('train/gradient/dVstar_dx',
                            dVstar_dx.item(), step+task_id*n_iter)
        self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
        self.writer.add_scalar('train/gradient/dH_dtheta',
                            grad_norm.item(), step+task_id*n_iter)
        dictum["train"+str(step+task_id*n_iter)] = ( V, dV, dVstar_dx, dVstar_dtheta, \
            H,\
            grad_norm, grad_norm )

        ## Validation Metric calculations on the total exp_replay
        if step %100==0:
            sum_delta_x=0.
            V_star_max=[]
            dVstar_dx=[]
            dVstar_dtheta=[]
            H=[]
            dV=[]
            loader_1, loader_2= valloader
            for (batch_x, batch_ex) in zip(loader_1, loader_2):
                (x, y) = batch_x
                (exp_x, exp_y) = batch_ex
                x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                delta_x = jnp.abs(exp_x-x)
                sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
                delta_x = (delta_x/sum_delta_x)
                data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
                _, losses = self.return_Hamiltonian_mse_AWB(params,data,notABTrain = True)         
                (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
                V_star_max.append(v)
                dVstar_dx.append(dvstar_dx)
                dVstar_dtheta.append(dvstar_dtheta)
                H.append(h)
                dV.append(dv)
            V_star_max=np_.mean(V_star_max)
            dVstar_dx= np_.mean(dVstar_dx)
            dVstar_dtheta=np_.mean(dVstar_dtheta)
            dV=np_.mean(dV)
            H=np_.mean(H)
            # #print(H,  dVstar_dx, dVstar_dtheta)
            # pbar.set_postfix({"Valid/MSE:": V_star_max,
            #               "Train/dVstar_dx:": dVstar_dx,
            #               "Train/dVstar_dtheta:": dVstar_dtheta,
            #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
            #               "Train/||dH_dtheta||:": grad_norm\
            #             })

            #if notABTrain:
            self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
            self.writer.add_scalar('Valid/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
            self.writer.add_scalar('Valid/Loss/dV', dV.item(), step+task_id*n_iter)
            self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
            self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
            
            dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
            V_star_max+dVstar_dx+dVstar_dtheta )
            
    ## Test Metric calculations on the total exp_replay
    sum_delta_x=0.
    V_star_max=[]
    dVstar_dx=[]
    dVstar_dtheta=[]
    H=[]
    loader_1, loader_2= valloader
    for (batch_x, batch_ex) in zip(loader_1, loader_2):
        (x, y) = batch_x
        (exp_x, exp_y) = batch_ex
        x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    
        delta_x = (exp_x-x)
        sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
        delta_x = (delta_x/sum_delta_x)
        data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
        _, losses = self.return_Hamiltonian_mse_AWB(params,data,notABTrain)         
        (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
        V_star_max.append(v)
        dVstar_dx.append(dvstar_dx)
        dVstar_dtheta.append(dvstar_dtheta)
        H.append(h)
        
            
    V_star_max=np_.mean(V_star_max)
    dVstar_dx= np_.mean(dVstar_dx)
    dVstar_dtheta=np_.mean(dVstar_dtheta)
    H=np_.mean(H)
    self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter)
    self.writer.add_scalar('test/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
    self.writer.add_scalar('test/gradient/dVstar_dx',
                        dVstar_dx.item(), step+task_id*n_iter)
    self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
    dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H)
    
    self.writer.flush()
    return params, static, optim, dictum

#===============CL Train Class AWB Function (i.e. with flag)=================#
def train__CL__class(self, train__, params,\
                    static, optim, n_iter=1000,\
                    save_iter=10, task_id=0,\
                    config={}, dictum = {},notABTrain = True):

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
        # #print("Now the flag is ", flag)
        # jax.value_and_grad(self.return_loss_function_CL, has_aux=True)
        # grad_loss_fn_inner = jax.value_and_grad(self.return_loss_function_CL_inner)
        # grad_loss_fn_inner_mod = jax.value_and_grad(self.return_loss_function_CL_inner_mod)
        # start_iter_inner = task_id*n_iter*inner_iter
        sum_delta_x =0.
        mm=0.
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
            grad, losses =  self.return_Hamiltonian_class_AWB(params, data, notABTrain)    
            (H, V, dV, dVstar_dtheta, dVstar_dx)  = losses         
            grad_leav = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
            updates, opt_state = optim.update(grad, opt_state, params)
            params =  optax.apply_updates(params, updates)
            # print("the details", task_id, step, step+task_id*n_iter )
            
            
            
            
            pbar.set_postfix({"Train/Cross:": V,
                              "Train/dVstar_dx:": dVstar_dx,
                              "Train/dVstar_dtheta:": dVstar_dtheta,
                              "Train/H:":  H,
                              "Train/||dH_dtheta||:": grad_norm,
                              "Train/Metric:": mm
                            })
            
            self.writer.add_scalar('train/Loss/H', H.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/Loss/Cross', V.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/Loss/dV', dV.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dx',
                                   dVstar_dx.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dH_dtheta',
                                   grad_norm.item(), step+task_id*n_iter)
            
            
                        
            dictum["train"+str(step+task_id*n_iter)] = ( V, dV, dVstar_dx, dVstar_dtheta, \
                H,\
                grad_norm, grad_norm )

            ## Validation Metric calculations on the total exp_replay
            if step %100==0:
                sum_delta_x=0.
                V_star_max=[]
                dVstar_dx=[]
                dVstar_dtheta=[]
                H=[]
                dV=[]
                metrics =[]
                loader_1, loader_2= valloader
                for (batch_x, batch_ex) in zip(loader_1, loader_2):
                    (x, y) = batch_x
                    (exp_x, exp_y) = batch_ex
                    x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    delta_x = jnp.abs(exp_x-x)
                    sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
                    delta_x = (delta_x/sum_delta_x)
                    data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
                    _, losses = self.return_Hamiltonian_class_AWB(params,data, notABTrain)         
                    (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
                    V_star_max.append(v)
                    dVstar_dx.append(dvstar_dx)
                    dVstar_dtheta.append(dvstar_dtheta)
                    H.append(h)
                    dV.append(dv)
                    
                    metrics.append(self.return_metric_AWB(params, static, data=(exp_x, exp_y), notABTrain=notABTrain))
                                   
                V_star_max=np_.mean(V_star_max)
                dVstar_dx= np_.mean(dVstar_dx)
                dVstar_dtheta=np_.mean(dVstar_dtheta)
                dV=np_.mean(dV)
                H=np_.mean(H)
                mm=(sum(metrics)/len(metrics))
                # print(len(metrics), sum(metrics), mm)
                # #print(H,  dVstar_dx, dVstar_dtheta)
                # pbar.set_postfix({"Valid/MSE:": V_star_max,
                #               "Train/dVstar_dx:": dVstar_dx,
                #               "Train/dVstar_dtheta:": dVstar_dtheta,
                #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
                #               "Train/||dH_dtheta||:": grad_norm\
                #             })
                
                self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/Loss/Cross', V_star_max.item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/Loss/dV', dV.item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                self.writer.add_scalar('valid/metrics/ACC', mm.item(), step+task_id*n_iter)
                
                dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta, mm.item() )
                
        ## Test Metric calculations on the total exp_replay
        sum_delta_x=0.
        V_star_max=[]
        dVstar_dx=[]
        dVstar_dtheta=[]
        H=[]
        metrics=[]
        loader_1, loader_2= testloader
        for (batch_x, batch_ex) in zip(loader_1, loader_2):
            (x, y) = batch_x
            (exp_x, exp_y) = batch_ex
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
            delta_x = (exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)
            data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
            _, losses = self.return_Hamiltonian_class_AWB(params,data, notABTrain)         
            (h, v, dv, dvstar_dtheta, dvstar_dx) = losses  
            V_star_max.append(v)
            dVstar_dx.append(dvstar_dx)
            dVstar_dtheta.append(dvstar_dtheta)
            H.append(h)
            metrics.append(self.return_metric_AWB(params, static, data = (exp_x, exp_y),notABTrain = notABTrain))
            
        V_star_max=np_.mean(V_star_max)
        dVstar_dx= np_.mean(dVstar_dx)
        dVstar_dtheta=np_.mean(dVstar_dtheta)
        H=np_.mean(H)
        mm = sum(metrics)/len(metrics)
        self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter)
        self.writer.add_scalar('test/Loss/Cross', V_star_max.item(), step+task_id*n_iter)
        self.writer.add_scalar('test/gradient/dVstar_dx',
                            dVstar_dx.item(), step+task_id*n_iter)
        self.writer.add_scalar('test/metrics/ACC',
                            mm.item(), step+task_id*n_iter)
        self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
        dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H, mm)
        
        self.writer.flush()
        return params, static, optim, dictum
    


        
                
