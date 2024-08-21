from torch.utils.tensorboard import SummaryWriter
import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np_
import pickle
import optax
import torch_geometric as pyg
from utils.utils import *
jax.config.update("jax_enable_x64", True)

class Trainer(eqx.Module):
    writer: SummaryWriter
    loss: str
    problem:str
    metric: str
    dict: dict()

    def __init__(self, logdir="runs", Loss='mse', metric='mse', problem='vectors'):
        self.writer= SummaryWriter(logdir)
        self.loss=Loss
        self.problem=problem
        self.metric=metric
        self.dict={}
    #---------------------------------------------- Vectors & matrices
    #------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
        return -jnp.mean(y * pred_y)

    @eqx.filter_jit
    def loss_fn_mse(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model)(x))**2)
    
    @eqx.filter_jit
    def accuracy_vectors(self,params, statics, x, y):
        model = eqx.combine(params, statics)
        pred = jnp.argmax( jax.nn.softmax(jax.vmap(model)(x)), axis=1) 
        y = jnp.argmax( y, axis=1) 
        return jnp.mean(pred == y)
    
    @eqx.filter_jit
    def mse_vectors(self,params, statics,  x, y):
        model = eqx.combine(params, statics)
        return jnp.mean( optax.l2_loss(y, jax.vmap(model)(x)) )


    #------------------------------------------------------------ Graphs 
    #-------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        logits = jnp.stack([ model(x[i], adj[i]).T for i in range(len(x))]) 
        pred_y = jnp.stack(logits)        
        y=y.astype(jnp.int64)
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(y * pred_y)
    
    @eqx.filter_jit
    def loss_fn_mse_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model(x, adj)))**2)
    
    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, adj, b, n):
        model = eqx.combine(params, statics)
        array_log = [ model(x[i], adj[i], b[i], n[i]) for i in range(len(x))]
        logits = jnp.concatenate(array_log)
        return jax.nn.log_softmax(logits, axis = 1)
    
    
    # -------------------------------------------------------------------
    @eqx.filter_jit
    def mse_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jnp.mean((jax.vmap(model)(x, adj) - y)**2 )
    

    # ------------------------------------------------------------ Graphs
    

    @eqx.filter_jit
    def get_pred(self, params, statics, x):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x)

    @eqx.filter_jit
    def get_pred_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x, adj)
    # -------------------------------------------------------------------
    def return_loss_grad(self, params, batch, static):
        if self.problem=='vectors':
            (x, y) = batch
            if self.loss == 'class':
                grads =jax.grad(self.loss_fn_class)(params, static, x, y)
                loss = self.loss_fn_class(params, static, x, y)
            elif self.loss=='mse':
                grads= jax.grad(self.loss_fn_mse)(params, static, x, y)
                loss  =self.loss_fn_mse(params, static, x, y)
        elif self.problem== 'graph':
            (x, y, adj) = batch
            if self.loss == 'class':
                grads  =jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
                loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
            elif self.loss=='mse':
                grads  =jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
                loss =  self.loss_fn_mse_graph(params, static, x, y, adj=adj)
        return loss, grads
    
    def writer(self, dict, epoch, string_scalers= ['train'], metric_scaler=['training_loss', 'validation_loss', 'loss', 'acc']):
        for (string, metric) in zip(string_scalers, metric_scaler):
            self.writer.add_scalar(str(string), dict[metric], epoch)
        pickle.dump( dict['params'], open("best_ckpt.pkl"), "wb")

    # # -------------------------------------------------------------------
    # @eqx.filter_jit
    # def return_V_star_vector_mse(self, x, params, data):
    #     statics, y = data
    #     model = eqx.combine(params, statics)
    #     if self.loss == "class":
    #         # #print("1The problem is", self.problem, "The loss is", self.loss)
    #         y=y.astype(jnp.int64)
    #         pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
    #         pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    #         return -jnp.mean(y * pred_y)
    #     elif self.loss=="mse":
    #         return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
            
    # # -------------------------------------------------------------------
    # @eqx.filter_jit
    # def return_V_star_vector_class(self, x, params, data):
    #     statics, y = data
    #     model = eqx.combine(params, statics)
    #     y=y.astype(jnp.int64)
    #     pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
    #     pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    #     return -jnp.mean(y * pred_y)
                              
                              
        # if self.problem=='vectors':
        #     # #print("The problem is", self.problem, "The loss is", self.loss)
        #     statics, y = data
        #     model = eqx.combine(params, statics)
        #     if self.loss == "class":
        #         # #print("1The problem is", self.problem, "The loss is", self.loss)
        #         y=y.astype(jnp.int64)
        #         pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
        #         pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        #         return -jnp.mean(y * pred_y)
        #     elif self.loss=="mse":
        #         return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
        # elif self.problem=='graph':
        #     statics, y, adj, batch, n_nodes = data
        #     model = eqx.combine(params, statics)
        #     if self.loss == 'class':
        #         logits = [model(x[i], adj[i], batch[i], n_nodes[i])\
        #             for i in range(len(x)) ] 
        #         pred_y = jnp.concatenate(logits)        
        #         y=jnp.concatenate(y).astype(jnp.int64)
        #         # print("value", y, pred_y)
        #         # print("shape",  y.shape, pred_y.shape)
        #         return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y,y))
        
        #                     elif self.loss=='mse':
        #         logits = jnp.stack([ model(x[i], adj[i]).T for i in range(len(x))]) 
        #         pred_y = jnp.stack(logits)        
        #         return jnp.mean((y-pred_y)**2)
        # #print("I am not going where I am suppsoed to")
    # -------------------------------------------------------------------
    def return_metric(self, params, statics, data):
        model = eqx.combine(params, statics)
        if self.problem=='vectors':
            x, y = data
            if self.metric == 'class':
                y=y.astype(jnp.int64)
                pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
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


        # ----------------------------------------------------------------------------------
        # elif self.problem== 'graphs':
        #     # #print("I came here and wemnt to calculate the loss")
        #     (x, y, adj) = batch
        #     if self.loss == 'class':
        #         grads  =jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
        #         loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
        #     elif self.loss=='mse':
        #         grads  =jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
        #         loss =  self.loss_fn_mse_graph(params, static, x, y, adj=adj)
        # return loss
    # -------------------------------------------------------------------
    # The following is the code for graphs.
    # -------------------------------------------------------------------
    # def return_jvp_graph(self, params,  data):
    #     (fun, ex, eadj, delta_theta, xdot, adjdot, static, ey, eb, en) = data
    #     primal = (params, ex, eadj)
    #     tangents= (delta_theta, xdot, adjdot)
    #     aux = ( static, ey, eb, en) 
    #     primal_val, tang_val = eqx.filter_jvp(fun, primal,\
    #                                           tangents, data = aux) 
    #     return primal_val+tang_val, (primal_val.item(), tang_val.item())
    
    def return_Hamiltonian_graph(self, params, data):
        static, (batch, batch_ex, deltax, delta_adj) = data
        x = jnp.float64(jnp.array(batch.x.numpy()))
        y = jnp.int64(jnp.array(batch.y.numpy()))
        adj = jnp.float64(jnp.array(batch.adj.numpy()))
        b = jnp.array(batch.batch.numpy())
        n = jnp.array(batch.n_nodes.numpy())
        ex = jnp.float64(jnp.array(batch_ex.x.numpy()))
        ey = jnp.int64(jnp.array(batch_ex.y.numpy()))
        eadj = jnp.float64(jnp.array(batch_ex.adj.numpy()))
        eb = jnp.array(batch_ex.batch.numpy())
        en = jnp.array(batch_ex.n_nodes.numpy())
        extra = (y, b, n)
        def return_V_star_graph(params, xx, xxadj):
            (yy, bb, nn) = extra
            model = eqx.combine(params, static)
            pred_y = model(xx, xxadj, bb, nn )
            # print(jnp.dtype(yy), jnp.dtype(pred_y))
            loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels( pred_y,yy ))
            return loss
        
        def norm_param(x):
            return (x*(-1*1e-04/jnp.sqrt(jnp.linalg.norm(x**2))))
        xdot = deltax
        zero_dx = jnp.zeros(xdot.shape)
        delta_theta= jax.grad(return_V_star_graph,argnums=(0))(params, x, adj) 
        wdot= jax.tree_util.tree_map(norm_param, delta_theta)
        zero_dtheta = jax.tree_util.tree_map(np.zeros_like, delta_theta)
        adjdot = delta_adj
        zero_dadj = jnp.zeros(adjdot.shape)
        extra = (ey, eb, en)
        grad_V = jax.grad(return_V_star_graph, argnums=(0))(params, ex, eadj)
        
        V, f_jvp = jax.linearize(return_V_star_graph, params, ex, eadj)
        grad_dV = jax.grad(f_jvp)(wdot, xdot, adjdot)
        dV= f_jvp(wdot, xdot, adjdot)
        def combine_grad(x, y, z, factor=1e-05):
            return (x+y)+factor*z
        grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)
        return grad,((V+dV), V, dV, \
                    f_jvp(wdot, zero_dx, zero_dadj),\
                    f_jvp(zero_dtheta, xdot, zero_dadj),\
                    f_jvp(zero_dtheta, zero_dx, adjdot)) 
    # -----------------------------------------------------------------------------------------------
    def train__CL__graph(self, train__, params, static,\
                        optim, n_iter=1000, save_iter=5, dictum = {}, 
                        task_id=0, config={}):
        # #print(x)
        # #print(y)
        # x = x.numpy().astype(np_.float64)
        # adj = adj.numpy().astype(np_.float64)
        # y = jax.nn.one_hot(jnp.array(y.astype(np_.int32)), 7)
        memory_train, test, train = train__
        trainiter = iter(train)
        expiter = iter(memory_train)
        opt_state = optim.init(params)
        from tqdm import tqdm
        pbar = tqdm(range(n_iter), dynamic_ncols=True)
        # sum_delta_x =0
        V_star_max=[]
        dVstar_dx=[]
        dVstar_dtheta=[]
        dVstar_dadj=[]
        H=[]
        metrics=[]
        import torch_geometric.transforms as T
        transforms = T.Compose([T.GCNNorm(), T.ToDense(), T.NormalizeFeatures()])
        
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
            trainiter = iter(train)
            expiter = iter(memory_train)
            for batch, batch_ex in zip(trainiter, expiter):
                batch = transforms(batch)
                batch_ex=transforms(batch_ex)
                # ---------------------------------------------------------------------
                # How do you ensure that these distances are reliable or rather not flipping 
                # all over the place
                # First problem, a distance metric, that does not care about the size of the nodes.
                x = batch.x.numpy()
                exp_x = batch_ex.x.numpy()
                # var__=jnp.sqrt(jnp.linalg.norm(  (jnp.mean( x, axis =0)-jnp.mean(exp_x, axis = 0 ) )**2 ) )
                # #print(var__)
                # delta_x = np_.random.normal(0, var__, exp_x.shape) 
                # var__= ( jnp.sqrt( jnp.linalg.norm(batch.adj.numpy()**2) )- jnp.sqrt(jnp.linalg.norm(batch_ex.adj.numpy()**2) ) )**2
                # #print(var__)
                # delta_adj = np_.random.normal(0, var__, batch_ex.adj.shape) 
                # #--------------------------------------------------------------------------
                delta_x = np_.random.normal(0, var_x, exp_x.shape)       
                delta_adj = np_.random.normal(0, var_adj, batch_ex.adj.shape)   
                data = (static, (batch, batch_ex, delta_x, delta_adj) )
                grad, losses = self.return_Hamiltonian_graph(params, data)         
                (h, V, dV, dv_dtheta, dv_dx, dv_dadj)     = losses
                updates, opt_state = optim.update(grad, opt_state, params)
                params =  optax.apply_updates(params, updates)
                # ------------------------------------------------------------------
                #Updated the parameters, now working on storing and viewing things.
                V_star_max.append(V)
                dVstar_dx.append(dv_dx)
                dVstar_dtheta.append(dv_dtheta)
                # print(jnp.linalg.norm(delta_adj), dvstar_dadj, dvstar_dx, dvstar_dtheta)
                dVstar_dadj.append(dv_dadj)
                 # +config['flag'][0]*dvstar_dx+config['flag'][0]*dvstar_dadj+config['flag'][1]*dvstar_dtheta    
                H.append(h)
                metrics.append( self.return_metric(params, static, data = (batch, batch_ex)))
            if step % save_iter ==0:    
                # print("going into the save iteration")
                # ------------------------------------------------------
                V_star_maxtr=np_.mean(dV)
                dVstar_dxtr= np_.mean(dVstar_dx)
                dVstar_dthetatr=np_.mean(dVstar_dtheta)
                dVstar_dadjtr=np_.mean(dVstar_dadj)
                Htr=np_.mean(H)
                metricstr=np_.mean(metrics)
                
                # V_star_max=[]
                # dVstar_dx=[]
                # dVstar_dtheta=[]
                # dVstar_dadj=[]
                # H=[]
                metrics=[]
                iter_t = iter(train)
                for batch in iter_t:
                    batch = transforms(batch)
                    # # print("after the initial step")
                    # x = jnp.array(batch.x.numpy())
                    # y = jnp.array(batch.y.numpy())
                    # adj = jnp.array(batch.adj.numpy())
                    # b = jnp.array(batch.batch.numpy())
                    # n = jnp.array(batch.n_nodes.numpy())
                    # ex = jnp.array(batch.x.numpy())
                    # ey = jnp.array(batch.y.numpy())
                    # eadj = jnp.array(batch.adj.numpy())
                    # eb = jnp.array(batch.batch.numpy())
                    # en = jnp.array(batch.n_nodes.numpy())
                    # #-------------------------------------------------------------------------------------------------------------
                    # # print("Term1")
                    # #-----------------------------------------------------------------------------------------------------------
                    # # The directions overwhich I calculate my jacobian 
                    # #---------------------------------------------------------------------------------
                    # # min_shape = min(exp_x.shape[0], x.shape[0])
                    # deltax = np_.random.normal(0, 1e-05+jnp.sqrt(jnp.linalg.norm( jnp.mean( x, axis =0)\
                    #                         - jnp.mean(x, axis = 0 ) ) ), x.shape) 
                    
                    # delta_adj = np_.random.normal(0,1e-05+jnp.linalg.norm( jnp.linalg.norm(
                    #                                 batch.adj.numpy())\
                    #                         - jnp.linalg.norm(batch_ex.adj.numpy()) ) , batch.adj.shape) 
                    
                    # xdot = jnp.float32(deltax)
                    # delta_theta = jax.grad(self.return_V_star_graph,argnums=(0) )(params, x, adj, data = (static, y,  b, n) ) 
                    # adjdot = jnp.float32(delta_adj)
                    # data = (self.return_V_star_graph, ex, eadj,  delta_theta, xdot, adjdot, static, ey, eb, en)
                    # h, (V, dV) = self.return_jvp_graph(params,  data)      
                    # # print("after_hamiltonian")
                    # V_star_max.append(V)
                    # dVstar_dx.append(dV)
                    # dVstar_dtheta.append(dV)
                    # dVstar_dadj.append(dV)
                    # H.append(h)  
                    metrics.append( self.return_metric(params, static, data = (batch, batch)) )
                # ------------------------------------------------------
                # V_star_max=np_.mean(V_star_max)
                # dVstar_dx= np_.mean(dVstar_dx)
                # dVstar_dtheta=np_.mean(dVstar_dtheta)
                # dVstar_dadj=np_.mean(dVstar_dadj)
                # H=np_.mean(H)
                metrics=np_.mean(metrics)
                pbar.set_postfix( 
                                {
                                "H:": Htr,
                                "dV": V_star_maxtr.item(),
                                "train/Metric:": metricstr,
                                "test/Metric:": metrics,
                                "dvstar_dx": round(dVstar_dxtr,6),
                                "dVstar_dtheta:": round(dVstar_dthetatr,6),
                                "dVstar_dadj:": round(dVstar_dadjtr,6),
                                })
                # ------------------------------------------------------
                # self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter )
                # self.writer.add_scalar('test/Loss/cross entropy', V_star_max.item(), step+task_id*n_iter )
                # self.writer.add_scalar('test/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                # self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                # self.writer.add_scalar('test/gradient/dVstar_dadj', dVstar_dadj.item(), step+task_id*n_iter)
                self.writer.add_scalar('train/Loss/H', Htr.item(),step+task_id*n_iter )
                self.writer.add_scalar('train/Loss/cross entropy', V_star_maxtr.item(), step+task_id*n_iter )
                self.writer.add_scalar('train/gradient/dVstar_dx', dVstar_dxtr.item(), step+task_id*n_iter)
                self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dthetatr.item(), step+task_id*n_iter)
                self.writer.add_scalar('train/gradient/dVstar_dadj', dVstar_dadjtr.item(), step+task_id*n_iter)
                self.writer.add_scalar('train/metric', metricstr, task_id)
                self.writer.add_scalar('test/metric', metrics, task_id)
                dictum["train"+str(step+task_id*n_iter)] =\
                ( V_star_maxtr, dVstar_dxtr, dVstar_dthetatr, dVstar_dadjtr, Htr, metricstr, metrics)
                # dictum["test"+str(step+task_id*n_iter)] =\
                # ( V_star_max,dVstar_dx, dVstar_dtheta,\
                # V_star_max+dVstar_dx+dVstar_dtheta, metrics)
                
                V_star_max=[]
                dVstar_dx=[]
                dVstar_dtheta=[]
                dVstar_dadj=[]
                H=[]
                metrics=[]
        self.writer.flush()
        return params, static, optim, dictum 



### The following is the software block for mse based optimization\
# and classification based optimization, check the graph one thoroughly 
# and then uncomment this
    # #-----------------------------------------------------------
    # # The following is the code for MSE
    # # -------------------------------------------------------------------
    # # -------------------------------------------------------------------
    # @eqx.filter_jit
    # def return_V_star_vector_mse(self, params, x, data=None):
    #     statics, y = data
    #     model = eqx.combine(params, statics)
    #     return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
    
    # @eqx.filter_jit
    # def return_jvp_mse(self, params,  data):
    #     (fun, ex,  delta_theta, xdot,  static, ey) = data
    #     primal = (params, ex)
    #     tangents= (delta_theta, xdot)
    #     aux = ( static, ey) 
    #     primal_val, tang_val = eqx.filter_jvp(fun, primal, tangents, data = aux)
    #     return (primal_val+1e-5*tang_val), (primal_val.item(), tang_val.item())       
    
    # def return_Hamiltonian_mse(self, params, data):
    #     static, (x, y, exp_x, exp_y, deltax, flag) = data 
    #     xdot = jnp.float32(deltax)
    #     delta_theta   = jax.grad(self.return_V_star_vector_mse,argnums=(1))(x, params, (static, y))            
    #     data = (self.return_V_star_vector_mse, exp_x, delta_theta, xdot, static, exp_y)
    #     return eqx.filter_grad(self.return_jvp_graph, has_aux=True)(params,  data)      


    # # -------------------------------------------------------------------
    # @eqx.filter_jit
    # def return_V_star_vector_class(self, x, params, data):
    #     statics, y = data
    #     model = eqx.combine(params, statics)
    #     y=y.astype(jnp.int64)
    #     pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
    #     pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    #     return -jnp.mean(y * pred_y)
                              
    # @eqx.filter_jit
    # def return_jvp_class(self, params,  data):
    #     (fun, ex,  delta_theta, xdot,  static, ey) = data
    #     primal = (params, ex)
    #     tangents= (delta_theta, xdot)
    #     aux = ( static, ey) 
    #     primal_val, tang_val = eqx.filter_jvp(fun, primal, tangents, data = aux)
    #     return (primal_val+1e-5*tang_val), (primal_val.item(), tang_val.item())       
    
    # def return_Hamiltonian_class(self, params, data):
    #     static, (x, y, exp_x, exp_y, deltax) = data 
    #     xdot = jnp.float32(deltax)
    #     delta_theta   = jax.grad(self.return_V_star_vector_class,argnums=(1))(x, params, (static, y))            
    #     data = (self.return_V_star_vector_class, exp_x, delta_theta, xdot, static, exp_y)
    #     return eqx.filter_grad(self.return_jvp_graph, has_aux=True)(params,  data)      
    
                      


    # def train__CL__reg(self, trainloader, exploader, valloader, testloader, params,\
    #                 static, optim_outer, n_iter=1000,\
    #                 save_iter=10, task_id=0,\
    #                 config={}, dictum = {}):
    #     trainiter = iter(trainloader)
    #     expiter = iter(exploader)
    #     # optim_inner_x, optim_inner_mod = optim_inner
    #     batch = next(trainiter)
    #     x, y = batch
    #     x = x.numpy().astype(np_.float64)
    #     y = y.numpy().astype(np_.float64)
    #     batch = (x, y)
    #     opt_state = optim_outer.init(params)
    #     from tqdm import tqdm 
    #     pbar = tqdm(range(n_iter))
    #     if task_id>0:
    #         flag=config["flag"]
    #     else:
    #         flag=config["flag"]
    #     # #print("Now the flag is ", flag)
    #     # jax.value_and_grad(self.return_loss_function_CL, has_aux=True)
    #     # grad_loss_fn_inner = jax.value_and_grad(self.return_loss_function_CL_inner)
    #     # grad_loss_fn_inner_mod = jax.value_and_grad(self.return_loss_function_CL_inner_mod)
    #     # start_iter_inner = task_id*n_iter*inner_iter
    #     sum_delta_x =0.
    #     for step in pbar:
            
    #         try:
    #             batch = next(trainiter)
    #         except StopIteration:
    #             trainiter = iter(trainloader)
    #             batch = next(trainiter)
    #         try:
    #             batch_ex = next(expiter)
    #         except StopIteration:
    #             expiter = iter(exploader)
    #             batch_ex = next(expiter)

    #         (x, y) = batch
    #         (exp_x, exp_y) = batch_ex
    #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            
    #         # #print(exp_x.shape, x.shape)
            
    #         delta_x = jnp.abs(exp_x-x)
    #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #         delta_x = (delta_x/sum_delta_x)            
    #         data = static, ( x, y, exp_x, exp_y, delta_x, flag)             
    #         grad, losses = jax.grad(self.return_Hamiltonian_mse,\
    #                     argnums=(0), has_aux=True)(params,data)         
    #         (V_star_max, dVstar_dx, dVstar_dtheta)  = losses         
    #         grad_leav = jax.tree_util.tree_leaves(grad)
    #         grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
            
            
    #         updates, opt_state = optim_outer.update(grad, opt_state, params)
    #         params =  optax.apply_updates(params, updates)
            
            
    #         # print("the details", task_id, step, step+task_id*n_iter )
    #         pbar.set_postfix({"Train/MSE:": V_star_max,
    #                           "Train/dVstar_dx:": dVstar_dx,
    #                           "Train/dVstar_dtheta:": dVstar_dtheta,
    #                           "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
    #                           "Train/||dH_dtheta||:": grad_norm,
    #                           "Train/Metric:": grad_norm
    #                         })
    #         self.writer.add_scalar('train/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dVstar_dx',
    #                                dVstar_dx.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dH_dtheta',
    #                                grad_norm.item(), step+task_id*n_iter)
            
            
                        
    #         dictum["train"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
    #             V_star_max+dVstar_dx+dVstar_dtheta,\
    #             grad_norm, grad_norm )

    #         ## Validation Metric calculations on the total exp_replay
    #         if step %100==0:
    #             sum_delta_x=0.
    #             V_star_max=[]
    #             dVstar_dx=[]
    #             dVstar_dtheta=[]
    #             H=[]
    #             loader_1, loader_2= valloader
    #             for (batch_x, batch_ex) in zip(loader_1, loader_2):
    #                 (x, y) = batch_x
    #                 (exp_x, exp_y) = batch_ex
    #                 x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 delta_x = jnp.abs(exp_x-x)
    #                 sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #                 delta_x = (delta_x/sum_delta_x)
    #                 data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
    #                 h, losses = self.return_Hamiltonian_mse(params,data)         
    #                 (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
    #                 V_star_max.append(V_star_max_b)
    #                 dVstar_dx.append(dVstar_dx_b)
    #                 dVstar_dtheta.append(dVstar_dtheta_b)
    #                 H.append(h)
                    
    #             V_star_max=np_.mean(V_star_max)
    #             dVstar_dx= np_.mean(dVstar_dx)
    #             dVstar_dtheta=np_.mean(dVstar_dtheta)
    #             H=np_.mean(H)
    #             # #print(H,  dVstar_dx, dVstar_dtheta)
    #             # pbar.set_postfix({"Valid/MSE:": V_star_max,
    #             #               "Train/dVstar_dx:": dVstar_dx,
    #             #               "Train/dVstar_dtheta:": dVstar_dtheta,
    #             #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
    #             #               "Train/||dH_dtheta||:": grad_norm\
    #             #             })
                
    #             self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
    #             self.writer.add_scalar('Valid/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
    #             self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
    #             self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                
    #             dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
    #             V_star_max+dVstar_dx+dVstar_dtheta )
                
    #     ## Test Metric calculations on the total exp_replay
    #     sum_delta_x=0.
    #     V_star_max=[]
    #     dVstar_dx=[]
    #     dVstar_dtheta=[]
    #     H=[]
    #     loader_1, loader_2= valloader
    #     for (batch_x, batch_ex) in zip(loader_1, loader_2):
    #         (x, y) = batch_x
    #         (exp_x, exp_y) = batch_ex
    #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
    #         delta_x = (exp_x-x)
    #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #         delta_x = (delta_x/sum_delta_x)
    #         data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
    #         h, losses = self.return_Hamiltonian_mse(params,data)         
    #         (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
    #         V_star_max.append(V_star_max_b)
    #         dVstar_dx.append(dVstar_dx_b)
    #         dVstar_dtheta.append(dVstar_dtheta_b)
    #         H.append(h)
            
                
                
    #     V_star_max=np_.mean(V_star_max)
    #     dVstar_dx= np_.mean(dVstar_dx)
    #     dVstar_dtheta=np_.mean(dVstar_dtheta)
    #     H=np_.mean(H)
    #     self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),step+task_id*n_iter)
    #     self.writer.add_scalar('test/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
    #     self.writer.add_scalar('test/gradient/dVstar_dx',
    #                         dVstar_dx.item(), step+task_id*n_iter)
    #     self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
    #     dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H)
        
    #     self.writer.flush()
        
        
    #     return params, static, optim_outer, dictum
    
  
    # def train__CL__class(self, trainloader, exploader, valloader, testloader, params,\
    #                 static, optim_outer, n_iter=1000,\
    #                 save_iter=10, task_id=0,\
    #                 config={}, dictum = {}):
    #     trainiter = iter(trainloader)
    #     expiter = iter(exploader)
    #     # optim_inner_x, optim_inner_mod = optim_inner
    #     batch = next(trainiter)
    #     x, y = batch
    #     x = x.numpy().astype(np_.float64)
    #     y = y.numpy().astype(np_.float64)
    #     batch = (x, y)
    #     opt_state = optim_outer.init(params)
    #     from tqdm import tqdm 
    #     pbar = tqdm(range(n_iter))
    #     if task_id>0:
    #         flag=config["flag"]
    #     else:
    #         flag=config["flag"]
    #     # #print("Now the flag is ", flag)
    #     # jax.value_and_grad(self.return_loss_function_CL, has_aux=True)
    #     # grad_loss_fn_inner = jax.value_and_grad(self.return_loss_function_CL_inner)
    #     # grad_loss_fn_inner_mod = jax.value_and_grad(self.return_loss_function_CL_inner_mod)
    #     # start_iter_inner = task_id*n_iter*inner_iter
    #     sum_delta_x =0.
    #     metrics=0.0
    #     for step in pbar:
            
    #         try:
    #             batch = next(trainiter)
    #         except StopIteration:
    #             trainiter = iter(trainloader)
    #             batch = next(trainiter)
    #         try:
    #             batch_ex = next(expiter)
    #         except StopIteration:
    #             expiter = iter(exploader)
    #             batch_ex = next(expiter)

    #         (x, y) = batch
    #         (exp_x, exp_y) = batch_ex
    #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            
    #         #--------------------------------------------------
    #         # print("in the train loop", exp_x.shape, x.shape)
    #         delta_x = jnp.abs(exp_x-x)
    #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #         delta_x = (delta_x/sum_delta_x)            
    #         data = static, ( x, y, exp_x, exp_y, delta_x, flag)             
    #         grad, losses = jax.grad(self.return_Hamiltonian_class,\
    #                     argnums=(0), has_aux=True)(params,data)         
    #         (V_star_max, dVstar_dx, dVstar_dtheta)  = losses         
    #         grad_leav = jax.tree_util.tree_leaves(grad)
    #         grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
    #         updates, opt_state = optim_outer.update(grad, opt_state, params)
    #         params =  optax.apply_updates(params, updates)
            
            
    #         # metric+= (1/100)*self.return_metric(x, params, (static, y)  )
    #         pbar.set_postfix({"Train/Cross Entropy:": V_star_max,
    #                           "Train/dVstar_dx:": dVstar_dx,
    #                           "Train/dVstar_dtheta:": dVstar_dtheta,
    #                           "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
    #                           "Train/||dH_dtheta||:": grad_norm,
    #                           "Train/Metric:": metrics
    #                         })
    #         self.writer.add_scalar('train/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/Loss/cross entropy', V_star_max.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dVstar_dx',
    #                                dVstar_dx.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/gradient/dH_dtheta',
    #                                grad_norm.item(), step+task_id*n_iter)
    #         self.writer.add_scalar('train/metric',
    #                                metrics, step+task_id*n_iter)
            
    #         dictum["train"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
    #             V_star_max+dVstar_dx+dVstar_dtheta, metrics)
            
            
    #         ## Validation Metric calculations on the total exp_replay
    #         if step %100==0:
    #             sum_delta_x=0.
    #             V_star_max=[]
    #             dVstar_dx=[]
    #             dVstar_dtheta=[]
    #             H=[]
    #             metrics=[]
    #             loader_1, loader_2= valloader
    #             for (batch_x, batch_ex) in zip(loader_1, loader_2):
    #                 (x, y) = batch_x
    #                 (exp_x, exp_y) = batch_ex
    #                 x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #                 exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                
    #                 delta_x = jnp.abs(exp_x-x)
    #                 sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #                 delta_x = (delta_x/sum_delta_x)
    #                 data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
    #                 h, losses = self.return_Hamiltonian_class(params,data)         
    #                 (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
    #                 V_star_max.append(V_star_max_b)
    #                 dVstar_dx.append(dVstar_dx_b)
    #                 dVstar_dtheta.append(dVstar_dtheta_b)
    #                 H.append(h)
                    
                    
    #                 metrics.append( self.return_metric(params, static, (exp_x, exp_y)  ) )
                    
    #             V_star_max=np_.mean(V_star_max)
    #             dVstar_dx= np_.mean(dVstar_dx)
    #             dVstar_dtheta=np_.mean(dVstar_dtheta)
    #             H=np_.mean(H)
    #             metrics= np_.mean(metrics)
    #             self.writer.add_scalar('Valid/Loss/H', (H).item(), step+task_id*n_iter)
    #             self.writer.add_scalar('Valid/Loss/cross entropy', V_star_max.item(), step+task_id*n_iter)
    #             self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
    #             self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
    #             self.writer.add_scalar('valid/metrics', metrics, step+task_id*n_iter)
    #             dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
    #             V_star_max+dVstar_dx+dVstar_dtheta, metrics)
    #     ## Test Metric calculations on the total exp_replay
    #     sum_delta_x=0.
    #     V_star_max=[]
    #     dVstar_dx=[]
    #     dVstar_dtheta=[]
    #     H=[]
    #     metrics=[]
    #     loader_1, loader_2= valloader
    #     for (batch_x, batch_ex) in zip(loader_1, loader_2):
    #         (x, y) = batch_x
    #         (exp_x, exp_y) = batch_ex
    #         x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
    #         exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
    #         delta_x = (exp_x-x)
    #         sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
    #         delta_x = (delta_x/sum_delta_x)
    #         data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
    #         h, losses = self.return_Hamiltonian_class(params,data)         
    #         (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
    #         V_star_max.append(V_star_max_b)
    #         dVstar_dx.append(dVstar_dx_b)
    #         dVstar_dtheta.append(dVstar_dtheta_b)
    #         H.append(h)
    #         metrics.append( self.return_metric(params, static, (exp_x, exp_y)  ) )
                
                
    #     V_star_max=np_.mean(V_star_max)
    #     dVstar_dx= np_.mean(dVstar_dx)
    #     dVstar_dtheta=np_.mean(dVstar_dtheta)
    #     H=np_.mean(H)
    #     metrics=np_.mean(metrics)
    #     self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),task_id)
    #     self.writer.add_scalar('test/Loss/cross entropy', V_star_max.item(), task_id)
    #     self.writer.add_scalar('test/gradient/dVstar_dx',
    #                         dVstar_dx.item(), task_id)
    #     self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), task_id)
    #     self.writer.add_scalar('test/metric', metrics, task_id)
    #     dictum["test"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
    #             V_star_max+dVstar_dx+dVstar_dtheta, metrics)
        
    #     self.writer.flush()
    #     return params, static, optim_outer, dictum

    
    # def return_batch(self, loader, iterator, batch_size):
    #     try:
    #         data= next(iterator)
    #         if data.y.shape[0]<batch_size:
    #             task_iter = iter(loader)
    #             return next(iterator)
    #         return data
    #     except StopIteration:
    #         task_iter = iter(loader)
    #         return next(iterator)
        
    

            
            
    # def evaluate__(self, epoch, batch, params, static):
        
    #     # --- Get Loss
    #     if self.problem=='vectors':
    #         (x, y) = batch
    #         if self.loss == 'class':
    #             loss = self.loss_fn_class(params, static, x, y)
    #         elif self.loss=='mse':
    #             loss, grads = self.return_loss_grad(params, (x,y), static)
    #             grads = jax.tree_util.tree_leaves(grads)
    #             grads = jnp.mean(jnp.asarray([jnp.linalg.norm(g) for g in grads]))
            
    #     elif self.problem== 'graphs':
    #         (x, y, adj) = batch
    #         if self.loss == 'class':
    #             loss  =self.loss_fn_class_graph(params, static, x, y, adj=adj)
    #         elif self.loss=='mse':
    #             loss  =self.loss_fn_mse_graph(params, static, x, y, adj=adj)

    #     # --- Get score
    #     if self.problem == 'vectors':
    #         (x, y) = batch
    #         if self.loss == 'class':
    #             score =self.accuracy_vectors(params, static, x, y)
    #         elif self.loss=='mse':
    #             score =self.mse_vectors(params, static, x, y)
    #     elif self.problem== 'graphs':
    #         (x, y, adj) = batch
    #         if self.loss == 'class':
    #             score = self.accuracy_graphs(params, static, x, y, adj=adj)
    #         elif self.loss=='mse':
    #             score =self.accuracy_graphs(params, static, x, y, adj=adj)
                
    #     # --- Get prediction
    #     if self.problem=='vectors':
    #         (x, _) = batch
    #         pred= self.get_pred(params, static, x)
    #     if self.problem == 'graphs':
    #         (x, _) = batch
    #         pred = self.get_pred(params, static, x)
                                

    #     return loss, score, pred, grads













        # V_star_max = self.return_V_star_graph(x, params, adj, data)
        # #-------------------------------------------------------------------------------------------------------------
        # # print("Term2")
        # #-----------------------------------------------------------------------------------------------------------
        # # Term 2
        # # print("before scaling", jnp.sqrt(jnp.linalg.norm(deltax)**2))
        # data = (static, ey,  eb, en, ex, ey, eadj, eb, en)
        # dVstar_dx = jax.grad(self.return_V_star_graph,argnums=(0))\
        #     (ex, params, eadj, data)     
        # deltax = deltax/jnp.sqrt(jnp.linalg.norm(deltax)**2)
        # dVstar_dx = dVstar_dx/jnp.sqrt(jnp.linalg.norm(dVstar_dx)**2)
        # dVstar_dx_deltax = jnp.sum(jnp.trace( jnp.inner(dVstar_dx, deltax ) )) 
        
        # #-------------------------------------------------------------------------------------------------------------
        # # print("Term3")
        # #-----------------------------------------------------------------------------------------------------------
        # # Term 3
        # # print("before scaling", jnp.sqrt(jnp.linalg.norm(delta_adj)**2))
        # data = (static, ey, eb, en, ex, ey, eadj, eb, en)
        # dVstar_dadj = jax.grad(self.return_V_star_graph,argnums=(2))(ex, params, eadj, data)
        # # delta_adj = delta_adj
        # # dVstar_dadj = dVstar_dadj
        # # print("the norms", jnp.sqrt(jnp.linalg.norm(delta_adj)**2), jnp.sqrt(jnp.linalg.norm(dVstar_dadj)**2))
        
        # print(delta_adj.shape, dVstar_dadj.shape)
        
        # dVstar_dx_deltaadj = jnp.sum( jnp.trace( jnp.inner(dVstar_dadj, delta_adj ) )) 
        # #-------------------------------------------------------------------------------------------------------------
        # # print("Term4")
        # #-------------------------------------------------------------------------------------------------------------
        # # Term 4
        
        # data = (static, y,  b, n, ex, ey, eadj, eb, en)
        # dVstar_dtheta = jax.grad(self.return_V_star_graph, argnums=(1))(x, params, adj, data)
        # dVstar_dtheta = jax.tree_util.tree_leaves(dVstar_dtheta)
        # delta_theta   = jax.tree_util.tree_leaves(delta_theta)
        # dVstar_dtheta_delta_theta=0.
        # for (dVstar_dw, delta_w)  in zip(dVstar_dtheta,  delta_theta):
        #     if len(delta_w.shape)>1:
        #         inner = jnp.inner(dVstar_dw, delta_w)
        #         dVstar_dtheta_delta_theta+=jnp.sum(jnp.trace(inner/jnp.linalg.norm(inner)) )
        #     else:
        #         inner = jnp.inner(dVstar_dw, delta_w)
        #         dVstar_dtheta_delta_theta+=jnp.sum(inner/jnp.linalg.norm(inner))
        
        # # dVstar_dx_deltax=0.0
        # # dVstar_dtheta_delta_theta=0.0
        # return  (V_star_max+flag[0]*dVstar_dx_deltax+flag[0]*dVstar_dx_deltaadj+flag[1]*dVstar_dtheta_delta_theta),\
        #         (V_star_max, dVstar_dx_deltax, dVstar_dtheta_delta_theta, dVstar_dx_deltaadj)
    
    
    
    # # -------------------------------------------------------------------
    # def return_Hamiltonian_mse(self, params, data):
    #     static, (x, y, exp_x, exp_y, deltax, flag) = data
    #     # #print("x", x.shape, "exp_x", exp_x.shape)
    #     x_tog = jnp.concatenate( [ x, exp_x])
    #     y_tog = jnp.concatenate( [ y, exp_y])      
    
    #     # Term 1 
    #     V_star_max = self.return_V_star_vector_mse(x_tog, params, (static, y_tog))
    #     #print(V_star_max)
    #     # Term 2
    #     dVstar_dx = jax.grad(self.return_V_star_vector_mse,argnums=(0))(exp_x, params, (static, exp_y))
    #     deltax = deltax/jnp.sqrt(jnp.linalg.norm(deltax)**2)
    #     dVstar_dx = dVstar_dx/jnp.sqrt(jnp.linalg.norm(dVstar_dx)**2)

    #     # #print("shape", dVstar_dx.shape)
    #     dVstar_dx_deltax = jnp.sum(jnp.trace( jnp.inner(dVstar_dx, deltax ) )) 
    #     # Term 3
    #     delta_theta   = jax.grad(self.return_V_star_vector_mse,argnums=(1))(x, params, (static, y))
    #     dVstar_dtheta = jax.grad(self.return_V_star_vector_mse, argnums=(1))(exp_x, params, (static, exp_y))
    #     dVstar_dtheta = jax.tree_util.tree_leaves(dVstar_dtheta )
    #     delta_theta   = jax.tree_util.tree_leaves(delta_theta )
    #     dVstar_dtheta_delta_theta=0.
    #     for (dVstar_dw, delta_w)  in zip(dVstar_dtheta,  delta_theta):
    #         # #print("the operand", dVstar_dw.shape, delta_w.shape)
    #         # delta_w = delta_w/jnp.sqrt(jnp.linalg.norm(delta_w)**2)
    #         dVstar_dw = dVstar_dw/jnp.sqrt(jnp.linalg.norm(dVstar_dw)**2)
                    
    #         if len(delta_w.shape)>1:
    #             inner = jnp.inner(dVstar_dw, delta_w)
    #             inner = inner
    #             # #print(inner.shape)
    #             dVstar_dtheta_delta_theta+=jnp.sum(jnp.trace(inner) )
    #         else:
    #             inner = jnp.inner(dVstar_dw, delta_w)
    #             # #print(inner.shape)
    #             dVstar_dtheta_delta_theta+=jnp.sum(inner)
    #     return  (V_star_max+flag[0]*dVstar_dx_deltax+flag[1]*dVstar_dtheta_delta_theta),\
    #             (V_star_max, dVstar_dx_deltax, dVstar_dtheta_delta_theta)
    
    #     #        dou_V_star_max_x = jnp.mean(optax.l2_loss(y_tog, jax.vmap(model)(x_tog_)) -
    #     #     optax.l2_loss(y_tog, jax.vmap(model)(x_tog)) )
    #     # dou_V_star_max_theta = jnp.mean(optax.l2_loss(y_tog, jax.vmap(model)(x_tog))
    #     #                   - optax.l2_loss(y_tog, jax.vmap(modelN)(x_tog)))
    #     # H = V_star_max + dou_V_star_max_x + dou_V_star_max_theta
    #     # -------------------------------------------------------------------
    # @eqx.filter_jit
    # def return_Hamiltonian_class(self, params, data):
    #     static, (x, y, exp_x, exp_y, deltax, flag) = data
    #     # #print("x", x.shape, "exp_x", exp_x.shape)
    #     x_tog = jnp.concatenate( [ x, exp_x])
    #     y_tog = jnp.concatenate( [ y, exp_y])      
    #     # Term 1 
    #     V_star_max = self.return_V_star_vector_class(x_tog, params, (static, y_tog))
        
        
    #     #print(V_star_max)
    #     # Term 2
    #     dVstar_dx = jax.grad(self.return_V_star_vector_class,argnums=(0))(exp_x, params, (static, exp_y))
    #     # #print("shape", dVstar_dx.shape)
    #     deltax = deltax/jnp.sqrt(jnp.linalg.norm(deltax)**2)
    #     dVstar_dx = dVstar_dx/jnp.sqrt(jnp.linalg.norm(dVstar_dx)**2)
        
    #     dVstar_dx_deltax = jnp.sum(jnp.trace( jnp.inner(dVstar_dx, deltax ) ))
        
        
    #     # Term 3
    #     delta_theta   = jax.grad(self.return_V_star_vector_class,argnums=(1))(x_tog, params, (static, y_tog))
    #     dVstar_dtheta = jax.grad(self.return_V_star_vector_class, argnums=(1))(exp_x, params, (static, exp_y))
    #     dVstar_dtheta = jax.tree_util.tree_leaves(dVstar_dtheta )
    #     delta_theta   = jax.tree_util.tree_leaves(delta_theta )
    #     dVstar_dtheta_delta_theta=0.

    #     for (dVstar_dw, delta_w)  in zip(dVstar_dtheta,  delta_theta):
    #         # #print("the operand", dVstar_dw.shape, delta_w.shape)
    #         # delta_w = delta_w/jnp.sqrt(jnp.linalg.norm(delta_w)**2)
    #         dVstar_dw = dVstar_dw/jnp.sqrt(jnp.linalg.norm(dVstar_dw)**2)
    #         if len(delta_w.shape)>1:
    #             inner = jnp.inner(dVstar_dw, delta_w)
    #             dVstar_dtheta_delta_theta+=jnp.sum(jnp.trace(inner) )
    #         else:
    #             inner = jnp.inner(dVstar_dw, delta_w)
    #             dVstar_dtheta_delta_theta+=jnp.sum(inner)
    #     return  (V_star_max+flag[0]*dVstar_dx_deltax+flag[1]*dVstar_dtheta_delta_theta),\
    #             (V_star_max, dVstar_dx_deltax, dVstar_dtheta_delta_theta)
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    

    