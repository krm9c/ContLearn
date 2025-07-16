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


        
                
