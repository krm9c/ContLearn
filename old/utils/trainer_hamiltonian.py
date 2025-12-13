"""Hamiltonian computation methods for continual learning."""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class HamiltonianMixin:
    """Mixin class containing Hamiltonian computation methods for continual learning."""

    # ---------------------------------------------------------------------------------------------
    def return_Hamiltonian_graph(self, params, data, notABTrain):
        if (notABTrain == True): # Train as normal
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
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
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
        else: #Train only AB...
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
                pred_y = model.get_AWBT(xx, xxadj, bb, nn )
                # print(jnp.dtype(yy), jnp.dtype(pred_y))
                loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels( pred_y,yy ))
                return loss

            def norm_param(x):
                return (x*(-1*1e-04/jnp.sqrt(jnp.linalg.norm(x**2))))

            xdot = deltax
            zero_dx = jnp.zeros(xdot.shape)
            delta_theta= jax.grad(return_V_star_graph,argnums=(0))(params, x, adj)
            wdot= jax.tree_util.tree_map(norm_param, delta_theta)
            zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
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

    #==================================================================================================================#
    #--------------------------------return_Hamiltonian_mse for Training W only not A,B--------------------------------------------------------------#
    def return_Hamiltonian_mse(self, params, data,notABTrain = True):
        statics, (x, y, exp_x, exp_y, deltax, flag)  = data
        extra=y
        #model = eqx.combine(params, statics)
        #pred_y = jax.vmap(model)(x)s
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


    def return_Hamiltonian_class(self, params, data, notABTrain=True):
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
