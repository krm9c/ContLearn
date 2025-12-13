"""Loss and metric functions for the Trainer class."""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class LossMixin:
    """Mixin class containing loss and metric computation methods."""

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

    @eqx.filter_jit
    def accuracy_graphs_AWBT(self, params, statics, x, adj, b, n):
        model = eqx.combine(params, statics)
        array_log = [ model.get_AWBT(x[i], adj[i], b[i], n[i]) for i in range(len(x))]
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

    # -------------------------------------------------------------------
    def return_metric(self, params, statics, data, notABTrain = True):
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
                preds = jax.vmap(model)(x)
                # Squeeze predictions if they have extra dimension (batch, 1, features) -> (batch, features)
                if preds.ndim == 3 and preds.shape[1] == 1:
                    preds = jnp.squeeze(preds, axis=1)
                return jnp.mean(optax.l2_loss(y, preds))
        elif self.problem== 'graph':
                batch, batch_ex = data
                x_tog   = [batch.x.numpy(), batch_ex.x.numpy()]
                y_tog   = [batch.y.numpy(), batch_ex.y.numpy()]
                adj_tog = [batch.adj.numpy(), batch_ex.adj.numpy()]
                b_tog   = [batch.batch.numpy(), batch_ex.batch.numpy()]
                n_tog   = [batch.n_nodes.numpy(), batch_ex.n_nodes.numpy()]

                if self.loss == 'class':
                    if (notABTrain == True):
                        yhat = self.accuracy_graphs(params, statics,\
                                x_tog, adj_tog, b_tog, n_tog)
                    else:
                        yhat = self.accuracy_graphs_AWBT(params, statics,\
                                x_tog, adj_tog, b_tog, n_tog)
                    pred_y = jnp.argmax(yhat, axis=1)
                    return jnp.mean(jnp.concatenate
                                    (y_tog)==pred_y)
