#Trainer file changes for AWB

#===============return_metric_AWB function==================#
def return_metric(self, params, statics, data, notABTrain):
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
                
