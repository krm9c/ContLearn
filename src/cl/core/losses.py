"""
Loss and metric functions for continual learning.

This module provides loss computation and metric evaluation for:
- Regression (MSE loss)
- Classification (Cross-entropy loss)
- Graph classification

All methods support both standard forward pass and AWB (Adaptive Weight Basis) mode.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class LossMixin:
    """Mixin class containing loss and metric computation methods."""

    # ========== Vector/Matrix Loss Functions ==========

    @eqx.filter_jit
    def loss_fn_class(self, params, statics, x, y):
        """Cross-entropy loss for classification (vectors)."""
        model = eqx.combine(params, statics)
        pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
        return -jnp.mean(y * pred_y)

    @eqx.filter_jit
    def loss_fn_mse(self, params, statics, x, y):
        """MSE loss for regression (vectors)."""
        model = eqx.combine(params, statics)
        preds = jax.vmap(model)(x)
        # Added by Claude: squeeze extra dimension if present (batch, 1, output) -> (batch, output)
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = jnp.squeeze(preds, axis=1)
        return jnp.mean((y - preds)**2)

    @eqx.filter_jit
    def accuracy_vectors(self, params, statics, x, y):
        """Accuracy metric for classification (vectors).

        Args:
            params: Model parameters
            statics: Static model components
            x: Input features, shape (batch, ...)
            y: Class labels - either scalar indices shape (batch,) or one-hot shape (batch, num_classes)

        Returns:
            Accuracy as float between 0 and 1
        """
        model = eqx.combine(params, statics)
        preds = jax.vmap(model)(x)
        # Squeeze extra dimension if present (batch, 1, classes) -> (batch, classes)
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = jnp.squeeze(preds, axis=1)
        pred = jnp.argmax(jax.nn.softmax(preds), axis=1)
        # Handle both one-hot encoded (batch, num_classes) and scalar labels (batch,)
        if y.ndim == 2 and y.shape[1] > 1:
            # One-hot encoded: convert to class indices
            y = jnp.argmax(y, axis=1)
        elif y.ndim == 2 and y.shape[1] == 1:
            # Shape (batch, 1): squeeze to (batch,)
            y = jnp.squeeze(y, axis=-1)
        # else: already shape (batch,) with scalar indices
        return jnp.mean(pred == y)

    @eqx.filter_jit
    def mse_vectors(self, params, statics, x, y):
        """MSE metric for regression (vectors)."""
        model = eqx.combine(params, statics)
        preds = jax.vmap(model)(x)
        # Added by Claude: squeeze extra dimension if present (batch, 1, output) -> (batch, output)
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = jnp.squeeze(preds, axis=1)
        return jnp.mean(optax.l2_loss(y, preds))

    # ========== Graph Loss Functions ==========

    @eqx.filter_jit
    def loss_fn_class_graph(self, params, statics, x, y, adj=None):
        """Cross-entropy loss for graph classification."""
        model = eqx.combine(params, statics)
        logits = jnp.stack([model(x[i], adj[i]).T for i in range(len(x))])
        pred_y = jnp.stack(logits)
        y = y.astype(jnp.int64)
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(y * pred_y)

    @eqx.filter_jit
    def loss_fn_mse_graph(self, params, statics, x, y, adj=None):
        """MSE loss for graph regression."""
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model(x, adj)))**2)

    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, adj, b, n):
        """Accuracy computation for graph classification (standard forward)."""
        model = eqx.combine(params, statics)
        array_log = [model(x[i], adj[i], b[i], n[i]) for i in range(len(x))]
        logits = jnp.concatenate(array_log)
        return jax.nn.log_softmax(logits, axis=1)

    @eqx.filter_jit
    def accuracy_graphs_AWBT(self, params, statics, x, adj, b, n):
        """Accuracy computation for graph classification (AWB forward)."""
        model = eqx.combine(params, statics)
        array_log = [model.get_AWBT(x[i], adj[i], b[i], n[i]) for i in range(len(x))]
        logits = jnp.concatenate(array_log)
        return jax.nn.log_softmax(logits, axis=1)

    @eqx.filter_jit
    def mse_graphs(self, params, statics, x, y, adj):
        """MSE metric for graph regression."""
        model = eqx.combine(params, statics)
        return jnp.mean((jax.vmap(model)(x, adj) - y)**2)

    # ========== Prediction Methods ==========

    @eqx.filter_jit
    def get_pred(self, params, statics, x):
        """Get predictions for vectors."""
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x)

    @eqx.filter_jit
    def get_pred_graphs(self, params, statics, x, y, adj):
        """Get predictions for graphs."""
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x, adj)

    # ========== Unified Interface ==========

    def return_loss_grad(self, params, batch, static):
        """Compute loss and gradient for a batch.

        Args:
            params: Trainable model parameters
            batch: Input batch (varies by problem type)
            static: Static model components

        Returns:
            Tuple of (loss, gradients)
        """
        if self.problem == 'vectors':
            (x, y) = batch
            if self.loss == 'class':
                grads = jax.grad(self.loss_fn_class)(params, static, x, y)
                loss = self.loss_fn_class(params, static, x, y)
            elif self.loss == 'mse':
                grads = jax.grad(self.loss_fn_mse)(params, static, x, y)
                loss = self.loss_fn_mse(params, static, x, y)
        elif self.problem == 'graph':
            (x, y, adj) = batch
            if self.loss == 'class':
                grads = jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
                loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
            elif self.loss == 'mse':
                grads = jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
                loss = self.loss_fn_mse_graph(params, static, x, y, adj=adj)
        return loss, grads

    def return_metric(self, params, statics, data, notABTrain=True):
        """Compute evaluation metric based on problem type.

        Args:
            params: Trainable model parameters
            statics: Static model components
            data: Input data (format depends on problem type)
                  For vectors: (x, y) where y is scalar class indices shape (batch,)
            notABTrain: True for standard forward, False for AWB forward

        Returns:
            Metric value (accuracy for classification, MSE for regression)
        """
        model = eqx.combine(params, statics)

        if self.problem == 'vectors':
            x, y = data
            if self.metric == 'class':
                # Ensure y is 1D int64 array of class indices
                y = y.astype(jnp.int64)
                if y.ndim == 2:
                    y = jnp.squeeze(y, axis=-1)
                if notABTrain:
                    preds = jax.vmap(model)(x)
                else:
                    preds = jax.vmap(model.get_AWBT)(x)
                # Squeeze extra dimension if present (batch, 1, classes) -> (batch, classes)
                if preds.ndim == 3 and preds.shape[1] == 1:
                    preds = jnp.squeeze(preds, axis=1)
                pred_y = jnp.argmax(jax.nn.log_softmax(preds), 1)
                return jnp.mean(y == pred_y)
            elif self.metric == 'mse':
                preds = jax.vmap(model)(x)
                # Squeeze predictions if they have extra dimension
                if preds.ndim == 3 and preds.shape[1] == 1:
                    preds = jnp.squeeze(preds, axis=1)
                return jnp.mean(optax.l2_loss(y, preds))

        elif self.problem == 'graph':
            batch, batch_ex = data
            x_tog = [batch.x.numpy(), batch_ex.x.numpy()]
            y_tog = [batch.y.numpy(), batch_ex.y.numpy()]
            adj_tog = [batch.adj.numpy(), batch_ex.adj.numpy()]
            b_tog = [batch.batch.numpy(), batch_ex.batch.numpy()]
            n_tog = [batch.n_nodes.numpy(), batch_ex.n_nodes.numpy()]

            if self.loss == 'class':
                if notABTrain:
                    yhat = self.accuracy_graphs(params, statics, x_tog, adj_tog, b_tog, n_tog)
                else:
                    yhat = self.accuracy_graphs_AWBT(params, statics, x_tog, adj_tog, b_tog, n_tog)
                pred_y = jnp.argmax(yhat, axis=1)
                return jnp.mean(jnp.concatenate(y_tog) == pred_y)
