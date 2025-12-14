"""
Hamiltonian computation methods for continual learning.

This module provides gradient computation using Hamiltonian formulation that combines:
- Current task loss gradient (delta_theta)
- Experience replay gradient (grad_V)
- Regularization term gradient (grad_dV) for continual learning

The final gradient is a convex combination:
    grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV

Where alpha, beta, gamma are configurable weights (grad_weights in config).

The Hamiltonian H = V + dV where:
- V is the loss on experience data
- dV is the change in loss due to parameter and input perturbations

Each method supports two modes:
- notABTrain=True: Standard training (train W with A/B frozen)
- notABTrain=False: AWB training (train A/B with W frozen)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

# Added by Claude: Default gradient combination weights
# [alpha, beta, gamma] for [current_task, experience_replay, hamiltonian_regularization]
DEFAULT_GRAD_WEIGHTS = [0.01, 0.98, 0.1]


class HamiltonianMixin:
    """Mixin class containing Hamiltonian computation methods for continual learning."""

    def return_Hamiltonian_graph(self, params, data, notABTrain, grad_weights=None):
        """Compute Hamiltonian gradient for graph classification.

        Args:
            params: Trainable model parameters
            data: Tuple of (static, (batch, batch_ex, deltax, delta_adj))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
                         [current_task, experience_replay, hamiltonian_regularization]

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx, dV_dadj))
        """
        # Added by Claude: Use provided weights or defaults
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        static, (batch, batch_ex, deltax, delta_adj) = data

        # Extract data from batches
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

        # Define loss function based on training mode
        if notABTrain:
            def return_V_star_graph(params, xx, xxadj):
                (yy, bb, nn) = extra
                model = eqx.combine(params, static)
                pred_y = model(xx, xxadj, bb, nn)
                loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, yy))
                return loss
        else:
            def return_V_star_graph(params, xx, xxadj):
                (yy, bb, nn) = extra
                model = eqx.combine(params, static)
                pred_y = model.get_AWBT(xx, xxadj, bb, nn)
                loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, yy))
                return loss

        def norm_param(x):
            return (x * (-1 * 1e-04 / jnp.sqrt(jnp.linalg.norm(x**2))))

        # Compute perturbation directions
        xdot = deltax
        zero_dx = jnp.zeros(xdot.shape)
        delta_theta = jax.grad(return_V_star_graph, argnums=(0))(params, x, adj)
        wdot = jax.tree_util.tree_map(norm_param, delta_theta)
        zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)
        adjdot = delta_adj
        zero_dadj = jnp.zeros(adjdot.shape)

        # Switch to experience data
        extra = (ey, eb, en)
        grad_V = jax.grad(return_V_star_graph, argnums=(0))(params, ex, eadj)

        # Linearize and compute directional derivatives
        V, f_jvp = jax.linearize(return_V_star_graph, params, ex, eadj)
        grad_dV = jax.grad(f_jvp)(wdot, xdot, adjdot)
        dV = f_jvp(wdot, xdot, adjdot)

        # Added by Claude: Use configurable gradient weights
        def combine_grad(x, y, z):
            return alpha * x + beta * y + gamma * z

        grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)

        return grad, (
            (V + dV),  # H = V + dV
            V,         # V (loss on experience)
            dV,        # dV (total perturbation effect)
            f_jvp(wdot, zero_dx, zero_dadj),    # dV_dtheta
            f_jvp(zero_dtheta, xdot, zero_dadj), # dV_dx
            f_jvp(zero_dtheta, zero_dx, adjdot)  # dV_dadj
        )

    def return_Hamiltonian_mse(self, params, data, notABTrain=True, grad_weights=None):
        """Compute Hamiltonian gradient for MSE regression.

        Args:
            params: Trainable model parameters
            data: Tuple of (statics, (x, y, exp_x, exp_y, deltax, flag))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
                         [current_task, experience_replay, hamiltonian_regularization]

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx))
        """
        # Added by Claude: Use provided weights or defaults
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        statics, (x, y, exp_x, exp_y, deltax, flag) = data
        extra = y

        if notABTrain:
            def return_V_star_vector_mse(params, x):
                y = extra
                model = eqx.combine(params, statics)
                pred_y = jax.vmap(model)(x)
                pred_y = pred_y.squeeze(1)
                return jnp.mean(optax.l2_loss(y, pred_y))
        else:
            def return_V_star_vector_mse(params, x):
                y = extra
                model = eqx.combine(params, statics)
                pred_y = jax.vmap(model.getAWB)(x)
                return jnp.mean(optax.l2_loss(y, pred_y))

        def norm_param(x):
            return (x * -1)

        # Compute perturbation directions
        xdot = deltax
        zero_dx = jnp.zeros(xdot.shape)
        delta_theta = jax.grad(return_V_star_vector_mse, argnums=(0))(params, x)
        wdot = jax.tree_util.tree_map(norm_param, delta_theta)
        zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

        # Switch to experience data
        extra = exp_y
        grad_V = jax.grad(return_V_star_vector_mse, argnums=(0))(params, exp_x)

        # Linearize: produces linear approximation using jvp and partial eval
        V, f_jvp = jax.linearize(return_V_star_vector_mse, params, exp_x)
        grad_dV = jax.grad(f_jvp)(wdot, xdot)
        dV = f_jvp(wdot, xdot)

        # Added by Claude: Use configurable gradient weights
        def combine_grad(x, y, z):
            return alpha * x + beta * y + gamma * z

        grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)

        return grad, (
            (V + dV),  # H
            V,         # V
            dV,        # dV
            f_jvp(wdot, zero_dx),   # dV_dtheta
            f_jvp(zero_dtheta, xdot) # dV_dx
        )

    def return_Hamiltonian_class(self, params, data, notABTrain=True, grad_weights=None):
        """Compute Hamiltonian gradient for classification.

        Args:
            params: Trainable model parameters
            data: Tuple of (statics, (x, y, exp_x, exp_y, deltax, flag))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
                         [current_task, experience_replay, hamiltonian_regularization]

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx))
        """
        # Added by Claude: Use provided weights or defaults
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        statics, (x, y, exp_x, exp_y, deltax, flag) = data
        extra = y

        if notABTrain:
            def return_V_star_class(params, x):
                y = extra
                model = eqx.combine(params, statics)
                y = y.astype(jnp.int64)
                pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))
        else:
            def return_V_star_class(params, x):
                y = extra
                model = eqx.combine(params, statics)
                y = y.astype(jnp.int64)
                pred_y = jax.vmap(model.get_AWBT)(x)
                pred_y = jax.nn.log_softmax(pred_y)
                return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))

        def norm_param(x):
            return (-1 * x)

        # Compute perturbation directions
        xdot = deltax
        zero_dx = jnp.zeros(xdot.shape)
        delta_theta = jax.grad(return_V_star_class, argnums=(0))(params, x)
        wdot = jax.tree_util.tree_map(norm_param, delta_theta)
        zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

        # Switch to experience data
        extra = exp_y
        grad_V = jax.grad(return_V_star_class, argnums=(0))(params, exp_x)

        # Linearize
        V, f_jvp = jax.linearize(return_V_star_class, params, exp_x)
        grad_dV = jax.grad(f_jvp)(wdot, xdot)
        dV = f_jvp(wdot, xdot)

        # Added by Claude: Use configurable gradient weights
        def combine_grad(x, y, z):
            return alpha * x + beta * y + gamma * z

        grad = jax.tree_util.tree_map(combine_grad, delta_theta, grad_V, grad_dV)

        return grad, (
            (V + dV),  # H
            V,         # V
            dV,        # dV
            f_jvp(wdot, zero_dx),    # dV_dtheta
            f_jvp(zero_dtheta, xdot)  # dV_dx
        )
