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

Performance Note:
    This module uses @eqx.filter_jit for JIT compilation of core computation functions.
    This enables high GPU utilization by compiling the entire Hamiltonian computation
    into fused GPU kernels, avoiding Python overhead between operations.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from functools import partial

# Default gradient combination weights
# [alpha, beta, gamma] for [current_task, experience_replay, hamiltonian_regularization]
DEFAULT_GRAD_WEIGHTS = [0.01, 0.98, 0.1]


# =============================================================================
# JIT-Compiled Loss Functions (Step 1)
# =============================================================================
# These are pure JAX functions that can be efficiently JIT-compiled.
# Separate functions for standard vs AWB modes to avoid Python conditionals in JIT.

@eqx.filter_jit
def _loss_mse_standard(params, static, x, y):
    """JIT-compiled MSE loss for standard training.

    Args:
        params: Trainable model parameters (PyTree)
        static: Frozen model components (PyTree)
        x: Input features (JAX array, shape [batch, input_dim])
        y: Target values (JAX array, shape [batch])

    Returns:
        Scalar loss value
    """
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model)(x)
    # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
    if pred_y.ndim == 3 and pred_y.shape[1] == 1:
        pred_y = jnp.squeeze(pred_y, axis=1)
    return jnp.mean(optax.l2_loss(y, pred_y))


@eqx.filter_jit
def _loss_mse_awb(params, static, x, y):
    """JIT-compiled MSE loss for AWB training (uses model.getAWB)."""
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model.getAWB)(x)
    # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
    if pred_y.ndim == 3 and pred_y.shape[1] == 1:
        pred_y = jnp.squeeze(pred_y, axis=1)
    return jnp.mean(optax.l2_loss(y, pred_y))


@eqx.filter_jit
def _loss_class_standard(params, static, x, y):
    """JIT-compiled classification loss for standard training.

    Args:
        params: Trainable model parameters (PyTree)
        static: Frozen model components (PyTree)
        x: Input features (JAX array, shape [batch, ...])
        y: Target labels (JAX array, shape [batch], int64)

    Returns:
        Scalar loss value
    """
    model = eqx.combine(params, static)
    pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))


@eqx.filter_jit
def _loss_class_awb(params, static, x, y):
    """JIT-compiled classification loss for AWB training (uses model.get_AWBT)."""
    model = eqx.combine(params, static)
    pred_y = jax.vmap(model.get_AWBT)(x)
    pred_y = jax.nn.log_softmax(pred_y)
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))


@eqx.filter_jit
def _loss_graph_standard(params, static, x, adj, b, n, y):
    """JIT-compiled graph classification loss for standard training.

    Args:
        params: Trainable model parameters (PyTree)
        static: Frozen model components (PyTree)
        x: Node features (JAX array)
        adj: Adjacency matrix (JAX array)
        b: Batch assignment vector (JAX array)
        n: Node counts per graph (JAX array)
        y: Target labels (JAX array, int64)

    Returns:
        Scalar loss value
    """
    model = eqx.combine(params, static)
    pred_y = model(x, adj, b, n)
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))


@eqx.filter_jit
def _loss_graph_awb(params, static, x, adj, b, n, y):
    """JIT-compiled graph classification loss for AWB training (uses model.get_AWBT)."""
    model = eqx.combine(params, static)
    pred_y = model.get_AWBT(x, adj, b, n)
    return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_y, y))


# =============================================================================
# JIT-Compiled Hamiltonian Core Functions (Step 2)
# =============================================================================
# These functions compute the full Hamiltonian gradient with all operations
# fused into a single JIT-compiled kernel for maximum GPU efficiency.

@eqx.filter_jit
def _hamiltonian_core_mse_standard(params, static, x, y, exp_x, exp_y, deltax,
                                    alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for MSE regression (standard training).

    Computes: grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV

    Args:
        params: Trainable parameters
        static: Frozen model components
        x, y: Current task batch
        exp_x, exp_y: Experience replay batch
        deltax: Input perturbation
        alpha, beta, gamma: Gradient combination weights
        sqrt_param_count: Pre-computed sqrt(param_count) for normalization
        dV_scale: Additional dV scaling factor

    Returns:
        Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx))
    """
    # Loss function for current task (closed over y)
    def loss_fn_curr(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model)(xx)
        # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = jnp.squeeze(pred, axis=1)
        return jnp.mean(optax.l2_loss(y, pred))

    # Loss function for experience data (closed over exp_y)
    def loss_fn_exp(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model)(xx)
        # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = jnp.squeeze(pred, axis=1)
        return jnp.mean(optax.l2_loss(exp_y, pred))

    # Compute delta_theta (gradient on current task)
    delta_theta = jax.grad(loss_fn_curr)(params, x)

    # Compute wdot (negative gradient direction for perturbation)
    wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    # Compute grad_V (gradient on experience data)
    grad_V = jax.grad(loss_fn_exp)(params, exp_x)

    # Linearize at experience data for directional derivatives
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x)

    # Compute directional derivatives
    zero_dx = jnp.zeros_like(deltax)
    grad_dV = jax.grad(f_jvp)(wdot, deltax)
    dV_raw = f_jvp(wdot, deltax)

    # Normalize dV values
    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax) / sqrt_param_count) * dV_scale

    # Combine gradients: grad = alpha * delta_theta + beta * grad_V + gamma * grad_dV
    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx)


@eqx.filter_jit
def _hamiltonian_core_mse_awb(params, static, x, y, exp_x, exp_y, deltax,
                               alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for MSE regression (AWB training)."""
    # Loss function for current task using AWB forward (closed over y)
    def loss_fn_curr(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model.getAWB)(xx)
        # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = jnp.squeeze(pred, axis=1)
        return jnp.mean(optax.l2_loss(y, pred))

    # Loss function for experience data using AWB forward (closed over exp_y)
    def loss_fn_exp(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model.getAWB)(xx)
        # Squeeze extra dimension if present: (batch, 1, output) -> (batch, output)
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = jnp.squeeze(pred, axis=1)
        return jnp.mean(optax.l2_loss(exp_y, pred))

    delta_theta = jax.grad(loss_fn_curr)(params, x)
    wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    grad_V = jax.grad(loss_fn_exp)(params, exp_x)
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x)

    zero_dx = jnp.zeros_like(deltax)
    grad_dV = jax.grad(f_jvp)(wdot, deltax)
    dV_raw = f_jvp(wdot, deltax)

    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax) / sqrt_param_count) * dV_scale

    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx)


@eqx.filter_jit
def _hamiltonian_core_class_standard(params, static, x, y, exp_x, exp_y, deltax,
                                      alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for classification (standard training)."""
    def loss_fn_curr(p, xx):
        model = eqx.combine(p, static)
        pred = jax.nn.log_softmax(jax.vmap(model)(xx))
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, y))

    def loss_fn_exp(p, xx):
        model = eqx.combine(p, static)
        pred = jax.nn.log_softmax(jax.vmap(model)(xx))
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, exp_y))

    delta_theta = jax.grad(loss_fn_curr)(params, x)
    wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    grad_V = jax.grad(loss_fn_exp)(params, exp_x)
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x)

    zero_dx = jnp.zeros_like(deltax)
    grad_dV = jax.grad(f_jvp)(wdot, deltax)
    dV_raw = f_jvp(wdot, deltax)

    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax) / sqrt_param_count) * dV_scale

    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx)


@eqx.filter_jit
def _hamiltonian_core_class_awb(params, static, x, y, exp_x, exp_y, deltax,
                                 alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for classification (AWB training)."""
    def loss_fn_curr(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model.get_AWBT)(xx)
        pred = jax.nn.log_softmax(pred)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, y))

    def loss_fn_exp(p, xx):
        model = eqx.combine(p, static)
        pred = jax.vmap(model.get_AWBT)(xx)
        pred = jax.nn.log_softmax(pred)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, exp_y))

    delta_theta = jax.grad(loss_fn_curr)(params, x)
    wdot = jax.tree_util.tree_map(lambda g: -g, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    grad_V = jax.grad(loss_fn_exp)(params, exp_x)
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x)

    zero_dx = jnp.zeros_like(deltax)
    grad_dV = jax.grad(f_jvp)(wdot, deltax)
    dV_raw = f_jvp(wdot, deltax)

    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax) / sqrt_param_count) * dV_scale

    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx)


@eqx.filter_jit
def _hamiltonian_core_graph_standard(params, static, x, y, adj, b, n,
                                      exp_x, exp_y, exp_adj, exp_b, exp_n,
                                      deltax, delta_adj,
                                      alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for graph classification (standard training)."""
    def loss_fn_curr(p, xx, xxadj):
        model = eqx.combine(p, static)
        pred = model(xx, xxadj, b, n)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, y))

    def loss_fn_exp(p, xx, xxadj):
        model = eqx.combine(p, static)
        pred = model(xx, xxadj, exp_b, exp_n)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, exp_y))

    # Compute delta_theta
    delta_theta = jax.grad(loss_fn_curr)(params, x, adj)

    # Compute wdot with scaling
    def norm_param(g):
        return g * (-1e-04 / jnp.sqrt(jnp.linalg.norm(g**2) + 1e-8))
    wdot = jax.tree_util.tree_map(norm_param, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    # Compute grad_V
    grad_V = jax.grad(loss_fn_exp)(params, exp_x, exp_adj)

    # Linearize
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x, exp_adj)

    zero_dx = jnp.zeros_like(deltax)
    zero_dadj = jnp.zeros_like(delta_adj)

    grad_dV = jax.grad(f_jvp)(wdot, deltax, delta_adj)
    dV_raw = f_jvp(wdot, deltax, delta_adj)

    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx, zero_dadj) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax, zero_dadj) / sqrt_param_count) * dV_scale
    dV_dadj = (f_jvp(zero_dtheta, zero_dx, delta_adj) / sqrt_param_count) * dV_scale

    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx, dV_dadj)


@eqx.filter_jit
def _hamiltonian_core_graph_awb(params, static, x, y, adj, b, n,
                                 exp_x, exp_y, exp_adj, exp_b, exp_n,
                                 deltax, delta_adj,
                                 alpha, beta, gamma, sqrt_param_count, dV_scale):
    """JIT-compiled Hamiltonian core for graph classification (AWB training)."""
    def loss_fn_curr(p, xx, xxadj):
        model = eqx.combine(p, static)
        pred = model.get_AWBT(xx, xxadj, b, n)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, y))

    def loss_fn_exp(p, xx, xxadj):
        model = eqx.combine(p, static)
        pred = model.get_AWBT(xx, xxadj, exp_b, exp_n)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred, exp_y))

    delta_theta = jax.grad(loss_fn_curr)(params, x, adj)

    def norm_param(g):
        return g * (-1e-04 / jnp.sqrt(jnp.linalg.norm(g**2) + 1e-8))
    wdot = jax.tree_util.tree_map(norm_param, delta_theta)
    zero_dtheta = jax.tree_util.tree_map(jnp.zeros_like, delta_theta)

    grad_V = jax.grad(loss_fn_exp)(params, exp_x, exp_adj)
    V, f_jvp = jax.linearize(loss_fn_exp, params, exp_x, exp_adj)

    zero_dx = jnp.zeros_like(deltax)
    zero_dadj = jnp.zeros_like(delta_adj)

    grad_dV = jax.grad(f_jvp)(wdot, deltax, delta_adj)
    dV_raw = f_jvp(wdot, deltax, delta_adj)

    dV = (dV_raw / sqrt_param_count) * dV_scale
    dV_dtheta = (f_jvp(wdot, zero_dx, zero_dadj) / sqrt_param_count) * dV_scale
    dV_dx = (f_jvp(zero_dtheta, deltax, zero_dadj) / sqrt_param_count) * dV_scale
    dV_dadj = (f_jvp(zero_dtheta, zero_dx, delta_adj) / sqrt_param_count) * dV_scale

    grad = jax.tree_util.tree_map(
        lambda dt, gv, gdv: alpha * dt + beta * gv + gamma * gdv,
        delta_theta, grad_V, grad_dV
    )

    return grad, (V + dV, V, dV, dV_dtheta, dV_dx, dV_dadj)


# =============================================================================
# Helper function to compute parameter count as JAX scalar
# =============================================================================

def _get_param_count_jax(params):
    """Compute parameter count as JAX scalar (for use in JIT).

    Args:
        params: PyTree of parameters

    Returns:
        JAX scalar with total parameter count
    """
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.array(sum(leaf.size for leaf in leaves), dtype=jnp.float64)


class HamiltonianMixin:
    """Mixin class containing Hamiltonian computation methods for continual learning.

    This class provides JIT-optimized Hamiltonian gradient computation for:
    - MSE regression (return_Hamiltonian_mse)
    - Classification (return_Hamiltonian_class)
    - Graph classification (return_Hamiltonian_graph)

    Each method dispatches to pre-compiled JIT functions for high GPU utilization.
    """

    def _count_parameters(self, params):
        """Count total number of parameters in pytree.

        Args:
            params: PyTree of parameters

        Returns:
            Total parameter count as float
        """
        leaves = jax.tree_util.tree_leaves(params)
        return float(sum(leaf.size for leaf in leaves))

    def _get_sqrt_param_count(self, params):
        """Get sqrt of parameter count as JAX scalar for JIT compatibility.

        Args:
            params: PyTree of parameters

        Returns:
            JAX scalar with sqrt(param_count)
        """
        leaves = jax.tree_util.tree_leaves(params)
        count = sum(leaf.size for leaf in leaves)
        return jnp.sqrt(jnp.array(count, dtype=jnp.float64))

    def _normalize_dV(self, dV, params, normalize=True, scale_factor=1.0):
        """Normalize dV by parameter count to prevent scaling with model size.

        Args:
            dV: Raw dV value
            params: Model parameters (for counting)
            normalize: Whether to apply normalization
            scale_factor: Additional manual scaling factor

        Returns:
            Normalized dV
        """
        if not normalize:
            return dV * scale_factor

        param_count = self._count_parameters(params)
        dV_normalized = dV / jnp.sqrt(param_count)
        dV_normalized = dV_normalized * scale_factor

        return dV_normalized

    def return_Hamiltonian_graph(self, params, data, notABTrain, grad_weights=None,
                                 normalize_dV=True, dV_scale=1.0):
        """Compute Hamiltonian gradient for graph classification.

        Uses JIT-compiled core functions for high GPU utilization.

        Args:
            params: Trainable model parameters
            data: Tuple of (static, (batch, batch_ex, deltax, delta_adj))
                  OR (static, (x, y, adj, b, n, exp_x, exp_y, exp_adj, exp_b, exp_n, deltax, delta_adj))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
            normalize_dV: Whether to normalize dV by parameter count (default True)
            dV_scale: Additional scaling factor for dV (default 1.0)

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx, dV_dadj))
        """
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        static, batch_data = data

        # Check if data is already extracted (new format) or needs extraction (old format)
        if len(batch_data) == 12:
            # New format: already extracted JAX arrays
            (x, y, adj, b, n, exp_x, exp_y, exp_adj, exp_b, exp_n, deltax, delta_adj) = batch_data
        else:
            # Old format: torch_geometric batch objects (for backward compatibility)
            (batch, batch_ex, deltax, delta_adj) = batch_data
            x = jnp.asarray(batch.x.numpy(), dtype=jnp.float64)
            y = jnp.asarray(batch.y.numpy(), dtype=jnp.int64)
            adj = jnp.asarray(batch.adj.numpy(), dtype=jnp.float64)
            b = jnp.asarray(batch.batch.numpy())
            n = jnp.asarray(batch.n_nodes.numpy())

            exp_x = jnp.asarray(batch_ex.x.numpy(), dtype=jnp.float64)
            exp_y = jnp.asarray(batch_ex.y.numpy(), dtype=jnp.int64)
            exp_adj = jnp.asarray(batch_ex.adj.numpy(), dtype=jnp.float64)
            exp_b = jnp.asarray(batch_ex.batch.numpy())
            exp_n = jnp.asarray(batch_ex.n_nodes.numpy())

            deltax = jnp.asarray(deltax, dtype=jnp.float64)
            delta_adj = jnp.asarray(delta_adj, dtype=jnp.float64)

        # Compute sqrt(param_count) for normalization
        sqrt_param_count = self._get_sqrt_param_count(params)

        # Dispatch to JIT-compiled core function (Python branching outside JIT)
        if notABTrain:
            return _hamiltonian_core_graph_standard(
                params, static, x, y, adj, b, n,
                exp_x, exp_y, exp_adj, exp_b, exp_n,
                deltax, delta_adj,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )
        else:
            return _hamiltonian_core_graph_awb(
                params, static, x, y, adj, b, n,
                exp_x, exp_y, exp_adj, exp_b, exp_n,
                deltax, delta_adj,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )

    def return_Hamiltonian_mse(self, params, data, notABTrain=True, grad_weights=None,
                               normalize_dV=True, dV_scale=1.0):
        """Compute Hamiltonian gradient for MSE regression.

        Uses JIT-compiled core functions for high GPU utilization.

        Args:
            params: Trainable model parameters
            data: Tuple of (statics, (x, y, exp_x, exp_y, deltax, flag))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
            normalize_dV: Whether to normalize dV by parameter count (default True)
            dV_scale: Additional scaling factor for dV (default 1.0)

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx))
        """
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        statics, (x, y, exp_x, exp_y, deltax, flag) = data

        # Compute sqrt(param_count) for normalization
        sqrt_param_count = self._get_sqrt_param_count(params)

        # Dispatch to JIT-compiled core function (Python branching outside JIT)
        if notABTrain:
            return _hamiltonian_core_mse_standard(
                params, statics, x, y, exp_x, exp_y, deltax,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )
        else:
            return _hamiltonian_core_mse_awb(
                params, statics, x, y, exp_x, exp_y, deltax,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )

    def return_Hamiltonian_class(self, params, data, notABTrain=True, grad_weights=None,
                                 normalize_dV=True, dV_scale=1.0):
        """Compute Hamiltonian gradient for classification.

        Uses JIT-compiled core functions for high GPU utilization.

        Args:
            params: Trainable model parameters
            data: Tuple of (statics, (x, y, exp_x, exp_y, deltax, flag))
            notABTrain: True for standard training, False for AWB A/B training
            grad_weights: Optional [alpha, beta, gamma] weights for gradient combination
            normalize_dV: Whether to normalize dV by parameter count (default True)
            dV_scale: Additional scaling factor for dV (default 1.0)

        Returns:
            Tuple of (grad, (H, V, dV, dV_dtheta, dV_dx))
        """
        if grad_weights is None:
            grad_weights = DEFAULT_GRAD_WEIGHTS
        alpha, beta, gamma = grad_weights

        statics, (x, y, exp_x, exp_y, deltax, flag) = data

        # Ensure labels are int64 for cross-entropy
        y = y.astype(jnp.int64)
        exp_y = exp_y.astype(jnp.int64)

        # Compute sqrt(param_count) for normalization
        sqrt_param_count = self._get_sqrt_param_count(params)

        # Dispatch to JIT-compiled core function (Python branching outside JIT)
        if notABTrain:
            return _hamiltonian_core_class_standard(
                params, statics, x, y, exp_x, exp_y, deltax,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )
        else:
            return _hamiltonian_core_class_awb(
                params, statics, x, y, exp_x, exp_y, deltax,
                jnp.array(alpha), jnp.array(beta), jnp.array(gamma),
                sqrt_param_count, jnp.array(dV_scale)
            )
