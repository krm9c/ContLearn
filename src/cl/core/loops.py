"""
Training loop methods for continual learning.

This module provides the unified training loop that works across:
- Regression (MSE loss)
- Classification (Cross-entropy loss)
- Graph classification

The training loop handles:
- Hamiltonian-based gradient computation
- Experience replay
- Metric recording at save intervals
- Both standard and AWB training modes
"""

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np_
import optax
from tqdm import tqdm


# Graph transform pipeline (lazy import to avoid dependency when not needed)
_GRAPH_TRANSFORMS = None


def get_graph_transforms():
    """Lazily load graph transforms to avoid torch_geometric import when not needed."""
    global _GRAPH_TRANSFORMS
    if _GRAPH_TRANSFORMS is None:
        import torch_geometric.transforms as T

        class RemoveEdgeAttr:
            """Remove edge_attr so ToDense creates 2D adjacency matrix."""
            def __call__(self, data):
                data.edge_attr = None
                return data

        _GRAPH_TRANSFORMS = T.Compose([
            RemoveEdgeAttr(),
            T.GCNNorm(),
            T.ToDense(),
            T.NormalizeFeatures()
        ])
    return _GRAPH_TRANSFORMS


class TrainingLoopsMixin:
    """Mixin class containing training loop methods for continual learning."""

    def _clip_gradients(self, grads, max_norm=None):
        """Clip gradients by global norm.

        Added by Claude: Following Pascanu et al. (2013) recommendations for
        gradient clipping in recurrent networks, applicable to continual learning
        with potentially unstable gradients.

        Reference:
        - Pascanu et al., "On the difficulty of training recurrent neural networks",
          ICML 2013

        Args:
            grads: PyTree of gradients
            max_norm: Maximum gradient norm (None = no clipping)

        Returns:
            Tuple of (clipped_grads, global_norm, was_clipped)
        """
        if max_norm is None or max_norm <= 0:
            # No clipping
            grad_leaves = jax.tree_util.tree_leaves(grads)
            global_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in grad_leaves]))
            return grads, global_norm, False

        # Compute global norm
        grad_leaves = jax.tree_util.tree_leaves(grads)
        global_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in grad_leaves]))

        # Compute clipping coefficient
        clip_coef = max_norm / (global_norm + 1e-6)
        clip_coef = jnp.minimum(clip_coef, 1.0)

        # Apply clipping
        clipped_grads = jax.tree_map(lambda g: g * clip_coef, grads)
        was_clipped = global_norm > max_norm

        return clipped_grads, global_norm, was_clipped

    def _compute_metrics_on_sampled_batches(self, params, static, loader,
                                            num_batches=10, problem_type='vectors',
                                            notABTrain=True, transforms=None):
        """Efficiently compute metrics on N sampled batches from a loader.

        Args:
            params: Model parameters
            static: Static model components
            loader: Data loader (tuple of current_loader, exp_loader)
            num_batches: Number of batches to sample (default 10)
            problem_type: 'vectors' or 'graph'
            notABTrain: Whether using normal training (True) or AWB training (False)
            transforms: Transform pipeline for graph data

        Returns:
            Tuple of (current_task_metric, experience_metric)
        """
        current_metrics = []
        exp_metrics = []

        if problem_type == 'graph':
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

                    data_current = (current_batch, current_batch)
                    metric_current = self.return_metric(params, static, data=data_current, notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    data_exp = (exp_batch, exp_batch)
                    metric_exp = self.return_metric(params, static, data=data_exp, notABTrain=notABTrain)
                    exp_metrics.append(metric_exp)
            else:
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
            # Vector data
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

                    x_curr, y_curr = current_batch
                    x_curr = jnp.array(x_curr.numpy().astype(np_.float64))
                    y_curr = jnp.array(y_curr.numpy().astype(np_.float64))
                    y_curr = jnp.squeeze(y_curr)
                    if y_curr.ndim == 1:
                        y_curr = jnp.expand_dims(y_curr, axis=-1)

                    metric_current = self.return_metric(params, static, data=(x_curr, y_curr), notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    x_exp, y_exp = exp_batch
                    x_exp = jnp.array(x_exp.numpy().astype(np_.float64))
                    y_exp = jnp.array(y_exp.numpy().astype(np_.float64))
                    y_exp = jnp.squeeze(y_exp)
                    if y_exp.ndim == 1:
                        y_exp = jnp.expand_dims(y_exp, axis=-1)

                    metric_exp = self.return_metric(params, static, data=(x_exp, y_exp), notABTrain=notABTrain)
                    exp_metrics.append(metric_exp)
            else:
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

    def _compute_perturbation_variance(self, trainloader, exploader, problem_type):
        """Pre-compute variance for perturbation sampling.

        Uses the mean difference approach for feature variance.
        For graphs, also computes adjacency variance.

        Args:
            trainloader: Current task data loader
            exploader: Experience replay data loader
            problem_type: 'vectors' or 'graph'

        Returns:
            Tuple of (var_x, var_adj) where var_adj is 0 for vectors
        """
        var_x_list, var_adj_list = [], []
        transforms = get_graph_transforms() if problem_type == 'graph' else None

        for batch, batch_ex in zip(iter(trainloader), iter(exploader)):
            if problem_type == 'graph':
                batch = transforms(batch)
                batch_ex = transforms(batch_ex)
                x = jnp.array(batch.x.numpy())
                exp_x = jnp.array(batch_ex.x.numpy())
                var_adj_list.append(
                    (jnp.sqrt(jnp.linalg.norm(batch.adj.numpy()**2)) -
                     jnp.sqrt(jnp.linalg.norm(batch_ex.adj.numpy()**2)))**2
                )
            else:
                (x, _) = batch
                (exp_x, _) = batch_ex
                min_batch = min(x.shape[0], exp_x.shape[0])
                x = jnp.array(x.numpy()[:min_batch], dtype=jnp.float64)
                exp_x = jnp.array(exp_x.numpy()[:min_batch], dtype=jnp.float64)

            # Compute feature variance
            var_x_list.append(
                jnp.sqrt(jnp.linalg.norm((jnp.mean(x, axis=0) - jnp.mean(exp_x, axis=0))**2))
            )

        var_x = sum(var_x_list) / max(len(var_x_list), 1)
        var_adj = 1e-3 * (sum(var_adj_list) / max(len(var_adj_list), 1)) if var_adj_list else 0.0
        return var_x, var_adj

    def train__CL(self, train__, params, static, opt_state, optim,
                  n_iter=1000, save_iter=10, task_id=0, config={},
                  record_dict={}, notABTrain=True,
                  problem_type='vectors', loss_type='classification',
                  phase='main', record_training=True, global_iteration_offset=0):
        """Unified continual learning training loop for all problem types.

        Args:
            train__: Tuple of (trainloader, exploader, valloader, testloader)
            params: Model parameters (trainable)
            static: Static model components (frozen)
            opt_state: Optimizer state - passed in and returned for continuity across tasks
            optim: Optimizer instance (for .update())
            n_iter: Number of epochs
            save_iter: Save metrics every N epochs
            task_id: Current task ID
            config: Configuration dict (must contain 'flag' for regularization)
            record_dict: Dictionary to record metrics
            notABTrain: True for normal training, False for AWB A/B training
            problem_type: 'vectors' or 'graph'
            loss_type: 'classification' or 'regression'
            phase: Training phase - 'main' (Step 5/standard), 'ab' (AB training), 'preliminary' (not recorded)
            record_training: Whether to record metrics (False for preliminary/warmup phases)
            global_iteration_offset: Offset for global iteration counter (to continue from previous phase)

        Returns:
            Tuple of (params, static, opt_state, record_dict)
        """
        trainloader, exploader, valloader, testloader = train__
        flag = config.get("flag", [1.0, 1.0])
        # Added by Claude: Get gradient combination weights from config
        # [alpha, beta, gamma] for [current_task, experience_replay, hamiltonian_regularization]
        grad_weights = config.get("grad_weights", None)  # None uses defaults in hamiltonian.py

        # Added by Claude: Get dV normalization and gradient clipping settings
        normalize_dV = config.get("normalize_dV", True)  # Default: enabled
        dV_scale = config.get("dV_scale_factor", 1.0)  # Default: no extra scaling
        gradient_clip_norm = config.get("gradient_clip_norm", None)  # Default: no clipping

        pbar = tqdm(range(n_iter), dynamic_ncols=True)

        # Select Hamiltonian function based on problem/loss type
        if problem_type == 'graph':
            hamiltonian_fn = self.return_Hamiltonian_graph
            transforms = get_graph_transforms()
        elif loss_type == 'regression':
            hamiltonian_fn = self.return_Hamiltonian_mse
            transforms = None
        else:  # classification
            hamiltonian_fn = self.return_Hamiltonian_class
            transforms = None

        # Pre-compute variance for perturbation
        var_x, var_adj = self._compute_perturbation_variance(
            trainloader, exploader, problem_type
        )

        # JIT-compile optimizer step to avoid recompilation overhead
        @jax.jit
        def optimizer_step(grad, opt_state, params):
            updates, new_opt_state = optim.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        for epoch in pbar:
            trainiter = iter(trainloader)
            expiter = iter(exploader)

            # Epoch accumulators
            epoch_H, epoch_V, epoch_dV = [], [], []
            epoch_dV_dtheta, epoch_dV_dx = [], []
            epoch_dV_dadj = []  # Only used for graph
            epoch_grad_norm, epoch_train_metrics = [], []

            # Inner loop: iterate over batches
            for batch, batch_ex in zip(trainiter, expiter):
                if problem_type == 'graph':
                    # Graph data processing
                    batch = transforms(batch)
                    batch_ex = transforms(batch_ex)
                    delta_x = np_.random.normal(0, var_x, batch_ex.x.numpy().shape)
                    delta_adj = np_.random.normal(0, var_adj, batch_ex.adj.shape)
                    data = (static, (batch, batch_ex, delta_x, delta_adj))
                else:
                    # Vector data processing (MLP, CNN)
                    # Use asarray for zero-copy when possible
                    (x, y) = batch
                    (exp_x, exp_y) = batch_ex
                    min_batch = min(exp_x.shape[0], x.shape[0])

                    # Convert PyTorch tensors to JAX arrays
                    # Note: .numpy() on CPU tensor is a view, asarray minimizes copies
                    x = jnp.asarray(x.numpy()[:min_batch], dtype=jnp.float64)
                    exp_x = jnp.asarray(exp_x.numpy()[:min_batch], dtype=jnp.float64)

                    if loss_type == 'regression':
                        y = jnp.asarray(y.numpy()[:min_batch], dtype=jnp.float64)
                        exp_y = jnp.asarray(exp_y.numpy()[:min_batch], dtype=jnp.float64)
                    else:  # classification
                        y = jnp.asarray(y.numpy()[:min_batch], dtype=jnp.int64)
                        exp_y = jnp.asarray(exp_y.numpy()[:min_batch], dtype=jnp.int64)

                    # Generate perturbations (numpy is fine here, small overhead)
                    delta_x = jnp.asarray(np_.random.normal(0, var_x, exp_x.shape))
                    data = (static, (x, y, exp_x, exp_y, delta_x, flag))

                # Compute Hamiltonian gradient (with configurable gradient weights and dV normalization)
                grad, losses = hamiltonian_fn(params, data, notABTrain,
                                             grad_weights=grad_weights,
                                             normalize_dV=normalize_dV,
                                             dV_scale=dV_scale)

                # Unpack losses
                if problem_type == 'graph':
                    (H, V, dV, dV_dtheta, dV_dx, dV_dadj) = losses
                    epoch_dV_dadj.append(float(dV_dadj))
                else:
                    (H, V, dV, dV_dtheta, dV_dx) = losses

                # Added by Claude: Apply gradient clipping if enabled
                grad, grad_norm, was_clipped = self._clip_gradients(grad, max_norm=gradient_clip_norm)

                # Update parameters (using JIT-compiled optimizer step)
                params, opt_state = optimizer_step(grad, opt_state, params)

                # Accumulate loss metrics
                epoch_H.append(float(H))
                epoch_V.append(float(V))
                epoch_dV.append(float(dV))
                epoch_dV_dtheta.append(float(dV_dtheta))
                epoch_dV_dx.append(float(dV_dx))
                epoch_grad_norm.append(float(grad_norm))

                # Compute training metric
                if problem_type == 'graph':
                    train_metric = self.return_metric(params, static, data=(batch, batch_ex), notABTrain=notABTrain)
                else:
                    train_metric = self.return_metric(params, static, data=(x, y), notABTrain=notABTrain)
                epoch_train_metrics.append(float(train_metric))

            # End of epoch: log metrics (at save_iter intervals and at the last epoch)
            is_last_epoch = (epoch == n_iter - 1)
            if (epoch % save_iter == 0 and epoch > 0) or is_last_epoch:
                H_avg = np_.mean(epoch_H)
                V_avg = np_.mean(epoch_V)
                dV_avg = np_.mean(epoch_dV)
                dV_dtheta_avg = np_.mean(epoch_dV_dtheta)
                dV_dx_avg = np_.mean(epoch_dV_dx)
                grad_norm_avg = np_.mean(epoch_grad_norm)
                train_metric_avg = np_.mean(epoch_train_metrics)

                # Compute test metrics
                test_current, test_exp = self._compute_metrics_on_sampled_batches(
                    params, static, testloader, num_batches=10,
                    problem_type=problem_type, notABTrain=notABTrain,
                    transforms=transforms
                )

                # Build loss dict for recording
                losses_dict = {
                    'H': H_avg, 'V': V_avg, 'dV': dV_avg,
                    'dV_dx': dV_dx_avg, 'dV_dtheta': dV_dtheta_avg,
                }
                if problem_type == 'graph':
                    losses_dict['dV_dadj'] = np_.mean(epoch_dV_dadj)

                # Progress bar display
                loss_name = 'MSE' if loss_type == 'regression' else 'CE'
                pbar.set_postfix_str(
                    f"{loss_name}={V_avg:.6e} H={H_avg:.6e} dV_dx={dV_dx_avg:.6e} "
                    f"dV_dθ={dV_dtheta_avg:.6e} ||∇||={grad_norm_avg:.6e} | "
                    f"Tr={train_metric_avg:.4f} Te/Cur={test_current:.4f} Te/Exp={test_exp:.4f}"
                )

                # Added by Claude: Phase-aware recording using new task-based structure
                # Also maintain backward compatibility with old 'iterations' dict
                if record_training:
                    model = eqx.combine(params, static)

                    # Compute global iteration for backward compatibility
                    global_iteration = global_iteration_offset + epoch

                    if phase == 'main':
                        # Main training: record to tasks[task_id]['main_training']
                        self.record_main_training_epoch(
                            record_dict=record_dict,
                            task_id=task_id,
                            global_iteration=global_iteration,
                            epoch=epoch,
                            losses=losses_dict,
                            gradients={'grad_norm': grad_norm_avg},
                            metrics={
                                'train': train_metric_avg,
                                'test_current': float(test_current),
                                'test_experience': float(test_exp),
                            },
                            model=model
                        )

                        # Added by Claude: Also record to old 'iterations' format for backward compatibility
                        if 'iterations' not in record_dict:
                            record_dict['iterations'] = {}
                        record_dict['iterations'][global_iteration] = self.record_metrics(
                            iteration=global_iteration,
                            step=epoch,
                            task_id=task_id,
                            losses=losses_dict,
                            gradients={'grad_norm': grad_norm_avg},
                            metrics={
                                'train': train_metric_avg,
                                'test_current': float(test_current),
                                'test_experience': float(test_exp),
                            },
                            model=model
                        )

                    elif phase == 'ab':
                        # AB training: record to tasks[task_id]['ab_training']
                        self.record_ab_training_epoch(
                            record_dict=record_dict,
                            task_id=task_id,
                            iteration=epoch,  # Local AB iteration
                            losses=losses_dict,
                            model=model
                        )

                        # Added by Claude: Also record to old 'iterations' format for backward compatibility
                        if 'iterations' not in record_dict:
                            record_dict['iterations'] = {}
                        ab_global_iteration = global_iteration  # Use same global iteration tracking
                        record_dict['iterations'][ab_global_iteration] = self.record_metrics(
                            iteration=ab_global_iteration,
                            step=epoch,
                            task_id=task_id,
                            losses=losses_dict,
                            gradients={'grad_norm': 0.0},  # Not tracked for AB
                            metrics={
                                'train': 0.0,
                                'test_current': 0.0,
                                'test_experience': 0.0,
                            },
                            model=model
                        )

                    elif phase == 'preliminary':
                        # Preliminary phase: record to old 'iterations' format for compute_avg_loss
                        if 'iterations' not in record_dict:
                            record_dict['iterations'] = {}
                        record_dict['iterations'][global_iteration] = self.record_metrics(
                            iteration=global_iteration,
                            step=epoch,
                            task_id=task_id,
                            losses=losses_dict,
                            gradients={'grad_norm': grad_norm_avg},
                            metrics={
                                'train': train_metric_avg,
                                'test_current': float(test_current),
                                'test_experience': float(test_exp),
                            },
                            model=model
                        )

        return params, static, opt_state, record_dict
