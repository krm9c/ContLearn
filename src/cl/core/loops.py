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
import time

# Added by Claude: Profiling support
from .profiling import profile, profile_section

# Added by Claude: Async checkpointing support
from .async_checkpoint import AsyncCheckpointManager

# Added by Claude: JAX-native data pipeline for GPU utilization
from ..datasets.jax_dataloader import PrefetchDataLoader, DualPrefetchDataLoader


# Graph transform pipeline (lazy import to avoid dependency when not needed)
_GRAPH_TRANSFORMS = None


def get_graph_transforms():
    """Lazily load graph transforms to avoid torch_geometric import when not needed.

    Transform pipeline matches old working code:
    T.Compose([T.GCNNorm(), T.ToDense(), T.NormalizeFeatures()])
    """
    global _GRAPH_TRANSFORMS
    if _GRAPH_TRANSFORMS is None:
        import torch_geometric.transforms as T

        _GRAPH_TRANSFORMS = T.Compose([
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

                    # Fixed by Claude: Pass single batch for evaluation, not tuple (batch, batch)
                    # return_metric expects single batch for eval mode, tuple for training mode
                    metric_current = self.return_metric(params, static, data=current_batch, notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    metric_exp = self.return_metric(params, static, data=exp_batch, notABTrain=notABTrain)
                    exp_metrics.append(metric_exp)
            else:
                iterator = iter(loader)
                for i in range(num_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    batch = transforms(batch)
                    # Fixed by Claude: Pass single batch for evaluation, not tuple
                    metric = self.return_metric(params, static, data=batch, notABTrain=notABTrain)
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
                    x_curr = jnp.array(x_curr.numpy(), dtype=jnp.float64)
                    # Keep labels as int64 for classification, float64 for regression
                    # Labels should be 1D vector of class indices, not expanded to (batch, 1)
                    if self.metric == 'class':
                        y_curr = jnp.array(y_curr.numpy(), dtype=jnp.int64)
                    else:
                        y_curr = jnp.array(y_curr.numpy(), dtype=jnp.float64)

                    metric_current = self.return_metric(params, static, data=(x_curr, y_curr), notABTrain=notABTrain)
                    current_metrics.append(metric_current)

                    # Added by Claude: Handle variable-length batch formats
                    if len(exp_batch) == 2:
                        x_exp, y_exp = exp_batch
                    elif len(exp_batch) == 3:
                        x_exp, y_exp, _ = exp_batch  # Ignore task_id or other metadata
                    else:
                        # Assume first two elements are (x, y)
                        x_exp, y_exp = exp_batch[0], exp_batch[1]

                    x_exp = jnp.array(x_exp.numpy(), dtype=jnp.float64)
                    # Keep labels as int64 for classification, float64 for regression
                    if self.metric == 'class':
                        y_exp = jnp.array(y_exp.numpy(), dtype=jnp.int64)
                    else:
                        y_exp = jnp.array(y_exp.numpy(), dtype=jnp.float64)

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
                    x = jnp.array(x.numpy(), dtype=jnp.float64)
                    # Keep labels as int64 for classification, float64 for regression
                    if self.metric == 'class':
                        y = jnp.array(y.numpy(), dtype=jnp.int64)
                    else:
                        y = jnp.array(y.numpy(), dtype=jnp.float64)
                    metric = self.return_metric(params, static, data=(x, y), notABTrain=notABTrain)
                    current_metrics.append(metric)
                    exp_metrics.append(metric)

        current_mean = np_.mean(current_metrics) if current_metrics else 0.0
        exp_mean = np_.mean(exp_metrics) if exp_metrics else 0.0
        return current_mean, exp_mean

    def _compute_perturbation_variance(self, trainloader, exploader, problem_type, max_batches=5):
        """Pre-compute variance for perturbation sampling.

        Uses the mean difference approach for feature variance.
        For graphs, also computes adjacency variance.

        Args:
            trainloader: Current task data loader
            exploader: Experience replay data loader
            problem_type: 'vectors' or 'graph'
            max_batches: Maximum batches to sample for variance estimation (default 5)
                        5 batches × 2048 batch_size = 10,240 samples (statistically sufficient)

        Returns:
            Tuple of (var_x, var_adj) where var_adj is 0 for vectors
        """
        var_x_list, var_adj_list = [], []
        transforms = get_graph_transforms() if problem_type == 'graph' else None

        # Fixed by Claude: Handle empty experience loader (Task 0 for graph datasets)
        # Try to check if exploader has any data
        exp_iter_test = iter(exploader)
        try:
            _ = next(exp_iter_test)
            has_experience = True
        except StopIteration:
            # Empty experience loader - return default small variance
            return 1e-3, 1e-3 if problem_type == 'graph' else 0.0

        # Experience loader has data, proceed normally
        # Added by Claude: Sample only max_batches for efficiency (sufficient for variance estimation)
        batch_count = 0
        for batch, batch_ex in zip(iter(trainloader), iter(exploader)):
            if batch_count >= max_batches:
                break
            batch_count += 1

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
                # Added by Claude: Handle both PyTorch tensors and JAX arrays
                if hasattr(x, 'numpy'):
                    # PyTorch tensor - convert to JAX
                    x = jnp.array(x.numpy()[:min_batch], dtype=jnp.float64)
                    exp_x = jnp.array(exp_x.numpy()[:min_batch], dtype=jnp.float64)
                else:
                    # Already JAX array - just slice and ensure dtype
                    x = jnp.array(x[:min_batch], dtype=jnp.float64)
                    exp_x = jnp.array(exp_x[:min_batch], dtype=jnp.float64)

            # Compute feature variance
            var_x_list.append(
                jnp.sqrt(jnp.linalg.norm((jnp.mean(x, axis=0) - jnp.mean(exp_x, axis=0))**2))
            )

        var_x = sum(var_x_list) / max(len(var_x_list), 1)
        var_adj = 1e-3 * (sum(var_adj_list) / max(len(var_adj_list), 1)) if var_adj_list else 0.0
        return var_x, var_adj

    # Added by Claude: Profile training loop
    @profile("Training Loop")
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
            save_iter: Save metrics to record_dict every N epochs (use eval_interval for test metrics)
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

        # Added by Claude: Wrap dataloaders with JAX prefetching for GPU utilization
        # This overlaps data loading with computation, eliminating CPU-GPU transfer bottleneck
        # prefetch_size=3 means 3 batches are loading while GPU processes current batch
        use_prefetch = config.get("use_jax_prefetch", True)  # Enable by default
        prefetch_size = config.get("prefetch_size", 3)  # 3 batches queued

        if use_prefetch:
            # Added by Claude: Pass loss_type to ensure correct dtype conversion
            trainloader = PrefetchDataLoader(trainloader, prefetch_size=prefetch_size, loss_type=loss_type)
            exploader = PrefetchDataLoader(exploader, prefetch_size=prefetch_size, loss_type=loss_type)
            # Don't prefetch test loaders - they're only used at eval_interval

        # Added by Claude: Get gradient combination weights from config
        # [alpha, beta, gamma] for [current_task, experience_replay, hamiltonian_regularization]
        grad_weights = config.get("grad_weights", None)  # None uses defaults in hamiltonian.py

        # Added by Claude: Get dV normalization and gradient clipping settings
        normalize_dV = config.get("normalize_dV", True)  # Default: enabled
        dV_scale = config.get("dV_scale_factor", 1.0)  # Default: no extra scaling
        gradient_clip_norm = config.get("gradient_clip_norm", None)  # Default: no clipping

        # Added by Claude: Separate logging/evaluation intervals for performance
        log_interval = config.get("log_interval", 1)  # Progress bar update frequency
        eval_interval = config.get("eval_interval", save_iter)  # Test metric computation frequency

        # Added by Claude: Initialize async checkpoint manager if periodic checkpointing enabled
        checkpoint_interval = config.get("checkpoint_interval", 0)
        checkpoint_manager = None
        if checkpoint_interval > 0 and config.get("model_path"):
            checkpoint_dir = f"{config['model_path']}_checkpoints"
            checkpoint_manager = AsyncCheckpointManager(
                save_dir=checkpoint_dir,
                max_checkpoints=config.get("max_checkpoints", 3),
                memory_limit_gb=config.get("checkpoint_memory_limit_gb", 8.0),
                enable_async=config.get("async_checkpointing", True)
            )

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

        # Added by Claude: Initialize JAX PRNG key for GPU-accelerated random generation
        # Use deterministic seed based on task_id for reproducibility
        rng_seed = config.get('random_seed', 42) + task_id * 1000
        rng_key = jax.random.PRNGKey(rng_seed)

        # Added by Claude: Profile JAX pre-conversion (Phase 3 optimization)
        profiling_enabled = config.get('profiling_enabled', False)
        if profiling_enabled:
            from .profiling import format_memory_stats
            print(f"\n[PROFILE] JAX pre-conversion starting... | {format_memory_stats()}")
            preconv_start = time.time()

        # Added by Claude: Phase 3 - Pre-convert train/exp loaders to JAX (ONCE per task)
        # Eliminates 10,000+ PyTorch→JAX conversions per task (only for vector data)
        # NOTE: Skip conversion if using PrefetchDataLoader (data already JAX arrays on GPU)
        if problem_type == 'vectors':
            if use_prefetch:
                # PrefetchDataLoader already converts to JAX and transfers to GPU
                # Just materialize the iterators into lists for multiple epochs
                # Added by Claude: Diagnostic logging
                print(f"[DEBUG] Using PrefetchDataLoader path (use_prefetch=True)")
                train_batches_jax = list(trainloader)
                exp_batches_jax = list(exploader)
                # Added by Claude: Check first batch device
                if train_batches_jax:
                    x_sample, y_sample = train_batches_jax[0]
                    print(f"[DEBUG] First train batch device: {list(x_sample.devices())}")
            else:
                # Original path: Manually convert PyTorch tensors to JAX
                # Added by Claude: Diagnostic logging
                print(f"[DEBUG] Using manual conversion path (use_prefetch=False)")
                print(f"[DEBUG] JAX default device: {jax.devices()[0]}")
                print(f"[DEBUG] All JAX devices: {jax.devices()}")

                train_batches_jax = []
                for i, batch in enumerate(trainloader):
                    x, y = batch[0], batch[1]
                    x_jax = jnp.array(x.numpy(), dtype=jnp.float64)
                    if loss_type == 'regression':
                        y_jax = jnp.array(y.numpy(), dtype=jnp.float64)
                    else:
                        y_jax = jnp.array(y.numpy(), dtype=jnp.int64)
                    train_batches_jax.append((x_jax, y_jax))
                    # Added by Claude: Log first batch device
                    if i == 0:
                        print(f"[DEBUG] First converted batch device: {x_jax.device}")

                exp_batches_jax = []
                for batch in exploader:
                    x, y = batch[0], batch[1]
                    x_jax = jnp.array(x.numpy(), dtype=jnp.float64)
                    if loss_type == 'regression':
                        y_jax = jnp.array(y.numpy(), dtype=jnp.float64)
                    else:
                        y_jax = jnp.array(y.numpy(), dtype=jnp.int64)
                    exp_batches_jax.append((x_jax, y_jax))

            if profiling_enabled:
                preconv_elapsed = time.time() - preconv_start
                print(f"[PROFILE] JAX pre-conversion complete: {preconv_elapsed:.2f}s | {format_memory_stats()}")
                print(f"[PROFILE]   Train batches: {len(train_batches_jax)}")
                print(f"[PROFILE]   Exp batches: {len(exp_batches_jax)}")

        # JIT-compile optimizer step to avoid recompilation overhead
        @jax.jit
        def optimizer_step(grad, opt_state, params):
            updates, new_opt_state = optim.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        # JIT-compile metric computation for vectors (classification/regression)
        @jax.jit
        def compute_metric_class(params, static, x, y):
            model = eqx.combine(params, static)
            preds = jax.vmap(model)(x)
            if preds.ndim == 3 and preds.shape[1] == 1:
                preds = jnp.squeeze(preds, axis=1)
            pred_y = jnp.argmax(jax.nn.log_softmax(preds), 1)
            # Fixed by Claude: Handle both one-hot encoded and index labels
            # If y is one-hot encoded (2D), convert to class indices
            if y.ndim == 2:
                y = jnp.argmax(y, axis=1)
            return jnp.mean(y == pred_y)

        @jax.jit
        def compute_metric_mse(params, static, x, y):
            model = eqx.combine(params, static)
            preds = jax.vmap(model)(x)
            if preds.ndim == 3 and preds.shape[1] == 1:
                preds = jnp.squeeze(preds, axis=1)
            return jnp.mean(optax.l2_loss(y, preds))

        # Select metric function based on loss type
        if loss_type == 'regression':
            compute_metric = compute_metric_mse
        else:
            compute_metric = compute_metric_class

        for epoch in pbar:
            # Added by Claude: Profile first epoch first batch
            if epoch == 0 and profiling_enabled:
                print(f"\n[PROFILE] Starting first epoch...")
                first_batch_start = time.time()
                first_batch_done = False

            # Added by Claude: Use pre-converted JAX data for vectors, original loaders for graphs
            if problem_type == 'vectors':
                trainiter = iter(train_batches_jax)
                expiter = iter(exp_batches_jax)
            else:
                trainiter = iter(trainloader)
                expiter = iter(exploader)

            # Epoch accumulators
            epoch_H, epoch_V, epoch_dV = [], [], []
            epoch_dV_dtheta, epoch_dV_dx = [], []
            epoch_dV_dadj = []  # Only used for graph
            epoch_grad_norm, epoch_train_metrics = [], []

            # Inner loop: iterate over batches
            batch_idx = 0
            for batch, batch_ex in zip(trainiter, expiter):
                if problem_type == 'graph':
                    # Graph data processing
                    batch = transforms(batch)
                    batch_ex = transforms(batch_ex)
                    # Added by Claude: GPU-accelerated random using JAX (replaces NumPy random)
                    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
                    delta_x = jax.random.normal(subkey1, batch_ex.x.numpy().shape) * var_x
                    delta_adj = jax.random.normal(subkey2, batch_ex.adj.shape) * var_adj
                    data = (static, (batch, batch_ex, delta_x, delta_adj))
                else:
                    # Vector data processing (MLP, CNN)
                    # Added by Claude: Data already in JAX format from pre-conversion, just unpack
                    (x, y) = batch
                    (exp_x, exp_y) = batch_ex
                    min_batch = min(exp_x.shape[0], x.shape[0])

                    # Slice to min batch size (data already correct dtype from pre-conversion)
                    x = x[:min_batch]
                    y = y[:min_batch]
                    exp_x = exp_x[:min_batch]
                    exp_y = exp_y[:min_batch]

                    # Added by Claude: GPU-accelerated random using JAX (replaces NumPy random)
                    rng_key, subkey = jax.random.split(rng_key)
                    delta_x = jax.random.normal(subkey, exp_x.shape) * var_x
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

                # Compute training metric (using JIT-compiled function for vectors)
                if problem_type == 'graph':
                    # Graph uses non-JIT method (different data structure)
                    train_metric = self.return_metric(params, static, data=(batch, batch_ex), notABTrain=notABTrain)
                else:
                    # Vectors use JIT-compiled metric function
                    train_metric = compute_metric(params, static, x, y)
                epoch_train_metrics.append(float(train_metric))

                # Added by Claude: Report first batch time
                if epoch == 0 and batch_idx == 0 and profiling_enabled and not first_batch_done:
                    first_batch_elapsed = time.time() - first_batch_start
                    print(f"[PROFILE] First batch complete: {first_batch_elapsed:.2f}s | {format_memory_stats()}")
                    first_batch_done = True

                batch_idx += 1

            # End of epoch: compute averages and conditionally log/evaluate
            # Added by Claude: Separate intervals for logging (cheap) vs evaluation (expensive)
            is_last_epoch = (epoch == n_iter - 1)
            should_log = (epoch % log_interval == 0 and epoch > 0) or is_last_epoch
            # Added by Claude: Skip test evaluation during A/B training (expensive due to AWB transformations)
            # A/B training focuses on optimizing A and B matrices, test metrics not critical during this phase
            if phase == 'ab':
                should_eval = is_last_epoch  # Only evaluate on final epoch of A/B training
            else:
                should_eval = (epoch % eval_interval == 0 and epoch > 0) or is_last_epoch
            should_record = (epoch % save_iter == 0 and epoch > 0) or is_last_epoch

            # Always compute training metric averages for progress bar
            if should_log:
                H_avg = np_.mean(epoch_H)
                V_avg = np_.mean(epoch_V)
                dV_avg = np_.mean(epoch_dV)
                dV_dtheta_avg = np_.mean(epoch_dV_dtheta)
                dV_dx_avg = np_.mean(epoch_dV_dx)
                grad_norm_avg = np_.mean(epoch_grad_norm)
                train_metric_avg = np_.mean(epoch_train_metrics)

                # Compute expensive test metrics only at eval_interval (not every log_interval)
                if should_eval:
                    test_current, test_exp = self._compute_metrics_on_sampled_batches(
                        params, static, testloader, num_batches=10,
                        problem_type=problem_type, notABTrain=notABTrain,
                        transforms=transforms
                    )
                else:
                    # Skip expensive test evaluation, use placeholder values for progress bar
                    test_current, test_exp = 0.0, 0.0

                # Build loss dict for progress bar
                losses_dict = {
                    'H': H_avg, 'V': V_avg, 'dV': dV_avg,
                    'dV_dx': dV_dx_avg, 'dV_dtheta': dV_dtheta_avg,
                }
                if problem_type == 'graph':
                    losses_dict['dV_dadj'] = np_.mean(epoch_dV_dadj)

                # Progress bar display (always update when logging)
                loss_name = 'MSE' if loss_type == 'regression' else 'CE'
                if should_eval:
                    # Full display with test metrics
                    pbar.set_postfix_str(
                        f"{loss_name}={V_avg:.6e} H={H_avg:.6e} dV_dx={dV_dx_avg:.6e} "
                        f"dV_dθ={dV_dtheta_avg:.6e} ||∇||={grad_norm_avg:.6e} | "
                        f"Tr={train_metric_avg:.4f} Te/Cur={test_current:.4f} Te/Exp={test_exp:.4f}"
                    )
                else:
                    # Lightweight display without test metrics
                    pbar.set_postfix_str(
                        f"{loss_name}={V_avg:.6e} H={H_avg:.6e} dV_dx={dV_dx_avg:.6e} "
                        f"dV_dθ={dV_dtheta_avg:.6e} ||∇||={grad_norm_avg:.6e} | "
                        f"Tr={train_metric_avg:.4f}"
                    )

                # Added by Claude: Record to dict only at save_iter intervals (not every log)
                # This reduces I/O overhead while maintaining progress bar updates
                if record_training and should_record and should_eval:
                    # Added by Claude: Eigenvalues only recorded for AB training, not main training
                    # Main training doesn't need eigenvalue tracking (expensive operation)
                    if phase == 'ab':
                        # AB training: compute eigenvalues to monitor AB matrix evolution
                        model = eqx.combine(params, static)
                    else:
                        # Main/preliminary training: skip eigenvalue computation
                        model = None

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

                    # Added by Claude: Periodic checkpointing (async, non-blocking)
                    if checkpoint_manager is not None and (epoch % checkpoint_interval == 0 and epoch > 0):
                        checkpoint_model = eqx.combine(params, static)
                        checkpoint_manager.save_checkpoint(
                            model=checkpoint_model,
                            record_dict=record_dict,
                            task_id=task_id,
                            epoch=epoch,
                            prefix=f"{phase}_checkpoint"
                        )

        # Added by Claude: Wait for any pending checkpoint saves before returning
        if checkpoint_manager is not None:
            checkpoint_manager.wait_all(timeout=30.0)
            checkpoint_manager.shutdown()

        return params, static, opt_state, record_dict
