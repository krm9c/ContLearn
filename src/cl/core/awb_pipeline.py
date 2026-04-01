"""
AWB (Adaptive Weight Basis) Pipeline Orchestrator.

This module implements the generic 5-step AWB algorithm for continual learning:
    STEP 1: Preliminary training on new task
    STEP 2: Decide if architecture change is needed
    STEP 3a: Architecture search (if change needed)
    STEP 3b: Train A/B matrices with W frozen
    STEP 4: Compute V = A @ W @ B^T
    STEP 5: Train V with A/B frozen (warmup + main training)

The orchestrator is model-agnostic - it calls model-specific operations via
the AWBOperations interface. This allows the same pipeline to work for:
- MLP (fully connected networks)
- CNN (convolutional networks)
- GCN (graph convolutional networks)
- Transformers (attention-based models) - future

Recording control ensures clean separation of training phases:
- Preliminary, warmup, and architecture search: NOT recorded
- AB training: Recorded separately in record_dict['tasks'][task_id]['ab_training']
- Main training: Recorded normally in record_dict['iterations']
"""

from typing import Dict, Any, Tuple, Optional
import subprocess
import equinox as eqx
import optax
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Batch
import time

# Added by Claude: Profiling support
from .profiling import profile, profile_section, get_gpu_memory_usage
from .async_checkpoint import AsyncCheckpointManager

from .awb_operations import AWBOperations
from .awb import (
    should_change_arch,
    compute_avg_loss,
    compute_ab_threshold,
)
from ..config.constants import (
    DEFAULT_AWB_PRELIMINARY_EPOCHS,
    DEFAULT_AWB_AB_TRAINING_EPOCHS,
    DEFAULT_AWB_AB_WARMUP_EPOCHS,
    DEFAULT_AWB_AB_MAX_ITERATIONS,
    DEFAULT_AWB_AVERAGING_WINDOW,
    DEFAULT_LR,
)


def create_balanced_validation_set(loader, validation_ratio=0.2, batch_size=64):
    """Create a balanced validation set from a data loader.

    Samples validation_ratio% of data from each class to ensure balanced representation.
    This is used for AWB architecture search to avoid using full training data.

    Handles both vector datasets (MNIST, CIFAR) and graph datasets (synthetic graphs).

    Args:
        loader: PyTorch DataLoader or torch_geometric DataLoader to sample from
        validation_ratio: Fraction of data to use for validation (default 0.2 = 20%)
        batch_size: Batch size for validation loader

    Returns:
        DataLoader with balanced validation set
    """
    # Detect if this is a graph dataset or vector dataset
    # Check the first batch to determine loader type
    first_batch = next(iter(loader))
    is_graph = isinstance(first_batch, Batch)

    if is_graph:
        # Handle graph datasets (torch_geometric)
        all_graphs = []
        for batch in loader:
            # Extract individual graphs from the batch
            all_graphs.extend(batch.to_data_list())

        # Group graphs by class
        class_to_graphs = {}
        for graph in all_graphs:
            label = int(graph.y.item())
            if label not in class_to_graphs:
                class_to_graphs[label] = []
            class_to_graphs[label].append(graph)

        # Sample validation graphs from each class
        val_graphs = []
        for cls, graphs in class_to_graphs.items():
            n_samples = len(graphs)
            n_val = max(1, int(n_samples * validation_ratio))
            # Randomly sample graphs for this class
            perm = torch.randperm(n_samples)
            val_graphs.extend([graphs[i] for i in perm[:n_val]])

        # Create validation loader
        val_loader = GeometricDataLoader(
            val_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )

        print(f"  Created balanced validation set: {len(val_graphs)} graphs from {len(class_to_graphs)} classes")
        return val_loader

    else:
        # Handle vector datasets (MNIST, CIFAR)
        all_x, all_y = [], []
        for batch_x, batch_y in loader:
            all_x.append(batch_x)
            all_y.append(batch_y)

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # Group indices by class
        unique_classes = torch.unique(all_y)
        val_indices = []

        for cls in unique_classes:
            cls_indices = torch.where(all_y == cls)[0]
            n_samples = len(cls_indices)
            n_val = max(1, int(n_samples * validation_ratio))  # At least 1 sample per class

            # Randomly sample indices for this class
            perm = torch.randperm(n_samples)
            val_idx = cls_indices[perm[:n_val]]
            val_indices.append(val_idx)

        # Combine all validation indices
        val_indices = torch.cat(val_indices)

        # Create validation dataset
        val_x = all_x[val_indices]
        val_y = all_y[val_indices]

        val_dataset = TensorDataset(val_x, val_y)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False  # Keep all validation samples
        )

        print(f"  Created balanced validation set: {len(val_indices)} samples from {len(unique_classes)} classes")
        return val_loader


# Added by Claude: Profile entire AWB task pipeline
@profile("AWB Task Pipeline")
def run_awb_task(
    task_id: int,
    trainer,
    model: eqx.Module,
    train_data: Tuple,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    config: Dict[str, Any],
    record_dict: Dict[str, Any],
    awb_ops: AWBOperations,
    problem_type: str,
    loss_type: str,
    previous_task_loss: float = None,
    resume_state: Optional[Dict[str, Any]] = None
) -> Tuple[eqx.Module, optax.GradientTransformation, optax.OptState, Dict[str, Any]]:
    """Execute complete AWB 5-step pipeline for one task.

    Args:
        task_id: Current task ID (must be >= 1, task 0 uses standard CL)
        trainer: Trainer instance
        model: Current model (Equinox module)
        train_data: Tuple of (trainloader, exploader, valloader, testloader)
        optim: Optimizer
        opt_state: Optimizer state
        config: Configuration dictionary
        record_dict: Recording dictionary
        awb_ops: Model-specific AWB operations
        problem_type: 'vectors' or 'graph'
        loss_type: 'regression' or 'classification'
        previous_task_loss: Loss from previous task

    Returns:
        Tuple of (model, optim, opt_state, record_dict)
    """
    if task_id < 1:
        raise ValueError(f"AWB pipeline only for tasks >= 1, got task_id={task_id}")

    print(f"\n{'='*70}")
    print(f"AWB PIPELINE - Task {task_id}")
    print(f"{'='*70}")

    # Extract configuration
    preliminary_epochs = config.get('awb_preliminary_epochs', DEFAULT_AWB_PRELIMINARY_EPOCHS)
    ab_training_epochs = config.get('awb_ab_training_epochs', DEFAULT_AWB_AB_TRAINING_EPOCHS)
    ab_warmup_epochs = config.get('awb_ab_warmup_epochs', DEFAULT_AWB_AB_WARMUP_EPOCHS)
    ab_max_iterations = config.get('awb_ab_max_iterations', DEFAULT_AWB_AB_MAX_ITERATIONS)
    averaging_window = config.get('awb_averaging_window', DEFAULT_AWB_AVERAGING_WINDOW)
    epochs_per_task = config.get('epochs_per_task', 500)
    save_iter = config.get('save_iter', 50)

    trainloader, exploader, valloader, testloader = train_data

    # Added by Claude: Initialize task metadata for architecture history
    task_metadata = {
        'preliminary_loss': None,
        'architecture_changed': False,
        'change_reason': 'no_change',
        'searched_architectures': [],
        'loss_ratio': None,
        'search_time': 0.0,
        'compute': {},
    }

    def _mem_snapshot():
        mem = get_gpu_memory_usage()
        if mem is None:
            return None
        used_gb, peak_gb = mem
        return {'used_gb': float(used_gb), 'peak_gb': float(peak_gb)}

    def _gpu_util_snapshot():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                return None
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                return None
            values = []
            for line in lines:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
            if not values:
                return None
            return {'avg_pct': float(sum(values) / len(values)), 'max_pct': float(max(values))}
        except Exception:
            return None

    def _record_step_time(
        step_key: str,
        start_time: float,
        mem_start: Optional[Dict[str, float]] = None,
        util_start: Optional[Dict[str, float]] = None,
    ):
        elapsed = time.time() - start_time
        entry = {'time_s': float(elapsed)}
        if mem_start is not None:
            entry['mem_start'] = mem_start
        mem_end = _mem_snapshot()
        if mem_end is not None:
            entry['mem_end'] = mem_end
        if util_start is not None:
            entry['gpu_util_start'] = util_start
        util_end = _gpu_util_snapshot()
        if util_end is not None:
            entry['gpu_util_end'] = util_end
        task_metadata['compute'][step_key] = entry
        return elapsed

    task_start_time = time.time()

    resume_phase = resume_state.get('phase') if resume_state is not None else None
    resume_after_search = resume_phase == 'awb_search'
    resume_after_ab = resume_phase == 'ab'
    current_iterations = len(record_dict.get('iterations', {}))
    if resume_after_search:
        print("\n[RESUME] Skipping preliminary + search (using saved architecture)")
    if resume_after_ab:
        print("\n[RESUME] Skipping preliminary + search + AB training (using saved A/B)")

    # STEP 1: Preliminary Training
    if not resume_after_search and not resume_after_ab:
        print(f"\n[STEP 1] Preliminary training ({preliminary_epochs} epochs) - Recorded temporarily")
        prelim_mem_start = _mem_snapshot()
        prelim_util_start = _gpu_util_snapshot()
        prelim_start = time.time()
        params, static = awb_ops.partition_for_standard_training(model)

        params, static, opt_state, record_dict = trainer.train__CL(
            train__=(trainloader, exploader, valloader, testloader),
            params=params, static=static, opt_state=opt_state, optim=optim,
            n_iter=preliminary_epochs, save_iter=save_iter,
            task_id=task_id, config=config, record_dict=record_dict,
            problem_type=problem_type, loss_type=loss_type,
            phase='preliminary', record_training=True, global_iteration_offset=current_iterations
        )

        model = eqx.combine(params, static)
        _record_step_time('preliminary', prelim_start, prelim_mem_start, prelim_util_start)

    if not resume_after_search and not resume_after_ab:
        # Added by Claude: Get preliminary loss from last recorded iteration
        iterations_dict = record_dict.get('iterations', {})
        last_prelim_iter = current_iterations + preliminary_epochs - 1
        if last_prelim_iter in iterations_dict:
            last_record = iterations_dict[last_prelim_iter]
            trainWLoss = last_record['losses']['V'] if isinstance(last_record, dict) and 'losses' in last_record else last_record[0]
        else:
            # Fallback: try to find the most recent iteration
            available_iters = sorted([k for k in iterations_dict.keys() if k >= current_iterations])
            if available_iters:
                last_record = iterations_dict[available_iters[-1]]
                trainWLoss = last_record['losses']['V'] if isinstance(last_record, dict) and 'losses' in last_record else last_record[0]
            else:
                trainWLoss = 0.0

        print(f"  Preliminary loss: {trainWLoss:.6f}")

        # Added by Claude: Store preliminary loss in metadata
        task_metadata['preliminary_loss'] = trainWLoss

        # STEP 2: Decision
        print(f"\n[STEP 2] Architecture change decision")
        if previous_task_loss is None:
            previous_task_loss = trainWLoss
            print(f"  WARNING: No previous task loss, using preliminary loss")

        # Added by Claude: Store loss ratio in metadata
        task_metadata['loss_ratio'] = trainWLoss / previous_task_loss if previous_task_loss > 0 else 1.0

        print(f"  Current: {trainWLoss:.6f}, Previous: {previous_task_loss:.6f}")

        # Added by Claude: Read threshold from config
        threshold_high = config.get('awb_loss_ratio_threshold')

        change_arch = should_change_arch(trainWLoss, previous_task_loss,
                                         threshold_high=threshold_high)

        # Added by Claude: DEBUG HACK - Force architecture change for profiling
        if config.get('force_arch_change', False):
            print(f"  [DEBUG] Forcing architecture change (force_arch_change=True)")
            change_arch = True

        print(f"  Decision: {'CHANGE' if change_arch else 'KEEP'}")

        original_arch = awb_ops.get_model_architecture(model)
        saved_weights = awb_ops.save_weights(model)
    else:
        trainWLoss = resume_state.get('preliminary_loss', None)
        if trainWLoss is None:
            trainWLoss = 0.0
            print("  [RESUME] Missing preliminary loss; defaulting to 0.0")

        previous_task_loss = resume_state.get('previous_task_loss', previous_task_loss)
        if previous_task_loss is None:
            previous_task_loss = trainWLoss
        task_metadata['preliminary_loss'] = trainWLoss
        task_metadata['loss_ratio'] = trainWLoss / previous_task_loss if previous_task_loss > 0 else 1.0
        original_arch = awb_ops.get_model_architecture(model)
        saved_weights = awb_ops.save_weights(model)
        if resume_after_search:
            new_arch = resume_state.get('awb_best_arch')
            if new_arch is None:
                raise ValueError("Resume requested after search but awb_best_arch is missing")
            change_arch = new_arch != original_arch
        else:
            new_arch = original_arch
            change_arch = True

    if change_arch:
        # STEP 3a: Architecture Search
        if not resume_after_search and not resume_after_ab:
            print(f"\n[STEP 3a] Architecture search - NOT recorded")

        if not resume_after_search and not resume_after_ab:
            # Added by Claude: Create balanced validation sets (20% of experience replay) for arch search
            # This is more efficient than using full training data and prevents overfitting
            validation_ratio = config.get('awb_validation_ratio', 0.2)
            val_batch_size = config.get('batch_size', 64)

            val_trainloader = create_balanced_validation_set(trainloader, validation_ratio, val_batch_size)
            val_exploader = create_balanced_validation_set(exploader, validation_ratio, val_batch_size)

            # Fixed by Claude: Unpack testloader tuple (test_curr_loader, test_exp_loader)
            test_curr, test_exp = testloader

            # Added by Claude: Profile architecture search
            # Use validation sets for architecture search instead of full training data
            arch_mem_start = _mem_snapshot()
            arch_util_start = _gpu_util_snapshot()
            arch_start = time.time()
            with profile_section("Architecture Search"):
                new_arch = awb_ops.search_architecture(
                    model, task_id, trainWLoss, val_trainloader, val_exploader,
                    test_curr, test_exp, config, trainer
                )
            task_metadata['search_time'] = _record_step_time('arch_search', arch_start, arch_mem_start, arch_util_start)
            print(f"  Original: {original_arch}")
            print(f"  Optimal: {new_arch}")

            model = awb_ops.restore_weights(model, saved_weights)
        else:
            new_arch = resume_state.get('awb_best_arch')
            if new_arch is None:
                if resume_after_ab:
                    new_arch = original_arch
                else:
                    raise ValueError("Resume requested after search but awb_best_arch is missing")
            print(f"  Original: {original_arch}")
            if resume_after_ab:
                print(f"  Optimal (resumed): {new_arch}")
            else:
                print(f"  Optimal (resumed): {new_arch}")

        # Optional: checkpoint after search to allow resume at Step 3b
        if not resume_after_search and not resume_after_ab and config.get('awb_checkpoint_after_search', True):
            if 'tasks' not in record_dict:
                record_dict['tasks'] = {}
            if task_id not in record_dict['tasks']:
                record_dict['tasks'][task_id] = {}
            record_dict['tasks'][task_id].update(task_metadata)

            checkpoint_dir = f"{config.get('model_path', 'outputs/model')}_checkpoints"
            checkpoint_manager = AsyncCheckpointManager(
                save_dir=checkpoint_dir,
                max_checkpoints=config.get("max_checkpoints", 3),
                memory_limit_gb=config.get("checkpoint_memory_limit_gb", 8.0),
                enable_async=config.get("async_checkpointing", True)
            )
            checkpoint_metadata = {
                'task_id': task_id,
                'epoch': preliminary_epochs,
                'phase': 'awb_search',
                'notABTrain': True,
                'arch_info': awb_ops.get_model_architecture(model),
                'awb_best_arch': new_arch,
                'awb_original_arch': original_arch,
                'preliminary_loss': trainWLoss,
                'previous_task_loss': previous_task_loss,
            }
            checkpoint_manager.save_checkpoint(
                model=model,
                record_dict=record_dict,
                task_id=task_id,
                epoch=preliminary_epochs,
                prefix="awb_search_checkpoint",
                opt_state=opt_state,
                config_snapshot=config,
                metadata=checkpoint_metadata
            )
            checkpoint_manager.wait_all(timeout=30.0)
            checkpoint_manager.shutdown()

        if new_arch != original_arch or resume_after_ab:
            # Added by Claude: Update metadata for architecture change
            task_metadata['architecture_changed'] = True
            task_metadata['change_reason'] = 'loss_ratio_threshold'

            # Added by Claude: Check if we should skip A/B transfer (Condition 3)
            skip_transfer = config.get('awb_skip_transfer', False)

            if skip_transfer:
                # CONDITION 3: Skip A/B training, use random init
                print(f"\n[STEP 3b] Skipping A/B training (awb_skip_transfer=True)")
                print(f"  Using random initialization for new architecture")

                # Set AB matrices (random init) and compute V = A @ W @ B.T
                model = awb_ops.set_AB_matrices(model, original_arch, new_arch)

                # STEP 4: Compute V
                print(f"\n[STEP 4] Compute V = A @ W @ B^T (random A/B)")
                model = awb_ops.compute_V(model)
                params, static = awb_ops.partition_for_standard_training(model)

                # STEP 5: Train V with random initialization
                print(f"\n[STEP 5] Train V (random init): warmup {ab_warmup_epochs} + main {epochs_per_task}")
                from ..runners.generic_runner import create_optimizer
                optim = create_optimizer(config)
                opt_state = optim.init(params)

            else:
                # STEP 3b: Train A/B
                if not resume_after_ab:
                    print(f"\n[STEP 3b] Train A/B matrices ({ab_training_epochs} epochs) - Recorded separately")

                if not resume_after_ab:
                    # Added by Claude: Profile A/B training
                    ab_mem_start = _mem_snapshot()
                    ab_util_start = _gpu_util_snapshot()
                    ab_training_start = time.time()

                    model = awb_ops.set_AB_matrices(model, original_arch, new_arch)
                    diff_model, static_model = awb_ops.partition_for_AB_training(model)
                    trainer.initialize_ab_training(record_dict, task_id)

                    ab_lr = config.get('awb_ab_lr', DEFAULT_LR)
                    ab_optim = optax.adam(ab_lr)
                    ab_opt_state = ab_optim.init(diff_model)

                    diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
                        train__=(trainloader, exploader, valloader, testloader),
                        params=diff_model, static=static_model, opt_state=ab_opt_state, optim=ab_optim,
                        n_iter=ab_training_epochs, save_iter=save_iter,
                        task_id=task_id, config=config, record_dict=record_dict,
                        notABTrain=False, problem_type=problem_type, loss_type=loss_type,
                        phase='ab', record_training=True, global_iteration_offset=0
                    )

                    ab_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=0, epochs=ab_training_epochs, window=averaging_window)
                    
                    print(f"  AB loss: {ab_loss:.6f}")
                    # Optional: Continue AB training
                    ab_threshold = compute_ab_threshold(trainWLoss, previous_task_loss)
                    ab_iter = 1
                    while (trainWLoss * ab_threshold < ab_loss) and (ab_iter < ab_max_iterations):
                        print(f"  Continuing AB training (iter {ab_iter + 1})")
                        diff_model, static_model, ab_opt_state, record_dict = trainer.train__CL(
                            train__=(trainloader, exploader, valloader, testloader),
                            params=diff_model, static=static_model, opt_state=ab_opt_state, optim=ab_optim,
                            n_iter=ab_training_epochs, save_iter=save_iter,
                            task_id=task_id, config=config, record_dict=record_dict,
                            notABTrain=False, problem_type=problem_type, loss_type=loss_type,
                            phase='ab', record_training=True, global_iteration_offset=0
                        )
                        ab_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=0,
                                                   epochs=ab_training_epochs, window=averaging_window)
                        ab_iter += 1

                    model = eqx.combine(diff_model, static_model)

                    _record_step_time('ab_training', ab_training_start, ab_mem_start, ab_util_start)

                    # Added by Claude: Report A/B training time
                    if config.get('profiling_enabled'):
                        from .profiling import format_memory_stats
                        ab_elapsed = time.time() - ab_training_start
                        print(f"[PROFILE] A/B training total: {ab_elapsed:.2f}s | {format_memory_stats()}")

                # STEP 4: Compute V
                print(f"\n[STEP 4] Compute V = A @ W @ B^T")
                compute_v_mem_start = _mem_snapshot()
                compute_v_util_start = _gpu_util_snapshot()
                compute_v_start = time.time()

                # DEBUG: Verify loss BEFORE and AFTER compute_V
                import jax.numpy as jnp
                debug_batch = next(iter(trainloader))
                if problem_type == 'graph':
                    from ..core.loops import get_graph_transforms
                    transforms = get_graph_transforms()
                    debug_batch = transforms(debug_batch)
                    x_d = jnp.asarray(debug_batch.x.numpy(), dtype=jnp.float64)
                    y_d = jnp.asarray(debug_batch.y.numpy(), dtype=jnp.int64)
                    adj_d = jnp.asarray(debug_batch.adj.numpy(), dtype=jnp.float64)
                    b_d = jnp.asarray(debug_batch.batch.numpy())
                    n_d = jnp.asarray(debug_batch.n_nodes.numpy())

                    # BEFORE compute_V: loss with get_AWBT
                    pred_before = model.get_AWBT(x_d, adj_d, b_d, n_d)
                    loss_before = float(jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_before, y_d)))
                    print(f"  [DEBUG] Loss BEFORE compute_V (get_AWBT): {loss_before:.6f}")

                model = awb_ops.compute_V(model)
                params, static = awb_ops.partition_for_standard_training(model)

                _record_step_time('compute_v', compute_v_start, compute_v_mem_start, compute_v_util_start)

                if problem_type == 'graph':
                    model_debug = eqx.combine(params, static)
                    # AFTER compute_V: loss with standard __call__
                    pred_after = model_debug(x_d, adj_d, b_d, n_d)
                    loss_after = float(jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_after, y_d)))
                    print(f"  [DEBUG] Current task - BEFORE (get_AWBT): {loss_before:.6f}")
                    print(f"  [DEBUG] Current task - AFTER (__call__): {loss_after:.6f}")
                    print(f"  [DEBUG] Current task - Difference: {abs(loss_before - loss_after):.10f}")

                    # Also check EXPERIENCE data
                    exp_batch = next(iter(exploader))
                    exp_batch = transforms(exp_batch)
                    x_e = jnp.asarray(exp_batch.x.numpy(), dtype=jnp.float64)
                    y_e = jnp.asarray(exp_batch.y.numpy(), dtype=jnp.int64)
                    adj_e = jnp.asarray(exp_batch.adj.numpy(), dtype=jnp.float64)
                    b_e = jnp.asarray(exp_batch.batch.numpy())
                    n_e = jnp.asarray(exp_batch.n_nodes.numpy())

                    # Experience loss AFTER compute_V
                    pred_exp = model_debug(x_e, adj_e, b_e, n_e)
                    loss_exp = float(jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_exp, y_e)))
                    print(f"  [DEBUG] Experience - AFTER (__call__): {loss_exp:.6f}")

                # STEP 5: Train V
                print(f"\n[STEP 5] Train V: warmup {ab_warmup_epochs} + main {epochs_per_task}")

                # DEBUG: Print weight shapes after compute_V (GCN only)
                if problem_type == 'graph':
                    print(f"  [DEBUG] Weight shapes after compute_V:")
                    print(f"    gcn_layers[0].weight: {model_debug.gcn_layers[0].weight.shape}")
                    print(f"    gcn_layers[1].weight: {model_debug.gcn_layers[1].weight.shape}")
                    print(f"    feed_layers[0].weight: {model_debug.feed_layers[0].weight.shape}")

                # Added by Claude: Compute LR reduction for V training based on parameter expansion
                # The issue: First gradient step causes loss explosion (0.93→53.14) because
                # V weights exist in a low-rank subspace and gradients point away from it.
                # Solution: Reduce LR proportionally to sqrt(param_expansion) for V warmup.
                import jax.tree_util
                param_leaves = jax.tree_util.tree_leaves(params)
                new_param_count = sum(p.size for p in param_leaves)

                # Get V training LR reduction factor from config (default: auto-compute from expansion)
                v_lr_factor = config.get('awb_v_lr_factor', None)
                if v_lr_factor is None:
                    if problem_type == 'graph':
                        # Auto-compute based on expansion ratio (GCN only)
                        # Use original param count from architecture before search
                        orig_gcn_sizes = [model_debug.gcn_sizes[0]] + [32] * (len(model_debug.gcn_sizes) - 1)  # Rough estimate
                        orig_feed_sizes = [32, 32, 16, model_debug.feed_sizes[-1]]
                        # Better: use the saved original sizes
                        orig_param_count = sum(
                            orig_gcn_sizes[i] * orig_gcn_sizes[i+1] + orig_gcn_sizes[i+1]
                            for i in range(len(orig_gcn_sizes)-1)
                        ) + sum(
                            orig_feed_sizes[i] * orig_feed_sizes[i+1] + orig_feed_sizes[i+1]
                            for i in range(len(orig_feed_sizes)-1)
                        )
                        expansion_ratio = new_param_count / max(orig_param_count, 1)
                        # Use 1/sqrt(expansion) as the LR factor (inspired by Xavier scaling)
                        import math
                        v_lr_factor = 1.0 / math.sqrt(expansion_ratio)
                        v_lr_factor = max(v_lr_factor, 0.01)  # Minimum 1% of original LR
                        print(f"  [V-TRAIN] Parameter expansion: {orig_param_count:,} → {new_param_count:,} ({expansion_ratio:.1f}x)")
                        print(f"  [V-TRAIN] Auto LR factor: {v_lr_factor:.4f}")
                    else:
                        v_lr_factor = 1.0
                        print("  [V-TRAIN] Auto LR factor disabled for non-graph models; using 1.0")
                else:
                    print(f"  [V-TRAIN] Using config LR factor: {v_lr_factor}")

                # Create optimizer with reduced LR for V training
                from ..runners.generic_runner import create_optimizer, create_optimizer_with_lr
                base_lr = config.get('lr', 0.001)
                v_training_lr = base_lr * v_lr_factor
                print(f"  [V-TRAIN] Base LR: {base_lr}, V training LR: {v_training_lr:.6f}")

                optim = create_optimizer_with_lr(config, v_training_lr)
                opt_state = optim.init(params)

            # Added by Claude: Create V training config with reduced LR for diagnostics
            v_train_config = {**config, 'lr': v_training_lr}

            if ab_warmup_epochs > 0:
                warmup_mem_start = _mem_snapshot()
                warmup_util_start = _gpu_util_snapshot()
                warmup_start = time.time()
                params, static, opt_state, record_dict = trainer.train__CL(
                    train__=(trainloader, exploader, valloader, testloader),
                    params=params, static=static, opt_state=opt_state, optim=optim,
                    n_iter=ab_warmup_epochs, save_iter=save_iter,
                    task_id=task_id, config=v_train_config, record_dict=record_dict,
                    problem_type=problem_type, loss_type=loss_type,
                    phase='warmup', record_training=False,
                    global_iteration_offset=task_id * epochs_per_task
                )
                _record_step_time('warmup', warmup_start, warmup_mem_start, warmup_util_start)

            main_mem_start = _mem_snapshot()
            main_util_start = _gpu_util_snapshot()
            main_start = time.time()
            params, static, opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=params, static=static, opt_state=opt_state, optim=optim,
                n_iter=epochs_per_task, save_iter=save_iter,
                task_id=task_id, config=v_train_config, record_dict=record_dict,
                problem_type=problem_type, loss_type=loss_type,
                phase='main', record_training=True,
                global_iteration_offset=task_id * epochs_per_task
            )
            model = eqx.combine(params, static)
            _record_step_time('main', main_start, main_mem_start, main_util_start)

        else:
            # Added by Claude: Same architecture found by search
            task_metadata['architecture_changed'] = False
            task_metadata['change_reason'] = 'search_found_same'

            # Same architecture, standard training
            print(f"  Same architecture - standard training")
            params, static = awb_ops.partition_for_standard_training(model)
            from ..runners.generic_runner import create_optimizer
            optim = create_optimizer(config)
            opt_state = optim.init(params)

            total_epochs = ab_warmup_epochs + epochs_per_task
            standard_mem_start = _mem_snapshot()
            standard_util_start = _gpu_util_snapshot()
            standard_start = time.time()
            params, static, opt_state, record_dict = trainer.train__CL(
                train__=(trainloader, exploader, valloader, testloader),
                params=params, static=static, opt_state=opt_state, optim=optim,
                n_iter=total_epochs, save_iter=save_iter,
                task_id=task_id, config=config, record_dict=record_dict,
                problem_type=problem_type, loss_type=loss_type,
                phase='main', record_training=True,
                global_iteration_offset=task_id * epochs_per_task
            )
            model = eqx.combine(params, static)
            _record_step_time('standard_training', standard_start, standard_mem_start, standard_util_start)
    else:
        # Added by Claude: No architecture change needed
        task_metadata['architecture_changed'] = False
        task_metadata['change_reason'] = 'loss_ratio_below_threshold'

        # No change, standard training
        print(f"  No change - standard training")
        params, static = awb_ops.partition_for_standard_training(model)
        from ..runners.generic_runner import create_optimizer
        optim = create_optimizer(config)
        opt_state = optim.init(params)

        total_epochs = ab_warmup_epochs + epochs_per_task
        standard_mem_start = _mem_snapshot()
        standard_util_start = _gpu_util_snapshot()
        standard_start = time.time()
        params, static, opt_state, record_dict = trainer.train__CL(
            train__=(trainloader, exploader, valloader, testloader),
            params=params, static=static, opt_state=opt_state, optim=optim,
            n_iter=total_epochs, save_iter=save_iter,
            task_id=task_id, config=config, record_dict=record_dict,
            problem_type=problem_type, loss_type=loss_type,
            phase='main', record_training=True,
            global_iteration_offset=task_id * epochs_per_task
        )
        model = eqx.combine(params, static)
        _record_step_time('standard_training', standard_start, standard_mem_start, standard_util_start)

    final_loss = compute_avg_loss(record_dict.get('iterations', {}), task_id=task_id,
                                   epochs=epochs_per_task, window=averaging_window)
    print(f"\n[AWB COMPLETE] Task {task_id} final loss: {final_loss:.6f}")
    print(f"{'='*70}\n")

    task_metadata['compute']['total'] = {'time_s': float(time.time() - task_start_time)}

    # Added by Claude: Store task metadata for architecture history tracking
    if 'tasks' not in record_dict:
        record_dict['tasks'] = {}
    if task_id not in record_dict['tasks']:
        record_dict['tasks'][task_id] = {}
    record_dict['tasks'][task_id].update(task_metadata)

    return model, optim, opt_state, record_dict


__all__ = ['run_awb_task']
