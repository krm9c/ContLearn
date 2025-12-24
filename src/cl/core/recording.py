"""
Unified recording system for training metrics and model analysis.

This module provides a centralized recording system that:
- Records all training/test metrics at each save_iter
- Computes and records eigenvalues of A/B matrices grouped by layer
- Maintains consistent format across all problem types (vectors/graphs)
- Supports easy extension for new metrics/datasets
- Saves data indexed by iteration with metadata
"""

import jax.numpy as jnp
import numpy as np
import pickle
import os
from typing import Dict, Any, Optional


class RecordingMixin:
    """Mixin class for unified metric recording across all training functions.

    This mixin provides centralized recording that replaces both TensorBoard
    logging and the previous tuple-based dictum structure.
    """

    def _compute_eigenvalues(self, model, combined=False):
        """Compute eigenvalues of A/B matrices (AWB mode) or weight matrices (standard mode).

        For AWB-enabled models: computes eigenvalues of A and B matrices.
        For standard models: computes eigenvalues of weight matrices (W).

        Args:
            model: The model (combined params + static)
            combined: If True, return model; if False, extract from self

        Returns:
            dict: Eigenvalues organized as:
                AWB mode:
                {
                    'A': {'layer_0': array, 'layer_1': array, ...},
                    'B': {'layer_0': array, 'layer_1': array, ...}
                }
                Standard mode (weights stored in 'A' key for plot compatibility):
                {
                    'A': {'layer_0': array, 'layer_1': array, ...},
                    'B': {}
                }
        """
        eigenvalues = {'A': {}, 'B': {}}

        try:
            # Added by Claude: Check if AWB is enabled and has valid A/B matrices
            awb_enabled = getattr(model, 'awb_enabled', False)
            has_valid_AB = (hasattr(model, 'A') and hasattr(model, 'B') and
                           model.A is not None and model.B is not None)

            # MLP: Compute eigenvalues based on AWB mode
            if hasattr(model, 'layers'):
                if awb_enabled and has_valid_AB and isinstance(model.A, list):
                    # AWB mode: compute eigenvalues of A and B matrices
                    for i, (A_matrix, B_matrix) in enumerate(zip(model.A, model.B)):
                        if A_matrix is not None and A_matrix.size > 0:
                            try:
                                eigs_A = jnp.linalg.eigvals(A_matrix @ A_matrix.T)
                                eigenvalues['A'][f'layer_{i}'] = np.array(eigs_A)
                            except:
                                eigenvalues['A'][f'layer_{i}'] = np.array([])

                        if B_matrix is not None and B_matrix.size > 0:
                            try:
                                eigs_B = jnp.linalg.eigvals(B_matrix @ B_matrix.T)
                                eigenvalues['B'][f'layer_{i}'] = np.array(eigs_B)
                            except:
                                eigenvalues['B'][f'layer_{i}'] = np.array([])
                else:
                    # Standard mode: compute eigenvalues of weight matrices
                    # Store in 'A' key for compatibility with plotting code
                    for i, layer in enumerate(model.layers):
                        if hasattr(layer, 'weight') and layer.weight is not None:
                            W = layer.weight
                            if W.size > 0:
                                try:
                                    # Compute W @ W.T eigenvalues (singular values squared)
                                    eigs_W = jnp.linalg.eigvals(W @ W.T)
                                    eigenvalues['A'][f'layer_{i}'] = np.array(eigs_W)
                                except:
                                    eigenvalues['A'][f'layer_{i}'] = np.array([])

            # CNN/CNN3D: A_conv, B_conv, A_feed, B_feed (AWB mode only)
            if hasattr(model, 'A_conv') and hasattr(model, 'B_conv'):
                for i, (A_matrix, B_matrix) in enumerate(zip(model.A_conv, model.B_conv)):
                    if A_matrix is not None and A_matrix.size > 0:
                        try:
                            eigs_A = jnp.linalg.eigvals(A_matrix @ A_matrix.T)
                            eigenvalues['A'][f'conv_{i}'] = np.array(eigs_A)
                        except:
                            eigenvalues['A'][f'conv_{i}'] = np.array([])

                    if B_matrix is not None and B_matrix.size > 0:
                        try:
                            eigs_B = jnp.linalg.eigvals(B_matrix @ B_matrix.T)
                            eigenvalues['B'][f'conv_{i}'] = np.array(eigs_B)
                        except:
                            eigenvalues['B'][f'conv_{i}'] = np.array([])

            if hasattr(model, 'A_feed') and hasattr(model, 'B_feed'):
                for i, (A_matrix, B_matrix) in enumerate(zip(model.A_feed, model.B_feed)):
                    if A_matrix is not None and A_matrix.size > 0:
                        try:
                            eigs_A = jnp.linalg.eigvals(A_matrix @ A_matrix.T)
                            eigenvalues['A'][f'feed_{i}'] = np.array(eigs_A)
                        except:
                            eigenvalues['A'][f'feed_{i}'] = np.array([])

                    if B_matrix is not None and B_matrix.size > 0:
                        try:
                            eigs_B = jnp.linalg.eigvals(B_matrix @ B_matrix.T)
                            eigenvalues['B'][f'feed_{i}'] = np.array(eigs_B)
                        except:
                            eigenvalues['B'][f'feed_{i}'] = np.array([])

            # myNN (GNN): A_gcn, B_gcn, A_feed, B_feed
            if hasattr(model, 'A_gcn') and hasattr(model, 'B_gcn'):
                for i, (A_matrix, B_matrix) in enumerate(zip(model.A_gcn, model.B_gcn)):
                    if A_matrix is not None and A_matrix.size > 0:
                        try:
                            eigs_A = jnp.linalg.eigvals(A_matrix @ A_matrix.T)
                            eigenvalues['A'][f'gcn_{i}'] = np.array(eigs_A)
                        except:
                            eigenvalues['A'][f'gcn_{i}'] = np.array([])

                    if B_matrix is not None and B_matrix.size > 0:
                        try:
                            eigs_B = jnp.linalg.eigvals(B_matrix @ B_matrix.T)
                            eigenvalues['B'][f'gcn_{i}'] = np.array(eigs_B)
                        except:
                            eigenvalues['B'][f'gcn_{i}'] = np.array([])

        except Exception as e:
            # If eigenvalue computation fails, return empty dict
            pass

        return eigenvalues

    def record_metrics(self,
                      iteration: int,
                      step: int,
                      task_id: int,
                      losses: Dict[str, float],
                      gradients: Dict[str, float],
                      metrics: Dict[str, float],
                      model,
                      extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record all metrics for a single iteration in unified format.

        Args:
            iteration: Iteration number (key for the record)
            step: Current step within task
            task_id: Current task ID
            losses: Dict of loss values (H, V, dV, dV_dx, dV_dtheta, dV_dadj, etc.)
            gradients: Dict of gradient norms
            metrics: Dict with train, test_current, test_experience metrics
            model: The model (for eigenvalue extraction)
            extra_metrics: Optional dict for dataset-specific metrics

        Returns:
            dict: Complete record for this iteration
        """
        record = {
            'losses': losses.copy(),
            'gradients': gradients.copy(),
            'metrics': metrics.copy(),
            'eigenvalues': self._compute_eigenvalues(model),
            'step': step,
            'task_id': task_id,
            'global_step': step + task_id * self.n_iter if hasattr(self, 'n_iter') else step,
        }

        # Add any extra dataset-specific metrics
        if extra_metrics is not None:
            record['extra_metrics'] = extra_metrics.copy()

        return record

    def initialize_record_dict(self, config: Dict[str, Any], run_id: int = 0) -> Dict[str, Any]:
        """Initialize the recording dictionary with metadata.

        Args:
            config: Configuration dictionary containing problem/dataset info
            run_id: The run/repetition number for this experiment

        Returns:
            dict: Initialized recording structure with task-based format
        """
        # Added by Claude: New task-based recording structure
        # - Global iterations: track progress across all tasks
        # - Within-task epochs: track progress within each task
        # - AB iterations: separate counter for AB training phases
        # Also maintain 'iterations' dict for backward compatibility with compute_avg_loss
        record_dict = {
            'metadata': {
                'problem': config.get('problem', 'unknown'),
                'prob': config.get('prob', 'unknown'),
                'dataset': config.get('data', 'unknown'),
                'network': config.get('network', 'unknown'),
                'loss_function': config.get('loss', 'unknown'),
                'metric_function': config.get('metric', 'unknown'),
                'n_tasks': config.get('n_task', 1),
                'epochs_per_task': config.get('epochs_per_task', 0),
                'save_iter': config.get('save_iter', 10),
                'learning_rate': config.get('lr', 0),
                'batch_size': config.get('batch_size', config.get('batch', 0)),
                'awb_enabled': config.get('awb_enabled', False),
                'run_id': run_id,
            },
            'tasks': {},
            'architecture_history': {},
            'iterations': {},  # Backward compatibility with existing code
            # Added by Claude: Per-task performance matrix for CL metrics
            # Format: task_performance_matrix[j][i] = performance on task i after training task j
            # This enables computing ACC, BWT, FWT, Forgetting metrics from literature
            'task_performance_matrix': {}
        }
        return record_dict

    def initialize_task(self, record_dict: Dict[str, Any], task_id: int, arch_info: Dict[str, Any]):
        """Initialize a new task's recording structure.

        Args:
            record_dict: The recording dictionary
            task_id: Task ID to initialize
            arch_info: Architecture information (sizes, filter_size, etc.)
        """
        # Added by Claude: Initialize task with main_training structure
        record_dict['tasks'][task_id] = {
            'main_training': {
                'iterations': [],      # Global iteration numbers
                'epochs': [],          # Within-task epoch numbers
                'H': [],
                'V': [],
                'dV': [],
                'dV_dx': [],
                'dV_dtheta': [],
                'grad_norm': [],
                'train_metric': [],
                'test_current': [],
                'test_experience': [],
                'eigenvalues': {'A': {}, 'B': {}}
            },
            'phase_info': {
                'type': 'standard',
                'total_epochs': 0
            },
            'arch_changed': False,
            'architecture': arch_info
        }

    def record_preliminary_summary(self, record_dict: Dict[str, Any], task_id: int,
                                   n_epochs: int, warmup_epochs: int,
                                   final_loss: float, decision: str):
        """Record summary of preliminary phase (not detailed metrics).

        Args:
            record_dict: The recording dictionary
            task_id: Task ID
            n_epochs: Total preliminary epochs
            warmup_epochs: Warmup epochs within preliminary
            final_loss: Final loss after preliminary training
            decision: 'arch_change' or 'no_change'
        """
        # Added by Claude: Record preliminary phase summary
        record_dict['tasks'][task_id]['preliminary'] = {
            'n_epochs': n_epochs,
            'warmup_epochs': warmup_epochs,
            'final_loss': float(final_loss),
            'decision': decision
        }

    def initialize_ab_training(self, record_dict: Dict[str, Any], task_id: int):
        """Initialize AB training recording for a task.

        Args:
            record_dict: The recording dictionary
            task_id: Task ID
        """
        # Added by Claude: Initialize AB training structure with separate iteration counter
        record_dict['tasks'][task_id]['ab_training'] = {
            'iterations': [],      # AB-specific iteration numbers (local to AB phase)
            'H': [],
            'V': [],
            'ab_eigenvalues': {'A': {}, 'B': {}}
        }

    def record_ab_training_epoch(self, record_dict: Dict[str, Any], task_id: int,
                                 iteration: int, losses: Dict[str, float], model):
        """Record a single epoch of AB training with AB eigenvalues.

        Args:
            record_dict: The recording dictionary
            task_id: Task ID
            iteration: AB training iteration number (local to AB phase)
            losses: Loss dictionary
            model: Model for eigenvalue extraction
        """
        # Added by Claude: Auto-initialize task and AB training if not exists
        if task_id not in record_dict.get('tasks', {}):
            arch_info = {'sizes': 'unknown'}
            self.initialize_task(record_dict, task_id, arch_info)
        if 'ab_training' not in record_dict['tasks'][task_id]:
            self.initialize_ab_training(record_dict, task_id)

        # Added by Claude: Record AB training epoch
        ab = record_dict['tasks'][task_id]['ab_training']
        ab['iterations'].append(iteration)
        ab['H'].append(float(losses.get('H', 0.0)))
        ab['V'].append(float(losses.get('V', 0.0)))

        # Record A/B eigenvalues during AB training
        eigs = self._compute_eigenvalues(model)
        for matrix_type in ['A', 'B']:
            for layer_name, eig_values in eigs[matrix_type].items():
                if layer_name not in ab['ab_eigenvalues'][matrix_type]:
                    ab['ab_eigenvalues'][matrix_type][layer_name] = []
                ab['ab_eigenvalues'][matrix_type][layer_name].append(eig_values)

    def record_main_training_epoch(self, record_dict: Dict[str, Any], task_id: int,
                                   global_iteration: int, epoch: int,
                                   losses: Dict[str, float], gradients: Dict[str, float],
                                   metrics: Dict[str, float], model):
        """Record a single epoch of main training (Step 5 or standard).

        Args:
            record_dict: The recording dictionary
            task_id: Task ID
            global_iteration: Global iteration number (across all tasks)
            epoch: Within-task epoch number
            losses: Loss dictionary
            gradients: Gradient dictionary
            metrics: Metrics dictionary
            model: Model for eigenvalue extraction
        """
        # Added by Claude: Auto-initialize task if not exists (for backward compatibility)
        if task_id not in record_dict.get('tasks', {}):
            # Get minimal architecture info from model
            arch_info = {'sizes': 'unknown'}  # Placeholder
            self.initialize_task(record_dict, task_id, arch_info)

        # Added by Claude: Record main training epoch with both global and local tracking
        task = record_dict['tasks'][task_id]['main_training']
        task['iterations'].append(global_iteration)
        task['epochs'].append(epoch)
        task['H'].append(float(losses.get('H', 0.0)))
        task['V'].append(float(losses.get('V', 0.0)))
        task['dV'].append(float(losses.get('dV', 0.0)))
        task['dV_dx'].append(float(losses.get('dV_dx', 0.0)))
        task['dV_dtheta'].append(float(losses.get('dV_dtheta', 0.0)))
        task['grad_norm'].append(float(gradients.get('grad_norm', 0.0)))
        task['train_metric'].append(float(metrics.get('train', 0.0)))
        task['test_current'].append(float(metrics.get('test_current', 0.0)))
        task['test_experience'].append(float(metrics.get('test_experience', 0.0)))

        # Record eigenvalues
        eigs = self._compute_eigenvalues(model)
        for matrix_type in ['A', 'B']:
            for layer_name, eig_values in eigs[matrix_type].items():
                if layer_name not in task['eigenvalues'][matrix_type]:
                    task['eigenvalues'][matrix_type][layer_name] = []
                task['eigenvalues'][matrix_type][layer_name].append(eig_values)

    def record_task_performance(self, record_dict: Dict[str, Any], current_task_id: int,
                                task_performances: Dict[int, float]):
        """Record performance on all tasks after training current_task_id.

        This builds the performance matrix A_{i,j} needed for CL metrics:
        - ACC (Average Accuracy)
        - BWT (Backward Transfer)
        - F (Average Forgetting)
        - FWT (Forward Transfer)

        Args:
            record_dict: The recording dictionary
            current_task_id: The task that was just trained (j in A_{i,j})
            task_performances: Dict mapping task_id -> performance metric
                               e.g., {0: 0.95, 1: 0.92, 2: 0.89}
                               Performance on tasks 0,1,2 after training task 2

        Example:
            After training task 2:
            task_performances = {
                0: 0.93,  # accuracy on task 0 after training task 2
                1: 0.91,  # accuracy on task 1 after training task 2
                2: 0.95   # accuracy on task 2 after training task 2
            }

        Matrix format enables computing:
            ACC = (1/T) Σ_{i=0}^{T-1} A[T-1][i]
            BWT = (1/(T-1)) Σ_{i=0}^{T-2} (A[T-1][i] - A[i][i])
            F = (1/(T-1)) Σ_{i=0}^{T-2} max_{j>=i} (A[i][i] - A[j][i])
        """
        # Added by Claude: Record per-task performance matrix
        if 'task_performance_matrix' not in record_dict:
            record_dict['task_performance_matrix'] = {}

        # Store performance on all tasks after training current_task_id
        record_dict['task_performance_matrix'][current_task_id] = {
            str(task_id): float(performance)
            for task_id, performance in task_performances.items()
        }

    def save_record_dict(self, record_dict: Dict[str, Any], base_path: str):
        """Save the recording dictionary to file using problem/dataset name.

        Args:
            record_dict: The complete recording dictionary
            base_path: Base path for saving (typically config['model_path'])
        """
        metadata = record_dict['metadata']

        # Create filename: problem_dataset_network[_awb]_run{run_id}_records.pkl
        run_id = metadata.get('run_id', 0)
        awb_suffix = "_awb" if metadata.get('awb_enabled', False) else ""
        filename = f"{metadata['prob']}_{metadata['dataset']}_{metadata['network']}{awb_suffix}_run{run_id}_records.pkl"

        # Use the base_path directory
        if base_path:
            save_dir = os.path.dirname(base_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename

        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(record_dict, f)

        print(f"Records saved to: {filepath}")
        return filepath

    @staticmethod
    def save_all_runs(all_runs_records: Dict[str, Any], base_path: str, config: Dict[str, Any]):
        """Save all experiment runs to a single file.

        Args:
            all_runs_records: Dictionary with run IDs as keys and record_dicts as values
            base_path: Base path for saving
            config: Configuration dictionary for metadata
        """
        # Create a consolidated structure with runs at the top level
        consolidated = {
            'runs': all_runs_records,
            'metadata': {
                'total_runs': len(all_runs_records),
                'problem': config.get('problem', 'unknown'),
                'prob': config.get('prob', 'unknown'),
                'dataset': config.get('data', 'unknown'),
                'network': config.get('network', 'unknown'),
                'awb_enabled': config.get('awb_enabled', False),
            }
        }

        # Create filename with AWB suffix if enabled
        awb_suffix = "_awb" if config.get('awb_enabled', False) else ""
        filename = f"{config.get('prob', 'unknown')}_{config.get('data', 'unknown')}_{config.get('network', 'unknown')}{awb_suffix}_allruns.pkl"

        # Use the base_path directory
        if base_path:
            save_dir = os.path.dirname(base_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename

        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(consolidated, f)

        print(f"All runs records saved to: {filepath}")
        return filepath
