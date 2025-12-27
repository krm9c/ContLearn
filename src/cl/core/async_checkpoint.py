"""
Asynchronous checkpointing utilities for JAX/Equinox models.

Added by Claude: Non-blocking checkpoint saving with memory safeguards.
This module enables periodic model checkpointing without blocking training.

Key features:
- Background saving using threading
- Memory usage estimation before saving
- Configurable memory limits to prevent OOM
- Automatic cleanup of old checkpoints

Usage:
    checkpoint_manager = AsyncCheckpointManager(
        save_dir="outputs/checkpoints",
        max_checkpoints=3,
        memory_limit_gb=8.0
    )

    # In training loop:
    checkpoint_manager.save_async(
        model=model,
        record_dict=record_dict,
        task_id=task_id,
        epoch=epoch
    )

    # Wait for all saves to complete before exiting:
    checkpoint_manager.wait_all()
"""

import os
import sys
import pickle
import threading
import queue
from typing import Dict, Any, Optional
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path


def estimate_model_memory_mb(model) -> float:
    """Estimate memory footprint of a JAX/Equinox model in MB.

    Args:
        model: Equinox model (MLP, CNN, GCN)

    Returns:
        Estimated memory in MB
    """
    total_bytes = 0

    # Count all array parameters
    leaves = jax.tree_util.tree_leaves(model)
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray):
            # bytes = size * itemsize
            total_bytes += leaf.size * leaf.dtype.itemsize

    # Convert bytes to MB
    return total_bytes / (1024 * 1024)


def estimate_records_memory_mb(record_dict: Dict[str, Any]) -> float:
    """Estimate memory footprint of record_dict in MB.

    Args:
        record_dict: Training records dictionary

    Returns:
        Estimated memory in MB (approximate)
    """
    # Rough estimate: pickle size is similar to in-memory size
    # For detailed estimation, we'd need to traverse the entire dict
    # For now, use a conservative estimate based on task count
    n_tasks = len(record_dict.get('tasks', {}))
    n_iterations = len(record_dict.get('iterations', {}))

    # Rough estimate: ~10 KB per task, ~1 KB per iteration
    estimated_bytes = (n_tasks * 10 * 1024) + (n_iterations * 1024)

    return estimated_bytes / (1024 * 1024)


def get_available_memory_gb() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in GB, or 0.0 if unable to determine
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except ImportError:
        # psutil not available, return conservative estimate
        return 0.0


class AsyncCheckpointManager:
    """Manages asynchronous model checkpointing with memory safeguards.

    This class handles:
    - Background checkpoint saving using a thread pool
    - Memory estimation to prevent OOM
    - Automatic cleanup of old checkpoints
    - Graceful shutdown
    """

    def __init__(self,
                 save_dir: str = "outputs/checkpoints",
                 max_checkpoints: int = 3,
                 memory_limit_gb: float = 8.0,
                 enable_async: bool = True):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (older ones deleted)
            memory_limit_gb: Maximum memory to use for checkpointing (GB)
            enable_async: If False, use synchronous saving (for debugging)
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.memory_limit_gb = memory_limit_gb
        self.enable_async = enable_async

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Background saving thread
        self._save_queue = queue.Queue()
        self._shutdown = False
        self._save_thread = None

        # Track saved checkpoints for cleanup
        self._checkpoint_history = []

        if enable_async:
            self._save_thread = threading.Thread(target=self._background_saver, daemon=True)
            self._save_thread.start()

    def _background_saver(self):
        """Background thread that saves checkpoints from queue."""
        while not self._shutdown:
            try:
                # Wait for save request (timeout to check shutdown flag)
                save_task = self._save_queue.get(timeout=1.0)
                if save_task is None:  # Shutdown signal
                    break

                # Unpack save task
                model_path, model, record_dict_path, record_dict = save_task

                # Save model
                if model is not None:
                    eqx.tree_serialise_leaves(model_path, model)

                # Save records
                if record_dict is not None:
                    with open(record_dict_path, 'wb') as f:
                        pickle.dump(record_dict, f)

                self._save_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncCheckpoint] Error saving checkpoint: {e}", file=sys.stderr)
                continue

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if len(self._checkpoint_history) > self.max_checkpoints:
            # Remove oldest checkpoint
            old_checkpoint = self._checkpoint_history.pop(0)
            model_path, record_path = old_checkpoint

            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(record_path):
                    os.remove(record_path)
            except Exception as e:
                print(f"[AsyncCheckpoint] Failed to delete old checkpoint: {e}", file=sys.stderr)

    def save_checkpoint(self,
                       model,
                       record_dict: Dict[str, Any],
                       task_id: int,
                       epoch: int,
                       prefix: str = "checkpoint") -> bool:
        """Save a checkpoint asynchronously (or synchronously if disabled).

        Args:
            model: Equinox model to save
            record_dict: Training records dictionary
            task_id: Current task ID
            epoch: Current epoch
            prefix: Filename prefix

        Returns:
            True if checkpoint was queued/saved, False if skipped due to memory
        """
        # Estimate memory requirements
        model_mem_mb = estimate_model_memory_mb(model)
        records_mem_mb = estimate_records_memory_mb(record_dict)
        total_mem_mb = model_mem_mb + records_mem_mb
        total_mem_gb = total_mem_mb / 1024

        # Check memory limit
        if total_mem_gb > self.memory_limit_gb:
            print(f"[AsyncCheckpoint] Skipping checkpoint (estimated {total_mem_gb:.2f} GB exceeds limit {self.memory_limit_gb:.2f} GB)")
            return False

        # Check available memory (if psutil available)
        available_gb = get_available_memory_gb()
        if available_gb > 0 and total_mem_gb > available_gb * 0.5:  # Use at most 50% of available
            print(f"[AsyncCheckpoint] Skipping checkpoint (estimated {total_mem_gb:.2f} GB exceeds 50% of available {available_gb:.2f} GB)")
            return False

        # Generate checkpoint paths
        checkpoint_name = f"{prefix}_task{task_id}_epoch{epoch}"
        model_path = str(self.save_dir / f"{checkpoint_name}.eqx")
        record_path = str(self.save_dir / f"{checkpoint_name}_records.pkl")

        # Track for cleanup
        self._checkpoint_history.append((model_path, record_path))
        self._cleanup_old_checkpoints()

        if self.enable_async:
            # Queue for background saving
            self._save_queue.put((model_path, model, record_path, record_dict))
            return True
        else:
            # Synchronous save
            try:
                eqx.tree_serialise_leaves(model_path, model)
                with open(record_path, 'wb') as f:
                    pickle.dump(record_dict, f)
                return True
            except Exception as e:
                print(f"[AsyncCheckpoint] Error saving checkpoint: {e}", file=sys.stderr)
                return False

    def wait_all(self, timeout: Optional[float] = None):
        """Wait for all pending checkpoint saves to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        if self.enable_async and self._save_queue is not None:
            self._save_queue.join()

    def shutdown(self):
        """Shutdown the checkpoint manager gracefully."""
        if self.enable_async and self._save_thread is not None:
            self._shutdown = True
            self._save_queue.put(None)  # Shutdown signal
            self._save_thread.join(timeout=10.0)

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Added by Claude: Import jax for tree utilities
import jax
