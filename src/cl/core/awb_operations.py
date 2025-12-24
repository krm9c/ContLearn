"""
Abstract interface for model-specific AWB operations.

This module defines the contract that each model type must implement to support
the AWB (Adaptive Weight Basis) pipeline. The strategy pattern separates:
- WHEN to execute AWB steps (awb_pipeline.py)
- HOW to execute model-specific operations (this interface + implementations)

Models supporting AWB must implement:
- Architecture search logic
- A/B matrix initialization
- Model partitioning for different training phases
- V = A @ W @ B^T computation

Current implementations:
- MLPAWBOps (src/cl/models/mlp.py)
- CNNAWBOps (src/cl/models/cnn.py)
- GCNAWBOps (src/cl/models/gcn.py)

Future implementations:
- TransformerAWBOps (for attention-based models)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import equinox as eqx


class AWBOperations(ABC):
    """Abstract base class for model-specific AWB operations.

    Each model type (MLP, CNN, GCN, Transformer, etc.) must implement this
    interface to support the AWB pipeline. The pipeline orchestrator calls
    these methods at the appropriate times during AWB training.

    The interface separates control flow (the 5-step AWB algorithm) from
    model-specific tensor operations (architecture search, A/B matrix setup, etc.).
    """

    @abstractmethod
    def search_architecture(
        self,
        model: eqx.Module,
        task_id: int,
        baseline_loss: float,
        dataloader_curr,
        dataloader_exp,
        test_loader_curr,
        test_loader_exp,
        config: Dict[str, Any],
        trainer=None
    ) -> Any:
        """Search for optimal architecture for current task.

        This is STEP 3a of the AWB pipeline. The search creates fresh candidate
        models with random initialization, trains each for a fixed number of epochs,
        and returns the architecture specification that achieves lowest loss.

        Args:
            model: Current model (used to extract baseline architecture)
            task_id: Current task ID
            baseline_loss: Loss from preliminary training (baseline for comparison)
            dataloader_curr: Current task training data
            dataloader_exp: Experience replay data
            test_loader_curr: Current task test data
            test_loader_exp: Experience replay test data
            config: Configuration dictionary
            trainer: Optional Trainer instance (created if None)

        Returns:
            Architecture specification (model-specific format)
            - MLP: [hidden_dim1, hidden_dim2, ...]
            - CNN: {'feed_sizes': [...], 'filter_size': int}
            - GCN: [hidden_dim1, hidden_dim2, ...]
            - Transformer: {'n_heads': int, 'embed_dim': int, 'n_layers': int}
        """
        pass

    @abstractmethod
    def set_AB_matrices(
        self,
        model: eqx.Module,
        original_arch: Any,
        new_arch: Any
    ) -> eqx.Module:
        """Initialize A/B matrices for architecture transition.

        This is the setup for STEP 3b of the AWB pipeline. Creates transformation
        matrices A and B that will map old architecture to new architecture:
        - A transforms from old output dimensions to new output dimensions
        - B transforms from old input dimensions to new input dimensions

        The old weights W are preserved in the model. During AB training:
        - get_AWBT computes: A @ W @ B^T for forward pass
        - A and B are trainable, W is frozen

        Args:
            model: Model with old architecture and weights W
            original_arch: Original architecture specification
            new_arch: New architecture specification (from search_architecture)

        Returns:
            Model with A/B matrices initialized (W unchanged)
        """
        pass

    @abstractmethod
    def partition_for_AB_training(
        self,
        model: eqx.Module
    ) -> Tuple[eqx.Module, eqx.Module]:
        """Partition model for AB training phase.

        This is used in STEP 3b of the AWB pipeline. Separates the model into:
        - Trainable parameters: A and B matrices only
        - Static (frozen) parameters: W (old weights) and everything else

        During AB training (notABTrain=False), only A/B are updated via gradient
        descent. The old weights W remain frozen.

        Args:
            model: Model with A, B, and W matrices

        Returns:
            Tuple of (trainable_params, static_params)
            - trainable_params: Contains only A and B matrices
            - static_params: Contains W and all other model parameters
        """
        pass

    @abstractmethod
    def compute_V(
        self,
        model: eqx.Module
    ) -> eqx.Module:
        """Compute transformed weights V = A @ W @ B^T.

        This is STEP 4 of the AWB pipeline. After AB training completes, we
        compute the effective weights V by applying the trained transformation
        matrices to the old weights:

        V = A @ W @ B^T

        The model is then updated to use V as the new weights. For biases:
        V_bias = A @ bias

        Args:
            model: Model with trained A, B matrices and old W weights

        Returns:
            Model with weights updated to V (A/B matrices still present but
            will be frozen in subsequent training)
        """
        pass

    @abstractmethod
    def partition_for_standard_training(
        self,
        model: eqx.Module
    ) -> Tuple[eqx.Module, eqx.Module]:
        """Partition model for standard training (STEP 5 and beyond).

        This is used in STEP 5 of the AWB pipeline and all subsequent training.
        Separates the model into:
        - Trainable parameters: V (transformed weights) and other parameters
        - Static (frozen) parameters: A and B matrices

        During standard training after AWB, only V is updated. The A/B matrices
        remain frozen, preserving the architecture transformation.

        Args:
            model: Model with V, A, and B matrices

        Returns:
            Tuple of (trainable_params, static_params)
            - trainable_params: Contains V and trainable parameters (excluding A/B)
            - static_params: Contains A and B matrices (frozen)
        """
        pass

    @abstractmethod
    def get_model_architecture(
        self,
        model: eqx.Module
    ) -> Any:
        """Extract architecture specification from model.

        Returns the current architecture in the same format expected by
        search_architecture() and set_AB_matrices().

        Args:
            model: Model instance

        Returns:
            Architecture specification (model-specific format)
        """
        pass

    @abstractmethod
    def save_weights(
        self,
        model: eqx.Module
    ) -> Any:
        """Save current model weights before architecture search.

        Architecture search creates fresh models with random initialization.
        We need to save the current weights to restore them after search completes.

        Args:
            model: Model instance

        Returns:
            Saved weights (model-specific format)
        """
        pass

    @abstractmethod
    def restore_weights(
        self,
        model: eqx.Module,
        saved_weights: Any
    ) -> eqx.Module:
        """Restore model weights after architecture search.

        After architecture search, we restore the original weights before
        initializing A/B matrices. This ensures we're transforming the trained
        weights, not random weights.

        Args:
            model: Model instance (possibly with different architecture)
            saved_weights: Weights saved by save_weights()

        Returns:
            Model with restored weights
        """
        pass


__all__ = ['AWBOperations']
