"""Evolve masker implementation.

This module provides the Evolve masker that implements evolutionary attention patterns.
The current implementation is a bare metal version that returns the previous mask.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
from scipy.stats import norm

from sparse_attention_hub.sparse_attention.research_attention.maskers import (
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    _get_num_key_value_groups,
    apply_inv_mask_sum,
    create_sampling_mask_with_per_head_budget,
    repeat_kv,
)


@dataclass
class EvolveMaskerConfig(MaskerConfig):
    """Configuration for EvolveMasker.

    This configuration class inherits from MaskerConfig and provides
    parameters for the attention mechanism evolved out of evolve.
    empty placeholder
    """

    pass


@MaskerRegistry.register(EvolveMaskerConfig)
class EvolveMasker(ResearchMasker):
    """Evolve masker for evolutionary attention computation.

    This masker implements evolutionary attention patterns that adapt over time.
    The current implementation is a bare metal version that returns the previous mask.

    Attributes:
        evolution_rate: The rate of evolution for attention patterns.
            This value is set from the configuration and controls how quickly
            the attention patterns evolve.

    Important Notes:
        - This is a bare metal implementation that simply returns the previous mask.
        - Future implementations will include evolutionary algorithms for attention pattern optimization.
        - The evolution_rate parameter is currently unused but will be utilized in future versions.

    Example:
        >>> config = EvolveMaskerConfig(evolution_rate=1.0)
        >>> masker = EvolveMasker(config)
        >>> # Use masker.add_mask() to apply evolutionary attention patterns
    """

    def __init__(self, config: EvolveMaskerConfig) -> None:
        """Initialize Evolve masker with configuration.

        Args:
            config: Configuration object containing the evolution rate and other
                parameters for the Evolve masker.

        Raises:
            ValueError: If the evolution_rate in config is negative.
                This validation is performed in the config's __post_init__ method.
        """
        self.base_rate_sampling = 0.1
        super().__init__(config)

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add random sampling mask to attention computation.

        This method implements random sampling of indices and adds them to the
        previous mask with probability 1.0.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).
            values: Value tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            attention_mask: Attention mask tensor indicating which positions are valid.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            previous_mask: Previous attention mask to merge with the new random sampling mask.
            **kwargs: Additional keyword arguments.

        Returns:
            A new Mask object representing the attention pattern after applying
            random sampling with probability 1.0.
        """
        # EVOLVE-BLOCK-START

        # Check if previous_mask is full mask, if so return full mask
        if previous_mask.is_full_mask():
            return previous_mask

        # Extract tensor dimensions
        batch_size: int = queries.shape[0]
        num_heads: int = queries.shape[1]
        seq_len_queries: int = queries.shape[2]
        seq_len_keys: int = keys.shape[2]

        # Calculate number of indices to sample per row based on base_rate_sampling
        num_indices_to_sample: int = int(self.base_rate_sampling * seq_len_keys)
        
        # Ensure at least 1 index is sampled if base_rate_sampling > 0
        if num_indices_to_sample == 0 and self.base_rate_sampling > 0:
            num_indices_to_sample = 1

        # Generate random indices for each row
        row_wise_idx: torch.Tensor = torch.randint(
            low=0,
            high=seq_len_keys,
            size=(batch_size, num_heads, seq_len_queries, num_indices_to_sample),
            device=keys.device,
            dtype=torch.long,
        )

        # Create data tensor with probability 1.0
        data: torch.Tensor = torch.full_like(
            row_wise_idx,
            self.base_rate_sampling,
            dtype=previous_mask.dtype,
            device=keys.device,
        )

        # Create mask shape
        mask_shape: tuple[int, int, int, int] = (
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
        )

        # Create mask from row-wise indices
        this_mask: Mask = Mask.create_from_row_wise_idx(
            shape=mask_shape,
            row_wise_idx=row_wise_idx,
            data=data,
            mask_type="index",
            dtype=previous_mask.dtype,
        )

        # EVOLVE-BLOCK-END
        # Merge this_mask with previous mask and return the new mask
        return previous_mask.merge_mask(this_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "EvolveMasker":
        """Create Evolve masker instance from configuration.

        Args:
            config: Configuration for the Evolve masker.

        Returns:
            Instance of the Evolve masker.

        Raises:
            ValueError: If the config type is invalid.
        """
        # not checking for config type here since we will be replacing this masker class
        # with the new masker class in the evaluator.py file
        return cls(config)
