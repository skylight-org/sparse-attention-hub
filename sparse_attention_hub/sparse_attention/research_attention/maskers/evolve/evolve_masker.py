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
        
        # Extract dimensions from input tensors
        batch_size: int = keys.shape[0]
        num_heads: int = queries.shape[1]
        seq_len_keys: int = keys.shape[2]
        seq_len_queries: int = queries.shape[2]
        
        # Target density is 10% (0.1), allocate:
        # - Sink tokens: 0.1% (initial tokens for context)
        # - Local window: 0.2% (recent tokens for local context)
        # - Top-k selection: 9.7% (based on query-key similarity with position bias)
        sink_size: int = max(1, int(0.045 * seq_len_keys))
        local_size: int = max(1, int(0.045 * seq_len_keys))
        
        # Get previous mask to use for guidance
        previous_dense_mask: torch.Tensor = previous_mask.get_dense_mask()
        
        # Handle GQA: repeat keys if needed to match query heads
        num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
        keys_repeated: torch.Tensor = repeat_kv(keys, num_key_value_groups)
        
        # Create dense mask starting with sink and local regions
        dense_mask: torch.Tensor = torch.zeros(batch_size, num_heads, seq_len_queries, seq_len_keys, dtype=previous_mask.dtype, device=keys.device)
        dense_mask[:, :, :, :sink_size] = 1.0
        if local_size > 0:
            dense_mask[:, :, :, -local_size:] = 1.0
            
        this_mask: Mask = Mask.create_mask_from_dense_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            mask=dense_mask,
            dtype=previous_mask.dtype
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
