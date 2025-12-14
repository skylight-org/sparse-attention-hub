"""Efficient attention research backend implementation.

This module provides a hook into research attention and acts as a reference
for other backend implementations.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..utils.mask import Mask
from .base import EfficientAttention, EfficientAttentionConfig
from ..research_attention.base import ResearchAttention, ResearchAttentionConfig
from .backends.research_backend.base import SparseResearchBackend


@dataclass
class EfficientAttentionResearchBackendConfig(EfficientAttentionConfig):
    """Configuration class for efficient attention research backend."""

    pass


class EfficientAttentionResearchBackend(EfficientAttention, SparseResearchBackend):
    """Efficient attention implementation using research backend.

    This class extends both EfficientAttention and SparseResearchBackend to provide
    a hook into research attention. It acts as a reference implementation for other
    backend implementations.

    The class implements custom_attention by:
    1. Calling indexer_first or indexer_next based on call count
    2. Computing attention using the research backend
    """

    def __init__(self, sparse_attention_config: EfficientAttentionResearchBackendConfig) -> None:
        """Initialize efficient attention research backend.

        Args:
            sparse_attention_config: Configuration for the efficient attention mechanism.
        """
        EfficientAttention.__init__(self, sparse_attention_config)
        SparseResearchBackend.__init__(self)
        self.research_attention = ResearchAttention.create_from_config(sparse_attention_config.research_attention_config)

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute efficient attention mechanism using research backend.

        Args:
            module: The attention module.
            queries: Query tensor of shape (b, h, sq, d).
            keys: Key tensor of shape (b, h_kv, sk, d).
            values: Value tensor of shape (b, h_kv, sk, d).
            attention_mask: Optional attention mask of shape (b, h, sq, sk).
            scaling: Scaling factor for attention weights.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments, must include 'layer_idx'.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Get layer_idx from kwargs
        layer_idx: int = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        # Check if this is the first call for this layer
        call_count: Dict[int, int] = sparse_meta_data.get("call_count", {})
        is_first_call: bool = layer_idx not in call_count

        # Call appropriate indexer method
        sparse_attention_mask: Mask
        if is_first_call:
            sparse_attention_mask = self.indexer_first(
                query=queries,
                key=keys,
                value=values,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )
        else:
            sparse_attention_mask = self.indexer_next(
                query=queries,
                key=keys,
                value=values,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        # Call attention computation backend
        result: torch.Tensor = self.attention_computation_backend(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=False,
            **kwargs,
        )

        return (result, None)

    def indexer_first(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        module: nn.Module,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Initialize indexer for the first iteration.

        This method is called once at the beginning to set up the indexer state.
        It calls indexer_research to construct the sparse attention mask.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Sparse attention mask object.
        """
        return self.indexer_research(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )

    def indexer_next(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        module: nn.Module,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Update indexer for subsequent iterations.

        This method is called for each subsequent iteration after indexer_first.
        It calls indexer_research to construct the sparse attention mask.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Sparse attention mask object.
        """
        return self.indexer_research(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )


    def indexer_research(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        module: nn.Module,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Indexer for research attention.

        This method constructs a sparse attention mask by applying all maskers
        from the research attention configuration, following the same pattern
        as ResearchAttention.custom_attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Sparse attention mask object.
        """
        # Create an empty Mask object
        mask_shape: Tuple[int, int, int, int] = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            key.shape[2],
        )
        sparse_attention_mask: Mask = Mask.create_empty_mask(
            mask_shape, dtype=query.dtype, device=query.device
        )

        # Apply all maskers sequentially, each one on the output of the previous one
        for masker in self.research_attention.maskers:
            sparse_attention_mask = masker.add_mask(
                keys=key,
                queries=query,
                values=value,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_attention_mask,
                **kwargs,
            )

        return sparse_attention_mask

    @classmethod
    @abstractmethod
    def create_sample_data_first(
        cls, B: int, H: int, num_keys: int, d: int
    ) -> Tuple[
        ResearchAttentionConfig,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        nn.Module,
        Optional[torch.Tensor],
        float,
        float,
        Dict[Any, Any],
    ]:
        """Create sample data for the first iteration.

        Args:
            B: Batch size.
            H: Number of attention heads.
            num_keys: Number of key tokens (sequence length).
            d: Dimension per attention head.

        Returns:
            Tuple containing (in order of indexer_next parameters):
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig and LocalMaskerConfig.
                - query: Query tensor of shape (B, H, num_queries, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Dictionary containing sparse attention metadata.
        """
        pass

    @classmethod
    @abstractmethod
    def create_sample_data_next(
        cls, B: int, H: int, num_keys: int, d: int
    ) -> Tuple[
        ResearchAttentionConfig,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        nn.Module,
        Optional[torch.Tensor],
        float,
        float,
        Dict[Any, Any],
    ]:
        """Create sample data for subsequent iterations.

        Args:
            B: Batch size.
            H: Number of attention heads.
            num_keys: Number of key tokens (sequence length).
            d: Dimension per attention head.

        Returns:
            Tuple containing (in order of indexer_next parameters):
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig and LocalMaskerConfig.
                - query: Query tensor of shape (B, H, num_queries, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Dictionary containing sparse attention metadata.
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: EfficientAttentionResearchBackendConfig) -> "EfficientAttentionResearchBackend":
        """Create efficient attention research backend instance from configuration.

        Args:
            config: Configuration for the efficient attention research backend.

        Returns:
            Instance of the efficient attention research backend.

        Raises:
            TypeError: If config is not an EfficientAttentionResearchBackendConfig.
        """
        pass