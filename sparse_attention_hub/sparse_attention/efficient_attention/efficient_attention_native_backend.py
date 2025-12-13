"""Efficient attention native backend implementation.

This module provides a hook into native backend attention computation.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .base import EfficientAttention, EfficientAttentionConfig
from .backends.native_backend.base import SparseNativeBackend


@dataclass
class EfficientAttentionNativeBackendConfig(EfficientAttentionConfig):
    """Configuration class for efficient attention native backend."""

    pass


class EfficientAttentionNativeBackend(EfficientAttention, SparseNativeBackend):
    """Efficient attention implementation using native backend.

    This class extends both EfficientAttention and SparseNativeBackend to provide
    a hook into native backend attention computation.

    The class implements custom_attention by:
    1. Calling indexer_first or indexer_next based on call count
    2. Computing attention using the native backend
    3. Applying post_attention_transform
    """

    def __init__(self, sparse_attention_config: EfficientAttentionNativeBackendConfig) -> None:
        """Initialize efficient attention native backend.

        Args:
            sparse_attention_config: Configuration for the efficient attention mechanism.
        """
        EfficientAttention.__init__(self, sparse_attention_config)
        SparseNativeBackend.__init__(self)

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
        """Compute efficient attention mechanism using native backend.

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
        if is_first_call:
            sparse_list, sparse_len, weight_list = self.indexer_first(
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
            sparse_list, sparse_len, weight_list = self.indexer_next(
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
            sparse_list=sparse_list,
            sparse_len=sparse_len,
            weight_list=weight_list,
            return_attention_weights=False,
            **kwargs,
        )

        # Apply post_attention_transform
        result = self.post_attention_transform(result)

        return (result, None)

    @abstractmethod
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize indexer for the first iteration.

        This method is called once at the beginning to set up the indexer state.
        Subclasses should implement this method to return sparse_list, sparse_len, and weight_list.

        Args:
            query: Query tensor of shape (b, h, sq, d) or (b, h, d).
            key: Key tensor of shape (b, h_kv, sk, d).
            value: Value tensor of shape (b, h_kv, sk, d).
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (sparse_list, sparse_len, weight_list) where:
            - sparse_list: Tensor of shape (b, h, sk) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        pass

    @abstractmethod
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update indexer for subsequent iterations.

        This method is called for each subsequent iteration after indexer_first.
        Subclasses should implement this method to return sparse_list, sparse_len, and weight_list.

        Args:
            query: Query tensor of shape (b, h, sq, d) or (b, h, d).
            key: Key tensor of shape (b, h_kv, sk, d).
            value: Value tensor of shape (b, h_kv, sk, d).
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (sparse_list, sparse_len, weight_list) where:
            - sparse_list: Tensor of shape (b, h, sk) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: EfficientAttentionNativeBackendConfig) -> "EfficientAttentionNativeBackend":
        """Create efficient attention native backend instance from configuration.

        Args:
            config: Configuration for the efficient attention native backend.

        Returns:
            Instance of the efficient attention native backend.

        Raises:
            TypeError: If config is not an EfficientAttentionNativeBackendConfig.
        """
        pass

