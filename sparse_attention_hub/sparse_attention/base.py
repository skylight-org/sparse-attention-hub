"""Base classes for sparse attention mechanisms (bare metal)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class SparseAttentionConfig:
    """Configuration class for sparse attention mechanisms."""

    pass


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    sparse_attention_config: SparseAttentionConfig

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize sparse attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        self.sparse_attention_config = sparse_attention_config

    @abstractmethod
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
        """Compute custom attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    def pre_attention_hook_generator(self) -> None:
        """Generate pre-attention hooks for model integration."""
        pass

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "SparseAttention":
        """Create sparse attention instance from configuration.

        Args:
            config: Configuration for the sparse attention mechanism.

        Returns:
            Instance of the sparse attention mechanism.
        """
        # Import here to avoid circular imports
        from .efficient_attention import EfficientAttention, EfficientAttentionConfig
        from .research_attention import ResearchAttention, ResearchAttentionConfig

        # Check config type and route to appropriate create_from_config method
        if isinstance(config, ResearchAttentionConfig):
            return ResearchAttention.create_from_config(config)
        elif isinstance(config, EfficientAttentionConfig):
            return EfficientAttention.create_from_config(config)
        else:
            # Fallback to default behavior for base config
            return cls(config)
