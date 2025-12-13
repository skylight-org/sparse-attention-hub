"""Base classes for efficient attention mechanisms."""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ..base import SparseAttention, SparseAttentionConfig
from ..research_attention.base import ResearchAttentionConfig
from ..research_attention.maskers.base import ResearchMasker


# Type alias for masker class
MaskerClass = Type[ResearchMasker]


@dataclass
class EfficientAttentionConfig(SparseAttentionConfig):
    """Configuration class for efficient attention mechanisms."""

    research_attention_config: ResearchAttentionConfig


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize efficient attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        super().__init__(sparse_attention_config)

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
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    @abstractmethod
    def indexer_first(self, *args: Any, **kwargs: Any) -> None:
        """Initialize indexer for the first iteration.

        This method is called once at the beginning to set up the indexer state.
        Subclasses should implement this method to initialize any necessary state
        for indexing operations.
        """
        pass

    @abstractmethod
    def indexer_next(self, *args: Any, **kwargs: Any) -> None:
        """Update indexer for subsequent iterations.

        This method is called for each subsequent iteration after indexer_first.
        Subclasses should implement this method to update the indexer state.
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: EfficientAttentionConfig) -> "EfficientAttention":
        """Create efficient attention instance from configuration.

        Args:
            config: Configuration for the efficient attention mechanism.

        Returns:
            Instance of the efficient attention mechanism.

        Raises:
            TypeError: If config is not an EfficientAttentionConfig.
        """
        pass


class SparseAlgorithm(ABC):
    """Abstract base class for sparse attention algorithms.

    Subclasses should define the masker_classes field to specify which
    masker classes are compatible with this algorithm.
    """

    masker_classes: ClassVar[List[MaskerClass]]