"""Base fixed pattern masker implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from ....utils.mask import Mask
from ..base import MaskerConfig, ResearchMasker


@dataclass
class FixedMaskerConfig(MaskerConfig):
    """Base configuration for fixed pattern maskers."""

    pass


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    def __init__(self, config: FixedMaskerConfig) -> None:
        """Initialize fixed masker with configuration."""
        super().__init__(config)

    @abstractmethod
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
        """Add fixed mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "FixedMasker":
        """Create fixed masker instance from configuration."""
        if not isinstance(config, FixedMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@dataclass
class TopKMaskerConfig(FixedMaskerConfig):
    """Base configuration for top-K maskers."""

    heavy_size: Union[float, int]

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for TopKMaskerConfig.

        Raises:
            ValueError: If heavy_size is not greater than 0.
        """
        if not self.heavy_size > 0:
            raise ValueError(f"heavy_size must be > 0, got {self.heavy_size}")


class TopKMasker(FixedMasker):
    """Abstract base class for top-K maskers."""

    def __init__(self, config: TopKMaskerConfig) -> None:
        """Initialize top-K masker with configuration."""
        super().__init__(config)

    def _get_topk_indices_from_inactive_positions(
        self, scores: torch.Tensor, previous_mask: Mask, k: int
    ) -> torch.Tensor:
        """Get top-K indices from positions not already active in previous mask."""
        previous_dense_mask: torch.Tensor = previous_mask.get_dense_mask()
        masked_scores: torch.Tensor = scores.clone()
        masked_scores[previous_dense_mask != 0] = float("-inf")

        _: torch.Tensor
        top_k_indices: torch.Tensor
        _, top_k_indices = torch.topk(masked_scores, k=k, dim=-1, largest=True)
        return top_k_indices

    @abstractmethod
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
        """Add top-K mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "TopKMasker":
        """Create top-K masker instance from configuration."""
        if not isinstance(config, TopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@dataclass
class TopPMaskerConfig(FixedMaskerConfig):
    """Base configuration for top-P maskers."""

    top_p: float

    def __post_init__(self) -> None:
        """Validate top_p parameter."""
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in range [0, 1], got {self.top_p}")


class TopPMasker(FixedMasker):
    """Abstract base class for top-P maskers."""

    def __init__(self, config: TopPMaskerConfig) -> None:
        """Initialize top-P masker with configuration."""
        super().__init__(config)
        self.top_p: float = config.top_p

    @abstractmethod
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
        """Add top-P mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "TopPMasker":
        """Create top-P masker instance from configuration."""
        if not isinstance(config, TopPMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
