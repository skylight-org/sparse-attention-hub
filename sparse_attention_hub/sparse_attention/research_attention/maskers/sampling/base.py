"""Base sampling masker implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch

from ....utils.mask import Mask
from ..base import MaskerConfig, ResearchMasker


@dataclass
class SamplingMaskerConfig(MaskerConfig):
    """Base configuration for sampling maskers."""

    pass


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    def __init__(self, config: SamplingMaskerConfig) -> None:
        """Initialize sampling masker with configuration."""
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
        """Add sampling mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SamplingMasker":
        """Create sampling masker instance from configuration."""
        if not isinstance(config, SamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
