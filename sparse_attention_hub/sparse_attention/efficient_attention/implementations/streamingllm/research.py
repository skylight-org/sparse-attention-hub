"""StreamingLLM research backend implementation.

This module provides StreamingLLM implementation for the research backend.
StreamingLLM combines sink tokens and local attention patterns for efficient long-context inference.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ....research_attention.base import ResearchAttentionConfig
from ....research_attention.maskers.fixed import LocalMaskerConfig, SinkMaskerConfig
from ...efficient_attention_research_backend import (
    EfficientAttentionResearchBackend,
    EfficientAttentionResearchBackendConfig,
)


@dataclass
class StreamingLLMResearchBackendConfig(EfficientAttentionResearchBackendConfig):
    """Configuration class for StreamingLLM research backend.

    This configuration validates that the research_attention_config contains
    exactly SinkMasker and LocalMasker maskers.
    """

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for StreamingLLMResearchBackendConfig.

        Validates that the masker classes in research_attention_config are
        SinkMasker and LocalMasker.

        Raises:
            ValueError: If masker classes are not SinkMasker and LocalMasker.
        """
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        masker_configs: List = self.research_attention_config.masker_configs
        if len(masker_configs) != 2:
            raise ValueError(
                f"StreamingLLM requires exactly 2 maskers (SinkMasker and LocalMasker), "
                f"got {len(masker_configs)} maskers"
            )

        config_types: List[Type] = [type(config) for config in masker_configs]
        expected_types: set = {SinkMaskerConfig, LocalMaskerConfig}

        if set(config_types) != expected_types:
            raise ValueError(
                f"StreamingLLM requires SinkMaskerConfig and LocalMaskerConfig, "
                f"got {[t.__name__ for t in config_types]}"
            )


class StreamingLLMResearchBackend(EfficientAttentionResearchBackend):
    """StreamingLLM implementation using research backend.

    This class extends EfficientAttentionResearchBackend to provide StreamingLLM
    attention pattern using the research backend.

    StreamingLLM combines:
    - Sink tokens: First few tokens for global context
    - Local attention: Sliding window for recent context
    """

    def __init__(self, sparse_attention_config: StreamingLLMResearchBackendConfig) -> None:
        """Initialize StreamingLLM research backend.

        Args:
            sparse_attention_config: Configuration for the StreamingLLM mechanism.
        """
        EfficientAttentionResearchBackend.__init__(self, sparse_attention_config)

    @classmethod
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
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig and LocalMaskerConfig
                  with randomly generated sink_size and window_size (between 1 and num_keys/2).
                - query: Query tensor of shape (B, H, 1, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Dictionary containing sparse attention metadata.
        """
        sink_size: int = random.randint(1, max(1, num_keys // 2))
        window_size: int = random.randint(1, max(1, num_keys // 2))
        research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=sink_size),
                LocalMaskerConfig(window_size=window_size),
            ]
        )
        query: torch.Tensor = torch.randn(B, H, 1, d)
        key: torch.Tensor = torch.randn(B, H, num_keys, d)
        value: torch.Tensor = torch.randn(B, H, num_keys, d)

        class DummyAttentionModule(nn.Module):
            """Dummy attention module for sample data."""

            def __init__(self) -> None:
                """Initialize dummy attention module."""
                super().__init__()
                self.training = False

        module: nn.Module = DummyAttentionModule()
        attention_mask: Optional[torch.Tensor] = None
        scaling: float = 1.0 / (d ** 0.5)
        dropout: float = 0.0
        sparse_meta_data: Dict[Any, Any] = {
        }
        return (
            research_attention_config,
            query,
            key,
            value,
            module,
            attention_mask,
            scaling,
            dropout,
            sparse_meta_data,
        )

    @classmethod
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
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig and LocalMaskerConfig
                  with randomly generated sink_size and window_size (between 1 and num_keys/2).
                - query: Query tensor of shape (B, H, 1, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Dictionary containing sparse attention metadata.
        """
        sink_size: int = random.randint(1, max(1, num_keys // 2))
        window_size: int = random.randint(1, max(1, num_keys // 2))
        research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=sink_size),
                LocalMaskerConfig(window_size=window_size),
            ]
        )
        query: torch.Tensor = torch.randn(B, H, 1, d)
        key: torch.Tensor = torch.randn(B, H, num_keys, d)
        value: torch.Tensor = torch.randn(B, H, num_keys, d)

        class DummyAttentionModule(nn.Module):
            """Dummy attention module for sample data."""

            def __init__(self) -> None:
                """Initialize dummy attention module."""
                super().__init__()
                self.training = False

        module: nn.Module = DummyAttentionModule()
        attention_mask: Optional[torch.Tensor] = None
        scaling: float = 1.0 / (d ** 0.5)
        dropout: float = 0.0
        sparse_meta_data: Dict[Any, Any] = {
        }
        return (
            research_attention_config,
            query,
            key,
            value,
            module,
            attention_mask,
            scaling,
            dropout,
            sparse_meta_data,
        )

    @classmethod
    def create_from_config(
        cls, config: StreamingLLMResearchBackendConfig
    ) -> "StreamingLLMResearchBackend":
        """Create StreamingLLM research backend instance from configuration.

        Args:
            config: Configuration for the StreamingLLM research backend.

        Returns:
            Instance of the StreamingLLM research backend.

        Raises:
            TypeError: If config is not a StreamingLLMResearchBackendConfig.
        """
        if not isinstance(config, StreamingLLMResearchBackendConfig):
            raise TypeError(f"Expected StreamingLLMResearchBackendConfig, got {type(config)}")
        return cls(config)

