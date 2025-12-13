"""StreamingLLM research backend implementation.

This module provides StreamingLLM implementation for the research backend.
StreamingLLM combines sink tokens and local attention patterns for efficient long-context inference.
"""

from dataclasses import dataclass
from typing import List, Type

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

