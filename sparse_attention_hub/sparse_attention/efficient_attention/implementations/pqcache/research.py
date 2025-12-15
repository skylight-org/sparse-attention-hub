"""PQCache research backend implementation.

This module provides PQCache implementation for the research backend.
PQCache combines sink tokens, local attention, and PQ-based top-K selection from middle tokens.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ....research_attention.base import ResearchAttentionConfig
from ....research_attention.maskers.fixed import (
    LocalMaskerConfig,
    PQCacheConfig,
    SinkMaskerConfig,
)
from ...efficient_attention_research_backend import (
    EfficientAttentionResearchBackend,
    EfficientAttentionResearchBackendConfig,
)


@dataclass
class PQCacheResearchBackendConfig(EfficientAttentionResearchBackendConfig):
    """Configuration class for PQCache research backend.

    This configuration validates that the research_attention_config contains
    exactly SinkMasker, LocalMasker, and PQCache maskers.
    """

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for PQCacheResearchBackendConfig.

        Validates that the masker classes in research_attention_config are
        SinkMasker, LocalMasker, and PQCache.

        Raises:
            ValueError: If masker classes are not SinkMasker, LocalMasker, and PQCache.
        """
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        masker_configs: List = self.research_attention_config.masker_configs
        if len(masker_configs) != 3:
            raise ValueError(
                f"PQCache requires exactly 3 maskers (SinkMasker, LocalMasker, and PQCache), "
                f"got {len(masker_configs)} maskers"
            )

        config_types: List[Type] = [type(config) for config in masker_configs]
        expected_types: set = {SinkMaskerConfig, LocalMaskerConfig, PQCacheConfig}

        if set(config_types) != expected_types:
            raise ValueError(
                f"PQCache requires SinkMaskerConfig, LocalMaskerConfig, and PQCacheConfig, "
                f"got {[t.__name__ for t in config_types]}"
            )


class PQCacheResearchBackend(EfficientAttentionResearchBackend):
    """PQCache implementation using research backend.

    This class extends EfficientAttentionResearchBackend to provide PQCache
    attention pattern using the research backend.

    PQCache combines:
    - Sink tokens: First few tokens for global context
    - Local attention: Sliding window for recent context
    - PQ-based top-K: Efficient selection from middle tokens using product quantization
    """

    def __init__(self, sparse_attention_config: PQCacheResearchBackendConfig) -> None:
        """Initialize PQCache research backend.

        Args:
            sparse_attention_config: Configuration for the PQCache mechanism.
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
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig,
                  LocalMaskerConfig, and PQCacheConfig with randomly generated parameters.
                - query: Query tensor of shape (B, H, 1, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Empty dictionary for sparse attention metadata.
        """
        sink_size: int = random.randint(1, max(1, num_keys // 4))
        window_size: int = random.randint(1, max(1, num_keys // 4))
        heavy_size: int = random.randint(1, max(1, num_keys // 4))
        
        # PQCache configuration parameters
        pq_group_factor: int = random.choice([1, 2, 4])  # Common factors for head_dim
        pq_bits: int = random.randint(4, 8)  # 16 to 256 centroids
        kmeans_iter: int = random.randint(5, 20)
        init_offset: int = sink_size  # Typically matches sink_size
        metric: str = random.choice(["euclidean", "ip"])
        
        research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=sink_size),
                LocalMaskerConfig(window_size=window_size),
                PQCacheConfig(
                    heavy_size=heavy_size,
                    pq_group_factor=pq_group_factor,
                    pq_bits=pq_bits,
                    kmeans_iter=kmeans_iter,
                    init_offset=init_offset,
                    metric=metric,
                ),
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
        scaling: float = 1.0 / (d**0.5)
        dropout: float = 0.0
        sparse_meta_data: Dict[Any, Any] = {}
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
        """Create sample data for subsequent iterations (decoding step).

        This method creates sample data similar to create_sample_data_first but also
        populates the sparse_meta_data with PQ cache fields (pq_centroids, pq_codebook,
        pq_ip2l2_phi) for num_keys - 1 number of keys to simulate a decoding step.

        Args:
            B: Batch size.
            H: Number of attention heads.
            num_keys: Number of key tokens (sequence length).
            d: Dimension per attention head.

        Returns:
            Tuple containing (in order of indexer_next parameters):
                - research_attention_config: ResearchAttentionConfig containing SinkMaskerConfig,
                  LocalMaskerConfig, and PQCacheConfig with randomly generated parameters.
                - query: Query tensor of shape (B, H, 1, d).
                - key: Key tensor of shape (B, H, num_keys, d).
                - value: Value tensor of shape (B, H, num_keys, d).
                - module: Dummy attention module.
                - attention_mask: Optional attention mask (set to None).
                - scaling: Scaling factor for attention (1.0 / sqrt(d)).
                - dropout: Dropout probability (0.0).
                - sparse_meta_data: Dictionary containing PQ cache metadata for decoding step.
        """
        sink_size: int = random.randint(1, max(1, num_keys // 4))
        window_size: int = random.randint(1, max(1, num_keys // 4))
        heavy_size: int = random.randint(1, max(1, num_keys // 4))
        
        # PQCache configuration parameters
        pq_group_factor: int = random.choice([1, 2, 4])  # Common factors for head_dim
        pq_bits: int = random.randint(4, 8)  # 16 to 256 centroids
        kmeans_iter: int = random.randint(5, 20)
        init_offset: int = sink_size  # Typically matches sink_size
        metric: str = random.choice(["euclidean", "ip"])
        
        research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=sink_size),
                LocalMaskerConfig(window_size=window_size),
                PQCacheConfig(
                    heavy_size=heavy_size,
                    pq_group_factor=pq_group_factor,
                    pq_bits=pq_bits,
                    kmeans_iter=kmeans_iter,
                    init_offset=init_offset,
                    metric=metric,
                ),
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
        scaling: float = 1.0 / (d**0.5)
        dropout: float = 0.0
        
        # Populate sparse_meta_data for decoding step
        # Simulate that PQ cache has already been built for num_keys - 1 keys
        n_quantized_keys: int = max(1, num_keys - 1 - init_offset)  # Keys after init_offset
        n_subvec: int = pq_group_factor
        subvec_d: int = d // pq_group_factor
        cent_cnt: int = 2**pq_bits
        
        # Layer index for testing (typically would be passed via kwargs)
        layer_idx: int = 0
        
        # Create PQ centroids: [B, H, n_subvec, cent_cnt, subvec_d]
        # If using IP metric, centroids are augmented with an extra dimension
        centroids_subvec_d: int = subvec_d + 1 if metric == "ip" else subvec_d
        pq_centroids: torch.Tensor = torch.randn(
            B, H, n_subvec, cent_cnt, centroids_subvec_d
        )
        
        # Create PQ codebook: [B, n_quantized_keys, H, n_subvec]
        # Each entry is an index into the centroids (0 to cent_cnt-1)
        pq_codebook: torch.Tensor = torch.randint(
            0, cent_cnt, (B, n_quantized_keys, H, n_subvec), dtype=torch.long
        )
        
        # Create IP-to-L2 phi values if using IP metric
        # Shape: [B * H * n_subvec] for batched processing
        pq_ip2l2_phi: Optional[torch.Tensor] = None
        if metric == "ip":
            pq_ip2l2_phi = torch.randn(B * H * n_subvec)
        
        sparse_meta_data: Dict[Any, Any] = {
            "pq_centroids": {layer_idx: pq_centroids},
            "pq_codebook": {layer_idx: pq_codebook},
            "pq_ip2l2_phi": {layer_idx: pq_ip2l2_phi},
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
        cls, config: PQCacheResearchBackendConfig
    ) -> "PQCacheResearchBackend":
        """Create PQCache research backend instance from configuration.

        Args:
            config: Configuration for the PQCache research backend.

        Returns:
            Instance of the PQCache research backend.

        Raises:
            TypeError: If config is not a PQCacheResearchBackendConfig.
        """
        if not isinstance(config, PQCacheResearchBackendConfig):
            raise TypeError(
                f"Expected PQCacheResearchBackendConfig, got {type(config)}"
            )
        return cls(config)

