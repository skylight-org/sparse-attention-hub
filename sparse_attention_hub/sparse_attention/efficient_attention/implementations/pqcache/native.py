"""PQCache native backend implementation.

This module provides PQCache implementation for the native backend.
PQCache combines sink tokens, local attention, and PQ-based top-K selection from middle tokens.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ....research_attention.base import ResearchAttention
from ....research_attention.maskers.fixed import (
    LocalMaskerConfig,
    PQCacheConfig,
    SinkMaskerConfig,
)
from ...efficient_attention_native_backend import (
    EfficientAttentionNativeBackend,
    EfficientAttentionNativeBackendConfig,
)


def __indexer_first(
    key: torch.Tensor,
    weight_list_dtype: torch.dtype,
    sink_size: int,
    window_size: int,
    heavy_size: int,
    pq_group_factor: int,
    pq_bits: int,
    kmeans_iter: int,
    init_offset: int,
    metric: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Indexer logic for PQCache pattern (first iteration).

    This function generates sparse_list, sparse_len, and weight_list for the
    PQCache attention pattern which attends to:
    1. First sink_size tokens (sink tokens)
    2. Top heavy_size tokens from middle section using PQ-based selection
    3. Last window_size tokens (local window)

    Args:
        key: Key tensor of shape (b, h_kv, sk, d).
        weight_list_dtype: Data type for weight_list tensor.
        sink_size: Number of sink tokens to attend to.
        window_size: Number of local window tokens to attend to.
        heavy_size: Number of heavy hitter tokens to select from middle section.
        pq_group_factor: Product quantization group factor (number of subvectors).
        pq_bits: Number of bits for PQ codebook (2^pq_bits centroids per subvector).
        kmeans_iter: Number of k-means iterations for building PQ codebook.
        init_offset: Offset for starting PQ-based selection.
        metric: Distance metric to use ("euclidean" or "ip" for inner product).

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi) where:
        - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
        - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
        - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        - pq_centroids: PQ centroids tensor for future iterations
        - pq_codebook: PQ codebook tensor for future iterations
        - pq_ip2l2_phi: Optional IP-to-L2 phi values for inner product metric
    """
    raise NotImplementedError("__indexer_first will be implemented later")


def __indexer_next(
    key: torch.Tensor,
    weight_list_dtype: torch.dtype,
    sink_size: int,
    window_size: int,
    heavy_size: int,
    pq_group_factor: int,
    pq_bits: int,
    kmeans_iter: int,
    init_offset: int,
    metric: str,
    pq_centroids: torch.Tensor,
    pq_codebook: torch.Tensor,
    pq_ip2l2_phi: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Indexer logic for PQCache pattern (subsequent iterations).

    This function generates sparse_list, sparse_len, and weight_list for the
    PQCache attention pattern which attends to:
    1. First sink_size tokens (sink tokens)
    2. Top heavy_size tokens from middle section using PQ-based selection
    3. Last window_size tokens (local window)

    Args:
        key: Key tensor of shape (b, h_kv, sk, d).
        weight_list_dtype: Data type for weight_list tensor.
        sink_size: Number of sink tokens to attend to.
        window_size: Number of local window tokens to attend to.
        heavy_size: Number of heavy hitter tokens to select from middle section.
        pq_group_factor: Product quantization group factor (number of subvectors).
        pq_bits: Number of bits for PQ codebook (2^pq_bits centroids per subvector).
        kmeans_iter: Number of k-means iterations for building PQ codebook.
        init_offset: Offset for starting PQ-based selection.
        metric: Distance metric to use ("euclidean" or "ip" for inner product).
        pq_centroids: PQ centroids tensor from previous iteration.
        pq_codebook: PQ codebook tensor from previous iteration.
        pq_ip2l2_phi: Optional IP-to-L2 phi values for inner product metric.

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi) where:
        - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
        - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
        - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        - pq_centroids: Updated PQ centroids tensor for future iterations
        - pq_codebook: Updated PQ codebook tensor for future iterations
        - pq_ip2l2_phi: Optional updated IP-to-L2 phi values for inner product metric
    """
    raise NotImplementedError("__indexer_next will be implemented later")


@dataclass
class PQCacheNativeBackendConfig(EfficientAttentionNativeBackendConfig):
    """Configuration class for PQCache native backend.

    This configuration validates that the research_attention_config contains
    exactly SinkMasker, LocalMasker, and PQCache maskers.
    """

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for PQCacheNativeBackendConfig.

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


class PQCacheNativeBackend(EfficientAttentionNativeBackend):
    """PQCache implementation using native backend.

    This class extends EfficientAttentionNativeBackend to provide PQCache
    attention pattern using the native backend.

    PQCache combines:
    - Sink tokens: First few tokens for global context
    - Local attention: Sliding window for recent context
    - PQ-based top-K: Efficient selection from middle tokens using product quantization
    """

    def __init__(self, sparse_attention_config: PQCacheNativeBackendConfig) -> None:
        """Initialize PQCache native backend.

        Args:
            sparse_attention_config: Configuration for the PQCache mechanism.
        """
        EfficientAttentionNativeBackend.__init__(self, sparse_attention_config)
        self.research_attention: ResearchAttention = ResearchAttention.create_from_config(
            sparse_attention_config.research_attention_config
        )
        
        # Extract configuration parameters from masker configs
        masker_configs: List = sparse_attention_config.research_attention_config.masker_configs
        
        # Extract SinkMasker config
        sink_config: Optional[SinkMaskerConfig] = next(
            (cfg for cfg in masker_configs if isinstance(cfg, SinkMaskerConfig)), None
        )
        if sink_config is None:
            raise ValueError("SinkMaskerConfig must be present")
        self.sink_size: int = int(sink_config.sink_size)
        
        # Extract LocalMasker config
        local_config: Optional[LocalMaskerConfig] = next(
            (cfg for cfg in masker_configs if isinstance(cfg, LocalMaskerConfig)), None
        )
        if local_config is None:
            raise ValueError("LocalMaskerConfig must be present")
        self.window_size: int = int(local_config.window_size)
        
        # Extract PQCache config
        pqcache_config: Optional[PQCacheConfig] = next(
            (cfg for cfg in masker_configs if isinstance(cfg, PQCacheConfig)), None
        )
        if pqcache_config is None:
            raise ValueError("PQCacheConfig must be present")
        self.heavy_size: int = int(pqcache_config.heavy_size)
        self.pq_group_factor: int = int(pqcache_config.pq_group_factor)
        self.pq_bits: int = int(pqcache_config.pq_bits)
        self.kmeans_iter: int = int(pqcache_config.kmeans_iter)
        self.init_offset: int = int(pqcache_config.init_offset)
        self.metric: str = str(pqcache_config.metric)

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

        This method calls the global __indexer_first function to generate the PQCache
        sparse attention pattern (sink + PQ-based top-K + local), and updates
        sparse_meta_data with the generated PQ cache.

        Args:
            query: Query tensor of shape (b, h, sq, d) or (b, h, d).
            key: Key tensor of shape (b, h_kv, sk, d).
            value: Value tensor of shape (b, h_kv, sk, d).
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments (must include 'layer_idx').

        Returns:
            Tuple of (sparse_list, sparse_len, weight_list) where:
            - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        indexer_function = globals().get("__indexer_first")
        
        # Extract layer_idx from kwargs
        layer_idx: int = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        
        # Call indexer function and get both sparsity details and metadata
        (
            sparse_list,
            sparse_len,
            weight_list,
            pq_centroids,
            pq_codebook,
            pq_ip2l2_phi,
        ) = indexer_function(
            key=key,
            weight_list_dtype=query.dtype,
            sink_size=self.sink_size,
            window_size=self.window_size,
            heavy_size=self.heavy_size,
            pq_group_factor=self.pq_group_factor,
            pq_bits=self.pq_bits,
            kmeans_iter=self.kmeans_iter,
            init_offset=self.init_offset,
            metric=self.metric,
        )
        
        # Update sparse_meta_data with PQ cache for this layer
        if "pq_centroids" not in sparse_meta_data:
            sparse_meta_data["pq_centroids"] = {}
        sparse_meta_data["pq_centroids"][layer_idx] = pq_centroids
        
        if "pq_codebook" not in sparse_meta_data:
            sparse_meta_data["pq_codebook"] = {}
        sparse_meta_data["pq_codebook"][layer_idx] = pq_codebook
        
        if "pq_ip2l2_phi" not in sparse_meta_data:
            sparse_meta_data["pq_ip2l2_phi"] = {}
        sparse_meta_data["pq_ip2l2_phi"][layer_idx] = pq_ip2l2_phi
        
        # Return only sparsity details
        return (sparse_list, sparse_len, weight_list)

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

        This method calls the global __indexer_next function to generate the PQCache
        sparse attention pattern (sink + PQ-based top-K + local), and updates
        sparse_meta_data with the updated PQ cache.

        Args:
            query: Query tensor of shape (b, h, sq, d) or (b, h, d).
            key: Key tensor of shape (b, h_kv, sk, d).
            value: Value tensor of shape (b, h_kv, sk, d).
            module: The attention module.
            attention_mask: Optional attention mask.
            scaling: Scaling factor for attention.
            dropout: Dropout probability.
            sparse_meta_data: Dictionary containing sparse attention metadata.
            **kwargs: Additional keyword arguments (must include 'layer_idx').

        Returns:
            Tuple of (sparse_list, sparse_len, weight_list) where:
            - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        indexer_function = globals().get("__indexer_next")
        
        # Extract layer_idx from kwargs
        layer_idx: int = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        
        # Extract PQ fields from sparse_meta_data for this layer
        pq_centroids: torch.Tensor = sparse_meta_data.get("pq_centroids", {}).get(layer_idx)
        pq_codebook: torch.Tensor = sparse_meta_data.get("pq_codebook", {}).get(layer_idx)
        pq_ip2l2_phi: Optional[torch.Tensor] = sparse_meta_data.get("pq_ip2l2_phi", {}).get(layer_idx)
        
        # Call indexer function and get both sparsity details and updated metadata
        (
            sparse_list,
            sparse_len,
            weight_list,
            updated_pq_centroids,
            updated_pq_codebook,
            updated_pq_ip2l2_phi,
        ) = indexer_function(
            key=key,
            weight_list_dtype=query.dtype,
            sink_size=self.sink_size,
            window_size=self.window_size,
            heavy_size=self.heavy_size,
            pq_group_factor=self.pq_group_factor,
            pq_bits=self.pq_bits,
            kmeans_iter=self.kmeans_iter,
            init_offset=self.init_offset,
            metric=self.metric,
            pq_centroids=pq_centroids,
            pq_codebook=pq_codebook,
            pq_ip2l2_phi=pq_ip2l2_phi,
        )
        
        # Update sparse_meta_data with updated PQ cache for this layer
        sparse_meta_data["pq_centroids"][layer_idx] = updated_pq_centroids
        sparse_meta_data["pq_codebook"][layer_idx] = updated_pq_codebook
        sparse_meta_data["pq_ip2l2_phi"][layer_idx] = updated_pq_ip2l2_phi
        
        # Return only sparsity details
        return (sparse_list, sparse_len, weight_list)

    @classmethod
    def create_from_config(
        cls, config: PQCacheNativeBackendConfig
    ) -> "PQCacheNativeBackend":
        """Create PQCache native backend instance from configuration.

        Args:
            config: Configuration for the PQCache native backend.

        Returns:
            Instance of the PQCache native backend.

        Raises:
            TypeError: If config is not a PQCacheNativeBackendConfig.
        """
        if not isinstance(config, PQCacheNativeBackendConfig):
            raise TypeError(f"Expected PQCacheNativeBackendConfig, got {type(config)}")
        return cls(config)

