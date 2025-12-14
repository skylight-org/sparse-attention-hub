"""StreamingLLM native backend implementation.

This module provides StreamingLLM implementation for the native backend.
StreamingLLM combines sink tokens and local attention patterns for efficient long-context inference.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import triton
import triton.language as tl
from torch import nn

from ....research_attention.base import ResearchAttention
from ....research_attention.maskers.fixed import LocalMaskerConfig, SinkMaskerConfig
from ...efficient_attention_native_backend import (
    EfficientAttentionNativeBackend,
    EfficientAttentionNativeBackendConfig,
)


@triton.jit
def _sparse_pattern_kernel(
    # Outputs
    sparse_list_ptr,
    weight_list_ptr,
    sparse_len_ptr,
    # Input params
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len_k: tl.constexpr,
    sink_size: tl.constexpr,
    window_size: tl.constexpr,
    # Strides
    sparse_list_stride_b: tl.constexpr,
    sparse_list_stride_h: tl.constexpr,
    sparse_list_stride_s: tl.constexpr,
    weight_list_stride_b: tl.constexpr,
    weight_list_stride_h: tl.constexpr,
    weight_list_stride_s: tl.constexpr,
    sparse_len_stride_b: tl.constexpr,
    sparse_len_stride_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel to generate sparse_list and weight_list for StreamingLLM pattern.

    This kernel generates:
    1. sparse_list: indices of tokens to attend to (sink + local)
    2. weight_list: weights for all tokens (1.0 for attended, 0.0 for others)
    3. sparse_len: number of attended tokens per batch/head
    """
    # Program ID: one per (batch, head) pair
    pid: tl.int32 = tl.program_id(0)

    # Decode batch and head indices
    batch_idx: tl.int32 = pid // num_heads
    head_idx: tl.int32 = pid % num_heads

    # Total number of attended tokens
    attended_len: tl.int32 = sink_size + window_size

    # Write sparse_len
    sparse_len_offset: tl.int64 = batch_idx * sparse_len_stride_b + head_idx * sparse_len_stride_h
    tl.store(sparse_len_ptr + sparse_len_offset, attended_len)

    # Generate sparse_list (sink + local tokens)
    # Process in blocks for efficiency
    for block_start in range(0, attended_len, BLOCK_SIZE):
        offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
        mask: tl.tensor = offsets < attended_len

        # Compute token indices:
        # First sink_size positions: [0, 1, 2, ..., sink_size-1]
        # Next window_size positions: [seq_len_k - window_size, ..., seq_len_k - 1]
        token_idx: tl.tensor = tl.where(
            offsets < sink_size,
            offsets,  # Sink tokens
            seq_len_k - window_size + (offsets - sink_size),  # Local window tokens
        )

        # Write to sparse_list
        sparse_list_offset: tl.int64 = (
            batch_idx * sparse_list_stride_b
            + head_idx * sparse_list_stride_h
            + offsets * sparse_list_stride_s
        )
        tl.store(sparse_list_ptr + sparse_list_offset, token_idx, mask=mask)

    # Generate weight_list (per-token weights)
    # Process sequence in blocks
    for block_start in range(0, seq_len_k, BLOCK_SIZE):
        offsets: tl.tensor = block_start + tl.arange(0, BLOCK_SIZE)
        mask: tl.tensor = offsets < seq_len_k

        # Weight is 1.0 if token is in sink (< sink_size) or local window (>= seq_len_k - window_size)
        # Otherwise 0.0
        is_sink: tl.tensor = offsets < sink_size
        is_local: tl.tensor = offsets >= (seq_len_k - window_size)
        weight: tl.tensor = tl.where(is_sink | is_local, 1.0, 0.0)

        # Write to weight_list
        weight_list_offset: tl.int64 = (
            batch_idx * weight_list_stride_b
            + head_idx * weight_list_stride_h
            + offsets * weight_list_stride_s
        )
        tl.store(weight_list_ptr + weight_list_offset, weight, mask=mask)

def __indexer(
    key: torch.Tensor,
    weight_list_dtype: torch.dtype,
    sink_size: int,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indexer logic for StreamingLLM pattern (sink + local attention) using Triton.

    This function generates sparse_list, sparse_len, and weight_list for the
    StreamingLLM attention pattern which attends to:
    1. First sink_size tokens (sink tokens)
    2. Last window_size tokens (local window)

    Args:
        key: Key tensor of shape (b, h_kv, sk, d).
        weight_list_dtype: Data type for weight_list tensor.
        sink_size: Number of sink tokens to attend to.
        window_size: Number of local window tokens to attend to.

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list) where:
        - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
        - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
        - weight_list: Tensor of shape (b, h, sk) containing weights for each token
    """
    batch_size: int = key.shape[0]
    num_heads: int = key.shape[1]
    seq_len_k: int = key.shape[2]
    device: torch.device = key.device

    # Handle case where sequence is shorter than sink + window
    if seq_len_k <= sink_size + window_size:
        # Use full attention - return all tokens
        sparse_list: torch.Tensor = (
            torch.arange(seq_len_k, device=device, dtype=torch.int32)
            .view(1, 1, seq_len_k)
            .repeat(batch_size, num_heads, 1)
        )
        sparse_len: torch.Tensor = torch.full(
            (batch_size, num_heads),
            seq_len_k,
            dtype=torch.int32,
            device=device,
        )
        weight_list: torch.Tensor = torch.ones(
            (batch_size, num_heads, seq_len_k),
            dtype=weight_list_dtype,
            device=device,
        )
        return (sparse_list, sparse_len, weight_list)

    # Total number of attended tokens
    attended_len: int = sink_size + window_size

    # Allocate output tensors
    new_sparse_list: torch.Tensor = torch.empty(
        (batch_size, num_heads, attended_len),
        dtype=torch.int32,
        device=device,
    )

    new_sparse_len: torch.Tensor = torch.empty(
        (batch_size, num_heads),
        dtype=torch.int32,
        device=device,
    )

    new_weight_list: torch.Tensor = torch.empty(
        (batch_size, num_heads, seq_len_k),
        dtype=weight_list_dtype,
        device=device,
    )

    # Launch kernel with one program per (batch, head) pair
    num_programs: int = batch_size * num_heads
    BLOCK_SIZE: int = 128  # Process 128 elements at a time

    _sparse_pattern_kernel[(num_programs,)](
        # Outputs
        new_sparse_list,
        new_weight_list,
        new_sparse_len,
        # Input params
        batch_size,
        num_heads,
        seq_len_k,
        sink_size,
        window_size,
        # Strides for sparse_list [B, H, S_attended]
        new_sparse_list.stride(0),
        new_sparse_list.stride(1),
        new_sparse_list.stride(2),
        # Strides for weight_list [B, H, S_full]
        new_weight_list.stride(0),
        new_weight_list.stride(1),
        new_weight_list.stride(2),
        # Strides for sparse_len [B, H]
        new_sparse_len.stride(0),
        new_sparse_len.stride(1),
        # Block size
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (new_sparse_list, new_sparse_len, new_weight_list)

__indexer_first = __indexer
__indexer_next = __indexer

@dataclass
class StreamingLLMNativeBackendConfig(EfficientAttentionNativeBackendConfig):
    """Configuration class for StreamingLLM native backend.

    This configuration validates that the research_attention_config contains
    exactly SinkMasker and LocalMasker maskers.
    """

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for StreamingLLMNativeBackendConfig.

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


class StreamingLLMNativeBackend(EfficientAttentionNativeBackend):
    """StreamingLLM implementation using native backend.

    This class extends EfficientAttentionNativeBackend to provide StreamingLLM
    attention pattern using the native backend.

    StreamingLLM combines:
    - Sink tokens: First few tokens for global context
    - Local attention: Sliding window for recent context
    """

    def __init__(self, sparse_attention_config: StreamingLLMNativeBackendConfig) -> None:
        """Initialize StreamingLLM native backend.

        Args:
            sparse_attention_config: Configuration for the StreamingLLM mechanism.
        """
        EfficientAttentionNativeBackend.__init__(self, sparse_attention_config)
        self.research_attention: ResearchAttention = ResearchAttention.create_from_config(
            sparse_attention_config.research_attention_config
        )
        # Extract sink_size and window_size from config
        masker_configs: List = sparse_attention_config.research_attention_config.masker_configs
        sink_config: SinkMaskerConfig = next(
            (cfg for cfg in masker_configs if isinstance(cfg, SinkMaskerConfig)), None
        )
        local_config: LocalMaskerConfig = next(
            (cfg for cfg in masker_configs if isinstance(cfg, LocalMaskerConfig)), None
        )
        if sink_config is None or local_config is None:
            raise ValueError("SinkMaskerConfig and LocalMaskerConfig must be present")
        self.sink_size: int = int(sink_config.sink_size)
        self.window_size: int = int(local_config.window_size)

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

        This method uses the optimized Triton kernel to generate the StreamingLLM
        sparse attention pattern (sink + local).

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
            - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        indexer_function = globals().get("__indexer_first")
        return indexer_function(
            key=key,
            weight_list_dtype=query.dtype,
            sink_size=self.sink_size,
            window_size=self.window_size,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update indexer for subsequent iterations.

        This method uses the optimized Triton kernel to generate the StreamingLLM
        sparse attention pattern (sink + local). For StreamingLLM, the pattern is
        deterministic and doesn't change between iterations.

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
            - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        
        indexer_function = globals().get("__indexer_next")
        return indexer_function(
            key=key,
            weight_list_dtype=query.dtype,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )

    @classmethod
    def create_from_config(
        cls, config: StreamingLLMNativeBackendConfig
    ) -> "StreamingLLMNativeBackend":
        """Create StreamingLLM native backend instance from configuration.

        Args:
            config: Configuration for the StreamingLLM native backend.

        Returns:
            Instance of the StreamingLLM native backend.

        Raises:
            TypeError: If config is not a StreamingLLMNativeBackendConfig.
        """
        if not isinstance(config, StreamingLLMNativeBackendConfig):
            raise TypeError(f"Expected StreamingLLMNativeBackendConfig, got {type(config)}")
        return cls(config)


