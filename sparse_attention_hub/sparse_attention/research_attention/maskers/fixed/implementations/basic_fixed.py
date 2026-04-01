"""Basic fixed pattern masker implementations."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import FixedMasker, FixedMaskerConfig


@dataclass
class LocalMaskerConfig(FixedMaskerConfig):
    """Configuration for LocalMasker."""

    window_size: Union[float, int]
    search_space: Dict[str, Any] = field(default_factory=dict)
    # add validation that window_size > 0

    def __post_init__(self) -> None:
        """Validate post-initialization constraints for LocalMaskerConfig.

        Raises:
            ValueError: If window_size is not greater than 0.
        """
        if not self.window_size > 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")


@MaskerRegistry.register(LocalMaskerConfig)
class LocalMasker(FixedMasker):
    """Local attention masker."""

    window_size: Union[float, int]

    def __init__(self, config: LocalMaskerConfig) -> None:
        """Initialize local masker with configuration."""
        super().__init__(config)
        self.window_size = config.window_size

    # @torch.compile
    def update_core(
        self, dense_mask: torch.Tensor, effective_window_size: int
    ) -> torch.Tensor:
        """Update the core of the masker."""
        i, j = torch.triu_indices(
            dense_mask.shape[2],
            dense_mask.shape[3],
            dense_mask.shape[3] - dense_mask.shape[2] - effective_window_size + 1,
            device=dense_mask.device,
            dtype=torch.long,
        )
        dense_mask[..., i, j] = 1.0
        i, j = torch.triu_indices(
            dense_mask.shape[2],
            dense_mask.shape[3],
            dense_mask.shape[3] - dense_mask.shape[2] + 1,
            device=dense_mask.device,
        )
        dense_mask[..., i, j] = 0.0
        return dense_mask

    def update_core_broadcast(
        self, dense_mask: torch.Tensor, effective_window_size: int
    ) -> torch.Tensor:
        """Update the core of the masker using broadcasting (optimized version).

        This implementation uses broadcasting instead of index selection for better
        performance and compilation efficiency.

        The mask creates a diagonal band pattern where positions with diagonal offset
        in the range [offset1, offset2) are masked out (set to 1.0), where:
        - offset1 = seq_len_keys - seq_len_queries - effective_window_size + 1
        - offset2 = seq_len_keys - seq_len_queries + 1
        - diagonal_offset = key_position - query_position

        Args:
            dense_mask: Dense mask tensor of shape [..., seq_len_queries, seq_len_keys]
            effective_window_size: Size of the local attention window

        Returns:
            Updated dense mask with local attention pattern applied
        """
        seq_len_queries: int = dense_mask.shape[-2]
        seq_len_keys: int = dense_mask.shape[-1]
        assert dense_mask is not None, "dense mask is NONE"
        # Create position indices for queries and keys
        query_pos: torch.Tensor = torch.arange(
            seq_len_queries, device=dense_mask.device
        ).unsqueeze(1)
        key_pos: torch.Tensor = torch.arange(
            seq_len_keys, device=dense_mask.device
        ).unsqueeze(0)

        # Calculate diagonal offset for each position
        diagonal_offset: torch.Tensor = key_pos - query_pos

        # Calculate the range of diagonals to mask
        offset1: int = seq_len_keys - seq_len_queries - effective_window_size + 1
        offset2: int = seq_len_keys - seq_len_queries + 1

        # Mask out diagonal band: offset1 <= diagonal < offset2
        # Positions in this range are masked (1.0), all others can attend (0.0)
        is_masked: torch.Tensor = (diagonal_offset >= offset1) & (
            diagonal_offset < offset2
        )

        # Update the mask: 1.0 for masked positions, 0.0 for attending positions
        # Broadcasting will handle the batch and head dimensions automatically
        dense_mask[...] = torch.where(is_masked, 1.0, 0.0)

        return dense_mask

    def get_updated_mask_1(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_window_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        """
        original implementation.
        """
        local_mask: Mask = self._create_local_mask(
            tensor_dims, effective_window_size, keys.device, previous_mask.dtype
        )
        return previous_mask.merge_mask(local_mask, inplace=False)

    def get_updated_mask_2(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_window_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        mask = previous_mask.get_dense_mask()
        mask = self.update_core_broadcast(mask, effective_window_size)
        return Mask.create_mask_from_dense_mask(mask.shape, mask, previous_mask.dtype)

    def get_updated_mask_3(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_window_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        mask = previous_mask.get_dense_mask()
        mask = self.update_core(mask, effective_window_size)
        return Mask.create_mask_from_dense_mask(mask.shape, mask, previous_mask.dtype)

    def get_updated_mask(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_window_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
        mode: Literal["sparse", "dense1", "dense2"] = "dense2",
    ) -> Mask:
        if mode == "sparse":
            return self.get_updated_mask_1(
                tensor_dims, effective_window_size, keys, previous_mask
            )
        elif mode == "dense1":
            return self.get_updated_mask_2(
                tensor_dims, effective_window_size, keys, previous_mask
            )
        elif mode == "dense2":
            return self.get_updated_mask_3(
                tensor_dims, effective_window_size, keys, previous_mask
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
        """Add local mask to enable local attention pattern.

        Uses the performance-optimized dense implementation (Dense2/triu) by default,
        which provides up to 1.8x speedup over sparse implementation.
        """
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_window_size: int = self._calculate_effective_window_size(
            tensor_dims.seq_len_keys
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims, effective_window_size):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )
        return self.get_updated_mask(
            tensor_dims, effective_window_size, keys, previous_mask,**kwargs
        )

    def _calculate_effective_window_size(self, seq_len_keys: int) -> int:
        """Calculate the effective window size based on configuration."""
        return self._calculate_effective_size(self.window_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, window_size: int
    ) -> bool:
        """Determine if full attention should be used instead of local attention."""
        return dims.seq_len_keys <= window_size + dims.seq_len_queries

    def _create_local_mask(
        self,
        dims: AttentionTensorDimensions,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create local attention mask using vectorized computation."""
        window_indices: torch.Tensor = self._compute_window_indices(
            dims, window_size, device
        )
        return self._create_mask_from_rowise_indices(
            dims, window_indices, device, dtype
        )

    def _compute_window_indices(
        self, dims: AttentionTensorDimensions, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute window indices for local attention using vectorized operations."""
        # For each query position, compute the start of its local window
        # Formula: window_start = seq_len_keys - (seq_len_queries - query_pos) - window_size + 1
        query_positions: torch.Tensor = torch.arange(
            dims.seq_len_queries, device=device, dtype=torch.long
        )
        window_starts: torch.Tensor = (
            dims.seq_len_keys - dims.seq_len_queries - window_size + query_positions + 1
        )
        # Create offset indices for the window
        window_offsets: torch.Tensor = torch.arange(
            window_size, device=device, dtype=torch.long
        )

        # Compute all window indices: window_start + offset for each query
        window_indices: torch.Tensor = window_starts.unsqueeze(
            1
        ) + window_offsets.unsqueeze(0)
        window_indices = torch.clamp(window_indices, 0, dims.seq_len_keys - 1)

        # Expand to match batch and head dimensions
        return (
            window_indices.unsqueeze(0)
            .unsqueeze(0)
            .expand(dims.batch_size, dims.num_heads, dims.seq_len_queries, window_size)
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "LocalMasker":
        """Create LocalMasker instance from configuration."""
        if not isinstance(config, LocalMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@MaskerRegistry.register(FixedMaskerConfig)
class CausalMasker(FixedMasker):
    """Causal attention masker."""

    def __init__(self, config: FixedMaskerConfig) -> None:
        """Initialize causal masker with configuration."""
        super().__init__(config)

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
        """Add causal mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "CausalMasker":
        """Create CausalMasker instance from configuration."""
        if not isinstance(config, FixedMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@dataclass
class SinkMaskerConfig(FixedMaskerConfig):
    """Configuration for SinkMasker."""

    sink_size: Union[float, int]
    search_space: Dict[str, Any] = field(default_factory=dict)
    is_gpt_oss: bool = False
    
    def __post_init__(self) -> None:
        """Validate post-initialization constraints for SinkMaskerConfig.

        Raises:
            ValueError: If sink_size is not greater than 0.
        """
        if not self.sink_size > 0:
            raise ValueError(f"sink_size must be > 0, got {self.sink_size}")


@MaskerRegistry.register(SinkMaskerConfig)
class SinkMasker(FixedMasker):
    """Sink attention masker."""

    sink_size: Union[float, int]

    def __init__(self, config: SinkMaskerConfig) -> None:
        """Initialize sink masker with configuration."""
        super().__init__(config)
        self.sink_size = config.sink_size
        self.is_gpt_oss = False

    def get_updated_mask_sparse(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_sink_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
        **kwargs,
    ) -> Mask:
        """
        original implementation.
        """
        sink_mask: Mask = self._create_sink_mask(
            tensor_dims, effective_sink_size, keys.device, previous_mask.dtype
        )
        # NOTE:
        # HuggingFace GPT-OSS handles sink tokens at attention level (not mask level).
        # This is a simplified approximation using masks.
        # Check if GPT-OSS mode is enabled
        if self.config.is_gpt_oss:
            mask = previous_mask.get_dense_mask().clone()

            seq_len_q = tensor_dims.seq_len_queries
            seq_len_k = tensor_dims.seq_len_keys

            causal_mask = torch.tril(
                torch.ones((seq_len_q, seq_len_k), device=mask.device)
            )

            mask = mask * causal_mask

            mask[..., :, :effective_sink_size] = torch.maximum(
                mask[..., :, :effective_sink_size],
                causal_mask[..., :effective_sink_size]
            )

            return Mask.create_mask_from_dense_mask(
                mask.shape,
                mask,
                previous_mask.dtype
            )

        return previous_mask.merge_mask(sink_mask, inplace=False)        
                
    def get_updated_mask_dense(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_sink_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        mask = previous_mask.get_dense_mask()
        mask[..., :effective_sink_size] = 1.0
        return Mask.create_mask_from_dense_mask(mask.shape, mask, previous_mask.dtype)

    def get_updated_mask(
        self,
        tensor_dims: AttentionTensorDimensions,
        effective_sink_size: int,
        keys: torch.Tensor,
        previous_mask: Mask,
        mode: Literal["sparse", "dense"] = "dense",
        **kwargs,
    ) -> Mask:
        if mode == "sparse":
            return self.get_updated_mask_sparse(
                tensor_dims, effective_sink_size, keys, previous_mask,**kwargs
            )
        elif mode == "dense":
            return self.get_updated_mask_dense(
                tensor_dims, effective_sink_size, keys, previous_mask
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
        """Add sink mask to enable sink attention pattern.

        Uses the performance-optimized dense implementation by default,
        which provides up to 3.8x speedup over sparse implementation.
        """
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_sink_size: int = self._calculate_effective_sink_size(
            tensor_dims.seq_len_keys
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims, effective_sink_size):
            shape = (
                tensor_dims.batch_size,
                tensor_dims.num_heads,
                tensor_dims.seq_len_queries,
                tensor_dims.seq_len_keys,
            )

            dense_mask = torch.ones(
                shape,
                device=previous_mask.device,
                dtype=previous_mask.dtype
            )

            return Mask.create_mask_from_dense_mask(
                shape,
                dense_mask,
                previous_mask.dtype
            )
        return self.get_updated_mask(
            tensor_dims, effective_sink_size, keys, previous_mask,mode="sparse",**kwargs
        )

    def _calculate_effective_sink_size(self, seq_len_keys: int) -> int:
        """Calculate the effective sink size based on configuration."""
        return self._calculate_effective_size(self.sink_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, sink_size: int
    ) -> bool:
        """Determine if full attention should be used instead of sink attention."""
        return dims.seq_len_keys <= sink_size

    def _create_sink_mask(
        self,
        dims: AttentionTensorDimensions,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create sink attention mask using vectorized computation."""
        sink_indices: torch.Tensor = self._compute_sink_indices(dims, sink_size, device)
        return self._create_mask_from_rowise_indices(dims, sink_indices, device, dtype)

    def _compute_sink_indices(
        self, dims: AttentionTensorDimensions, sink_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute sink indices for sink attention pattern."""
        # Create row_wise_idx with shape (b, h, sq, sink_size)
        # Each row contains indices [0, 1, ..., sink_size-1] (first sink_size positions)
        sink_indices: torch.Tensor = (
            torch.arange(sink_size, device=device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(dims.batch_size, dims.num_heads, dims.seq_len_queries, sink_size)
        )

        return sink_indices

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SinkMasker":
        """Create SinkMasker instance from configuration."""
        if not isinstance(config, SinkMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
