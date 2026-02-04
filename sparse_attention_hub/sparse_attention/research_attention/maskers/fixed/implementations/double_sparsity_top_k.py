"""Double sparsity top-K masker implementation."""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig
from .utils.common_utils import pseudo_quantize
from .utils.double_sparsity_utils import (
    extract_layer_channels,
    load_sorted_channels_from_file,
)


@dataclass
class DoubleSparsityTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for DoubleSparsityTopKMasker."""

    sorted_channel_file: str
    group_factor: int = 16
    label_bits: int = 4
    channel_selection: str = "qk_proj"
    # TTBOMU: the original repository does not work with k_proj and GQA
    search_space: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate post-initialization constraints."""
        super().__post_init__()

        if self.group_factor <= 0:
            raise ValueError(f"group_factor must be > 0, got {self.group_factor}")

        if not (0 < self.label_bits <= 16):
            raise ValueError(
                f"label_bits must be in range (0, 16], got {self.label_bits}"
            )

        # Check if sorted_channel_file exists
        if not os.path.exists(self.sorted_channel_file):
            raise ValueError(
                f"sorted_channel_file does not exist: {self.sorted_channel_file}"
            )


@MaskerRegistry.register(DoubleSparsityTopKMaskerConfig)
class DoubleSparsityTopKMasker(TopKMasker):
    """Double sparsity top-K masker."""

    heavy_size: Union[float, int]
    group_factor: int
    label_bits: int
    channel_selection: str
    sorted_channels: Dict[str, List[List[int]]]
    sorted_channel: Optional[torch.Tensor]

    def __init__(self, config: DoubleSparsityTopKMaskerConfig) -> None:
        """Initialize double sparsity top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.group_factor = config.group_factor
        self.label_bits = config.label_bits
        self.channel_selection = config.channel_selection
        # Load sorted channels from file
        self.sorted_channel = None

    def _ensure_sorted_channel_is_loaded(
        self, layer_idx: int, device: torch.device
    ) -> None:
        """Ensure sorted channel is loaded for the given layer."""
        if self.sorted_channel is None:
            config: DoubleSparsityTopKMaskerConfig = self.config  # type: ignore[assignment]
            self.sorted_channels = load_sorted_channels_from_file(
                config.sorted_channel_file
            )
            self.sorted_channel = extract_layer_channels(
                self.sorted_channels, layer_idx, self.channel_selection, device
            )

    def _compute_grouped_scores(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention scores using grouped channels.

        Args:
            keys: Key tensor
            queries: Query tensor
            attention_mask: Attention mask tensor
            layer_idx: Layer index

        Returns:
            Attention scores tensor
        """
        b, h, q_len, head_dim = queries.shape
        _, _, kv_len, _ = keys.shape
        self._ensure_sorted_channel_is_loaded(layer_idx, queries.device)
        sorted_query_states = queries.transpose(1, 2)
        sorted_key_states = keys.transpose(1, 2)
        sorted_query_states = torch.gather(
            sorted_query_states,
            -1,
            self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(b, q_len, -1, -1),
        ).transpose(1, 2)
        sorted_key_states = torch.gather(
            sorted_key_states,
            -1,
            self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(b, kv_len, -1, -1),
        ).transpose(1, 2)
        outlier_num = queries.shape[-1] // self.group_factor
        grouped_query = sorted_query_states[:, :, :, :outlier_num]
        grouped_key = sorted_key_states[:, :, :, :outlier_num]

        if self.label_bits < 16:
            grouped_query = pseudo_quantize(grouped_query, self.label_bits)
            grouped_key = pseudo_quantize(grouped_key, self.label_bits)

        grouped_attn_weights = torch.matmul(
            grouped_query, grouped_key.transpose(2, 3)
        ) / math.sqrt(head_dim // self.group_factor)

        if attention_mask is not None:
            grouped_attn_weights = (
                grouped_attn_weights + attention_mask[:, :, :, : keys.shape[2]]
            )

        return grouped_attn_weights

    def _apply_topk_selection(
        self, scores: torch.Tensor, previous_mask: Mask, effective_heavy_size: int
    ) -> torch.Tensor:
        """Apply top-K selection with offset constraints.

        Args:
            scores: Attention scores tensor
            previous_mask: Previous mask
            effective_heavy_size: Number of top-K positions to select

        Returns:
            Top-K indices tensor

        Important Comment:
            We want to remove the previously selected positions from the scores. However,
            since sparse_attention_hub uses per head mask and double sparsity uses
            aggregate prediction, we need to aggregate previous mask. We do it using max
            which means that if the token is selected in atleast one head it will be masked
            for top-k selection. We might want to change this later.
            In most practical cases, top-k masker is applied after sink and local maskers where
            the either a token is used / unused across heads.
        """
        previous_dense_mask = previous_mask.get_dense_mask()
        assert (
            previous_dense_mask.shape == scores.shape
        ), "previous_dense_mask and scores must have the same shape. Check the channel selection."
        scores[previous_dense_mask != 0] = float("-inf")
        _, top_k_indices = torch.topk(
            scores, k=effective_heavy_size, dim=-1, largest=True
        )
        return top_k_indices

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_size: int
    ) -> bool:
        """Determine if full attention should be used instead of double sparsity.
        we use full attention if for all tokens we do not have atleast heavy_size tokens to attend to.
        """
        return dims.seq_len_keys - dims.seq_len_queries <= heavy_size

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
        """Add double sparsity top-K mask to attention computation."""
        # Validate inputs

        layer_idx_raw: Any = kwargs.get("layer_idx")
        layer_idx: Optional[int] = (
            layer_idx_raw if isinstance(layer_idx_raw, int) else None
        )
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        # Handle GQA if needed
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        values = repeat_kv(values, ngroups)

        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_heavy_size: int = self._calculate_effective_size(
            self.heavy_size, tensor_dims.seq_len_keys
        )

        if self._should_use_full_attention(tensor_dims, effective_heavy_size):
            return self._create_full_mask(
                tensor_dims, previous_mask.dtype, previous_mask.device
            )

        # Compute grouped attention scores # b,1,q_len,kv_len
        grouped_scores = self._compute_grouped_scores(
            keys, queries, attention_mask, layer_idx
        )

        # Apply top-K selection,
        top_k_indices = self._apply_topk_selection(
            grouped_scores, previous_mask, effective_heavy_size
        )
        previous_dense_mask = previous_mask.get_dense_mask()
        previous_dense_mask.scatter_(dim=-1, index=top_k_indices, value=1.0)
        return Mask.create_mask_from_dense_mask(
            previous_dense_mask.shape, previous_dense_mask, dtype=previous_mask.dtype
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "DoubleSparsityTopKMasker":
        """Create DoubleSparsityTopKMasker instance from configuration."""
        if not isinstance(config, DoubleSparsityTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
