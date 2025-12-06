"""Utility functions for computing sparse attention masks for research attention.

This module exposes a simple helper function that mirrors the mask computation
logic used in ``ResearchAttention.custom_attention`` while decoupling it from
the attention module itself.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from sparse_attention_hub.sparse_attention.research_attention.base import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_masked_attention_output,
)

sparse_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
    masker_configs=[]
)

def custom_indexer_hub(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_meta_data: Dict[Any, Any],
    **kwargs: Dict[str, Any],
) -> Mask:
    """Compute the sparse attention mask used by research attention.

    This function mirrors the first part of
    ``ResearchAttention.custom_attention``: it creates an empty mask, applies
    all maskers specified in the provided ``ResearchAttentionConfig``, and
    optionally logs the resulting mask density. It **does not** perform the
    masked attention computation itself and only returns the final mask.

    Args:
        sparse_attention_config: Configuration for the research attention
            mechanism, including the list of masker configurations.
        module: Attention module associated with this computation. Included to
            keep the signature close to ``ResearchAttention.custom_attention``,
            but not used here since maskers currently do not depend on it.
        queries: Query tensor of shape ``(batch_size, num_heads, seq_len_q, d)``.
        keys: Key tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        values: Value tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        attention_mask: Optional dense attention mask of shape
            ``(batch_size, num_heads, seq_len_q, seq_len_k)``.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability used in attention.
        sparse_meta_data: Dictionary carrying auxiliary metadata for sparse
            attention and maskers.
        **kwargs: Additional keyword arguments forwarded to maskers (for
            example, ``layer_idx`` for logging).

    Returns:
        Final sparse attention ``Mask`` after applying all configured maskers.
    """
    # Create an empty Mask object with the same 4D shape as the attention scores
    mask_shape: Tuple[int, int, int, int] = (
        queries.shape[0],
        queries.shape[1],
        queries.shape[2],
        keys.shape[2],
    )
    sparse_attention_mask: Mask = Mask.create_empty_mask(
        mask_shape, dtype=queries.dtype, device=queries.device
    )

    # Instantiate and apply all maskers sequentially as in ResearchAttention.create_from_config
    masker_config: MaskerConfig
    for masker_config in sparse_attention_config.masker_configs:
        masker: ResearchMasker = ResearchMasker.create_masker_from_config(
            masker_config
        )
        sparse_attention_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            previous_mask=sparse_attention_mask,
            **kwargs,
        )

    return sparse_attention_mask


def custom_backend_hub(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Compute masked attention output given a precomputed sparse attention mask.

    This function mirrors the second part of
    ``ResearchAttention.custom_attention``: it takes the sparse attention mask
    and computes the masked attention output using
    ``get_masked_attention_output``. Attention weights and logging are omitted.

    Args:
        module: The attention module.
        queries: Query tensor of shape ``(batch_size, num_heads, seq_len_q, d)``.
        keys: Key tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        values: Value tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        attention_mask: Optional dense attention mask of shape
            ``(batch_size, num_heads, seq_len_q, seq_len_k)``.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability used in attention.
        sparse_attention_mask: Precomputed sparse attention :class:`Mask`.
        **kwargs: Additional keyword arguments forwarded to the backend.

    Returns:
        Masked attention output tensor.
    """
    attention_output: torch.Tensor = get_masked_attention_output(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=sparse_attention_mask,
        return_attention_weights=False,
        **kwargs,
    )

    return attention_output


def custom_attention(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_meta_data: Dict[Any, Any],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """End-to-end custom attention using indexer and backend helpers.

    This function composes :func:`custom_indexer_hub` and
    :func:`custom_backend_hub` to mirror the behavior of
    ``ResearchAttention.custom_attention`` while separating mask computation
    from the backend attention computation.

    Args:
        module: The attention module.
        queries: Query tensor of shape ``(batch_size, num_heads, seq_len_q, d)``.
        keys: Key tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        values: Value tensor of shape ``(batch_size, num_heads, seq_len_k, d)``.
        attention_mask: Optional dense attention mask of shape
            ``(batch_size, num_heads, seq_len_q, seq_len_k)``.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability used in attention.
        sparse_meta_data: Dictionary carrying auxiliary metadata for sparse
            attention and maskers.
        **kwargs: Additional keyword arguments forwarded to helpers.

    Returns:
        Masked attention output tensor.
    """
    sparse_attention_mask: Mask = custom_indexer_hub(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_meta_data=sparse_meta_data,
        **kwargs,
    )

    attention_output: torch.Tensor = custom_backend_hub(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=sparse_attention_mask,
        **kwargs,
    )

    return attention_output

