"""Utility helpers for sparse attention native backend.

This module contains conversion utilities to transform dense attention inputs
into the format expected by the sparse attention native backend.
"""

from typing import List, Optional, Tuple

import torch


SparseAttentionInputTuple = Tuple[
    torch.Tensor,  # query
    torch.Tensor,  # key
    torch.Tensor,  # value
    torch.Tensor,  # sparse_list
    torch.Tensor,  # sparse_len
    torch.Tensor,  # weight_list
]


def convert_dense_to_sparse_attention_inputs(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> SparseAttentionInputTuple:
    """Convert dense Q/K/V into arguments for sparse attention native backend.

    The dense queries are expected to have a single query position
    (``seq_len_q == 1``); the function flattens that dimension to match the
    sparse attention query layout.

    Args:
        queries: Query tensor of shape
            ``(batch_size, num_heads, 1, head_dim)``.
        keys: Key tensor of shape
            ``(batch_size, num_heads, seq_len_k, head_dim)``.
        values: Value tensor of shape
            ``(batch_size, num_heads, seq_len_k, head_dim)``.

    Returns:
        Tuple of arguments matching the signature expected by
        ``bias_sparse_attention_fwd``:
        (query, key, value, sparse_list, sparse_len, weight_list)
    """
    if queries.shape[2] != 1:
        raise ValueError(
            f"convert_dense_to_sparse_attention_inputs expects seq_len_q == 1, "
            f"got queries.shape={queries.shape}"
        )

    batch_size: int = queries.shape[0]
    num_heads: int = queries.shape[1]
    seq_len_k: int = keys.shape[2]
    head_dim: int = queries.shape[3]

    device: torch.device = queries.device
    dtype: torch.dtype = queries.dtype

    # Flatten the single query position: (b, h, 1, d) -> (b, h, d)
    query: torch.Tensor = queries[:, :, 0, :]

    # Ensure keys and values have the same number of heads as queries
    # For now, we assume no GQA (num_kv_heads == num_heads)
    if keys.shape[1] != num_heads:
        raise ValueError(
            f"Number of key heads ({keys.shape[1]}) must equal number of query heads ({num_heads})"
        )

    # For sparse attention backend with GQA support, keys/values are [B, Kv, S, D]
    # Since we're not using GQA here, Kv = H
    key: torch.Tensor = keys
    value: torch.Tensor = values

    # Create sparse_list that contains all token indices [0, 1, 2, ..., seq_len_k-1]
    # Shape: [B, H, S]
    sparse_list: torch.Tensor = (
        torch.arange(seq_len_k, device=device, dtype=torch.int32)
        .view(1, 1, seq_len_k)
        .repeat(batch_size, num_heads, 1)
    )

    # Set sparse_len to full sequence length for all batch/heads
    # Shape: [B, H]
    sparse_len: torch.Tensor = torch.full(
        (batch_size, num_heads),
        seq_len_k,
        dtype=torch.int32,
        device=device,
    )

    # Create uniform weights (all ones) for unbiased attention
    # Shape: [B, H, S]
    weight_list: torch.Tensor = torch.ones(
        (batch_size, num_heads, seq_len_k),
        dtype=dtype,
        device=device,
    )

    return (
        query,
        key,
        value,
        sparse_list,
        sparse_len,
        weight_list,
    )

