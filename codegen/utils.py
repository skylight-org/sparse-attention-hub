"""Utility helpers for code generation experiments.

This module contains small conversion utilities used to compare different
attention backends (e.g. dense vs. FlashInfer paged KV cache).
"""

from typing import List, Optional, Tuple

import torch


FlashInferInputTuple = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
    str,
    torch.dtype,
    bool,
    bool,
    str,
    Optional[List[str]],
]


def dense_to_paged_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense key/value tensors to a simple paged KV cache layout.

    The output layout matches the expectations of the FlashInfer wrapper used in
    this project, with a single page per sequence and ``kv_layout="NHD"``:

    - ``kv_cache`` has shape ``(num_pages, 2, page_size, num_kv_heads, head_dim)``,
      where the dimension of size ``2`` stores keys and values respectively.
    - ``kv_page_indptr`` and ``kv_page_indices`` describe one page per batch
      element.
    - ``kv_last_page_len`` stores the effective sequence length for each batch.

    Args:
        keys: Dense key tensor of shape
            ``(batch_size, num_kv_heads, seq_len_k, head_dim)``.
        values: Dense value tensor of shape
            ``(batch_size, num_kv_heads, seq_len_k, head_dim)``.
        page_size: Page size for the KV cache representation. Must be greater
            than or equal to ``seq_len_k``.

    Returns:
        Tuple of ``(kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len)``.
    """
    batch_size: int = keys.shape[0]
    num_kv_heads: int = keys.shape[1]
    seq_len_k: int = keys.shape[2]
    head_dim: int = keys.shape[3]

    if page_size < seq_len_k:
        raise ValueError(
            f"page_size must be >= seq_len_k, got page_size={page_size}, seq_len_k={seq_len_k}"
        )

    num_pages: int = batch_size

    kv_cache: torch.Tensor = torch.zeros(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=keys.dtype,
        device=keys.device,
    )

    # Pack keys and values into the paged cache (one page per sequence).
    # keys / values: (b, h, s, d) -> (b, s, h, d) to match (page_size, num_kv_heads, head_dim).
    keys_perm: torch.Tensor = keys.permute(0, 2, 1, 3)
    values_perm: torch.Tensor = values.permute(0, 2, 1, 3)

    kv_cache[:, 0, :seq_len_k, :, :] = keys_perm
    kv_cache[:, 1, :seq_len_k, :, :] = values_perm

    # One page per sequence:
    #   indptr: [0, 1, 2, ..., batch_size]
    #   indices: [0, 1, 2, ..., batch_size-1]
    kv_page_indptr: torch.Tensor = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=keys.device
    )
    kv_page_indices: torch.Tensor = torch.arange(
        0, batch_size, dtype=torch.int32, device=keys.device
    )

    kv_last_page_len: torch.Tensor = torch.full(
        (batch_size,),
        seq_len_k,
        dtype=torch.int32,
        device=keys.device,
    )

    return kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len


def convert_dense_to_flashinfer_inputs(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    data_type: Optional[torch.dtype] = None,
    use_cuda_graph: bool = False,
    use_tensor_cores: bool = False,
    backend: str = "auto",
    jit_args: Optional[List[str]] = None,
) -> FlashInferInputTuple:
    """Convert dense Q/K/V into arguments for FlashInfer custom attention.

    The dense queries are expected to have a single query position
    (``seq_len_q == 1``); the function flattens that dimension to match the
    FlashInfer query layout.

    Args:
        queries: Query tensor of shape
            ``(batch_size, num_qo_heads, 1, head_dim)``.
        keys: Key tensor of shape
            ``(batch_size, num_kv_heads, seq_len_k, head_dim)``.
        values: Value tensor of shape
            ``(batch_size, num_kv_heads, seq_len_k, head_dim)``.
        page_size: Page size to use for the paged KV cache.
        pos_encoding_mode: Positional encoding mode for FlashInfer.
        data_type: Optional data type override for FlashInfer internal
            computation. If ``None``, uses ``queries.dtype``.
        use_cuda_graph: Whether to enable CUDA Graphs.
        use_tensor_cores: Whether to enable tensor core kernels.
        backend: Backend selection string for FlashInfer.
        jit_args: Optional list of JIT arguments.

    Returns:
        Tuple of arguments matching the signature expected by
        ``custom_attention_flashinfer.custom_attention_flashinfer``.
    """
    if queries.shape[2] != 1:
        raise ValueError(
            f"convert_dense_to_flashinfer_inputs expects seq_len_q == 1, "
            f"got queries.shape={queries.shape}"
        )

    batch_size: int = queries.shape[0]
    num_qo_heads: int = queries.shape[1]
    head_dim: int = queries.shape[3]

    # Flatten the single query position: (b, h, 1, d) -> (b, h, d)
    q: torch.Tensor = queries[:, :, 0, :]

    kv_cache: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len = dense_to_paged_kv_cache(
        keys=keys,
        values=values,
        page_size=page_size,
    )

    num_kv_heads: int = keys.shape[1]
    # Ensure we are not silently mixing heads.
    if num_kv_heads != num_qo_heads:
        raise ValueError(
            f"Number of KV heads ({num_kv_heads}) must equal number of QO heads ({num_qo_heads})"
        )

    effective_dtype: torch.dtype = data_type if data_type is not None else queries.dtype

    return (
        q,
        kv_cache,
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode,
        effective_dtype,
        use_cuda_graph,
        use_tensor_cores,
        backend,
        jit_args,
    )






