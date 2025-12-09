"""FlashInfer-based backend for custom attention.

This module provides a thin wrapper around
``flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`` that is intended to be
used as the backend stage of custom attention when working with paged KV
cache layouts.
"""

from typing import List, Optional, Tuple

import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def custom_indexer_flashinfer(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    data_type: torch.dtype = torch.float16,
    use_cuda_graph: bool = False,
    use_tensor_cores: bool = False,
    backend: str = "auto",
    jit_args: Optional[List[str]] = None,
) -> Tuple[
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
]:
    """No-op indexer for FlashInfer backend that forwards all inputs unchanged.

    This function is a placeholder counterpart to :func:`custom_backend_flashinfer`
    and can be used where an indexer stage is expected. It simply returns all of
    its inputs as a tuple without modification.

    Returns:
        Tuple containing all input arguments in the same order.
    """
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
        data_type,
        use_cuda_graph,
        use_tensor_cores,
        backend,
        jit_args,
    )


def custom_backend_flashinfer(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    data_type: torch.dtype = torch.float16,
    use_cuda_graph: bool = False,
    use_tensor_cores: bool = False,
    backend: str = "auto",
    jit_args: Optional[List[str]] = None,
) -> torch.Tensor:
    """Run FlashInfer paged-KV decode backend and return attention output.

    This function encapsulates the common pattern of:

    1. Allocating a workspace buffer.
    2. Constructing a :class:`BatchDecodeWithPagedKVCacheWrapper`.
    3. Calling ``plan`` with the paged KV metadata and model hyperparameters.
    4. Calling ``run`` with the query tensor and KV cache to obtain the output.

    Args:
        q: Query tensor of shape ``(batch_size, num_qo_heads, head_dim)``.
        kv_cache: KV cache tensor of shape
            ``(num_pages, 2, page_size, num_kv_heads, head_dim)``.
        kv_page_indptr: Pointer tensor for paged KV cache.
        kv_page_indices: Indices tensor for paged KV cache.
        kv_last_page_len: Tensor of last-page lengths for each sequence.
        num_qo_heads: Number of query/output heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimension of each attention head.
        page_size: Page size used in the paged KV cache.
        pos_encoding_mode: Position encoding mode used by FlashInfer.
        data_type: Data type for internal computation.
        use_cuda_graph: Whether to enable CUDA Graphs in the wrapper.
        use_tensor_cores: Whether to enable tensor core kernels.
        backend: Backend selection string for FlashInfer (e.g. ``\"auto\"``).
        jit_args: Optional list of JIT arguments forwarded to the wrapper.

    Returns:
        Attention output tensor of shape ``(batch_size, num_qo_heads, head_dim)``.
    """
    # Allocate workspace buffer; size can be tuned based on model/workload.
    workspace_size: int = 128 * 1024 * 1024  # 128 MB
    workspace_buffer: torch.Tensor = torch.empty(
        workspace_size, dtype=torch.uint8, device=q.device
    )

    wrapper: BatchDecodeWithPagedKVCacheWrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_cuda_graph=use_cuda_graph,
        use_tensor_cores=use_tensor_cores,
        paged_kv_indptr_buffer=kv_page_indptr,
        paged_kv_indices_buffer=kv_page_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len,
        backend=backend,
        jit_args=jit_args,
    )

    wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=data_type,
    )

    output: torch.Tensor = wrapper.run(q, kv_cache)
    return output


def custom_attention_flashinfer(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    data_type: torch.dtype = torch.float16,
    use_cuda_graph: bool = False,
    use_tensor_cores: bool = False,
    backend: str = "auto",
    jit_args: Optional[List[str]] = None,
) -> torch.Tensor:
    """End-to-end FlashInfer attention using indexer and backend helpers.

    This function mirrors the composition pattern used in
    :mod:`custom_attention_hub`: it first calls
    :func:`custom_indexer_flashinfer` (currently a no-op) and then forwards the
    returned values into :func:`custom_backend_flashinfer` to produce the final
    attention output.

    Args:
        q: Query tensor of shape ``(batch_size, num_qo_heads, head_dim)``.
        kv_cache: KV cache tensor of shape
            ``(num_pages, 2, page_size, num_kv_heads, head_dim)``.
        kv_page_indptr: Pointer tensor for paged KV cache.
        kv_page_indices: Indices tensor for paged KV cache.
        kv_last_page_len: Tensor of last-page lengths for each sequence.
        num_qo_heads: Number of query/output heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimension of each attention head.
        page_size: Page size used in the paged KV cache.
        pos_encoding_mode: Position encoding mode used by FlashInfer.
        data_type: Data type for internal computation.
        use_cuda_graph: Whether to enable CUDA Graphs in the wrapper.
        use_tensor_cores: Whether to enable tensor core kernels.
        backend: Backend selection string for FlashInfer (e.g. ``\"auto\"``).
        jit_args: Optional list of JIT arguments forwarded to the wrapper.

    Returns:
        Attention output tensor of shape ``(batch_size, num_qo_heads, head_dim)``.
    """
    (
        q_i,
        kv_cache_i,
        kv_page_indptr_i,
        kv_page_indices_i,
        kv_last_page_len_i,
        num_qo_heads_i,
        num_kv_heads_i,
        head_dim_i,
        page_size_i,
        pos_encoding_mode_i,
        data_type_i,
        use_cuda_graph_i,
        use_tensor_cores_i,
        backend_i,
        jit_args_i,
    ) = custom_indexer_flashinfer(
        q=q,
        kv_cache=kv_cache,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=data_type,
        use_cuda_graph=use_cuda_graph,
        use_tensor_cores=use_tensor_cores,
        backend=backend,
        jit_args=jit_args,
    )

    output: torch.Tensor = custom_backend_flashinfer(
        q=q_i,
        kv_cache=kv_cache_i,
        kv_page_indptr=kv_page_indptr_i,
        kv_page_indices=kv_page_indices_i,
        kv_last_page_len=kv_last_page_len_i,
        num_qo_heads=num_qo_heads_i,
        num_kv_heads=num_kv_heads_i,
        head_dim=head_dim_i,
        page_size=page_size_i,
        pos_encoding_mode=pos_encoding_mode_i,
        data_type=data_type_i,
        use_cuda_graph=use_cuda_graph_i,
        use_tensor_cores=use_tensor_cores_i,
        backend=backend_i,
        jit_args=jit_args_i,
    )

    return output

