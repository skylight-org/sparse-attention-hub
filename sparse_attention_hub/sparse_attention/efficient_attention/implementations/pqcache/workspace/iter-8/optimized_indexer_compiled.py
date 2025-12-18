"""Iteration 8: torch.compile optimization for CPU overhead reduction.

This is the EASIEST approach to reduce CPU overhead with minimal code changes.

Expected Performance:
- Iteration 6: 518 μs (208 μs CUDA + 310 μs overhead)
- Iteration 8: 450-480 μs (208 μs CUDA + 240-270 μs overhead)
- Improvement: 40-70 μs (10-15% reduction)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def pq_score_kernel_v6(
    queries_ptr,
    centroids_ptr,
    codebook_ptr,
    scores_ptr,
    b,
    h,
    n_subvec,
    subvec_d,
    n_clustered,
    q_stride_b,
    q_stride_h,
    q_stride_subvec,
    q_stride_d,
    c_stride_b,
    c_stride_h,
    c_stride_subvec,
    c_stride_cent,
    c_stride_d,
    cb_stride_b,
    cb_stride_h,
    cb_stride_subvec,
    cb_stride_clustered,
    s_stride_b,
    s_stride_h,
    s_stride_clustered,
):
    """Optimized PQ score computation kernel from iteration 6."""
    BLOCK_KEYS: tl.constexpr = 256
    BLOCK_D: tl.constexpr = 64
    
    pid: tl.int32 = tl.program_id(0)
    num_key_blocks: tl.int32 = tl.cdiv(n_clustered, BLOCK_KEYS)
    query_id: tl.int32 = pid // num_key_blocks
    key_block_id: tl.int32 = pid % num_key_blocks
    
    batch_idx: tl.int32 = query_id // h
    head_idx: tl.int32 = query_id % h
    
    key_start: tl.int32 = key_block_id * BLOCK_KEYS
    key_offsets = key_start + tl.arange(0, BLOCK_KEYS)
    key_mask = key_offsets < n_clustered
    
    scores_accum = tl.zeros([BLOCK_KEYS], dtype=tl.float32)
    
    for subvec_idx in range(n_subvec):
        cb_base_ptr: tl.pointer_type(tl.int64) = (
            codebook_ptr
            + batch_idx * cb_stride_b
            + head_idx * cb_stride_h
            + subvec_idx * cb_stride_subvec
        )
        cb_indices = tl.load(
            cb_base_ptr + key_offsets * cb_stride_clustered,
            mask=key_mask,
            other=0,
        )
        
        q_base_ptr: tl.pointer_type(tl.float32) = (
            queries_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + subvec_idx * q_stride_subvec
        )
        
        c_base_ptr: tl.pointer_type(tl.float32) = (
            centroids_ptr
            + batch_idx * c_stride_b
            + head_idx * c_stride_h
            + subvec_idx * c_stride_subvec
        )
        
        for d_start in range(0, subvec_d, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < subvec_d
            
            q_vals = tl.load(
                q_base_ptr + d_offsets * q_stride_d,
                mask=d_mask,
                other=0.0,
            )
            
            cent_offsets = (
                c_base_ptr
                + cb_indices[:, None] * c_stride_cent
                + d_offsets[None, :] * c_stride_d
            )
            c_vals = tl.load(cent_offsets, mask=key_mask[:, None] & d_mask[None, :], other=0.0)
            
            scores_accum += tl.sum(q_vals[None, :] * c_vals, axis=1)
            
    scores_base_ptr: tl.pointer_type(tl.float32) = (
        scores_ptr
        + batch_idx * s_stride_b
        + head_idx * s_stride_h
    )
    tl.store(scores_base_ptr + key_offsets * s_stride_clustered, scores_accum, mask=key_mask)


def __indexer_next(
    query: torch.Tensor,
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
    """Iteration 6 implementation - base for torch.compile."""
    
    b, h, sq, d = query.shape
    sk = key.shape[2]
    
    cent_cnt = 2**pq_bits
    total_needed = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        sparse_list = torch.arange(sk, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        sparse_len = torch.full((b, h), sk, device=query.device, dtype=torch.long)
        weight_list = torch.ones((b, h, sk), device=query.device, dtype=weight_list_dtype)
        return sparse_list, sparse_len, weight_list, query, key, None
    
    n_subvec = pq_group_factor
    subvec_d = d // n_subvec
    n_clustered = pq_codebook.shape[1]
    
    queries = query.reshape(b, h, n_subvec, subvec_d)
    repeat_centroids = pq_centroids[:, :, :, :, :subvec_d]
    repeat_codebook = pq_codebook.permute(0, 2, 3, 1)
    
    scores = torch.zeros((b, h, n_clustered), device=query.device, dtype=torch.float32)
    
    grid = lambda meta: (b * h * triton.cdiv(n_clustered, 256),)
    
    pq_score_kernel_v6[grid](
        queries,
        repeat_centroids,
        repeat_codebook,
        scores,
        b,
        h,
        n_subvec,
        subvec_d,
        n_clustered,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        queries.stride(3),
        repeat_centroids.stride(0),
        repeat_centroids.stride(1),
        repeat_centroids.stride(2),
        repeat_centroids.stride(3),
        repeat_centroids.stride(4),
        repeat_codebook.stride(0),
        repeat_codebook.stride(1),
        repeat_codebook.stride(2),
        repeat_codebook.stride(3),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
    )
    
    actual_window_size = min(window_size, sk)
    window_start = max(0, sk - actual_window_size - init_offset)
    
    if window_start < n_clustered:
        scores[:, :, window_start:] = float('-inf')
    
    actual_heavy_size = min(heavy_size, n_clustered)
    topk_indices = torch.topk(scores, actual_heavy_size, dim=-1, largest=True, sorted=False).indices
    
    actual_sink_size = min(sink_size, sk)
    total_attended = actual_sink_size + actual_heavy_size + actual_window_size
    
    sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
    
    sink_indices = torch.arange(actual_sink_size, device=query.device, dtype=torch.long)
    sparse_list[:, :, :actual_sink_size] = sink_indices.view(1, 1, -1)
    
    heavy_indices = topk_indices + init_offset
    sparse_list[:, :, actual_sink_size:actual_sink_size+actual_heavy_size] = heavy_indices
    
    window_start_idx = sk - actual_window_size
    window_indices = torch.arange(window_start_idx, sk, device=query.device, dtype=torch.long)
    sparse_list[:, :, actual_sink_size+actual_heavy_size:] = window_indices.view(1, 1, -1)
    
    sparse_list = torch.sort(sparse_list, dim=-1).values
    
    sparse_len = torch.full((b, h), total_attended, device=query.device, dtype=torch.long)
    
    weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
    batch_indices = torch.arange(b, device=query.device, dtype=torch.long).view(b, 1, 1)
    head_indices = torch.arange(h, device=query.device, dtype=torch.long).view(1, h, 1)
    weight_list[batch_indices, head_indices, sparse_list] = 1.0
    
    return sparse_list, sparse_len, weight_list, query, key, None


# Create compiled version
__indexer_next_compiled = torch.compile(
    __indexer_next,
    mode="reduce-overhead",  # Optimize for latency
    fullgraph=False,          # Allow fallback for Triton kernels
)


# Export both versions
__all__ = ['__indexer_next', '__indexer_next_compiled']

