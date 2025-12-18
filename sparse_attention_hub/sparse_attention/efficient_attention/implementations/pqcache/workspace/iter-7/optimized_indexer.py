"""Optimized implementation of PQCache indexer_next logic - Iteration 7.

BIG CHANGES: Eliminate ALL unnecessary operations and intermediate tensors.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def ultra_fast_score_kernel(
    queries_ptr,
    centroids_ptr,
    codebook_ptr,
    scores_ptr,
    window_mask_val: tl.constexpr,
    b: tl.constexpr,
    h: tl.constexpr,
    n_subvec: tl.constexpr,
    subvec_d: tl.constexpr,
    n_clustered: tl.constexpr,
    window_start: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_subvec: tl.constexpr,
    q_stride_d: tl.constexpr,
    c_stride_b: tl.constexpr,
    c_stride_h: tl.constexpr,
    c_stride_subvec: tl.constexpr,
    c_stride_cent: tl.constexpr,
    c_stride_d: tl.constexpr,
    cb_stride_b: tl.constexpr,
    cb_stride_h: tl.constexpr,
    cb_stride_subvec: tl.constexpr,
    cb_stride_clustered: tl.constexpr,
    s_stride_b: tl.constexpr,
    s_stride_h: tl.constexpr,
    s_stride_clustered: tl.constexpr,
):
    """Ultra-fast score kernel with in-kernel masking."""
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
    
    # Apply window mask directly during computation
    window_mask = key_offsets < window_start
    valid_mask = key_mask & window_mask
    
    scores_accum = tl.zeros([BLOCK_KEYS], dtype=tl.float32)
    
    for subvec_idx in range(n_subvec):
        cb_base: tl.pointer_type(tl.int64) = (
            codebook_ptr + batch_idx * cb_stride_b + head_idx * cb_stride_h + subvec_idx * cb_stride_subvec
        )
        cb_indices = tl.load(cb_base + key_offsets * cb_stride_clustered, mask=key_mask, other=0)
        
        q_base: tl.pointer_type(tl.float32) = (
            queries_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + subvec_idx * q_stride_subvec
        )
        
        c_base: tl.pointer_type(tl.float32) = (
            centroids_ptr + batch_idx * c_stride_b + head_idx * c_stride_h + subvec_idx * c_stride_subvec
        )
        
        for d_start in range(0, subvec_d, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < subvec_d
            
            q_vals = tl.load(q_base + d_offsets * q_stride_d, mask=d_mask, other=0.0)
            
            cent_offsets = c_base + cb_indices[:, None] * c_stride_cent + d_offsets[None, :] * c_stride_d
            c_vals = tl.load(cent_offsets, mask=key_mask[:, None] & d_mask[None, :], other=0.0)
            
            scores_accum += tl.sum(q_vals[None, :] * c_vals, axis=1)
    
    # Mask invalid scores to -inf IN THE KERNEL
    scores_accum = tl.where(valid_mask, scores_accum, window_mask_val)
    
    scores_base: tl.pointer_type(tl.float32) = (
        scores_ptr + batch_idx * s_stride_b + head_idx * s_stride_h
    )
    tl.store(scores_base + key_offsets * s_stride_clustered, scores_accum, mask=key_mask)


@triton.jit
def ultra_fast_index_kernel(
    topk_ptr,  # [b, h, k]
    out_ptr,   # [b, h, total]
    weights_ptr,  # [b, h, sk]
    b: tl.constexpr,
    h: tl.constexpr,
    sk: tl.constexpr,
    sink_size: tl.constexpr,
    window_start: tl.constexpr,
    window_end: tl.constexpr,
    k: tl.constexpr,
    init_offset: tl.constexpr,
    total: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_h: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
):
    """Ultra-fast index generation - all in one kernel."""
    pid: tl.int32 = tl.program_id(0)
    batch_idx: tl.int32 = pid // h
    head_idx: tl.int32 = pid % h
    
    out_base: tl.pointer_type(tl.int64) = out_ptr + batch_idx * out_stride_b + head_idx * out_stride_h
    w_base: tl.pointer_type(tl.float32) = weights_ptr + batch_idx * w_stride_b + head_idx * w_stride_h
    
    # Write sink indices [0, 1, 2, ..., sink_size-1]
    for i in range(sink_size):
        tl.store(out_base + i, i)
        tl.store(w_base + i, 1.0)
    
    # Load and write heavy indices (already sorted from topk with sorted=True)
    topk_base: tl.pointer_type(tl.int64) = topk_ptr + batch_idx * topk_stride_b + head_idx * topk_stride_h
    for i in range(k):
        idx: tl.int64 = tl.load(topk_base + i)
        adjusted: tl.int64 = idx + init_offset
        tl.store(out_base + sink_size + i, adjusted)
        tl.store(w_base + adjusted, 1.0)
    
    # Write window indices
    window_size: tl.int32 = window_end - window_start
    for i in range(window_size):
        idx: tl.int64 = window_start + i
        tl.store(out_base + sink_size + k + i, idx)
        tl.store(w_base + idx, 1.0)


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
    """BIG CHANGES - Iteration 7: Eliminate everything unnecessary."""
    b, h, sq, d = query.shape
    _, h_kv, sk, _ = key.shape
    
    cent_cnt: int = 2**pq_bits
    total_needed: int = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        sparse_list = torch.arange(sk, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        sparse_len = torch.full((b, h), sk, device=query.device, dtype=torch.long)
        weight_list = torch.ones((b, h, sk), device=query.device, dtype=weight_list_dtype)
        return (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi)
    
    # Key quantization
    cached_codebook = pq_codebook
    cached_num_keys = cached_codebook.shape[1] if cached_codebook is not None else 0
    current_quantized_keys = sk - init_offset
    
    if current_quantized_keys > cached_num_keys:
        new_start = init_offset + cached_num_keys
        new_keys = key[:, :, new_start:, :]
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        n_subvec = pq_group_factor
        base_subvec_d = head_dim // n_subvec
        new_keys_reshaped = new_keys.reshape(bsz, kv_heads, n_new, n_subvec, base_subvec_d).transpose(2, 3)
        
        if metric == "ip":
            new_keys_flat = new_keys_reshaped.reshape(-1, n_new, base_subvec_d)
            phi_expanded = pq_ip2l2_phi.unsqueeze(1).unsqueeze(2)
            new_keys_squared_norm = torch.sum(new_keys_flat**2, dim=-1, keepdim=True)
            phi_minus_norm = torch.clamp(phi_expanded - new_keys_squared_norm, min=0.0)
            augment_dim = torch.sqrt(phi_minus_norm)
            new_keys_flat_aug = torch.cat([new_keys_flat, augment_dim], dim=-1)
            new_keys_reshaped = new_keys_flat_aug.reshape(bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1)
        
        new_keys_exp = new_keys_reshaped.unsqueeze(4)
        centroids_exp = pq_centroids.unsqueeze(3)
        distances = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)
        new_codes = torch.argmin(distances, dim=-1).permute(0, 3, 1, 2)
        codebook = new_codes if cached_codebook is None else torch.cat([cached_codebook, new_codes], dim=1)
    else:
        codebook = cached_codebook
    
    # Ultra-optimized path
    n_clustered = codebook.shape[1]
    n_subvec = pq_group_factor
    subvec_d = d // n_subvec
    num_key_value_groups = h // h_kv
    
    queries_reshaped = query.reshape(b, h, n_subvec, subvec_d)
    
    if num_key_value_groups == 1:
        repeat_centroids = pq_centroids
        repeat_codebook = codebook.permute(0, 2, 3, 1)
    else:
        repeat_centroids = pq_centroids[:, :, None, :, :, :].expand(b, h_kv, num_key_value_groups, n_subvec, cent_cnt, -1).reshape(b, h_kv * num_key_value_groups, n_subvec, cent_cnt, -1)
        codebook_permuted = codebook.permute(0, 2, 3, 1)
        repeat_codebook = codebook_permuted.unsqueeze(2).expand(b, h_kv, num_key_value_groups, n_subvec, n_clustered).reshape(b, h, n_subvec, n_clustered)
    
    repeat_centroids = repeat_centroids[..., :subvec_d]
    
    actual_sink_size = min(sink_size, sk)
    actual_window_size = min(window_size, sk)
    actual_heavy_size = min(heavy_size, n_clustered)
    window_start_in_quantized = max(0, sk - actual_window_size - init_offset)
    
    if actual_heavy_size > 0:
        # Compute scores with in-kernel masking
        scores = torch.empty((b, h, n_clustered), device=query.device, dtype=torch.float32)
        
        q_strides = queries_reshaped.stride()
        c_strides = repeat_centroids.stride()
        cb_strides = repeat_codebook.stride()
        s_strides = scores.stride()
        
        grid = (b * h * triton.cdiv(n_clustered, 256),)
        
        ultra_fast_score_kernel[grid](
            queries_reshaped, repeat_centroids, repeat_codebook, scores,
            float('-inf'),  # window_mask_val
            b, h, n_subvec, subvec_d, n_clustered, window_start_in_quantized,
            q_strides[0], q_strides[1], q_strides[2], q_strides[3],
            c_strides[0], c_strides[1], c_strides[2], c_strides[3], c_strides[4],
            cb_strides[0], cb_strides[1], cb_strides[2], cb_strides[3],
            s_strides[0], s_strides[1], s_strides[2],
        )
        
        # Top-K without sort (faster)
        _, topk_indices = torch.topk(scores, k=actual_heavy_size, dim=-1, largest=True, sorted=False)
        
        # Generate sparse list - concatenate and sort
        total_attended = actual_sink_size + actual_heavy_size + actual_window_size
        sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
        
        # Direct assignment
        sparse_list[:, :, :actual_sink_size] = torch.arange(actual_sink_size, device=query.device).view(1, 1, -1)
        sparse_list[:, :, actual_sink_size:actual_sink_size+actual_heavy_size] = topk_indices + init_offset
        window_start = sk - actual_window_size
        sparse_list[:, :, actual_sink_size+actual_heavy_size:] = torch.arange(window_start, sk, device=query.device).view(1, 1, -1)
        
        # Single sort
        sparse_list, _ = torch.sort(sparse_list, dim=2)
        
        # Weights
        weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
        batch_indices = torch.arange(b, device=query.device).view(b, 1, 1)
        head_indices = torch.arange(h, device=query.device).view(1, h, 1)
        weight_list[batch_indices, head_indices, sparse_list] = 1.0
    else:
        total_attended = actual_sink_size + actual_window_size
        sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
        sparse_list[:, :, :actual_sink_size] = torch.arange(actual_sink_size, device=query.device).view(1, 1, -1)
        window_start = sk - actual_window_size
        sparse_list[:, :, actual_sink_size:] = torch.arange(window_start, sk, device=query.device).view(1, 1, -1)
        
        weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
        batch_indices = torch.arange(b, device=query.device).view(b, 1, 1)
        head_indices = torch.arange(h, device=query.device).view(1, h, 1)
        weight_list[batch_indices, head_indices, sparse_list] = 1.0
    
    sparse_len = torch.full((b, h), total_attended, device=query.device, dtype=torch.long)
    
    return (sparse_list, sparse_len, weight_list, pq_centroids, codebook, pq_ip2l2_phi)
