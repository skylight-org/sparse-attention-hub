"""Optimized implementation of PQCache indexer_next logic - Iteration 5.

Ultra-aggressive optimization targeting 200 μs latency.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_KEYS": 256, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_KEYS": 128, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_KEYS": 256, "BLOCK_D": 32}, num_warps=8),
    ],
    key=["n_clustered", "subvec_d"],
)
@triton.jit
def pq_score_kernel_v5(
    queries_ptr,
    centroids_ptr,
    codebook_ptr,
    scores_ptr,
    b: tl.constexpr,
    h: tl.constexpr,
    n_subvec: tl.constexpr,
    subvec_d: tl.constexpr,
    n_clustered: tl.constexpr,
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
    BLOCK_KEYS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Ultra-optimized kernel v5 - specialized for sq=1."""
    pid: tl.int32 = tl.program_id(0)
    
    num_key_blocks: tl.int32 = tl.cdiv(n_clustered, BLOCK_KEYS)
    total_queries: tl.int32 = b * h
    
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
            c_vals = tl.load(
                cent_offsets,
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            
            dot_products = tl.sum(q_vals[None, :] * c_vals, axis=1)
            scores_accum += dot_products
    
    scores_base_ptr: tl.pointer_type(tl.float32) = (
        scores_ptr
        + batch_idx * s_stride_b
        + head_idx * s_stride_h
    )
    tl.store(
        scores_base_ptr + key_offsets * s_stride_clustered,
        scores_accum,
        mask=key_mask,
    )


@triton.jit
def generate_sorted_sparse_list_kernel(
    sparse_list_ptr,
    weight_list_ptr,
    topk_indices_ptr,
    b: tl.constexpr,
    h: tl.constexpr,
    sk: tl.constexpr,
    sink_size: tl.constexpr,
    window_size: tl.constexpr,
    heavy_size: tl.constexpr,
    init_offset: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_h: tl.constexpr,
    topk_stride_k: tl.constexpr,
    sparse_stride_b: tl.constexpr,
    sparse_stride_h: tl.constexpr,
    sparse_stride_attended: tl.constexpr,
    weight_stride_b: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_sk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate sparse list with indices already sorted."""
    pid: tl.int32 = tl.program_id(0)
    batch_idx: tl.int32 = pid // h
    head_idx: tl.int32 = pid % h
    
    actual_sink_size: tl.int32 = tl.minimum(sink_size, sk)
    actual_window_size: tl.int32 = tl.minimum(window_size, sk)
    window_start: tl.int32 = sk - actual_window_size
    
    sparse_base: tl.pointer_type(tl.int64) = (
        sparse_list_ptr
        + batch_idx * sparse_stride_b
        + head_idx * sparse_stride_h
    )
    weight_base: tl.pointer_type(tl.float32) = (
        weight_list_ptr
        + batch_idx * weight_stride_b
        + head_idx * weight_stride_h
    )
    topk_base: tl.pointer_type(tl.int64) = (
        topk_indices_ptr
        + batch_idx * topk_stride_b
        + head_idx * topk_stride_h
    )
    
    # Load all heavy indices first
    heavy_indices = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for i in range(heavy_size):
        topk_idx: tl.int64 = tl.load(topk_base + i * topk_stride_k)
        heavy_indices = tl.where(i == tl.arange(0, BLOCK_SIZE), topk_idx + init_offset, heavy_indices)
    
    # Sort heavy indices using simple bubble sort (only for small arrays)
    # For actual implementation, we'll rely on the fact that topk often returns somewhat sorted
    
    # Merge sink, heavy, and window indices in sorted order
    out_idx: tl.int32 = 0
    sink_idx: tl.int32 = 0
    heavy_idx: tl.int32 = 0
    window_idx: tl.int32 = window_start
    
    # Three-way merge to maintain sorted order
    for _ in range(actual_sink_size + heavy_size + actual_window_size):
        # Determine which index to write next
        use_sink: tl.int1 = sink_idx < actual_sink_size
        use_window: tl.int1 = window_idx < sk
        use_heavy: tl.int1 = heavy_idx < heavy_size
        
        sink_val: tl.int64 = sink_idx
        window_val: tl.int64 = window_idx
        heavy_val: tl.int64 = tl.load(topk_base + heavy_idx * topk_stride_k) + init_offset if use_heavy else sk
        
        # Find minimum
        min_val: tl.int64 = sink_val
        use_which: tl.int32 = 0  # 0=sink, 1=heavy, 2=window
        
        if use_sink and (not use_heavy or sink_val < heavy_val) and (not use_window or sink_val < window_val):
            min_val = sink_val
            use_which = 0
            sink_idx += 1
        elif use_heavy and (not use_window or heavy_val < window_val):
            min_val = heavy_val
            use_which = 1
            heavy_idx += 1
        elif use_window:
            min_val = window_val
            use_which = 2
            window_idx += 1
        
        if out_idx < actual_sink_size + heavy_size + actual_window_size:
            tl.store(sparse_base + out_idx * sparse_stride_attended, min_val)
            tl.store(weight_base + min_val * weight_stride_sk, 1.0)
            out_idx += 1


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
    """Ultra-optimized indexer - Iteration 5 targeting 200 μs."""
    b, h, sq, d = query.shape
    b_key, h_kv, sk, d_key = key.shape
    
    # Fast path checks
    cent_cnt: int = 2**pq_bits
    total_needed: int = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        sparse_list: torch.Tensor = torch.arange(sk, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        sparse_len: torch.Tensor = torch.full((b, h), sk, device=query.device, dtype=torch.long)
        weight_list: torch.Tensor = torch.ones((b, h, sk), device=query.device, dtype=weight_list_dtype)
        return (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi)
    
    # Handle incremental keys (keep as is - not the bottleneck)
    cached_codebook: torch.Tensor = pq_codebook
    cached_num_keys: int = cached_codebook.shape[1] if cached_codebook is not None else 0
    current_quantized_keys: int = sk - init_offset
    
    if current_quantized_keys > cached_num_keys:
        new_start: int = init_offset + cached_num_keys
        new_keys: torch.Tensor = key[:, :, new_start:, :]
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        
        n_subvec: int = pq_group_factor
        base_subvec_d: int = head_dim // n_subvec
        
        new_keys_reshaped: torch.Tensor = new_keys.reshape(bsz, kv_heads, n_new, n_subvec, base_subvec_d).transpose(2, 3)
        
        if metric == "ip":
            new_keys_flat: torch.Tensor = new_keys_reshaped.reshape(-1, n_new, base_subvec_d)
            phi_expanded: torch.Tensor = pq_ip2l2_phi.unsqueeze(1).unsqueeze(2)
            new_keys_squared_norm: torch.Tensor = torch.sum(new_keys_flat**2, dim=-1, keepdim=True)
            phi_minus_norm: torch.Tensor = phi_expanded - new_keys_squared_norm
            phi_minus_norm_clamped: torch.Tensor = torch.clamp(phi_minus_norm, min=0.0)
            augment_dim: torch.Tensor = torch.sqrt(phi_minus_norm_clamped)
            new_keys_flat_aug: torch.Tensor = torch.cat([new_keys_flat, augment_dim], dim=-1)
            new_keys_reshaped = new_keys_flat_aug.reshape(bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1)
        
        new_keys_exp: torch.Tensor = new_keys_reshaped.unsqueeze(4)
        centroids_exp: torch.Tensor = pq_centroids.unsqueeze(3)
        distances: torch.Tensor = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)
        new_codes: torch.Tensor = torch.argmin(distances, dim=-1).permute(0, 3, 1, 2)
        
        codebook: torch.Tensor = new_codes if cached_codebook is None else torch.cat([cached_codebook, new_codes], dim=1)
    else:
        codebook = cached_codebook
    
    # Score computation - optimized for sq=1
    n_clustered: int = codebook.shape[1]
    n_subvec: int = pq_group_factor
    subvec_d: int = d // n_subvec
    num_key_value_groups: int = h // h_kv
    
    # Squeeze out sq dimension for faster processing (sq=1)
    queries_reshaped: torch.Tensor = query.reshape(b, h, n_subvec, subvec_d)
    
    # Prepare centroids and codebook
    if num_key_value_groups == 1:
        repeat_centroids: torch.Tensor = pq_centroids
        repeat_codebook: torch.Tensor = codebook.permute(0, 2, 3, 1)
    else:
        repeat_centroids = pq_centroids[:, :, None, :, :, :].expand(b, h_kv, num_key_value_groups, n_subvec, cent_cnt, -1).reshape(b, h_kv * num_key_value_groups, n_subvec, cent_cnt, -1)
        codebook_permuted = codebook.permute(0, 2, 3, 1)
        repeat_codebook = codebook_permuted.unsqueeze(2).expand(b, h_kv, num_key_value_groups, n_subvec, n_clustered).reshape(b, h, n_subvec, n_clustered)
    
    repeat_centroids = repeat_centroids[..., :subvec_d]
    
    # Allocate output for scores
    scores: torch.Tensor = torch.empty((b, h, n_clustered), device=query.device, dtype=torch.float32)
    
    q_strides = queries_reshaped.stride()
    c_strides = repeat_centroids.stride()
    cb_strides = repeat_codebook.stride()
    s_strides = scores.stride()
    
    def grid(meta):
        return (b * h * triton.cdiv(n_clustered, meta["BLOCK_KEYS"]),)
    
    # Launch optimized kernel
    pq_score_kernel_v5[grid](
        queries_reshaped, repeat_centroids, repeat_codebook, scores,
        b=b, h=h, n_subvec=n_subvec, subvec_d=subvec_d, n_clustered=n_clustered,
        q_stride_b=q_strides[0], q_stride_h=q_strides[1], q_stride_subvec=q_strides[2], q_stride_d=q_strides[3],
        c_stride_b=c_strides[0], c_stride_h=c_strides[1], c_stride_subvec=c_strides[2], c_stride_cent=c_strides[3], c_stride_d=c_strides[4],
        cb_stride_b=cb_strides[0], cb_stride_h=cb_strides[1], cb_stride_subvec=cb_strides[2], cb_stride_clustered=cb_strides[3],
        s_stride_b=s_strides[0], s_stride_h=s_strides[1], s_stride_clustered=s_strides[2],
    )
    
    # Mask in-place
    actual_sink_size: int = min(sink_size, sk)
    actual_window_size: int = min(window_size, sk)
    window_start_in_quantized: int = max(0, sk - actual_window_size - init_offset)
    
    if window_start_in_quantized < n_clustered:
        scores[:, :, window_start_in_quantized:] = float('-inf')
    
    # Top-K selection
    actual_heavy_size: int = min(heavy_size, n_clustered)
    
    if actual_heavy_size > 0:
        # Use faster top-K with sorted=False (we'll sort ourselves if needed)
        _, topk_indices = torch.topk(scores, k=actual_heavy_size, dim=-1, largest=True, sorted=False)
        
        # Sort topk_indices to enable merging
        topk_indices, _ = torch.sort(topk_indices, dim=-1)
    else:
        topk_indices = torch.empty((b, h, 0), device=query.device, dtype=torch.long)
    
    # Generate sparse list - WITHOUT final sort operation
    total_attended: int = actual_sink_size + actual_heavy_size + actual_window_size
    sparse_list: torch.Tensor = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
    weight_list: torch.Tensor = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
    
    if actual_heavy_size > 0:
        # Simple concatenation without sort - indices should be roughly in order
        # Sink indices
        sink_indices: torch.Tensor = torch.arange(actual_sink_size, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        
        # Adjust heavy indices
        topk_adjusted: torch.Tensor = topk_indices + init_offset
        
        # Window indices
        window_start: int = sk - actual_window_size
        window_indices: torch.Tensor = torch.arange(window_start, sk, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        
        # Concatenate and sort in one go (still need sort for correctness, but sorted inputs make it faster)
        sparse_list = torch.cat([sink_indices, topk_adjusted, window_indices], dim=2)
        sparse_list, _ = torch.sort(sparse_list, dim=2)
        
        # Set weights
        batch_indices: torch.Tensor = torch.arange(b, device=query.device).view(b, 1, 1)
        head_indices: torch.Tensor = torch.arange(h, device=query.device).view(1, h, 1)
        weight_list[batch_indices, head_indices, sparse_list] = 1.0
    else:
        sink_indices = torch.arange(actual_sink_size, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        window_start = sk - actual_window_size
        window_indices = torch.arange(window_start, sk, device=query.device, dtype=torch.long).view(1, 1, -1).expand(b, h, -1)
        sparse_list = torch.cat([sink_indices, window_indices], dim=2)
        batch_indices = torch.arange(b, device=query.device).view(b, 1, 1)
        head_indices = torch.arange(h, device=query.device).view(1, h, 1)
        weight_list[batch_indices, head_indices, sparse_list] = 1.0
    
    sparse_len: torch.Tensor = torch.full((b, h), total_attended, device=query.device, dtype=torch.long)
    
    return (sparse_list, sparse_len, weight_list, pq_centroids, codebook, pq_ip2l2_phi)

