"""Optimized implementation of PQCache indexer_next logic - Iteration 3.

This module implements a more memory-efficient approach by reducing intermediate
tensor sizes and optimizing the critical path.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_KEYS": 128, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_KEYS": 256, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_KEYS": 128, "BLOCK_D": 32}, num_warps=4),
        triton.Config({"BLOCK_KEYS": 64, "BLOCK_D": 64}, num_warps=4),
    ],
    key=["n_clustered", "subvec_d"],
)
@triton.jit
def pq_score_kernel_v3(
    # Input pointers
    queries_ptr,  # [b, h, sq, n_subvec, subvec_d]
    centroids_ptr,  # [b, h, n_subvec, cent_cnt, subvec_d]
    codebook_ptr,  # [b, h, n_subvec, n_clustered]
    # Output pointer
    scores_ptr,  # [b, h, sq, n_clustered]
    # Dimensions
    b: tl.constexpr,
    h: tl.constexpr,
    sq: tl.constexpr,
    n_subvec: tl.constexpr,
    subvec_d: tl.constexpr,
    cent_cnt: tl.constexpr,
    n_clustered: tl.constexpr,
    # Strides for queries
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_sq: tl.constexpr,
    q_stride_subvec: tl.constexpr,
    q_stride_d: tl.constexpr,
    # Strides for centroids
    c_stride_b: tl.constexpr,
    c_stride_h: tl.constexpr,
    c_stride_subvec: tl.constexpr,
    c_stride_cent: tl.constexpr,
    c_stride_d: tl.constexpr,
    # Strides for codebook
    cb_stride_b: tl.constexpr,
    cb_stride_h: tl.constexpr,
    cb_stride_subvec: tl.constexpr,
    cb_stride_clustered: tl.constexpr,
    # Strides for scores
    s_stride_b: tl.constexpr,
    s_stride_h: tl.constexpr,
    s_stride_sq: tl.constexpr,
    s_stride_clustered: tl.constexpr,
    # Block sizes (autotuned)
    BLOCK_KEYS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Optimized Triton kernel v3 with vectorized dimension processing.
    
    Key improvements:
    - Vectorized loads across dimension (BLOCK_D)
    - Better register usage
    - Optimized for memory bandwidth
    """
    # Get program ID
    pid: tl.int32 = tl.program_id(0)
    
    # Compute which (b, h, sq, key_block) this program handles
    num_key_blocks: tl.int32 = tl.cdiv(n_clustered, BLOCK_KEYS)
    total_queries: tl.int32 = b * h * sq
    
    query_id: tl.int32 = pid // num_key_blocks
    key_block_id: tl.int32 = pid % num_key_blocks
    
    # Extract b, h, sq from query_id
    batch_idx: tl.int32 = query_id // (h * sq)
    rem: tl.int32 = query_id % (h * sq)
    head_idx: tl.int32 = rem // sq
    seq_idx: tl.int32 = rem % sq
    
    # Compute key range for this block
    key_start: tl.int32 = key_block_id * BLOCK_KEYS
    key_offsets = key_start + tl.arange(0, BLOCK_KEYS)
    key_mask = key_offsets < n_clustered
    
    # Initialize score accumulator
    scores_accum = tl.zeros([BLOCK_KEYS], dtype=tl.float32)
    
    # Loop over subvectors
    for subvec_idx in range(n_subvec):
        # Load codebook indices for this subvector: [BLOCK_KEYS]
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
        
        # Base pointers for query and centroids
        q_base_ptr: tl.pointer_type(tl.float32) = (
            queries_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + seq_idx * q_stride_sq
            + subvec_idx * q_stride_subvec
        )
        
        c_base_ptr: tl.pointer_type(tl.float32) = (
            centroids_ptr
            + batch_idx * c_stride_b
            + head_idx * c_stride_h
            + subvec_idx * c_stride_subvec
        )
        
        # Process dimensions in blocks for better vectorization
        for d_start in range(0, subvec_d, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < subvec_d
            
            # Load query values: [BLOCK_D]
            q_vals = tl.load(
                q_base_ptr + d_offsets * q_stride_d,
                mask=d_mask,
                other=0.0,
            )
            
            # Load centroid values for all keys: [BLOCK_KEYS, BLOCK_D]
            # Create 2D offsets for centroid loading
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
            
            # Compute dot products: [BLOCK_KEYS]
            # q_vals: [BLOCK_D], c_vals: [BLOCK_KEYS, BLOCK_D]
            # Result: [BLOCK_KEYS]
            dot_products = tl.sum(q_vals[None, :] * c_vals, axis=1)
            scores_accum += dot_products
    
    # Store results
    scores_base_ptr: tl.pointer_type(tl.float32) = (
        scores_ptr
        + batch_idx * s_stride_b
        + head_idx * s_stride_h
        + seq_idx * s_stride_sq
    )
    tl.store(
        scores_base_ptr + key_offsets * s_stride_clustered,
        scores_accum,
        mask=key_mask,
    )


def compute_pq_scores_triton_v3(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute PQ-based attention scores using optimized Triton kernel v3.
    
    Args:
        queries: Query tensor reshaped to [b, h, sq, n_subvec, subvec_d]
        centroids: Centroid tensor [b, h, n_subvec, cent_cnt, subvec_d]
        codebook: Codebook tensor [b, h, n_subvec, n_clustered]
    
    Returns:
        scores: Attention scores [b, h, sq, n_clustered]
    """
    b: int
    h: int
    sq: int
    n_subvec: int
    subvec_d: int
    b, h, sq, n_subvec, subvec_d = queries.shape
    
    _, _, _, cent_cnt, _ = centroids.shape
    _, _, _, n_clustered = codebook.shape
    
    # Allocate output
    scores: torch.Tensor = torch.empty(
        (b, h, sq, n_clustered), device=queries.device, dtype=torch.float32
    )
    
    # Get strides
    q_strides = queries.stride()
    c_strides = centroids.stride()
    cb_strides = codebook.stride()
    s_strides = scores.stride()
    
    # Launch kernel with autotuning
    def grid(meta):
        return (
            b * h * sq * triton.cdiv(n_clustered, meta["BLOCK_KEYS"]),
        )
    
    pq_score_kernel_v3[grid](
        queries,
        centroids,
        codebook,
        scores,
        b=b,
        h=h,
        sq=sq,
        n_subvec=n_subvec,
        subvec_d=subvec_d,
        cent_cnt=cent_cnt,
        n_clustered=n_clustered,
        q_stride_b=q_strides[0],
        q_stride_h=q_strides[1],
        q_stride_sq=q_strides[2],
        q_stride_subvec=q_strides[3],
        q_stride_d=q_strides[4],
        c_stride_b=c_strides[0],
        c_stride_h=c_strides[1],
        c_stride_subvec=c_strides[2],
        c_stride_cent=c_strides[3],
        c_stride_d=c_strides[4],
        cb_stride_b=cb_strides[0],
        cb_stride_h=cb_strides[1],
        cb_stride_subvec=cb_strides[2],
        cb_stride_clustered=cb_strides[3],
        s_stride_b=s_strides[0],
        s_stride_h=s_strides[1],
        s_stride_sq=s_strides[2],
        s_stride_clustered=s_strides[3],
    )
    
    return scores


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
    """Optimized indexer logic for PQCache pattern - Iteration 3.
    
    This version uses kernel v3 with vectorized dimension processing.
    """
    # Get dimensions
    b: int
    h: int
    sq: int
    d: int
    b, h, sq, d = query.shape
    
    b_key: int
    h_kv: int
    sk: int
    d_key: int
    b_key, h_kv, sk, d_key = key.shape
    
    # Check if should use full attention
    cent_cnt: int = 2**pq_bits
    total_needed: int = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        # Use full attention
        sparse_list: torch.Tensor = torch.arange(sk, device=query.device, dtype=torch.long)
        sparse_list = sparse_list.view(1, 1, -1).expand(b, h, -1)
        
        sparse_len: torch.Tensor = torch.full(
            (b, h), sk, device=query.device, dtype=torch.long
        )
        
        weight_list: torch.Tensor = torch.ones(
            (b, h, sk), device=query.device, dtype=weight_list_dtype
        )
        
        return (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi)
    
    # Handle incremental keys - update codebook if there are new keys
    cached_codebook: torch.Tensor = pq_codebook
    cached_num_keys: int = cached_codebook.shape[1] if cached_codebook is not None else 0
    
    current_quantized_keys: int = sk - init_offset
    
    # Check if there are new keys to quantize
    if current_quantized_keys > cached_num_keys:
        # Extract new keys
        new_start: int = init_offset + cached_num_keys
        new_keys: torch.Tensor = key[:, :, new_start:, :]
        
        bsz: int
        kv_heads: int
        n_new: int
        head_dim: int
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        
        n_subvec: int = pq_group_factor
        cent_cnt: int = 2**pq_bits
        base_subvec_d: int = head_dim // n_subvec
        
        # Reshape new_keys to subvectors
        new_keys_reshaped: torch.Tensor = new_keys.reshape(
            bsz, kv_heads, n_new, n_subvec, base_subvec_d
        ).transpose(2, 3)
        
        # If using IP metric, augment new_keys
        if metric == "ip":
            new_keys_flat: torch.Tensor = new_keys_reshaped.reshape(
                -1, n_new, base_subvec_d
            )
            phi_expanded: torch.Tensor = pq_ip2l2_phi.unsqueeze(1).unsqueeze(2)
            
            new_keys_squared_norm: torch.Tensor = torch.sum(
                new_keys_flat**2, dim=-1, keepdim=True
            )
            
            phi_minus_norm: torch.Tensor = phi_expanded - new_keys_squared_norm
            phi_minus_norm_clamped: torch.Tensor = torch.clamp(phi_minus_norm, min=0.0)
            augment_dim: torch.Tensor = torch.sqrt(phi_minus_norm_clamped)
            
            new_keys_flat_aug: torch.Tensor = torch.cat(
                [new_keys_flat, augment_dim], dim=-1
            )
            
            new_keys_reshaped = new_keys_flat_aug.reshape(
                bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1
            )
        
        # Compute distances to centroids
        new_keys_exp: torch.Tensor = new_keys_reshaped.unsqueeze(4)
        centroids_exp: torch.Tensor = pq_centroids.unsqueeze(3)
        
        distances: torch.Tensor = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)
        
        # Get nearest centroid
        new_codes: torch.Tensor = torch.argmin(distances, dim=-1)
        new_codes = new_codes.permute(0, 3, 1, 2)
        
        # Update codebook
        if cached_codebook is None:
            codebook: torch.Tensor = new_codes
        else:
            codebook = torch.cat([cached_codebook, new_codes], dim=1)
    else:
        codebook = cached_codebook
    
    # Compute PQ-based scores using optimized Triton kernel v3
    n_clustered: int = codebook.shape[1]
    n_subvec: int = pq_group_factor
    subvec_d: int = d // n_subvec
    
    # Calculate GQA repeat factor
    num_key_value_groups: int = h // h_kv
    
    # Reshape queries: [b, h, sq, n_subvec, subvec_d]
    queries_reshaped: torch.Tensor = query.reshape(
        b, h, sq, n_subvec, subvec_d
    )
    
    # Repeat centroids for GQA
    if num_key_value_groups == 1:
        repeat_centroids: torch.Tensor = pq_centroids
    else:
        repeat_centroids: torch.Tensor = (
            pq_centroids[:, :, None, :, :, :]
            .expand(b, h_kv, num_key_value_groups, n_subvec, cent_cnt, -1)
            .reshape(b, h_kv * num_key_value_groups, n_subvec, cent_cnt, -1)
        )
    
    # Extract only original dimensions
    repeat_centroids = repeat_centroids[..., :subvec_d]
    
    # Repeat codebook for GQA
    codebook_permuted: torch.Tensor = codebook.permute(0, 2, 3, 1)
    
    if num_key_value_groups == 1:
        repeat_codebook: torch.Tensor = codebook_permuted
    else:
        repeat_codebook = codebook_permuted.unsqueeze(2).expand(
            b, h_kv, num_key_value_groups, n_subvec, n_clustered
        ).reshape(b, h, n_subvec, n_clustered)
    
    # Use optimized Triton kernel v3 for score computation
    scores: torch.Tensor = compute_pq_scores_triton_v3(
        queries_reshaped,
        repeat_centroids,
        repeat_codebook,
    )
    
    # Select top-K indices
    actual_sink_size: int = min(sink_size, sk)
    actual_window_size: int = min(window_size, sk)
    
    # Create mask for scores
    mask_scores: torch.Tensor = scores.clone()
    
    window_start_in_quantized: int = max(0, sk - actual_window_size - init_offset)
    if window_start_in_quantized < n_clustered:
        mask_scores[:, :, :, window_start_in_quantized:] = torch.finfo(scores.dtype).min
    
    # Select top-K
    actual_heavy_size: int = min(heavy_size, n_clustered)
    if actual_heavy_size > 0:
        _, topk_indices = torch.topk(
            mask_scores, k=actual_heavy_size, dim=-1, largest=True
        )
        topk_indices_adjusted: torch.Tensor = topk_indices + init_offset
    else:
        topk_indices_adjusted = torch.empty(
            (b, h, sq, 0), device=query.device, dtype=torch.long
        )
    
    # Create sparse_list: [sink | heavy | window]
    sink_indices: torch.Tensor = torch.arange(
        actual_sink_size, device=query.device, dtype=torch.long
    )
    sink_indices = sink_indices.view(1, 1, 1, -1).expand(b, h, sq, -1)
    
    window_start: int = sk - actual_window_size
    window_indices: torch.Tensor = torch.arange(
        window_start, sk, device=query.device, dtype=torch.long
    )
    window_indices = window_indices.view(1, 1, 1, -1).expand(b, h, sq, -1)
    
    sparse_list: torch.Tensor = torch.cat(
        [sink_indices, topk_indices_adjusted, window_indices], dim=3
    )
    
    sparse_list, _ = torch.sort(sparse_list, dim=3)
    
    total_attended: int = sparse_list.shape[3]
    sparse_len: torch.Tensor = torch.full(
        (b, h), total_attended, device=query.device, dtype=torch.long
    )
    
    # weight_list
    weight_list: torch.Tensor = torch.zeros(
        (b, h, sk), device=query.device, dtype=weight_list_dtype
    )
    
    sparse_list_first_q: torch.Tensor = sparse_list[:, :, 0, :]
    
    batch_indices: torch.Tensor = torch.arange(b, device=query.device).view(b, 1, 1)
    head_indices: torch.Tensor = torch.arange(h, device=query.device).view(1, h, 1)
    weight_list[batch_indices, head_indices, sparse_list_first_q] = 1.0
    
    return (sparse_list_first_q, sparse_len, weight_list, pq_centroids, codebook, pq_ip2l2_phi)

