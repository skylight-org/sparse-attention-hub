"""Optimized implementation of PQCache indexer_next logic - Iteration 1.

This module contains an optimized version using Triton kernels for score computation.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def pq_score_kernel(
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
    # Block sizes
    BLOCK_CLUSTERED: tl.constexpr,
):
    """Triton kernel for computing PQ-based attention scores.
    
    For each (batch, head, query, key) combination:
    1. Compute Q @ Centroids.T for all subvectors
    2. Gather scores using codebook indices
    3. Sum across subvectors
    
    Each program processes a block of keys for a specific (batch, head, query) triple.
    """
    # Get program ID
    pid: tl.int32 = tl.program_id(0)
    
    # Compute which (b, h, sq) this program handles
    num_programs_per_query: tl.int32 = tl.cdiv(n_clustered, BLOCK_CLUSTERED)
    total_queries: tl.int32 = b * h * sq
    
    query_id: tl.int32 = pid // num_programs_per_query
    block_id: tl.int32 = pid % num_programs_per_query
    
    # Extract b, h, sq from query_id
    batch_idx: tl.int32 = query_id // (h * sq)
    rem: tl.int32 = query_id % (h * sq)
    head_idx: tl.int32 = rem // sq
    seq_idx: tl.int32 = rem % sq
    
    # Compute key indices this block handles
    key_start: tl.int32 = block_id * BLOCK_CLUSTERED
    key_offsets = key_start + tl.arange(0, BLOCK_CLUSTERED)
    key_mask = key_offsets < n_clustered
    
    # Initialize accumulator for scores
    scores_accum = tl.zeros([BLOCK_CLUSTERED], dtype=tl.float32)
    
    # Loop over subvectors to accumulate scores
    for subvec_idx in range(n_subvec):
        # Load query for this subvector: [subvec_d]
        q_base_ptr: tl.pointer_type(tl.float32) = (
            queries_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + seq_idx * q_stride_sq
            + subvec_idx * q_stride_subvec
        )
        
        # Load centroids for this subvector: [cent_cnt, subvec_d]
        c_base_ptr: tl.pointer_type(tl.float32) = (
            centroids_ptr
            + batch_idx * c_stride_b
            + head_idx * c_stride_h
            + subvec_idx * c_stride_subvec
        )
        
        # Load codebook indices for this subvector: [n_clustered]
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
        
        # Compute Q @ Centroids.T for this subvector
        # Inner loop over subvec_d dimension
        subvec_scores = tl.zeros([BLOCK_CLUSTERED], dtype=tl.float32)
        for d_idx in range(subvec_d):
            q_val: tl.float32 = tl.load(q_base_ptr + d_idx * q_stride_d)
            
            # Load centroid values for all centroids at this dimension
            # We need to gather centroids based on cb_indices
            # For each key, load the corresponding centroid value
            cent_offsets = cb_indices * c_stride_cent + d_idx * c_stride_d
            cent_vals = tl.load(
                c_base_ptr + cent_offsets,
                mask=key_mask,
                other=0.0,
            )
            
            subvec_scores += q_val * cent_vals
        
        # Accumulate to total scores
        scores_accum += subvec_scores
    
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


def compute_pq_scores_triton(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute PQ-based attention scores using Triton kernel.
    
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
    
    # Launch kernel
    BLOCK_CLUSTERED: int = 128
    num_keys_blocks: int = triton.cdiv(n_clustered, BLOCK_CLUSTERED)
    total_queries: int = b * h * sq
    grid: tuple = (total_queries * num_keys_blocks,)
    
    pq_score_kernel[grid](
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
        BLOCK_CLUSTERED=BLOCK_CLUSTERED,
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
    """Optimized indexer logic for PQCache pattern (subsequent iterations).
    
    This version uses a Triton kernel for the score computation step.
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
    
    # Compute PQ-based scores using Triton kernel
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
    
    # Use Triton kernel for score computation
    scores: torch.Tensor = compute_pq_scores_triton(
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

