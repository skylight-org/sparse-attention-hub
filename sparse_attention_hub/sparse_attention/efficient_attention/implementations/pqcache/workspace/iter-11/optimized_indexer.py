"""Iteration 11: Eliminate torch.arange with Custom Triton Kernel

**KEY OPTIMIZATION: Replace all torch.arange calls with fused Triton kernel!**

Problem:
- 4-5 torch.arange calls → 4-5 separate kernel launches (~72 μs)
- Advanced indexing weight_list[...] = 1.0 → expensive (~159 μs)
- Total overhead: ~230 μs

Solution:
- Single Triton kernel generates ALL indices and sets weights
- One kernel launch instead of 12+
- Expected savings: ~180-220 μs (40-46% faster!)
"""

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


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
    """PQ score computation kernel (same as iter-9)."""
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
def simple_arange_kernel(
    out_ptr,
    n,
    b: tl.constexpr,
    h: tl.constexpr,
    b_stride,
    h_stride,
):
    """Generate sequential indices [0, 1, 2, ..., n-1] for each (batch, head).
    
    Used for fast path when sk <= total_needed.
    """
    pid = tl.program_id(0)
    b_idx: tl.int32 = pid // h
    h_idx: tl.int32 = pid % h
    base = out_ptr + b_idx * b_stride + h_idx * h_stride
    
    for i in range(n):
        tl.store(base + i, i)


@triton.jit
def generate_indices_and_weights_kernel(
    # Outputs
    sparse_list_ptr,
    weight_list_ptr,
    # Inputs
    topk_indices_ptr,
    # Dimensions
    b: tl.constexpr,
    h: tl.constexpr,
    sk,
    sink_size,
    heavy_size,
    window_size,
    init_offset,
    total_attended,
    # Strides
    sparse_stride_b,
    sparse_stride_h,
    weight_stride_b,
    weight_stride_h,
    topk_stride_b,
    topk_stride_h,
):
    """Generate sparse indices and set weights in a single kernel.
    
    Replaces:
    - torch.arange(sink_size)
    - torch.arange(window_start, sk)
    - torch.arange(b), torch.arange(h)
    - weight_list[batch_indices, head_indices, sparse_list] = 1.0
    """
    BLOCK_SIZE: tl.constexpr = 256
    
    # Each program handles one (batch, head) pair
    pid = tl.program_id(0)
    batch_idx: tl.int32 = pid // h
    head_idx: tl.int32 = pid % h
    
    # Base pointers for this (batch, head)
    sparse_base = sparse_list_ptr + batch_idx * sparse_stride_b + head_idx * sparse_stride_h
    weight_base = weight_list_ptr + batch_idx * weight_stride_b + head_idx * weight_stride_h
    topk_base = topk_indices_ptr + batch_idx * topk_stride_b + head_idx * topk_stride_h
    
    window_start: tl.int32 = sk - window_size
    
    # Process sparse_list in blocks
    for block_start in range(0, total_attended, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_attended
        
        # Determine which region each offset belongs to
        in_sink = offsets < sink_size
        in_heavy = (offsets >= sink_size) & (offsets < sink_size + heavy_size)
        # in_window is everything else
        
        # Generate index values for each region
        # Sink region: [0, 1, 2, ..., sink_size-1]
        sink_vals = offsets
        
        # Heavy region: load from topk and add init_offset
        heavy_offset = offsets - sink_size
        heavy_vals = tl.load(topk_base + heavy_offset, mask=in_heavy, other=0)
        heavy_vals = heavy_vals + init_offset
        
        # Window region: [window_start, window_start+1, ..., sk-1]
        window_vals = window_start + (offsets - sink_size - heavy_size)
        
        # Select the appropriate value based on region
        sparse_vals = tl.where(in_sink, sink_vals,
                      tl.where(in_heavy, heavy_vals, window_vals))
        
        # Store to sparse_list
        tl.store(sparse_base + offsets, sparse_vals, mask=mask)
    
    # Set weights: scatter 1.0 to weight_list at sparse_list indices
    # This replaces: weight_list[batch, head, sparse_list] = 1.0
    for i in range(total_attended):
        # Load the index from sparse_list
        idx = tl.load(sparse_base + i)
        # Set weight to 1.0 at that index
        # Note: Multiple threads might write same location (OK, all write 1.0)
        tl.store(weight_base + idx, 1.0)


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
    """Iteration 11: No torch.arange - use Triton kernel instead!"""
    b, h, sq, d = query.shape
    _, h_kv, sk, _ = key.shape
    
    cent_cnt: int = 2**pq_bits
    total_needed: int = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        # Fast path: Use Triton kernel to generate [0, 1, 2, ..., sk-1]
        sparse_list = torch.empty((b, h, sk), device=query.device, dtype=torch.long)
        weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
        
        grid = (b * h,)
        simple_arange_kernel[grid](
            sparse_list, sk, b, h,
            sparse_list.stride(0), sparse_list.stride(1)
        )
        weight_list[:] = 1.0
        
        sparse_len = torch.full((b, h), sk, device=query.device, dtype=torch.long)
        return (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi)
    
    # Key quantization (same as iter-9)
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
    
    # PQ scoring (same as iter-9)
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
    
    scores = torch.empty((b, h, n_clustered), device=query.device, dtype=torch.float32)
    
    q_strides = queries_reshaped.stride()
    c_strides = repeat_centroids.stride()
    cb_strides = repeat_codebook.stride()
    s_strides = scores.stride()
    
    grid = (b * h * triton.cdiv(n_clustered, 256),)
    
    pq_score_kernel_v6[grid](
        queries_reshaped, repeat_centroids, repeat_codebook, scores,
        b, h, n_subvec, subvec_d, n_clustered,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        c_strides[0], c_strides[1], c_strides[2], c_strides[3], c_strides[4],
        cb_strides[0], cb_strides[1], cb_strides[2], cb_strides[3],
        s_strides[0], s_strides[1], s_strides[2],
    )
    
    # Mask and TopK (same as iter-9)
    actual_sink_size = min(sink_size, sk)
    actual_window_size = min(window_size, sk)
    window_start_in_quantized = max(0, sk - actual_window_size - init_offset)
    
    if window_start_in_quantized < n_clustered:
        scores[:, :, window_start_in_quantized:] = float('-inf')
    
    actual_heavy_size = min(heavy_size, n_clustered)
    
    if actual_heavy_size > 0:
        _, topk_indices = torch.topk(scores, k=actual_heavy_size, dim=-1, largest=True, sorted=False)
        
        # ✨ NEW: Use custom Triton kernel instead of torch.arange! ✨
        total_attended = actual_sink_size + actual_heavy_size + actual_window_size
        sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
        weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
        
        # Launch custom kernel
        grid = (b * h,)
        
        generate_indices_and_weights_kernel[grid](
            sparse_list,
            weight_list,
            topk_indices,
            b, h, sk,
            actual_sink_size,
            actual_heavy_size,
            actual_window_size,
            init_offset,
            total_attended,
            sparse_list.stride(0),
            sparse_list.stride(1),
            weight_list.stride(0),
            weight_list.stride(1),
            topk_indices.stride(0),
            topk_indices.stride(1),
        )
    else:
        # No heavy hitters - simpler case
        total_attended = actual_sink_size + actual_window_size
        sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
        weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
        
        # Still use kernel for consistency
        dummy_topk = torch.empty((b, h, 0), device=query.device, dtype=torch.long)
        grid = (b * h,)
        
        generate_indices_and_weights_kernel[grid](
            sparse_list,
            weight_list,
            dummy_topk,
            b, h, sk,
            actual_sink_size,
            0,  # no heavy
            actual_window_size,
            init_offset,
            total_attended,
            sparse_list.stride(0),
            sparse_list.stride(1),
            weight_list.stride(0),
            weight_list.stride(1),
            0, 0,  # dummy strides
        )
    
    sparse_len = torch.full((b, h), total_attended, device=query.device, dtype=torch.long)
    
    return (sparse_list, sparse_len, weight_list, pq_centroids, codebook, pq_ip2l2_phi)

