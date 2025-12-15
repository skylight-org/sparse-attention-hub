"""Imperative implementation of PQCache indexer_next logic.

This module contains the core imperative logic for PQCache's next iteration,
which handles incremental key quantization and PQ-based top-K selection.
"""

from typing import Optional, Tuple

import torch


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
    """Indexer logic for PQCache pattern (subsequent iterations).

    This function generates sparse_list, sparse_len, and weight_list for the
    PQCache attention pattern which attends to:
    1. First sink_size tokens (sink tokens)
    2. Top heavy_size tokens from middle section using PQ-based selection
    3. Last window_size tokens (local window)

    Args:
        query: Query tensor of shape (b, h, sq, d).
        key: Key tensor of shape (b, h_kv, sk, d).
        weight_list_dtype: Data type for weight_list tensor.
        sink_size: Number of sink tokens to attend to.
        window_size: Number of local window tokens to attend to.
        heavy_size: Number of heavy hitter tokens to select from middle section.
        pq_group_factor: Product quantization group factor (number of subvectors).
        pq_bits: Number of bits for PQ codebook (2^pq_bits centroids per subvector).
        kmeans_iter: Number of k-means iterations for building PQ codebook.
        init_offset: Offset for starting PQ-based selection.
        metric: Distance metric to use ("euclidean" or "ip" for inner product).
        pq_centroids: PQ centroids tensor from previous iteration.
        pq_codebook: PQ codebook tensor from previous iteration.
        pq_ip2l2_phi: Optional IP-to-L2 phi values for inner product metric.

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list, pq_centroids, pq_codebook, pq_ip2l2_phi) where:
        - sparse_list: Tensor of shape (b, h, attended_len) containing token indices to attend to
        - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
        - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        - pq_centroids: Updated PQ centroids tensor for future iterations
        - pq_codebook: Updated PQ codebook tensor for future iterations
        - pq_ip2l2_phi: Optional updated IP-to-L2 phi values for inner product metric
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
    
    # Check if should use full attention (same logic as research backend)
    # total_needed = heavy_size + init_offset + seq_len_queries + 2^pq_bits
    cent_cnt: int = 2**pq_bits
    total_needed: int = heavy_size + init_offset + sq + cent_cnt
    
    if sk <= total_needed:
        # Use full attention - attend to all tokens
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
    
    # Total keys that should be quantized (excluding sink)
    current_quantized_keys: int = sk - init_offset
    
    # Check if there are new keys to quantize
    if current_quantized_keys > cached_num_keys:
        # Extract new keys in the quantized region
        new_start: int = init_offset + cached_num_keys
        new_keys: torch.Tensor = key[:, :, new_start:, :]
        
        # Quantize new keys
        bsz: int
        kv_heads: int
        n_new: int
        head_dim: int
        bsz, kv_heads, n_new, head_dim = new_keys.shape
        
        n_subvec: int = pq_group_factor
        cent_cnt: int = 2**pq_bits
        base_subvec_d: int = head_dim // n_subvec
        
        # Reshape new_keys to subvectors
        # [bsz, kv_heads, n_new, head_dim] → [bsz, kv_heads, n_new, n_subvec, base_subvec_d]
        new_keys_reshaped: torch.Tensor = new_keys.reshape(
            bsz, kv_heads, n_new, n_subvec, base_subvec_d
        ).transpose(2, 3)
        # [bsz, kv_heads, n_subvec, n_new, base_subvec_d]
        
        # If using IP metric, augment new_keys
        if metric == "ip":
            # Flatten for augmentation
            new_keys_flat: torch.Tensor = new_keys_reshaped.reshape(
                -1, n_new, base_subvec_d
            )
            # Augment using phi
            # ip2l2_phi shape: [bsz * kv_heads * n_subvec]
            phi_expanded: torch.Tensor = pq_ip2l2_phi.unsqueeze(1).unsqueeze(2)
            # [bsz * kv_heads * n_subvec, 1, 1]
            
            # Augment: add extra dimension
            new_keys_squared_norm: torch.Tensor = torch.sum(
                new_keys_flat**2, dim=-1, keepdim=True
            )
            # [bsz * kv_heads * n_subvec, n_new, 1]
            
            phi_minus_norm: torch.Tensor = phi_expanded - new_keys_squared_norm
            # [bsz * kv_heads * n_subvec, n_new, 1]
            
            phi_minus_norm_clamped: torch.Tensor = torch.clamp(phi_minus_norm, min=0.0)
            augment_dim: torch.Tensor = torch.sqrt(phi_minus_norm_clamped)
            
            new_keys_flat_aug: torch.Tensor = torch.cat(
                [new_keys_flat, augment_dim], dim=-1
            )
            # [bsz * kv_heads * n_subvec, n_new, base_subvec_d + 1]
            
            # Reshape back
            new_keys_reshaped = new_keys_flat_aug.reshape(
                bsz, kv_heads, n_subvec, n_new, base_subvec_d + 1
            )
        
        # Compute distances to centroids
        # new_keys: [bsz, kv_heads, n_subvec, n_new, 1, subvec_d]
        # centroids: [bsz, kv_heads, n_subvec, 1, cent_cnt, subvec_d]
        new_keys_exp: torch.Tensor = new_keys_reshaped.unsqueeze(4)
        centroids_exp: torch.Tensor = pq_centroids.unsqueeze(3)
        
        # Euclidean distance
        distances: torch.Tensor = torch.sum((new_keys_exp - centroids_exp) ** 2, dim=-1)
        # [bsz, kv_heads, n_subvec, n_new, cent_cnt]
        
        # Get nearest centroid
        new_codes: torch.Tensor = torch.argmin(distances, dim=-1)
        # [bsz, kv_heads, n_subvec, n_new]
        new_codes = new_codes.permute(0, 3, 1, 2)
        # [bsz, n_new, kv_heads, n_subvec]
        
        # Update codebook
        if cached_codebook is None:
            codebook: torch.Tensor = new_codes
        else:
            codebook = torch.cat([cached_codebook, new_codes], dim=1)
    else:
        # No new keys
        codebook = cached_codebook
    
    # Compute PQ-based scores using queries and codebook
    # queries: [bsz, n_heads, seq_len_q, head_dim]
    # centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d (or subvec_d+1 if augmented)]
    # codebook: [bsz, n_clustered_keys, kv_heads, n_subvec]
    
    n_clustered: int = codebook.shape[1]
    n_subvec: int = pq_group_factor
    subvec_d: int = d // n_subvec
    
    # Calculate GQA repeat factor
    num_key_value_groups: int = h // h_kv
    
    # Reshape queries: [bsz, n_heads, seq_len_q, n_subvec, subvec_d]
    queries_reshaped: torch.Tensor = query.reshape(
        b, h, sq, n_subvec, subvec_d
    )
    # → [bsz, n_heads, n_subvec, seq_len_q, subvec_d]
    queries_trans: torch.Tensor = queries_reshaped.transpose(2, 3)
    
    # Repeat centroids for GQA: [bsz, kv_heads, ...] → [bsz, n_heads, ...]
    # Note: centroids may be augmented if IP metric was used during clustering,
    # but we only use the first subvec_d dimensions for scoring
    if num_key_value_groups == 1:
        repeat_centroids: torch.Tensor = pq_centroids
    else:
        # Manually repeat along head dimension for 5D tensor
        repeat_centroids: torch.Tensor = (
            pq_centroids[:, :, None, :, :, :]
            .expand(b, h_kv, num_key_value_groups, n_subvec, cent_cnt, -1)
            .reshape(b, h_kv * num_key_value_groups, n_subvec, cent_cnt, -1)
        )
    # [bsz, n_heads, n_subvec, cent_cnt, subvec_d_stored]
    
    # Extract only the original dimensions (ignore augmented dimension if present)
    repeat_centroids = repeat_centroids[..., :subvec_d]
    # [bsz, n_heads, n_subvec, cent_cnt, subvec_d]
    
    # Transpose for matmul: [bsz, n_heads, n_subvec, subvec_d, cent_cnt]
    repeat_centroids = repeat_centroids.transpose(3, 4)
    
    # Compute Q @ Centroids.T (inner product scores)
    qk_table: torch.Tensor = torch.matmul(queries_trans, repeat_centroids)
    # [bsz, n_heads, n_subvec, seq_len_q, cent_cnt]
    
    # Repeat codebook for GQA
    # codebook: [bsz, n_clustered, kv_heads, n_subvec]
    # → [bsz, kv_heads, n_subvec, n_clustered]
    codebook_permuted: torch.Tensor = codebook.permute(0, 2, 3, 1)
    
    # Repeat for all query heads
    if num_key_value_groups == 1:
        repeat_codebook: torch.Tensor = codebook_permuted
    else:
        repeat_codebook = codebook_permuted.unsqueeze(2).expand(
            b, h_kv, num_key_value_groups, n_subvec, n_clustered
        ).reshape(b, h, n_subvec, n_clustered)
    # [bsz, n_heads, n_subvec, n_clustered]
    
    # Gather scores using codebook indices
    # Expand codebook for all queries: [bsz, n_heads, n_subvec, 1, n_clustered]
    repeat_codebook_exp: torch.Tensor = repeat_codebook.unsqueeze(3).expand(
        -1, -1, -1, sq, -1
    )
    
    # Gather: for each query, get score to each key's assigned centroid
    gathered_scores: torch.Tensor = torch.gather(
        qk_table, dim=4, index=repeat_codebook_exp
    )
    # [bsz, n_heads, n_subvec, seq_len_q, n_clustered]
    
    # Sum across subvectors
    scores: torch.Tensor = gathered_scores.sum(dim=2)
    # [bsz, n_heads, seq_len_q, n_clustered]
    
    # Now select top-K indices from the quantized region
    # Create sink and window masks to exclude them from selection
    actual_sink_size: int = min(sink_size, sk)
    actual_window_size: int = min(window_size, sk)
    
    # The scores correspond to keys in [init_offset, init_offset + n_clustered)
    # We need to mask out positions that overlap with sink or window
    
    # Create a mask for positions that are NOT in sink or window
    # Positions in [init_offset, sk - window_size)
    mask_scores: torch.Tensor = scores.clone()
    
    # For each position in the quantized region, check if it's in the window
    # Window is [sk - window_size, sk), so positions [sk - window_size - init_offset, ...)
    # in the quantized region should be masked
    window_start_in_quantized: int = max(0, sk - actual_window_size - init_offset)
    if window_start_in_quantized < n_clustered:
        # Mask out window positions
        mask_scores[:, :, :, window_start_in_quantized:] = torch.finfo(scores.dtype).min
    
    # Select top-K
    actual_heavy_size: int = min(heavy_size, n_clustered)
    if actual_heavy_size > 0:
        _, topk_indices = torch.topk(
            mask_scores, k=actual_heavy_size, dim=-1, largest=True
        )
        # [bsz, n_heads, seq_len_q, actual_heavy_size]
        
        # Adjust indices to account for init_offset
        topk_indices_adjusted: torch.Tensor = topk_indices + init_offset
    else:
        topk_indices_adjusted = torch.empty(
            (b, h, sq, 0), device=query.device, dtype=torch.long
        )
    
    # Create sparse_list: [sink | heavy | window]
    # Sink indices
    sink_indices: torch.Tensor = torch.arange(
        actual_sink_size, device=query.device, dtype=torch.long
    )
    # Expand: [1, 1, 1, actual_sink_size] → [b, h, sq, actual_sink_size]
    sink_indices = sink_indices.view(1, 1, 1, -1).expand(b, h, sq, -1)
    
    # Window indices
    window_start: int = sk - actual_window_size
    window_indices: torch.Tensor = torch.arange(
        window_start, sk, device=query.device, dtype=torch.long
    )
    # Expand: [1, 1, 1, actual_window_size] → [b, h, sq, actual_window_size]
    window_indices = window_indices.view(1, 1, 1, -1).expand(b, h, sq, -1)
    
    # Combine: [b, h, sq, total_attended]
    sparse_list: torch.Tensor = torch.cat(
        [sink_indices, topk_indices_adjusted, window_indices], dim=3
    )
    
    # Sort the indices to match the expected order from research backend
    sparse_list, _ = torch.sort(sparse_list, dim=3)
    
    # sparse_len: number of tokens attended per batch/head/query
    total_attended: int = sparse_list.shape[3]
    sparse_len: torch.Tensor = torch.full(
        (b, h), total_attended, device=query.device, dtype=torch.long
    )
    
    # weight_list: uniform weights for all tokens
    weight_list: torch.Tensor = torch.zeros(
        (b, h, sk), device=query.device, dtype=weight_list_dtype
    )
    # Set weights to 1.0 for attended positions
    # For each query, mark the attended positions
    # We'll use the first query's sparse_list (assuming sq=1 for decoding)
    sparse_list_first_q: torch.Tensor = sparse_list[:, :, 0, :]
    # [b, h, total_attended]
    
    batch_indices: torch.Tensor = torch.arange(b, device=query.device).view(b, 1, 1)
    head_indices: torch.Tensor = torch.arange(h, device=query.device).view(1, h, 1)
    weight_list[batch_indices, head_indices, sparse_list_first_q] = 1.0
    
    # Return updated metadata
    return (sparse_list_first_q, sparse_len, weight_list, pq_centroids, codebook, pq_ip2l2_phi)

