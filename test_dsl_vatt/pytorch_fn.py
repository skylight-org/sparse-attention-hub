"""Budget computation for adaptive sampling.

This module provides the budget computation logic extracted from adaptive sampling.
"""

from typing import List, Tuple

import torch

HASH_P = 1000007
HASH_A = 2323
HASH_B = 2277
HASH_C = 1777

def hash_function(i: int, start_idx: int, end_idx: int, b: int, q_head: int) -> int:
    """Compute a deterministic hash-based index within a range.

    Args:
        i (int): Sample index within the base sample size.
        start_idx (int): Start index (inclusive) of the sampling window.
        end_idx (int): End index (exclusive) of the sampling window.
        b (int): Batch index.
        q_head (int): Query head index.

    Returns:
        int: Hashed index within [start_idx, end_idx).
    """
    sampling_range = end_idx - start_idx
    return (HASH_A * i + HASH_B * b + HASH_C * q_head) % HASH_P % sampling_range + start_idx

def ref_vatt_idx_computation(
    keys: torch.Tensor,
    queries: torch.Tensor,
    base_sample_size: int,
    max_sample_size: int,
    epsilon: float,
    delta_ppf: float,
    scaling: float,
    start_offset: int,
    end_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sparse indices and weights using adaptive sampling.

    Args:
        keys (torch.Tensor): Keys tensor of shape (B, kH, keys, D).
        queries (torch.Tensor): Queries tensor of shape (B, qH, 1, D).
        base_sample_size (int): Base number of samples for variance estimation.
        max_sample_size (int): Upper bound on sampled indices per head.
        epsilon (float): Relative error tolerance.
        delta_ppf (float): Normal PPF value for the confidence level.
        scaling (float): Scaling factor applied to dot products.
        start_offset (int): Number of keys to include from the start.
        end_offset (int): Number of keys to include from the end.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - sparse_lens: (B, qH, 1, 1) lengths per query head.
            - sparse_idx: (B, qH, 1, keys) hashed indices.
            - weights: (B, qH, 1, keys) sampling weights.
    """
    B, qH, queries_len, D = queries.shape
    B, kH, keys_len, D = keys.shape
    assert queries_len == 1
    ngroups = qH // kH
    assert qH % kH == 0
    assert start_offset + end_offset > 0
    sampling_range = keys_len - start_offset - end_offset
    start_idx = start_offset
    end_idx = keys_len - end_offset
    effective_max_sample_size: int = min(max_sample_size, sampling_range)


    #1.  computing the max_norm_constant and static denominator.
    max_norm_constant = torch.zeros(B, qH, 1, 1, device=queries.device, dtype=queries.dtype)
    static_denominator = torch.zeros(B, qH, 1, 1, device=queries.device, dtype=queries.dtype)

    for b in range(B):
        for q_head in range(qH):
            k_head = q_head % kH
            query_vector: torch.Tensor = queries[b, q_head, 0, :]
            scores_parts: List[torch.Tensor] = []

            if start_offset > 0:
                start_keys: torch.Tensor = keys[b, k_head, :start_offset, :]
                start_scores: torch.Tensor = torch.matmul(start_keys, query_vector) * scaling
                scores_parts.append(start_scores)

            if end_offset > 0:
                end_keys: torch.Tensor = keys[b, k_head, -end_offset:, :]
                end_scores: torch.Tensor = torch.matmul(end_keys, query_vector) * scaling
                scores_parts.append(end_scores)

            all_scores: torch.Tensor = (
                torch.cat(scores_parts, dim=0) if len(scores_parts) > 1 else scores_parts[0]
            )
            max_norm_constant[b, q_head, 0, 0] = all_scores.max()
            static_denominator[b, q_head, 0, 0] = torch.exp(
                all_scores - max_norm_constant[b, q_head, 0, 0]
            ).sum()

    #2.  computing the sparse_lens or also referred to as budget
    sparse_lens = torch.zeros(B, qH, 1, 1, device=queries.device, dtype=torch.int64)
    sparse_idx = torch.zeros(B, qH, 1, keys_len, device=queries.device, dtype=torch.int64)
    weights = torch.zeros(B, qH, 1, keys_len, device=queries.device, dtype=queries.dtype)

    for b in range(B):
        for q_head in range(qH):  # Fixed: should iterate over qH, not ngroups
            k_head = q_head % kH   # Map query head to key head
            query_vector = queries[b, q_head, 0, :]
            max_norm_const = max_norm_constant[b, q_head, 0, 0]
            # Use torch tensors instead of Python scalars for accumulation
            ex_x: torch.Tensor = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)
            ex_x2: torch.Tensor = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)
            
            for i in range(base_sample_size):
                idx = hash_function(i, start_idx, end_idx, b, q_head)
                key_vector = keys[b, k_head, idx, :]
                # Compute attention score (note: max_norm_constant should be per-row max for numerical stability)
                attention_score: torch.Tensor = torch.exp(torch.dot(query_vector, key_vector) * scaling - max_norm_const)
                ex_x = ex_x + attention_score
                ex_x2 = ex_x2 + attention_score ** 2
            
            ex: torch.Tensor = ex_x / base_sample_size
            ex2: torch.Tensor = ex_x2 / base_sample_size
            var: torch.Tensor = ex2 - ex ** 2
            # Clamp variance to avoid numerical issues
            var = torch.clamp(var, min=1e-8)
            
            estimated_denominator: torch.Tensor = ex * sampling_range
            # static_denominator should have shape (B, qH, 1, 1)
            total_denominator: torch.Tensor = static_denominator[b, q_head, 0, 0] + estimated_denominator
            epsilon_allowable_error: torch.Tensor = epsilon * total_denominator
            # Clamp epsilon_allowable_error to avoid division by zero
            epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)
            
            # Budget formula: (delta_ppf * std * sampling_range / epsilon_allowable_error) ** 2
            # Since var = std^2, we have: (delta_ppf * sampling_range / epsilon_allowable_error) ** 2 * var
            this_budget: torch.Tensor = (delta_ppf * sampling_range / epsilon_allowable_error) ** 2 * var
            sparse_lens[b, q_head, 0, 0] = torch.clamp(
                this_budget, min=base_sample_size, max=effective_max_sample_size
            ).long()
    #3.  computing the sparse_idx and weights
    for b in range(B):
        for q_head in range(qH):
            sparse_len: int = int(sparse_lens[b, q_head, 0, 0].item())
            for i in range(sparse_len):
                idx = hash_function(i, start_idx, end_idx, b, q_head)
                sparse_idx[b, q_head, 0, i] = idx
                weights[b, q_head, 0, i] = sampling_range / sparse_len
    return sparse_lens, sparse_idx, weights


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample data
    batch_size: int = 2
    num_query_heads: int = 4
    num_key_heads: int = 2
    seq_len_keys: int = 32000
    seq_len_queries: int = 1
    head_dim: int = 128
    base_sample_size: int = 32
    max_sample_size: int = 1024
    
    # Create keys tensor: (B, kH, keys, D)
    keys: torch.Tensor = torch.randn(batch_size, num_key_heads, seq_len_keys, head_dim)
    
    # Create queries tensor: (B, qH, 1, D)
    queries: torch.Tensor = torch.randn(batch_size, num_query_heads, seq_len_queries, head_dim)
    
    # Set parameters
    epsilon: float = 0.1
    delta: float = 0.05
    # Compute delta_ppf using scipy (for the example)
    from scipy.stats import norm
    delta_ppf: float = float(norm.ppf(1 - delta))
    scaling: float = 0.125  # 1 / sqrt(head_dim) is common
    start_offset: int = 128
    end_offset: int = 128
    
    
    print("Running ref_vatt_idx_computation with sample data...")
    print(f"Keys shape: {keys.shape}")
    print(f"Queries shape: {queries.shape}")
    print(f"Parameters: base_sample_size={base_sample_size}, epsilon={epsilon}, delta={delta}")
    print(f"delta_ppf={delta_ppf:.4f}, scaling={scaling}, offsets=({start_offset}, {end_offset})")
    print()
    
    # Run the function
    sparse_lens, sparse_idx, weights = ref_vatt_idx_computation(
        keys=keys,
        queries=queries,
        base_sample_size=base_sample_size,
        max_sample_size=max_sample_size,
        epsilon=epsilon,
        delta_ppf=delta_ppf,
        scaling=scaling,
        start_offset=start_offset,
        end_offset=end_offset,
    )
    
    print("Results:")
    print(f"Sparse lens shape: {sparse_lens.shape}")
    print(f"Sparse lens values:")
    print(sparse_lens.squeeze())
    print(f"\nSparse lens statistics:")
    print(f"  Min: {sparse_lens.min().item():.2f}")
    print(f"  Max: {sparse_lens.max().item():.2f}")
    print(f"  Mean: {sparse_lens.float().mean().item():.2f}")
    print(f"  Std: {sparse_lens.float().std().item():.2f}")
    print(f"\nSparse idx shape: {sparse_idx.shape}")
    print(f"Weights shape: {weights.shape}")
