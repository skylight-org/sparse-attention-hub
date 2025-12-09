"""Example custom indexer for sparse attention native backend.

This is a template/example showing how to write a custom indexer function.
The indexer can modify the sparse attention pattern by:
1. Changing which tokens are attended to (via sparse_list and sparse_len)
2. Adjusting per-token weights (via weight_list)

For now, this is a no-op indexer that returns all inputs unchanged.
"""

from typing import Any, Dict, Tuple

import torch


def __indexer(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    weight_list: torch.Tensor,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom indexer for sparse attention pattern.

    This function can be used to customize the sparse attention pattern by
    modifying which tokens are attended to and their weights.

    Args:
        queries: Query tensor of shape ``(batch_size, num_heads, head_dim)``.
        keys: Key tensor of shape ``(batch_size, num_heads, seq_len_k, head_dim)``.
        values: Value tensor of shape ``(batch_size, num_heads, seq_len_k, head_dim)``.
        sparse_list: Tensor of shape ``(batch_size, num_heads, seq_len_k)``
            containing token indices to attend to. Can be modified to change
            which tokens are attended.
        sparse_len: Tensor of shape ``(batch_size, num_heads)`` indicating
            the valid length in sparse_list. Can be reduced to attend to fewer tokens.
        weight_list: Tensor of shape ``(batch_size, num_heads, seq_len_k)``
            containing per-token weights. Can be modified to bias attention weights.
        **kwargs: Additional keyword arguments (unused in this example).

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list). In this no-op example,
        all inputs are returned unchanged.

    Example Usage:
        To use this indexer, pass it to the correctness check:
        
        ```bash
        python correctness.py --indexer-file example_indexer.py
        ```

    Example Modifications:
        # Example 1: Only attend to first 100 tokens
        sparse_len_modified = torch.minimum(sparse_len, torch.tensor(100))
        
        # Example 2: Increase weight of recent tokens (exponential decay)
        batch_size, num_heads, seq_len = sparse_list.shape
        positions = torch.arange(seq_len, device=weight_list.device)
        decay = torch.exp(-0.01 * (seq_len - positions))
        weight_list_modified = weight_list * decay.view(1, 1, -1)
        
        # Example 3: Only attend to even-indexed tokens
        mask = (sparse_list % 2) == 0
        # ... (more complex logic to filter sparse_list and update sparse_len)
    """
    # TODO: Implement your custom indexing logic here
    # For now, this is a no-op that returns inputs unchanged
    
    return sparse_list, sparse_len, weight_list

