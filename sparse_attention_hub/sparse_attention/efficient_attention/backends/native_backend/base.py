"""Native backend implementation for sparse attention."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ....utils.mask import Mask
from ..base import SparseBackend
from .bias_sparse_attention_backend import bias_sparse_attention_fwd


class SparseNativeBackend(SparseBackend):
    """Native backend implementation that uses bias_sparse_attention_fwd.

    This backend implements sparse attention computation using the native Triton
    implementation with biased sparse attention. It extends SparseBackend and
    provides implementations for attention computation and input transformation.

    Example:
        >>> backend = SparseNativeBackend()
        >>> sparse_list, sparse_len, weight_list = backend.convert_indexer_format(mask)
        >>> output = backend.attention_computation_backend(
        ...     module=module,
        ...     queries=queries,
        ...     keys=keys,
        ...     values=values,
        ...     attention_mask=attention_mask,
        ...     scaling=scaling,
        ...     dropout=dropout,
        ...     sparse_list=sparse_list,
        ...     sparse_len=sparse_len,
        ...     weight_list=weight_list,
        ... )
    """

    def attention_computation_backend(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_list: torch.Tensor,
        sparse_len: torch.Tensor,
        weight_list: torch.Tensor,
        return_attention_weights: bool = False,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Perform attention computation using bias_sparse_attention_fwd.

        This method calls bias_sparse_attention_fwd to compute the sparse attention output.

        Args:
            module: The attention module (nn.Module) - not used but kept for signature compatibility.
            queries: Query tensor of shape (b, h, sq, d) where sq should be 1 for native backend.
            keys: Key tensor of shape (b, h_kv, sk, d).
            values: Value tensor of shape (b, h_kv, sk, d).
            attention_mask: Optional attention mask - must be None or have all zero values.
                Raises ValueError if provided with non-zero values.
            scaling: Scaling factor for attention weights - not used (computed internally).
            dropout: Dropout probability - not used by native backend.
            sparse_list: Tensor of shape (b, h, sk) containing token indices to attend to.
            sparse_len: Tensor of shape (b, h) containing number of valid tokens per head.
            weight_list: Tensor of shape (b, h, sk) containing weights for each token.
            return_attention_weights: Whether to return attention weights (not supported, always False).
            **kwargs: Additional keyword arguments:
                - block_seq: Optional int for block sequence size (default: 256)

        Returns:
            Attention output tensor of shape (b, h, sq, d) or (b, h, d).
        """
        if return_attention_weights:
            raise NotImplementedError("Native backend does not support returning attention weights")

        # Native backend does not support attention_mask
        if attention_mask is not None and attention_mask.sum() != 0:
            raise ValueError(
                "Native backend does not support attention_mask. "
                "attention_mask must be None or have all zero values."
            )

        block_seq: int = kwargs.get("block_seq", 256)

        # Handle query shape: (b, h, sq, d) -> (b, h, d) where sq should be 1
        if queries.dim() == 4:
            if queries.shape[2] != 1:
                raise ValueError(
                    f"Native backend expects queries with seq_len_q == 1, got {queries.shape}"
                )
            query: torch.Tensor = queries[:, :, 0, :]  # (b, h, d)
        else:
            query = queries

        # Handle key/value shape: (b, h_kv, sk, d) -> (b, h_kv, sk, d) (already correct)
        key: torch.Tensor = keys
        value: torch.Tensor = values

        result: torch.Tensor = bias_sparse_attention_fwd(
            query=query,
            key=key,
            value=value,
            sparse_list=sparse_list,
            sparse_len=sparse_len,
            weight_list=weight_list,
            block_seq=block_seq,
        )

        # Return 3D tensor (b, h, d) - post_attention_transform will handle reshaping to 4D
        return result

    def convert_indexer_format(
        self,
        sparse_attention_mask: Mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sparse attention mask to native backend format.

        This function converts a Mask object to the native backend format:
        sparse_list, sparse_len, and weight_list.

        Args:
            sparse_attention_mask: Mask object of shape (b, h, sq, sk) where sq must be 1.

        Returns:
            Tuple of (sparse_list, sparse_len, weight_list) where:
            - sparse_list: Tensor of shape (b, h, sk) containing token indices to attend to
            - sparse_len: Tensor of shape (b, h) containing number of valid tokens per head
            - weight_list: Tensor of shape (b, h, sk) containing weights for each token
        """
        mask: Mask = sparse_attention_mask
        # Get mask shape and validate
        B: int = mask.shape[0]
        H: int = mask.shape[1]
        Sq: int = mask.shape[2]
        Sk: int = mask.shape[3]

        # Native backend only supports single query (sq = 1)
        if Sq != 1:
            raise ValueError(
                f"Native backend does not support multiple queries. Expected sq=1, got sq={Sq}."
            )

        device: torch.device = mask.device
        dtype: torch.dtype = mask.dtype

        # Get index representation for efficient processing
        indices, ptr, data = mask.get_index_mask()  # indices: (num_nonzero,), ptr: (B*H*Sq+1,), data: (num_nonzero,)

        # Initialize output tensors
        sparse_list: torch.Tensor = torch.zeros(
            (B, H, Sk), dtype=torch.int32, device=device
        )
        sparse_len: torch.Tensor = torch.zeros((B, H), dtype=torch.int32, device=device)
        weight_list: torch.Tensor = torch.zeros(
            (B, H, Sk), dtype=dtype, device=device
        )

        # Vectorized processing using index representation
        # Since Sq=1, ptr has size B*H+1, and each (b, h) corresponds to ptr[b*H + h]
        # The mask is flattened as view(-1), so for shape (B, H, Sq, Sk) with Sq=1:
        # flattened_idx = b * (H * Sk) + h * Sk + sk
        # So for row (b, h), the row starts at: row_start_flat = b * H * Sk + h * Sk

        num_rows: int = B * H
        # Compute num_active for all rows at once
        num_active_per_row: torch.Tensor = ptr[1:] - ptr[:-1]  # (B*H,)
        sparse_len.view(-1)[:] = torch.clamp(num_active_per_row, 0, Sk).to(torch.int32)

        # Compute row_start_flat for all rows: row_idx * Sk
        row_indices: torch.Tensor = torch.arange(num_rows, device=device, dtype=torch.long)
        row_start_flat: torch.Tensor = row_indices * Sk  # (B*H,)

        # Process each row to ensure correct sequential ordering
        # While we can vectorize some operations, we need to process rows sequentially
        # to ensure sparse_list has correct ordering and handle truncation properly
        element_indices: torch.Tensor = torch.arange(indices.numel(), device=device, dtype=torch.long)
        
        for row_idx in range(num_rows):
            b: int = row_idx // H
            h: int = row_idx % H
            start_idx: int = int(ptr[row_idx].item())
            end_idx: int = int(ptr[row_idx + 1].item())
            num_active: int = end_idx - start_idx

            if num_active == 0:
                sparse_len[b, h] = 0
                continue

            # Get indices and data for this row
            row_indices_flat: torch.Tensor = indices[start_idx:end_idx]
            row_data: torch.Tensor = data[start_idx:end_idx]
            
            # Convert flattened indices to column indices
            row_start_flat: int = row_idx * Sk
            col_indices: torch.Tensor = (row_indices_flat - row_start_flat).to(torch.int32)
            col_indices = torch.clamp(col_indices, 0, Sk - 1)

            # Convert mask values to weights
            active_weights: torch.Tensor = 1.0 / (row_data + 1e-6)

            # Limit to Sk elements if needed
            if num_active > Sk:
                col_indices = col_indices[:Sk]
                active_weights = active_weights[:Sk]
                num_active = Sk

            sparse_len[b, h] = num_active
            # Store indices sequentially in sparse_list
            sparse_list[b, h, :num_active] = col_indices
            # Store weights at the token index positions
            weight_list[b, h, col_indices] = active_weights

        return (sparse_list, sparse_len, weight_list)

    def check_correctness_with_research_backend(self, other_sparse_attention_mask: Mask, *args) -> bool:
        sparse_list, sparse_len, weight_list = args
        other_sparse_list, other_sparse_len, other_weight_list = self.convert_indexer_format(other_sparse_attention_mask)

        # match sparse_len
        if not torch.equal(sparse_len, other_sparse_len):
            return False
        if not torch.allclose(weight_list, other_weight_list, atol=1e-2, rtol=1e-2):
            return False
        for b in range(sparse_len.shape[0]):
            for h in range(sparse_len.shape[1]):
                if sparse_len[b, h] > 0:
                    # Check for SET equality (order-agnostic) since attention is a set operation
                    # Sort both lists before comparing to ensure we're checking semantic correctness
                    # rather than implementation details
                    curr_len = sparse_len[b, h]
                    sorted_sparse = torch.sort(sparse_list[b, h, :curr_len])[0]
                    sorted_other = torch.sort(other_sparse_list[b, h, :curr_len])[0]
                    if not torch.equal(sorted_sparse, sorted_other):
                        return False
        return True


    def post_attention_transform(
        self,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Transform attention output to ensure it's in (B, H, Q, D) format.

        This function ensures the attention output is in the correct format.
        If the output is (B, H, D), it adds the Q dimension to make it (B, H, 1, D).

        Args:
            attention_output: Attention output tensor of shape (B, H, Q, D) or (B, H, D).

        Returns:
            Attention output tensor of shape (B, H, Q, D).
        """
        if attention_output.dim() == 3:
            # (B, H, D) -> (B, H, 1, D)
            attention_output = attention_output.unsqueeze(2)
        elif attention_output.dim() == 4:
            # Already in (B, H, Q, D) format
            pass
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {attention_output.dim()}D tensor with shape {attention_output.shape}"
            )
        return attention_output

