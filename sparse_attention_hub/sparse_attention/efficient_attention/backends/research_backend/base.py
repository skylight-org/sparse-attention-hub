"""Research backend implementation for sparse attention."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from ....utils.mask import Mask
from ....utils.mask_attention_utils import get_masked_attention_output
from ..base import SparseBackend


class SparseResearchBackend(SparseBackend):
    """Research backend implementation that uses get_masked_attention_output.

    This backend implements sparse attention computation using the research attention
    mechanism with masked attention output computation. It extends SparseBackend and
    provides implementations for attention computation and input transformation.

    Example:
        >>> backend = SparseResearchBackend()
        >>> output = backend.attention_computation_backend(
        ...     module=module,
        ...     queries=queries,
        ...     keys=keys,
        ...     values=values,
        ...     attention_mask=attention_mask,
        ...     scaling=scaling,
        ...     dropout=dropout,
        ...     sparse_attention_mask=mask,
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
        sparse_attention_mask: Mask,
        return_attention_weights: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform attention computation using get_masked_attention_output.

        This method calls get_masked_attention_output to compute the masked attention output.

        Args:
            module: The attention module (nn.Module).
            queries: Query tensor of shape (b, h, sk, d).
            keys: Key tensor of shape (b, h_kv, sq, d).
            values: Value tensor of shape (b, h_kv, sq, d).
            attention_mask: Optional attention mask of shape (b, h, sq, sk).
            scaling: Scaling factor for attention weights.
            dropout: Dropout probability.
            sparse_attention_mask: Mask object for sparse attention.
            return_attention_weights: Whether to return attention weights (default: False).
            **kwargs: Additional keyword arguments forwarded to get_masked_attention_output.

        Returns:
            If return_attention_weights is False:
                Attention output tensor of shape (b, h, sq, d).
            If return_attention_weights is True:
                Tuple of (attention_output, attention_weights) where:
                - attention_output: tensor of shape (b, h, sq, d)
                - attention_weights: tensor of shape (b, h, sq, sk)
        """
        result: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=return_attention_weights,
            **kwargs,
        )

        return result

    def convert_indexer_format(self, sparse_attention_mask: Mask) -> Mask:
        """Convert sparse attention mask to research backend format.

        For the research backend, the mask is returned as-is since it uses
        the Mask object directly in attention computation.

        Args:
            sparse_attention_mask: Mask object representing the sparse attention pattern.

        Returns:
            The same Mask object, unchanged.
        """
        return sparse_attention_mask
    
    def check_correctness_with_research_backend(self, other_sparse_attention_mask: Mask, *args) -> bool:
        my_sparse_attention_mask = args[0]
        return torch.allclose(my_sparse_attention_mask.get_dense_mask(), other_sparse_attention_mask.get_dense_mask(), atol=1e-2, rtol=1e-2)


    def post_attention_transform(
        self,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Transform attention output to ensure it's in (B, H, Q, D) format.

        This function ensures the attention output is in the correct format.
        The output from get_masked_attention_output is (B, Q, H, D) after transpose,
        so we transpose it back to (B, H, Q, D).

        Args:
            attention_output: Attention output tensor of shape (B, Q, H, D).

        Returns:
            Attention output tensor of shape (B, H, Q, D).
        """
        if attention_output.dim() == 4:
            # get_masked_attention_output returns (B, Q, H, D) after transpose(1, 2)
            # We need to transpose back to (B, H, Q, D)
            attention_output = attention_output.transpose(1, 2).contiguous()
        return attention_output

