"""Correctness checks between dense and sparse attention native backends.

This module compares the outputs of:

* ``custom_attention_hub.custom_attention`` (dense / Mask-based backend), and
* ``bias_sparse_attention_fwd`` (Sparse Attention Native backend),

after converting inputs into the appropriate formats.
"""

import argparse
import importlib.util
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple
from unittest.mock import patch

import torch
from torch import nn


# Add project root to Python path (go up 4 levels: file -> sparse_attention_hub_native -> backends -> codegen -> project_root)
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codegen.custom_attention_hub import custom_attention as custom_attention_hub
from codegen.backends.sparse_attention_hub_native.bias_sparse_attention_backend import (  # noqa: E402
    bias_sparse_attention_fwd,
)
from codegen.backends.sparse_attention_hub_native.utils import (  # noqa: E402
    convert_dense_to_sparse_attention_inputs,
)


class DummyAttentionModule(nn.Module):
    """Minimal attention module used only to provide a ``training`` flag."""

    def __init__(self) -> None:
        """Initialize dummy attention module."""
        super().__init__()


def custom_indexer_sparse_native(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    weight_list: torch.Tensor,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Default (no-op) indexer for sparse attention native backend.

    This indexer simply returns the inputs unchanged, meaning all tokens are
    attended to with uniform weights.

    Args:
        queries: Query tensor of shape ``(batch_size, num_heads, head_dim)``.
        keys: Key tensor of shape ``(batch_size, num_heads, seq_len_k, head_dim)``.
        values: Value tensor of shape ``(batch_size, num_heads, seq_len_k, head_dim)``.
        sparse_list: Tensor of shape ``(batch_size, num_heads, seq_len_k)``
            containing token indices to attend to.
        sparse_len: Tensor of shape ``(batch_size, num_heads)`` indicating
            the valid length in sparse_list.
        weight_list: Tensor of shape ``(batch_size, num_heads, seq_len_k)``
            containing per-token weights.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple of (sparse_list, sparse_len, weight_list) unchanged.
    """
    return sparse_list, sparse_len, weight_list


def load_indexer_from_file(file_path: str) -> Callable:
    """Dynamically load a custom indexer function from a Python file.

    The file must contain a function named ``__indexer`` with the same signature
    as ``custom_indexer_sparse_native``.

    Args:
        file_path: Path to the Python file containing the ``__indexer`` function.

    Returns:
        The ``__indexer`` function from the loaded module.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file does not contain a ``__indexer`` function.
        ImportError: If the file cannot be imported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Indexer file not found: {file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_indexer_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the __indexer function
    if not hasattr(module, "__indexer"):
        raise AttributeError(
            f"Module {file_path} does not contain a '__indexer' function"
        )

    indexer_fn: Callable = getattr(module, "__indexer")
    return indexer_fn


def run_custom_attention_equivalence_check(
    num_trials: int = 10,
    custom_indexer_fn: Optional[Callable] = None,
    indexer_file: Optional[str] = None,
) -> bool:
    """Run numerical equivalence checks between dense and sparse native backends.

    The check uses synthetic data with a single query position (``seq_len_q=1``)
    so that the inputs can be mapped cleanly into sparse attention's decode layout.
    ``num_trials`` independent random initializations of K, Q and V are used.

    Args:
        num_trials: Number of random (K, Q, V) initializations to test.
        custom_indexer_fn: Optional custom indexer function to replace the default
            ``custom_indexer_sparse_native``. If provided, this function will be
            used to modify sparse_list, sparse_len, and weight_list before calling
            the backend. Must have the same signature as ``custom_indexer_sparse_native``.
        indexer_file: Optional path to a Python file containing a ``__indexer``
            function. If provided, this will be loaded and used as the custom
            indexer. Takes precedence over ``custom_indexer_fn`` if both are provided.

    Returns:
        True if all trials pass the equivalence check, False otherwise.
    """
    # Load indexer from file if provided
    if indexer_file is not None:
        custom_indexer_fn = load_indexer_from_file(indexer_file)
        print(f"Loaded custom indexer from: {indexer_file}")

    # Use default indexer if none provided
    if custom_indexer_fn is None:
        custom_indexer_fn = custom_indexer_sparse_native

    batch_size: int = 2
    num_heads: int = 4
    seq_len_q: int = 1
    seq_len_k: int = 16000
    head_dim: int = 32

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float16

    module: DummyAttentionModule = DummyAttentionModule().to(device=device)

    atol: float = 1e-3
    rtol: float = 1e-3

    for trial_idx in range(num_trials):
        queries: torch.Tensor = torch.randn(
            batch_size, num_heads, seq_len_q, head_dim, device=device, dtype=dtype
        )
        keys: torch.Tensor = torch.randn(
            batch_size, num_heads, seq_len_k, head_dim, device=device, dtype=dtype
        )
        values: torch.Tensor = torch.randn(
            batch_size, num_heads, seq_len_k, head_dim, device=device, dtype=dtype
        )

        attention_mask: Optional[torch.Tensor] = None

        scaling: float = 1.0 / (head_dim**0.5)
        dropout: float = 0.0

        sparse_meta_data: Dict[Any, Any] = {}

        # Dense / Mask-based path.
        output_hub: torch.Tensor = custom_attention_hub(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
        )

        # Sparse attention native path: convert inputs and call bias_sparse_attention_fwd
        (
            query_sparse,
            key_sparse,
            value_sparse,
            sparse_list,
            sparse_len,
            weight_list,
        ) = convert_dense_to_sparse_attention_inputs(
            queries=queries,
            keys=keys,
            values=values,
        )

        # Apply custom indexer to modify sparse_list, sparse_len, weight_list
        sparse_list_indexed: torch.Tensor
        sparse_len_indexed: torch.Tensor
        weight_list_indexed: torch.Tensor
        sparse_list_indexed, sparse_len_indexed, weight_list_indexed = custom_indexer_fn(
            queries=query_sparse,
            keys=key_sparse,
            values=value_sparse,
            sparse_list=sparse_list,
            sparse_len=sparse_len,
            weight_list=weight_list,
        )

        # Call sparse attention backend
        output_sparse: torch.Tensor = bias_sparse_attention_fwd(
            query=query_sparse,
            key=key_sparse,
            value=value_sparse,
            sparse_list=sparse_list_indexed,
            sparse_len=sparse_len_indexed,
            weight_list=weight_list_indexed,
        )

        # Sparse attention output is (batch, num_heads, head_dim)
        # custom_attention_hub output is (batch, seq_len_q, num_heads, head_dim)
        # For seq_len_q=1, we need to squeeze and transpose to match
        output_hub_reshaped: torch.Tensor = output_hub.squeeze(1)  # (batch, num_heads, head_dim)

        if output_hub_reshaped.shape != output_sparse.shape:
            print(
                f"[Trial {trial_idx}] Shape mismatch: "
                f"hub_reshaped={output_hub_reshaped.shape}, sparse={output_sparse.shape}"
            )
            return False

        if not torch.allclose(output_hub_reshaped, output_sparse, atol=atol, rtol=rtol):
            max_abs_diff: float = float(
                torch.max(torch.abs(output_hub_reshaped - output_sparse)).item()
            )
            print(
                f"[Trial {trial_idx}] Outputs are not numerically close: "
                f"max_abs_diff={max_abs_diff}, atol={atol}, rtol={rtol}"
            )
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run correctness checks between dense and sparse attention native backends."
    )
    parser.add_argument(
        "--indexer-file",
        type=str,
        default=None,
        help="Path to a Python file containing a '__indexer' function to use as custom indexer. "
        "If not provided, uses the default (no-op) indexer.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of random K/Q/V initializations to test (default: 10).",
    )

    args = parser.parse_args()

    # Run with specified configuration
    result: bool = run_custom_attention_equivalence_check(
        num_trials=args.num_trials,
        indexer_file=args.indexer_file,
    )

    indexer_type: str = (
        f"custom indexer from {args.indexer_file}"
        if args.indexer_file
        else "default indexer"
    )
    print(f"\nCustom attention equivalence check result ({indexer_type}): {result}")

