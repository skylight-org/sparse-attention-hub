"""Correctness checks between dense and FlashInfer custom attention backends.

This module compares the outputs of:

* ``custom_attention_hub.custom_attention`` (dense / Mask-based backend), and
* ``custom_attention_flashinfer.custom_attention_flashinfer`` (FlashInfer backend),

after converting inputs into the appropriate formats.
"""

from typing import Any, Callable, Dict, Optional
from unittest.mock import patch
import argparse
import importlib.util
import os
import sys

import torch
from torch import nn


PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from codegen.custom_attention_hub import custom_attention as custom_attention_hub  # noqa: E402
from codegen.custom_attention_flashinfer import (  # noqa: E402
    custom_attention_flashinfer,
)
from codegen.utils import convert_dense_to_flashinfer_inputs  # noqa: E402


class DummyAttentionModule(nn.Module):
    """Minimal attention module used only to provide a ``training`` flag."""

    def __init__(self) -> None:
        """Initialize dummy attention module."""
        super().__init__()


def load_indexer_from_file(file_path: str) -> Callable:
    """Dynamically load a custom indexer function from a Python file.

    The file must contain a function named ``__indexer`` with the same signature
    as ``custom_indexer_flashinfer``.

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
    """Run numerical equivalence checks between the two custom attention paths.

    The check uses synthetic data with a single query position (``seq_len_q=1``)
    so that the inputs can be mapped cleanly into FlashInfer's decode layout.
    ``num_trials`` independent random initializations of K, Q and V are used.

    Args:
        num_trials: Number of random (K, Q, V) initializations to test.
        custom_indexer_fn: Optional custom indexer function to replace the default
            ``custom_indexer_flashinfer``. If provided, this function will be
            patched into the FlashInfer attention pipeline to test custom indexing
            strategies. Must have the same signature as ``custom_indexer_flashinfer``.
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

        # FlashInfer path: convert inputs and call custom_attention_flashinfer.
        (
            q_f,
            kv_cache_f,
            kv_page_indptr_f,
            kv_page_indices_f,
            kv_last_page_len_f,
            num_qo_heads_f,
            num_kv_heads_f,
            head_dim_f,
            page_size_f,
            pos_encoding_mode_f,
            data_type_f,
            use_cuda_graph_f,
            use_tensor_cores_f,
            backend_f,
            jit_args_f,
        ) = convert_dense_to_flashinfer_inputs(
            queries=queries,
            keys=keys,
            values=values,
            page_size=seq_len_k,
        )

        # Call FlashInfer backend with optional custom indexer patching
        if custom_indexer_fn is not None:
            # Patch the custom_indexer_flashinfer function with the provided one
            with patch(
                "codegen.custom_attention_flashinfer.custom_indexer_flashinfer",
                custom_indexer_fn,
            ):
                output_flashinfer: torch.Tensor = custom_attention_flashinfer(
                    q=q_f,
                    kv_cache=kv_cache_f,
                    kv_page_indptr=kv_page_indptr_f,
                    kv_page_indices=kv_page_indices_f,
                    kv_last_page_len=kv_last_page_len_f,
                    num_qo_heads=num_qo_heads_f,
                    num_kv_heads=num_kv_heads_f,
                    head_dim=head_dim_f,
                    page_size=page_size_f,
                    pos_encoding_mode=pos_encoding_mode_f,
                    data_type=data_type_f,
                    use_cuda_graph=use_cuda_graph_f,
                    use_tensor_cores=use_tensor_cores_f,
                    backend=backend_f,
                    jit_args=jit_args_f,
                )
        else:
            # Use default indexer
            output_flashinfer: torch.Tensor = custom_attention_flashinfer(
                q=q_f,
                kv_cache=kv_cache_f,
                kv_page_indptr=kv_page_indptr_f,
                kv_page_indices=kv_page_indices_f,
                kv_last_page_len=kv_last_page_len_f,
                num_qo_heads=num_qo_heads_f,
                num_kv_heads=num_kv_heads_f,
                head_dim=head_dim_f,
                page_size=page_size_f,
                pos_encoding_mode=pos_encoding_mode_f,
                data_type=data_type_f,
                use_cuda_graph=use_cuda_graph_f,
                use_tensor_cores=use_tensor_cores_f,
                backend=backend_f,
                jit_args=jit_args_f,
            )

        # FlashInfer decode output is (batch, num_heads, head_dim)
        # custom_attention_hub output is (batch, seq_len_q, num_heads, head_dim)
        # For seq_len_q=1, we need to squeeze and transpose to match
        output_hub_reshaped: torch.Tensor = output_hub.squeeze(1)  # (batch, num_heads, head_dim)

        if output_hub_reshaped.shape != output_flashinfer.shape:
            print(
                f"[Trial {trial_idx}] Shape mismatch: "
                f"hub_reshaped={output_hub_reshaped.shape}, flashinfer={output_flashinfer.shape}"
            )
            return False

        if not torch.allclose(output_hub_reshaped, output_flashinfer, atol=atol, rtol=rtol):
            max_abs_diff: float = float(
                torch.max(torch.abs(output_hub_reshaped - output_flashinfer)).item()
            )
            print(
                f"[Trial {trial_idx}] Outputs are not numerically close: "
                f"max_abs_diff={max_abs_diff}, atol={atol}, rtol={rtol}"
            )
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run correctness checks between dense and FlashInfer attention backends."
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


