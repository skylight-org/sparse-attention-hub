"""CUDA-backed VATT idx computation."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple
import sys

import torch
from torch.utils.cpp_extension import load

try:
    from test_dsl_vatt.pytorch_fn import ref_vatt_idx_computation as _ref_vatt_idx
except ModuleNotFoundError:
    this_dir: Path = Path(__file__).resolve()
    repo_root: Path = this_dir.parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from test_dsl_vatt.pytorch_fn import ref_vatt_idx_computation as _ref_vatt_idx

_EXTENSION: Optional[ModuleType] = None


def _load_extension() -> ModuleType:
    """Load the CUDA extension for vatt idx computation."""
    global _EXTENSION
    module: Optional[ModuleType] = _EXTENSION
    if module is not None:
        return module

    this_dir: Path = Path(__file__).resolve().parent
    sources: list[str] = [
        str(this_dir / "binding.cpp"),
        str(this_dir / "vatt_idx_computation.cu"),
    ]
    module = load(
        name="vatt_idx_computation_cuda_ext",
        sources=sources,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )
    _EXTENSION = module
    return module


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
    """Compute sparse indices and weights using a CUDA kernel.

    Args:
        keys: Keys tensor of shape (B, kH, keys, D).
        queries: Queries tensor of shape (B, qH, 1, D).
        base_sample_size: Base number of samples for variance estimation.
        max_sample_size: Upper bound on sampled indices per head.
        epsilon: Relative error tolerance.
        delta_ppf: Normal PPF value for the confidence level.
        scaling: Scaling factor applied to dot products.
        start_offset: Number of keys to include from the start.
        end_offset: Number of keys to include from the end.

    Returns:
        Tuple containing (sparse_lens, sparse_idx, weights).
    """
    if not keys.is_cuda:
        return _ref_vatt_idx(
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

    module: ModuleType = _load_extension()
    sparse_lens, sparse_idx, weights = module.ref_vatt_idx_computation(
        keys,
        queries,
        base_sample_size,
        max_sample_size,
        epsilon,
        delta_ppf,
        scaling,
        start_offset,
        end_offset,
    )
    return sparse_lens, sparse_idx, weights
