"""Utility functions for codegen profiling and correctness checking."""

from typing import Any, Dict

import torch


def move_sparse_meta_data_to_device(
    sparse_meta_data: Dict[Any, Any], device: torch.device
) -> Dict[Any, Any]:
    """Recursively move all tensors in sparse_meta_data to the specified device.

    Args:
        sparse_meta_data: Dictionary potentially containing tensors at any nesting level.
        device: Target device to move tensors to.

    Returns:
        Dictionary with all tensors moved to the specified device.
    """
    if sparse_meta_data is None:
        return None

    result: Dict[Any, Any] = {}
    for key, value in sparse_meta_data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = move_sparse_meta_data_to_device(value, device)
        else:
            result[key] = value
    return result

