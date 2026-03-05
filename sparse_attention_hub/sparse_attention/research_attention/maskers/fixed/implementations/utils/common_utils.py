"""Common utility functions for masker implementations."""

import torch


def pseudo_quantize(tensor: torch.Tensor, q_bit: int) -> torch.Tensor:
    """Apply pseudo-quantization to reduce memory footprint.

    Args:
        tensor: Input tensor to quantize
        q_bit: Number of quantization bits

    Returns:
        Quantized (uncompressed)tensor
    """
    max_quant = 2**q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]

    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    dequantized: torch.Tensor = quantized / scale + min_val

    return dequantized
