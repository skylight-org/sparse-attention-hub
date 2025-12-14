"""Tests for backend correctness."""

from typing import Optional, Tuple

import pytest
import torch
from torch import nn

from sparse_attention_hub.sparse_attention.efficient_attention.backends.native_backend.base import (
    SparseNativeBackend,
)
from sparse_attention_hub.sparse_attention.efficient_attention.backends.research_backend.base import (
    SparseResearchBackend,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.fixture
def device() -> torch.device:
    """Fixture for CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping backend correctness test")
    return torch.device("cuda")


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture for float16 dtype."""
    return torch.float16


@pytest.fixture
def test_params() -> dict:
    """Fixture for test parameters."""
    return {
        "B": 4,
        "H": 32,
        "Q": 1,
        "K": 256,
        "D": 128,
    }


@pytest.fixture
def dummy_module() -> nn.Module:
    """Fixture for dummy attention module."""

    class DummyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.training = False

    return DummyModule()


@pytest.fixture
def attention_params(test_params: dict) -> dict:
    """Fixture for attention computation parameters."""
    D: int = test_params["D"]
    return {
        "scaling": 1.0 / (D ** 0.5),
        "dropout": 0.0,
    }


@pytest.fixture
def test_tensors(device: torch.device, dtype: torch.dtype, test_params: dict, request) -> dict:
    """Fixture for test tensors with parametrized seed."""
    # Get seed from parametrize if available, otherwise use default
    seed: int = getattr(request, "param", 42)
    
    torch.manual_seed(seed)
    B: int = test_params["B"]
    H: int = test_params["H"]
    Q: int = test_params["Q"]
    K: int = test_params["K"]
    D: int = test_params["D"]
    
    return {
        "queries": torch.randn(B, H, Q, D, device=device, dtype=dtype),
        "keys": torch.randn(B, H, K, D, device=device, dtype=dtype),
        "values": torch.randn(B, H, K, D, device=device, dtype=dtype),
    }


@pytest.mark.parametrize("test_tensors", [42 + i for i in range(10)], indirect=True)
def test_backend_correctness_with_attention_mask_none(
    device: torch.device,
    dtype: torch.dtype,
    test_params: dict,
    dummy_module: nn.Module,
    attention_params: dict,
    test_tensors: dict,
) -> None:
    """Test backend correctness with attention_mask=None.

    This test verifies that both research and native backends produce
    consistent outputs when attention_mask is None.
    """
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    
    _run_backend_correctness_test(
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=None,
        module=dummy_module,
        scaling=attention_params["scaling"],
        dropout=attention_params["dropout"],
        device=device,
        dtype=dtype,
        test_name="with attention_mask=None",
    )


@pytest.mark.parametrize("test_tensors", [42 + i for i in range(10)], indirect=True)
def test_backend_correctness_with_all_zero_attention_mask(
    device: torch.device,
    dtype: torch.dtype,
    test_params: dict,
    dummy_module: nn.Module,
    attention_params: dict,
    test_tensors: dict,
) -> None:
    """Test backend correctness with all-zero attention_mask.

    This test verifies that both research and native backends produce
    consistent outputs when attention_mask is all zeros.
    """
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    B: int = test_params["B"]
    H: int = test_params["H"]
    Q: int = test_params["Q"]
    K: int = test_params["K"]
    
    attention_mask_zeros: torch.Tensor = torch.zeros(B, H, Q, K, device=device, dtype=dtype)
    _run_backend_correctness_test(
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask_zeros,
        module=dummy_module,
        scaling=attention_params["scaling"],
        dropout=attention_params["dropout"],
        device=device,
        dtype=dtype,
        test_name="with all-zero attention_mask",
    )


@pytest.mark.parametrize("test_tensors", [42 + i for i in range(10)], indirect=True)
def test_backend_correctness_with_random_attention_mask_research_only(
    device: torch.device,
    dtype: torch.dtype,
    test_params: dict,
    dummy_module: nn.Module,
    attention_params: dict,
    test_tensors: dict,
) -> None:
    """Test research backend with random attention_mask.

    Native backend doesn't support attention_mask with non-zero values,
    so this test only runs the research backend.
    """
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    B: int = test_params["B"]
    H: int = test_params["H"]
    Q: int = test_params["Q"]
    K: int = test_params["K"]
    
    attention_mask: torch.Tensor = torch.rand(B, H, Q, K, device=device, dtype=dtype)
    _run_backend_correctness_test_research_only(
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        module=dummy_module,
        scaling=attention_params["scaling"],
        dropout=attention_params["dropout"],
        device=device,
        dtype=dtype,
        test_name="with random attention_mask (research backend only)",
    )


@pytest.mark.parametrize("test_tensors", [42 + i for i in range(10)], indirect=True)
def test_native_backend_attention_mask_exception(
    device: torch.device,
    dtype: torch.dtype,
    test_params: dict,
    dummy_module: nn.Module,
    attention_params: dict,
    test_tensors: dict,
) -> None:
    """Test that native backend raises ValueError for non-zero attention_mask."""
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    B: int = test_params["B"]
    H: int = test_params["H"]
    Q: int = test_params["Q"]
    K: int = test_params["K"]

    # Create a mask with entries between [0,1] with some zeros and others between [0,1]
    mask_shape: Tuple[int, int, int, int] = (B, H, Q, K)
    dense_mask_tensor: torch.Tensor = torch.rand(mask_shape, device=device, dtype=dtype)

    # Set some entries to zero to create sparsity
    zero_mask: torch.Tensor = torch.rand(mask_shape, device=device) > 0.3
    dense_mask_tensor = dense_mask_tensor * zero_mask

    # Ensure at least some non-zero entries per head and clamp mask values
    for b in range(B):
        for h in range(H):
            if dense_mask_tensor[b, h, 0].sum() == 0:
                num_active: int = torch.randint(1, min(100, K), (1,), device=device).item()
                indices: torch.Tensor = torch.randperm(K, device=device)[:num_active]
                dense_mask_tensor[b, h, 0, indices] = torch.rand(
                    num_active, device=device, dtype=dtype
                )
            # Clamp mask values to avoid very small values that cause numerical issues
            dense_mask_tensor[b, h, 0] = torch.clamp(dense_mask_tensor[b, h, 0], min=1e-3, max=1.0)

    mask: Mask = Mask.create_mask_from_dense_mask(
        shape=mask_shape, mask=dense_mask_tensor, dtype=dtype
    )

    # Create a non-zero attention_mask
    attention_mask: torch.Tensor = torch.rand(B, H, Q, K, device=device, dtype=dtype)

    # Run native backend
    native_backend: SparseNativeBackend = SparseNativeBackend()
    # Transform inputs to get sparse_list, sparse_len, weight_list
    (
        sparse_list,
        sparse_len,
        weight_list,
    ) = native_backend.convert_indexer_format(mask)

    # Verify that ValueError is raised
    with pytest.raises(ValueError, match="attention_mask"):
        native_backend.attention_computation_backend(
            module=dummy_module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=attention_params["scaling"],
            dropout=attention_params["dropout"],
            sparse_list=sparse_list,
            sparse_len=sparse_len,
            weight_list=weight_list,
            return_attention_weights=False,
        )


def _run_backend_correctness_test(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    module: nn.Module,
    scaling: float,
    dropout: float,
    device: torch.device,
    dtype: torch.dtype,
    test_name: str,
) -> None:
    """Run backend correctness test for a specific configuration.

    Args:
        queries: Query tensor of shape (B, H, Q, D).
        keys: Key tensor of shape (B, H, K, D).
        values: Value tensor of shape (B, H, K, D).
        attention_mask: Optional attention mask tensor.
        module: Attention module.
        scaling: Scaling factor for attention.
        dropout: Dropout probability.
        device: Device for tensors.
        dtype: Data type for tensors.
        test_name: Name of the test case for logging.
    """
    B: int = queries.shape[0]
    H: int = queries.shape[1]
    Q: int = queries.shape[2]
    K: int = keys.shape[2]

    # Create a mask with entries between [0,1] with some zeros and others between [0,1]
    mask_shape: Tuple[int, int, int, int] = (B, H, Q, K)
    dense_mask_tensor: torch.Tensor = torch.rand(mask_shape, device=device, dtype=dtype)

    # Set some entries to zero to create sparsity
    # Randomly set ~30% of entries to zero
    zero_mask: torch.Tensor = torch.rand(mask_shape, device=device) > 0.3
    dense_mask_tensor = dense_mask_tensor * zero_mask

    # Ensure at least some non-zero entries per head and clamp mask values to avoid numerical issues
    for b in range(B):
        for h in range(H):
            if dense_mask_tensor[b, h, 0].sum() == 0:
                # If all zeros, set a few random positions to non-zero
                num_active: int = torch.randint(1, min(100, K), (1,), device=device).item()
                indices: torch.Tensor = torch.randperm(K, device=device)[:num_active]
                dense_mask_tensor[b, h, 0, indices] = torch.rand(
                    num_active, device=device, dtype=dtype
                )
            # Clamp non-zero mask values to avoid very small values that cause numerical issues
            # Keep zeros as zeros, but clamp non-zero values to minimum of 1e-3
            non_zero_mask: torch.Tensor = dense_mask_tensor[b, h, 0] > 0
            if non_zero_mask.any():
                dense_mask_tensor[b, h, 0] = torch.where(
                    non_zero_mask,
                    torch.clamp(dense_mask_tensor[b, h, 0], min=1e-3, max=1.0),
                    dense_mask_tensor[b, h, 0]
                )
    mask: Mask = Mask.create_mask_from_dense_mask(
        shape=mask_shape, mask=dense_mask_tensor, dtype=dtype
    )

    # Run research backend
    research_backend: SparseResearchBackend = SparseResearchBackend()
    research_output: torch.Tensor = research_backend.attention_computation_backend(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=mask,
        return_attention_weights=False,
    )

    # Transform research output
    research_output_transformed: torch.Tensor = research_backend.post_attention_transform(
        research_output
    )

    # Run native backend
    native_backend: SparseNativeBackend = SparseNativeBackend()
    # Transform inputs to get sparse_list, sparse_len, weight_list
    (
        sparse_list,
        sparse_len,
        weight_list,
    ) = native_backend.convert_indexer_format(mask)

    native_output: torch.Tensor = native_backend.attention_computation_backend(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_list=sparse_list,
        sparse_len=sparse_len,
        weight_list=weight_list,
        return_attention_weights=False,
    )

    # Transform native output
    native_output_transformed: torch.Tensor = native_backend.post_attention_transform(
        native_output
    )

    # Check for NaN or Inf in outputs
    research_has_nan: bool = torch.isnan(research_output_transformed).any().item()
    research_has_inf: bool = torch.isinf(research_output_transformed).any().item()
    native_has_nan: bool = torch.isnan(native_output_transformed).any().item()
    native_has_inf: bool = torch.isinf(native_output_transformed).any().item()

    if research_has_nan or research_has_inf:
        pytest.fail(
            f"Research backend output contains NaN or Inf for {test_name}. "
            f"NaN: {research_has_nan}, Inf: {research_has_inf}"
        )
    if native_has_nan or native_has_inf:
        pytest.fail(
            f"Native backend output contains NaN or Inf for {test_name}. "
            f"NaN: {native_has_nan}, Inf: {native_has_inf}"
        )

    # Compare outputs
    diff: torch.Tensor = research_output_transformed - native_output_transformed
    max_err: float = diff.abs().max().item()
    mean_err: float = diff.abs().mean().item()

    # Check if error itself is NaN
    if torch.isnan(torch.tensor(max_err)) or torch.isnan(torch.tensor(mean_err)):
        pytest.fail(
            f"Error computation resulted in NaN for {test_name}. "
            f"This may indicate numerical instability in one of the backends."
        )

    print(
        f"[TEST] {test_name} - max error: {max_err:.6e}, mean error: {mean_err:.6e}"
    )

    # Allow tolerance for numerical differences between implementations
    assert mean_err < 1e-2, (
        f"Backend outputs differ too much for {test_name}: "
        f"mean error = {mean_err:.6e}"
    )
    assert max_err < 1e-2, (
        f"Backend outputs differ too much for {test_name}: "
        f"max error = {max_err:.6e}"
    )


def _run_backend_correctness_test_research_only(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    module: nn.Module,
    scaling: float,
    dropout: float,
    device: torch.device,
    dtype: torch.dtype,
    test_name: str,
) -> None:
    """Run backend correctness test for research backend only.

    This is used when native backend doesn't support the configuration
    (e.g., non-zero attention_mask).

    Args:
        queries: Query tensor of shape (B, H, Q, D).
        keys: Key tensor of shape (B, H, K, D).
        values: Value tensor of shape (B, H, K, D).
        attention_mask: Optional attention mask tensor.
        module: Attention module.
        scaling: Scaling factor for attention.
        dropout: Dropout probability.
        device: Device for tensors.
        dtype: Data type for tensors.
        test_name: Name of the test case for logging.
    """
    B: int = queries.shape[0]
    H: int = queries.shape[1]
    Q: int = queries.shape[2]
    K: int = keys.shape[2]

    # Create a mask with entries between [0,1] with some zeros and others between [0,1]
    mask_shape: Tuple[int, int, int, int] = (B, H, Q, K)
    dense_mask_tensor: torch.Tensor = torch.rand(mask_shape, device=device, dtype=dtype)

    # Set some entries to zero to create sparsity
    # Randomly set ~30% of entries to zero
    zero_mask: torch.Tensor = torch.rand(mask_shape, device=device) > 0.3
    dense_mask_tensor = dense_mask_tensor * zero_mask

    # Ensure at least some non-zero entries per head
    for b in range(B):
        for h in range(H):
            if dense_mask_tensor[b, h, 0].sum() == 0:
                # If all zeros, set a few random positions to non-zero
                num_active: int = torch.randint(1, min(100, K), (1,), device=device).item()
                indices: torch.Tensor = torch.randperm(K, device=device)[:num_active]
                dense_mask_tensor[b, h, 0, indices] = torch.rand(
                    num_active, device=device, dtype=dtype
                )

    mask: Mask = Mask.create_mask_from_dense_mask(
        shape=mask_shape, mask=dense_mask_tensor, dtype=dtype
    )

    # Run research backend only
    research_backend: SparseResearchBackend = SparseResearchBackend()
    research_output: torch.Tensor = research_backend.attention_computation_backend(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=mask,
        return_attention_weights=False,
    )

    # Transform research output
    research_output_transformed: torch.Tensor = research_backend.post_attention_transform(
        research_output
    )

    print(f"[TEST] {test_name} - research backend output shape: {research_output_transformed.shape}")



