"""Tests for StreamingLLM native backend indexer correctness.

This module tests the correctness of indexer_first and indexer_next methods
for StreamingLLM native backend with sink + local research attention config.
"""

from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from torch import nn

from sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm import (
    StreamingLLMNativeBackend,
    StreamingLLMNativeBackendConfig,
)
from sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research import (
    StreamingLLMResearchBackend,
    StreamingLLMResearchBackendConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.fixture
def device() -> torch.device:
    """Fixture for CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping StreamingLLM native backend test")
    return torch.device("cuda")


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture for float16 dtype."""
    return torch.float16


@pytest.fixture
def sink_size() -> int:
    """Fixture for sink size."""
    return 4


@pytest.fixture
def window_size() -> int:
    """Fixture for window size."""
    return 16


@pytest.fixture
def test_params() -> Dict[str, int]:
    """Fixture for test parameters."""
    return {
        "B": 2,
        "H": 8,
        "Q": 1,
        "K": 64,
        "D": 64,
    }


@pytest.fixture
def dummy_module() -> nn.Module:
    """Fixture for dummy attention module."""

    class DummyModule(nn.Module):
        """Dummy attention module for testing."""

        def __init__(self) -> None:
            """Initialize dummy attention module."""
            super().__init__()
            self.training = False

    return DummyModule()


@pytest.fixture
def attention_params(test_params: Dict[str, int]) -> Dict[str, float]:
    """Fixture for attention computation parameters."""
    D: int = test_params["D"]
    return {
        "scaling": 1.0 / (D ** 0.5),
        "dropout": 0.0,
    }


@pytest.fixture
def test_tensors(
    device: torch.device, dtype: torch.dtype, test_params: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """Fixture for test tensors."""
    torch.manual_seed(42)
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


@pytest.fixture
def native_backend(
    sink_size: int, window_size: int
) -> StreamingLLMNativeBackend:
    """Fixture for StreamingLLM native backend."""
    sink_config: SinkMaskerConfig = SinkMaskerConfig(sink_size=sink_size)
    local_config: LocalMaskerConfig = LocalMaskerConfig(window_size=window_size)

    research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
        masker_configs=[sink_config, local_config]
    )

    streamingllm_config: StreamingLLMNativeBackendConfig = StreamingLLMNativeBackendConfig(
        research_attention_config=research_attention_config
    )

    return StreamingLLMNativeBackend.create_from_config(streamingllm_config)


@pytest.fixture
def research_backend(
    sink_size: int, window_size: int
) -> StreamingLLMResearchBackend:
    """Fixture for StreamingLLM research backend."""
    sink_config: SinkMaskerConfig = SinkMaskerConfig(sink_size=sink_size)
    local_config: LocalMaskerConfig = LocalMaskerConfig(window_size=window_size)

    research_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(
        masker_configs=[sink_config, local_config]
    )

    streamingllm_config: StreamingLLMResearchBackendConfig = StreamingLLMResearchBackendConfig(
        research_attention_config=research_attention_config
    )

    return StreamingLLMResearchBackend.create_from_config(streamingllm_config)


def test_indexer_first_correctness(
    device: torch.device,
    dtype: torch.dtype,
    test_params: Dict[str, int],
    dummy_module: nn.Module,
    attention_params: Dict[str, float],
    test_tensors: Dict[str, torch.Tensor],
    native_backend: StreamingLLMNativeBackend,
    research_backend: StreamingLLMResearchBackend,
    sink_size: int,
    window_size: int,
) -> None:
    """Test correctness of indexer_first for StreamingLLM native backend.

    This test verifies that indexer_first produces the correct sparse attention
    pattern (sink + local) by comparing with the research backend.

    Args:
        device: Device for tensors.
        dtype: Data type for tensors.
        test_params: Test parameters dictionary.
        dummy_module: Dummy attention module.
        attention_params: Attention computation parameters.
        test_tensors: Dictionary of test tensors.
        native_backend: StreamingLLM native backend instance.
        research_backend: StreamingLLM research backend instance.
        sink_size: Size of sink tokens.
        window_size: Size of local window.
    """
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    B: int = test_params["B"]
    H: int = test_params["H"]
    K: int = test_params["K"]

    attention_mask: Optional[torch.Tensor] = None
    scaling: float = attention_params["scaling"]
    dropout: float = attention_params["dropout"]

    sparse_meta_data: Dict[str, int] = {
        "layer_idx": 0,
        "call_count": {},
    }
    kwargs: Dict[str, Any] = {}

    # Call native backend indexer_first
    sparse_list_first: torch.Tensor
    sparse_len_first: torch.Tensor
    weight_list_first: torch.Tensor
    sparse_list_first, sparse_len_first, weight_list_first = native_backend.indexer_first(
        query=queries,
        key=keys,
        value=values,
        module=dummy_module,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_meta_data=sparse_meta_data,
        **kwargs,
    )

    # Call research backend indexer_first
    mask_first: Mask = research_backend.indexer_first(
        query=queries,
        key=keys,
        value=values,
        module=dummy_module,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_meta_data=sparse_meta_data,
        **kwargs,
    )

    # Verify output shapes
    expected_attended_len: int = sink_size + window_size
    assert sparse_list_first.shape == (B, H, expected_attended_len), (
        f"sparse_list_first shape mismatch: expected {(B, H, expected_attended_len)}, "
        f"got {sparse_list_first.shape}"
    )
    assert sparse_len_first.shape == (B, H), (
        f"sparse_len_first shape mismatch: expected {(B, H)}, got {sparse_len_first.shape}"
    )
    assert weight_list_first.shape == (B, H, K), (
        f"weight_list_first shape mismatch: expected {(B, H, K)}, got {weight_list_first.shape}"
    )

    # Compare with research backend mask using convert_indexer_format
    # Convert research backend mask to native backend format
    (
        research_sparse_list,
        research_sparse_len,
        research_weight_list,
    ) = native_backend.convert_indexer_format(mask_first)

    # Compare sparse_len
    assert torch.equal(sparse_len_first, research_sparse_len), (
        f"sparse_len mismatch between native indexer and convert_indexer_format: "
        f"native={sparse_len_first}, research={research_sparse_len}"
    )

    # Compare sparse_list (need to handle ordering differences)
    for b in range(B):
        for h in range(H):
            native_len: int = int(sparse_len_first[b, h].item())
            research_len: int = int(research_sparse_len[b, h].item())
            assert native_len == research_len, (
                f"sparse_len mismatch for batch={b}, head={h}: "
                f"native={native_len}, research={research_len}"
            )

            if native_len > 0:
                native_indices: torch.Tensor = torch.sort(sparse_list_first[b, h, :native_len])[0]
                research_indices: torch.Tensor = torch.sort(
                    research_sparse_list[b, h, :research_len]
                )[0]
                assert torch.equal(native_indices, research_indices), (
                    f"sparse_list indices mismatch for batch={b}, head={h}: "
                    f"native={native_indices.tolist()}, research={research_indices.tolist()}"
                )

    # Compare weight_list
    # Note: convert_indexer_format converts mask values to weights as 1.0 / (mask_value + 1e-6)
    # For StreamingLLM mask (1.0 for attended, 0.0 for non-attended):
    # - Attended tokens: weight ≈ 1.0 / (1.0 + 1e-6) ≈ 0.999999
    # - Non-attended tokens: weight = 1.0 / (0.0 + 1e-6) = 1e6
    # Native indexer uses 1.0 for attended and 0.0 for non-attended
    # So we compare only at attended positions
    for b in range(B):
        for h in range(H):
            native_len: int = int(sparse_len_first[b, h].item())
            if native_len > 0:
                # Get attended token indices
                attended_indices: torch.Tensor = sparse_list_first[b, h, :native_len]
                # Compare weights at attended positions
                native_weights_at_attended: torch.Tensor = weight_list_first[b, h][attended_indices]
                research_weights_at_attended: torch.Tensor = research_weight_list[b, h][attended_indices]
                # Research weights should be approximately 1.0 (from 1.0 / (1.0 + 1e-6))
                assert torch.allclose(
                    native_weights_at_attended, research_weights_at_attended, atol=1e-3
                ), (
                    f"weight_list mismatch at attended positions for batch={b}, head={h}: "
                    f"native={native_weights_at_attended.tolist()}, "
                    f"research={research_weights_at_attended.tolist()}"
                )


def test_indexer_next_correctness(
    device: torch.device,
    dtype: torch.dtype,
    test_params: Dict[str, int],
    dummy_module: nn.Module,
    attention_params: Dict[str, float],
    test_tensors: Dict[str, torch.Tensor],
    native_backend: StreamingLLMNativeBackend,
    research_backend: StreamingLLMResearchBackend,
    sink_size: int,
    window_size: int,
) -> None:
    """Test correctness of indexer_next for StreamingLLM native backend.

    This test verifies that indexer_next produces the correct sparse attention
    pattern (sink + local) and that it matches indexer_first output.

    Args:
        device: Device for tensors.
        dtype: Data type for tensors.
        test_params: Test parameters dictionary.
        dummy_module: Dummy attention module.
        attention_params: Attention computation parameters.
        test_tensors: Dictionary of test tensors.
        native_backend: StreamingLLM native backend instance.
        research_backend: StreamingLLM research backend instance.
        sink_size: Size of sink tokens.
        window_size: Size of local window.
    """
    queries: torch.Tensor = test_tensors["queries"]
    keys: torch.Tensor = test_tensors["keys"]
    values: torch.Tensor = test_tensors["values"]
    B: int = test_params["B"]
    H: int = test_params["H"]
    K: int = test_params["K"]

    attention_mask: Optional[torch.Tensor] = None
    scaling: float = attention_params["scaling"]
    dropout: float = attention_params["dropout"]

    sparse_meta_data: Dict[str, int] = {
        "layer_idx": 0,
        "call_count": {0: 1},  # Simulate subsequent iteration
    }
    kwargs: Dict[str, Any] = {}

    # Call native backend indexer_next
    sparse_list_next: torch.Tensor
    sparse_len_next: torch.Tensor
    weight_list_next: torch.Tensor
    sparse_list_next, sparse_len_next, weight_list_next = native_backend.indexer_next(
        query=queries,
        key=keys,
        value=values,
        module=dummy_module,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_meta_data=sparse_meta_data,
        **kwargs,
    )
    # Call research backend indexer_next
    mask_next: Mask = research_backend.indexer_next(
        query=queries,
        key=keys,
        value=values,
        module=dummy_module,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_meta_data=sparse_meta_data,
        **kwargs,
    )

    # Verify output shapes
    expected_attended_len: int = sink_size + window_size
    assert sparse_list_next.shape == (B, H, expected_attended_len), (
        f"sparse_list_next shape mismatch: expected {(B, H, expected_attended_len)}, "
        f"got {sparse_list_next.shape}"
    )
    assert sparse_len_next.shape == (B, H), (
        f"sparse_len_next shape mismatch: expected {(B, H)}, got {sparse_len_next.shape}"
    )
    assert weight_list_next.shape == (B, H, K), (
        f"weight_list_next shape mismatch: expected {(B, H, K)}, got {weight_list_next.shape}"
    )

    # Compare with research backend mask using convert_indexer_format
    # Convert research backend mask to native backend format
    (
        research_sparse_list,
        research_sparse_len,
        research_weight_list,
    ) = native_backend.convert_indexer_format(mask_next)

    # Compare sparse_len
    assert torch.equal(sparse_len_next, research_sparse_len), (
        f"sparse_len mismatch between native indexer and convert_indexer_format: "
        f"native={sparse_len_next}, research={research_sparse_len}"
    )

    # Compare sparse_list (need to handle ordering differences)
    for b in range(B):
        for h in range(H):
            native_len: int = int(sparse_len_next[b, h].item())
            research_len: int = int(research_sparse_len[b, h].item())
            assert native_len == research_len, (
                f"sparse_len mismatch for batch={b}, head={h}: "
                f"native={native_len}, research={research_len}"
            )

            if native_len > 0:
                native_indices: torch.Tensor = torch.sort(sparse_list_next[b, h, :native_len])[0]
                research_indices: torch.Tensor = torch.sort(
                    research_sparse_list[b, h, :research_len]
                )[0]
                assert torch.equal(native_indices, research_indices), (
                    f"sparse_list indices mismatch for batch={b}, head={h}: "
                    f"native={native_indices.tolist()}, research={research_indices.tolist()}"
                )

    # Compare weight_list
    # Note: convert_indexer_format converts mask values to weights as 1.0 / (mask_value + 1e-6)
    # For StreamingLLM mask (1.0 for attended, 0.0 for non-attended):
    # - Attended tokens: weight ≈ 1.0 / (1.0 + 1e-6) ≈ 0.999999
    # - Non-attended tokens: weight = 1.0 / (0.0 + 1e-6) = 1e6
    # Native indexer uses 1.0 for attended and 0.0 for non-attended
    # So we compare only at attended positions
    for b in range(B):
        for h in range(H):
            native_len: int = int(sparse_len_next[b, h].item())
            if native_len > 0:
                # Get attended token indices
                attended_indices: torch.Tensor = sparse_list_next[b, h, :native_len]
                # Compare weights at attended positions
                native_weights_at_attended: torch.Tensor = weight_list_next[b, h][attended_indices]
                research_weights_at_attended: torch.Tensor = research_weight_list[b, h][attended_indices]
                # Research weights should be approximately 1.0 (from 1.0 / (1.0 + 1e-6))
                assert torch.allclose(
                    native_weights_at_attended, research_weights_at_attended, atol=1e-3
                ), (
                    f"weight_list mismatch at attended positions for batch={b}, head={h}: "
                    f"native={native_weights_at_attended.tolist()}, "
                    f"research={research_weights_at_attended.tolist()}"
                )
