import copy
import json
import os
import subprocess
import sys
import tempfile
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig

from sparse_attention_hub.sparse_attention.base import SparseAttention
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttention
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.double_sparsity_top_k import (  # noqa: E501
    DoubleSparsityTopKMasker,
    DoubleSparsityTopKMaskerConfig,
)

# Test Configuration
TEST_CONFIG = {
    "heavy_const": int(os.environ.get("HEAVY_CONST", 1024)),
    "group_factor": int(os.environ.get("GROUP_FACTOR", 8)),
    "label_bits": int(os.environ.get("LABEL_BITS", 4)),
    "past_sequence_length": int(os.environ.get("PAST_SEQUENCE_LENGTH", 2048)),
}

# Multiple test configurations for stress testing
TEST_CONFIGS = [
    {
        "past_sequence_length": past_seq_len,
        "group_factor": gf,
        "label_bits": lb,
        "heavy_const": hc,
    }
    for past_seq_len, gf, lb, hc in product(
        [8192, 16384, 32768],  # past_sequence_length
        [8, 16, 32],  # group_factor
        [4, 8, 16],  # label_bits
        [128, 512, 1024],  # heavy_const
    )
]


def _get_doublesparse_path() -> str:
    """Get the path to DoubleSparse repository, downloading it if necessary."""
    # Create a temporary directory for DoubleSparse
    temp_dir = tempfile.gettempdir()
    doublesparse_path = os.path.join(temp_dir, "DoubleSparse")

    # Check if DoubleSparse already exists
    if os.path.exists(doublesparse_path):
        return doublesparse_path

    # Download DoubleSparse from GitHub
    print(f"Downloading DoubleSparse from GitHub to {doublesparse_path}...")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/andy-yang-1/DoubleSparse.git",
                doublesparse_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ Successfully downloaded DoubleSparse to {doublesparse_path}")
        return doublesparse_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download DoubleSparse: {e.stderr}") from e


# Import the original implementation from DoubleSparse
_doublesparse_path = _get_doublesparse_path()
# _doublesparse_path = "/home/ubuntu/DoubleSparse"
sys.path.append(_doublesparse_path)


def get_custom_attention_function(sparse_attention: SparseAttention) -> Callable:
    """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

    Args:
        sparse_attention: The sparse attention instance

    Returns:
        custom_attention_fn: Callable with correct signature for HuggingFace
    """

    def custom_attention_callable(
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs: Dict[str, Any],
    ):
        """Custom attention callable for HuggingFace integration."""
        if hasattr(module, "layer_idx"):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                kwargs["layer_idx"] = layer_idx

        if "sparse_meta_data" in kwargs:
            sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
            kwargs.pop("sparse_meta_data", None)
        else:
            raise ValueError(
                "sparse_meta_data must be provided while calling model.forward()"
            )

        return sparse_attention.custom_attention(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )

    return custom_attention_callable


@pytest.fixture
def test_config() -> LlamaConfig:
    """Create a test Llama configuration matching Llama-3.1-8B-Instruct dimensions."""
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        attention_dropout=0.0,
        attention_bias=False,
    )


@pytest.fixture(params=TEST_CONFIGS)
def test_params(request) -> Dict[str, int]:
    """Parametrized fixture that provides different test configurations."""
    return request.param


@pytest.fixture
def channel_data() -> Dict[str, List[List[int]]]:
    """Create test channel data using the actual DoubleSparse configuration."""
    # Use the actual channel configuration from DoubleSparse
    config_path = os.path.join(
        _doublesparse_path, "config", "meta-llama", "Llama-3.1-8B-Instruct.json"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DoubleSparse config file not found at {config_path}")

    with open(config_path, "r") as f:
        channel_data = json.load(f)

    return channel_data


@pytest.fixture
def channel_file(channel_data: Dict[str, List[List[int]]]) -> str:
    """Create a temporary JSON file with channel data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(channel_data, f)
        return f.name


@pytest.fixture
def original_attention(
    test_config: LlamaConfig,
    channel_data: Dict[str, List[List[int]]],
    test_params: Dict[str, int],
) -> nn.Module:
    """Create the original implementation from DoubleSparse."""
    from evaluation.modify_llama import LlamaAttention_heavy_hitter

    # Test parameters from config
    heavy_const = test_params["heavy_const"]
    group_factor = test_params["group_factor"]
    label_bits = test_params["label_bits"]
    layer_idx = 4

    # Create the original attention module
    original_attention = LlamaAttention_heavy_hitter(test_config, layer_idx=layer_idx)

    # Set the parameters
    original_attention.heavy_const = heavy_const
    original_attention.group_factor = group_factor
    original_attention.label_bits = label_bits

    # Set the sorted channel data
    key = f"model.layers.{layer_idx}.self_attn.qk_proj"
    if key in channel_data:
        device = next(original_attention.parameters()).device
        original_attention.sorted_channel = torch.tensor(channel_data[key]).to(device)

    return original_attention


@pytest.fixture
def new_attention(
    test_config: LlamaConfig, channel_file: str, test_params: Dict[str, int]
) -> nn.Module:
    """Create the new implementation using DoubleSparsityTopKMasker."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import LlamaAttention

    # Test parameters from config
    heavy_const = test_params["heavy_const"]
    group_factor = test_params["group_factor"]
    label_bits = test_params["label_bits"]

    masker_config = DoubleSparsityTopKMaskerConfig(
        heavy_size=heavy_const,
        group_factor=group_factor,
        label_bits=label_bits,
        sorted_channel_file=channel_file,
    )
    masker = DoubleSparsityTopKMasker.create_from_config(masker_config)
    research_attention = ResearchAttention(
        sparse_attention_config=masker_config, maskers=[masker]
    )

    # Create custom attention function
    custom_attention_callable = get_custom_attention_function(research_attention)

    custom_name = (
        f"double_sparsity_top_k_masker_{heavy_const}_{group_factor}_{label_bits}"
    )
    ALL_ATTENTION_FUNCTIONS.register(custom_name, custom_attention_callable)
    ALL_MASK_ATTENTION_FUNCTIONS.register(custom_name, eager_mask)
    llama_attention = LlamaAttention(config=test_config, layer_idx=4)
    # Also set the _attn_implementation attribute
    llama_attention.config._attn_implementation = custom_name
    return llama_attention


@pytest.fixture
def batch_size() -> int:
    """Create test batch size."""
    return 1


@pytest.fixture
def small_sequence_length() -> int:
    """Create test sequence length for new tokens (query/new key values)."""
    return 1


@pytest.fixture
def past_sequence_length(test_params: Dict[str, int]) -> int:
    """Create test sequence length for past key/value tokens."""
    return test_params["past_sequence_length"]


@pytest.fixture
def tolerance() -> float:
    """Create test tolerance for numerical comparison."""
    return 1e-2


@pytest.fixture
def hidden_states(
    test_config: LlamaConfig, batch_size: int, small_sequence_length: int
) -> torch.Tensor:
    """Create test hidden states tensor with 32 tokens (query/new key values)."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(
        batch_size, small_sequence_length, test_config.hidden_size, generator=generator
    )


@pytest.fixture
def past_key_value(
    test_config: LlamaConfig, batch_size: int, past_sequence_length: int
) -> Optional[Cache]:
    """Create past key/value cache with 10240 tokens."""
    from transformers.cache_utils import DynamicCache

    # Create a dynamic cache
    cache = DynamicCache()

    # Create dummy key and value states for past tokens
    # Shape: (batch_size, num_key_value_heads, past_sequence_length, head_dim)
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    generator = torch.Generator().manual_seed(2)
    past_key_states = torch.randn(
        batch_size,
        test_config.num_key_value_heads,
        past_sequence_length,
        head_dim,
        generator=generator,
    )
    generator = torch.Generator().manual_seed(3)
    past_value_states = torch.randn(
        batch_size,
        test_config.num_key_value_heads,
        past_sequence_length,
        head_dim,
        generator=generator,
    )

    # Update the cache with past states using the correct signature
    cache_kwargs = {"cache_position": torch.arange(past_sequence_length)}
    cache.update(
        past_key_states, past_value_states, layer_idx=4, cache_kwargs=cache_kwargs
    )

    return cache


@pytest.fixture
def attention_mask(
    batch_size: int, small_sequence_length: int, past_sequence_length: int
) -> Optional[torch.Tensor]:
    """Create test attention mask for total sequence length of 10240+32=10272."""
    total_seq_len = past_sequence_length + small_sequence_length  # 10240 + 32 = 10272

    # Create causal mask for the total sequence length
    # Query tokens (32) can attend to all past tokens (10240) + themselves
    boolean_mask = torch.tril(
        torch.ones(small_sequence_length, total_seq_len, dtype=torch.bool),
        diagonal=total_seq_len - small_sequence_length,
    )
    attention_mask = torch.zeros(
        small_sequence_length, total_seq_len, dtype=torch.float16
    )
    attention_mask = attention_mask.masked_fill(~boolean_mask, float("-inf")).view(
        1, 1, small_sequence_length, total_seq_len
    )
    return None


@pytest.fixture
def position_ids(
    small_sequence_length: int, past_sequence_length: int
) -> Optional[torch.LongTensor]:
    """Create test position IDs for new tokens (starting from past_sequence_length)."""
    # Position IDs for the new 32 tokens should start from past_sequence_length
    start_pos = past_sequence_length
    end_pos = start_pos + small_sequence_length
    return torch.arange(start_pos, end_pos).unsqueeze(0)


@pytest.fixture
def position_embeddings(
    test_config: LlamaConfig, position_ids: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create position embeddings for the new attention implementation."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary_emb = LlamaRotaryEmbedding(config=test_config)

    # Create dummy value states with the correct shape for the query sequence length
    # Shape: (batch_size, num_attention_heads, query_seq_len, head_dim)
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    dummy_value_states = torch.randn(
        1, test_config.num_attention_heads, position_ids.shape[1], head_dim
    )

    # Use the position_ids directly - the rotary embedding will handle the slicing
    cos, sin = rotary_emb(dummy_value_states, position_ids)

    return cos, sin


@pytest.mark.unit
class TestDoubleSparsityCorrectness:
    """Test class for correctness comparison between implementations."""

    def test_compare_outputs(
        self,
        original_attention: nn.Module,
        new_attention: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache],
        tolerance: float,
    ) -> bool:
        """Compare the outputs of original and new attention implementations.

        Args:
            original_attention: The original LlamaAttention_heavy_hitter implementation
            new_attention: The new LlamaAttention with custom implementation
            hidden_states: Input hidden states tensor (32 tokens)
            attention_mask: Optional attention mask (32 x 10272)
            position_ids: Optional position IDs (starting from 10240)
            position_embeddings: Position embeddings for RoPE
            past_key_value: Past key/value cache (10240 tokens)
            tolerance: Tolerance for numerical comparison

        Returns:
            bool: True if outputs match within tolerance, False otherwise
        """
        # Set both models to eval mode for consistent behavior
        original_attention.eval()
        new_attention.eval()

        # Make sure that weights of original and new attention are the same
        parameter_dict = original_attention.state_dict()
        new_attention.load_state_dict(parameter_dict)
        new_dict = new_attention.state_dict()
        original_attention_dict = original_attention.state_dict()

        for key in original_attention_dict:
            if key not in new_dict:
                raise ValueError(f"Key {key} missing from new attention")
            assert torch.allclose(
                original_attention_dict[key], new_dict[key], atol=tolerance
            )
        print(
            "Shapes",
            hidden_states.shape,
            position_ids.shape,
            position_embeddings[0].shape,
            position_embeddings[1].shape,
            past_key_value.get_seq_length(),
        )
        print(
            "params",
            original_attention.label_bits,
            original_attention.group_factor,
            original_attention.heavy_const,
        )
        with torch.no_grad():
            # Get outputs from original implementation (uses position_ids and past_key_value)
            past_key_value_clone = copy.deepcopy(past_key_value)
            orig_attention_output, orig_attention_weights, _ = original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_clone,
                output_attentions=True,
                use_cache=False,
            )

            # Get outputs from new implementation (uses position_embeddings and past_key_values)
            past_key_value_clone = copy.deepcopy(past_key_value)
            new_attention_output, new_attention_weights = new_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_value_clone,
                output_attentions=True,
                use_cache=False,
                sparse_meta_data={},  # Add empty sparse_meta_data
            )
            print(
                "Settings",
                past_key_value.layers[4].get_seq_length(),
                original_attention.group_factor,
                original_attention.label_bits,
                original_attention.heavy_const,
            )
            assert torch.allclose(
                orig_attention_output, new_attention_output, atol=tolerance
            )
            assert torch.allclose(
                orig_attention_weights, new_attention_weights, atol=tolerance
            )
