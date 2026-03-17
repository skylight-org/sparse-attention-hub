import contextlib
import importlib.util
import math
import os
import shutil
import subprocess
import tempfile
import time
import types
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)

from sparse_attention_hub.sparse_attention.base import SparseAttention
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttention
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.quest_top_k import (
    QuestTopKMasker,
    QuestTopKMaskerConfig,
)

# ------------------------- Repo bootstrap -------------------------

QUEST_REPO = os.environ.get("QUEST_REPO", "https://github.com/mit-han-lab/Quest.git")
# Optionally pin to a known commit/branch/tag for stability:
QUEST_REF = os.environ.get("QUEST_REF", "")

CACHE_ROOT = Path(tempfile.gettempdir()) / "quest_cache"
FINAL_DIR = CACHE_ROOT / "Quest"
MODULE_REL = Path("evaluation/quest_attention.py")


def _atomic_clone_quest() -> Path:
    """
    Ensure Quest is present in FINAL_DIR with evaluation/quest_attention.py available.
    Uses an atomic temp-dir -> os.replace workflow to avoid races on CI.
    Returns the FINAL_DIR path.
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Fast path: already present AND file exists
    if (FINAL_DIR / MODULE_REL).exists():
        return FINAL_DIR

    tmp_dir = CACHE_ROOT / f"Quest_tmp_{os.getpid()}_{int(time.time() * 1000)}"
    try:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        # If you pin a ref, do a full clone + checkout; else shallow clone is fine
        if QUEST_REF:
            subprocess.run(
                ["git", "clone", QUEST_REPO, str(tmp_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            subprocess.run(
                ["git", "-C", str(tmp_dir), "checkout", QUEST_REF],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", QUEST_REPO, str(tmp_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        candidate = tmp_dir / MODULE_REL
        if not candidate.exists():
            raise RuntimeError(
                f"Quest clone OK but missing {MODULE_REL} at {candidate}"
            )

        # Replace existing atomically (best effort on same filesystem)
        if FINAL_DIR.exists():
            shutil.rmtree(FINAL_DIR)
        os.replace(str(tmp_dir), str(FINAL_DIR))
    except Exception:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)
        raise

    return FINAL_DIR


def _load_quest_forward():
    """
    Import Quest's evaluation/quest_attention.py by *file path* and return its `forward` callable.
    This does NOT rely on sys.path package imports, so it avoids race/namespace issues on CI.
    """
    repo = _atomic_clone_quest()
    mod_path = repo / MODULE_REL

    spec = importlib.util.spec_from_file_location("quest_attention_mod", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, "forward", None)
    if not callable(fn):
        raise ImportError("`forward` not found or not callable in quest_attention.py")
    return fn


def get_custom_attention_function(sparse_attention: SparseAttention) -> Callable:
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


# ------------------------- Test params -------------------------

TEST_CONFIG = {
    "past_sequence_length": int(os.environ.get("PAST_SEQUENCE_LENGTH", 2048)),
    "page_size": int(os.environ.get("PAGE_SIZE", 128)),
    "heavy_ratio": float(os.environ.get("HEAVY_RATIO", 0.05)),
}

TEST_CONFIGS = [
    {
        "past_sequence_length": past_seq_len,
        "page_size": page_sz,
        "heavy_ratio": heavy_ratio,
    }
    for past_seq_len, page_sz, heavy_ratio in product(
        [8192, 16384, 32768],
        [16, 32, 64, 128],
        [0.02, 0.05, 0.10],
    )
]


# ------------------------- Fixtures -------------------------


@pytest.fixture
def test_config() -> LlamaConfig:
    config = LlamaConfig(
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
    # Backwards/forwards compatibility across Transformers versions.
    if not hasattr(config, "rope_theta"):
        setattr(config, "rope_theta", 500000.0)
    return config


@pytest.fixture(params=TEST_CONFIGS)
def test_params(request) -> Dict[str, Any]:
    return request.param


@pytest.fixture
def batch_size() -> int:
    return 1


@pytest.fixture
def small_sequence_length() -> int:
    # single-token decode path
    return 1


@pytest.fixture
def past_sequence_length(test_params: Dict[str, Any]) -> int:
    return test_params["past_sequence_length"]


@pytest.fixture
def tolerance() -> float:
    return 1e-2


@pytest.fixture
def hidden_states(
    test_config: LlamaConfig, batch_size: int, small_sequence_length: int
) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randn(
        batch_size, small_sequence_length, test_config.hidden_size, generator=g
    )


@pytest.fixture
def past_key_value(
    test_config: LlamaConfig, batch_size: int, past_sequence_length: int
) -> Optional[Cache]:
    """
    Make a DynamicCache and ALSO stash the generated tensors on it so we
    don't depend on private HF internals in different versions.
    """
    cache = DynamicCache()
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    gk = torch.Generator().manual_seed(2)
    gv = torch.Generator().manual_seed(3)
    past_k = torch.randn(
        batch_size,
        test_config.num_key_value_heads,
        past_sequence_length,
        head_dim,
        generator=gk,
    )
    past_v = torch.randn(
        batch_size,
        test_config.num_key_value_heads,
        past_sequence_length,
        head_dim,
        generator=gv,
    )
    cache_kwargs = {"cache_position": torch.arange(past_sequence_length)}
    layer_idx = 32
    cache.update(past_k, past_v, layer_idx=layer_idx, cache_kwargs=cache_kwargs)

    # Stash for version-agnostic extraction
    cache._test_layer_idx = layer_idx
    cache._test_past_k = past_k
    cache._test_past_v = past_v
    return cache


@pytest.fixture
def attention_mask() -> Optional[torch.Tensor]:
    # Keep None; let sparsity machinery decide
    return None


@pytest.fixture
def position_ids(
    small_sequence_length: int, past_sequence_length: int
) -> torch.LongTensor:
    start_pos = past_sequence_length
    end_pos = start_pos + small_sequence_length
    return torch.arange(start_pos, end_pos).unsqueeze(0)


@pytest.fixture
def position_embeddings(test_config: LlamaConfig, position_ids: torch.LongTensor):
    rotary = LlamaRotaryEmbedding(config=test_config)
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    dummy = torch.randn(
        1, test_config.num_attention_heads, position_ids.shape[1], head_dim
    )
    cos, sin = rotary(dummy, position_ids)
    return cos, sin


def _effective_ratio(past_len: int, page_size: int, ratio: float) -> float:
    # Match Quest: pick at least 3 chunks worth of tokens
    min_tokens = 3 * page_size
    want_tokens = max(min_tokens, math.ceil(ratio * past_len))
    return want_tokens / max(1, past_len)


@pytest.fixture
def original_attention(
    request: pytest.FixtureRequest, test_config: LlamaConfig, test_params: Dict[str, Any]
) -> nn.Module:
    """
    Build a LlamaAttention and monkeypatch its forward with the upstream Quest implementation.
    """
    # Quest's upstream implementation calls `apply_rotary_pos_emb` with an older positional
    # signature where the 5th positional argument is `position_ids`. Newer Transformers
    # versions interpret the 5th positional argument as `unsqueeze_dim` (an int), causing:
    # `TypeError: unsqueeze(): argument 'dim' must be int, not Tensor`.
    #
    # Patch in a tiny compatibility shim for this test only.
    from transformers.models.llama import modeling_llama

    original_apply_rotary = modeling_llama.apply_rotary_pos_emb

    def _apply_rotary_pos_emb_compat(*args: Any, **kwargs: Any):
        if len(args) == 5 and torch.is_tensor(args[4]) and "unsqueeze_dim" not in kwargs:
            q, k, cos, sin, _position_ids = args
            return original_apply_rotary(q, k, cos, sin, unsqueeze_dim=1)
        return original_apply_rotary(*args, **kwargs)

    modeling_llama.apply_rotary_pos_emb = _apply_rotary_pos_emb_compat  # type: ignore[assignment]

    def _restore_apply_rotary() -> None:
        modeling_llama.apply_rotary_pos_emb = original_apply_rotary  # type: ignore[assignment]

    request.addfinalizer(_restore_apply_rotary)

    quest_forward = _load_quest_forward()

    attn = LlamaAttention(config=test_config, layer_idx=32)

    # Preserve original forward as "flash" fallback if Quest code expects it
    attn.flash_forward = attn.forward
    attn.forward = types.MethodType(quest_forward, attn)

    # Attributes expected by Quest code in evaluation/quest_attention.py
    attn.layer_id = 32
    attn.chunk_size = int(test_params["page_size"])
    attn.edge_budget = 0
    # token_budget = ceil(ratio * K), but ensure >= 3 chunks like Quest
    past_len = int(test_params["past_sequence_length"])
    ratio = float(test_params["heavy_ratio"])
    want = max(3 * attn.chunk_size, math.ceil(ratio * past_len))
    attn.token_budget = max(1, want)

    # Ensure HF attrs exist
    if not hasattr(attn, "num_heads"):
        attn.num_heads = attn.config.num_attention_heads
    if not hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = attn.config.num_key_value_heads
    if not hasattr(attn, "hidden_size"):
        attn.hidden_size = attn.config.hidden_size
    if not hasattr(attn, "head_dim"):
        attn.head_dim = attn.hidden_size // attn.num_heads
    if not hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = max(1, attn.num_heads // attn.num_key_value_heads)
    if not hasattr(attn, "rotary_emb"):
        attn.rotary_emb = LlamaRotaryEmbedding(test_config)

    return attn


@pytest.fixture
def new_attention(test_config: LlamaConfig, test_params: Dict[str, Any]) -> nn.Module:
    """
    Build LlamaAttention using ResearchAttention + QuestTopKMasker.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    past_len = int(test_params["past_sequence_length"])
    page_size = int(test_params["page_size"])
    ratio = float(test_params["heavy_ratio"])
    eff_ratio = _effective_ratio(past_len, page_size, ratio)

    cfg = QuestTopKMaskerConfig(
        heavy_size=eff_ratio,
        page_size=page_size,
    )
    masker = QuestTopKMasker.create_from_config(cfg)
    research_attention = ResearchAttention(
        sparse_attention_config=cfg, maskers=[masker]
    )
    custom_fn = get_custom_attention_function(research_attention)

    custom_name = f"quest_top_k_masker_{page_size}_{ratio}"
    ALL_ATTENTION_FUNCTIONS.register(custom_name, custom_fn)
    ALL_MASK_ATTENTION_FUNCTIONS.register(custom_name, eager_mask)

    llama_attention = LlamaAttention(config=test_config, layer_idx=32)
    llama_attention.config._attn_implementation = custom_name
    return llama_attention


# ------------------------- Helpers -------------------------


def _extract_layer_kv(dc: DynamicCache, layer_idx: int):
    """
    Robustly extract (past_k, past_v) from a DynamicCache across HF versions.
    Prefer the tensors we stashed in the fixture; otherwise try common layouts.
    """
    # Preferred: version-agnostic stashed tensors from our fixture
    if hasattr(dc, "_test_past_k") and hasattr(dc, "_test_past_v"):
        return dc._test_past_k, dc._test_past_v

    # Common newer layout: top-level lists
    if hasattr(dc, "key_cache") and hasattr(dc, "value_cache"):
        if layer_idx < len(dc.key_cache) and dc.key_cache[layer_idx] is not None:
            return dc.key_cache[layer_idx], dc.value_cache[layer_idx]

    # Some builds: dc.layers is a list of per-layer caches/tuples
    if hasattr(dc, "layers"):
        if layer_idx < len(dc.layers) and dc.layers[layer_idx] is not None:
            lyr = dc.layers[layer_idx]
            if isinstance(lyr, tuple) and len(lyr) >= 2:
                return lyr[0], lyr[1]
            if hasattr(lyr, "key_cache") and hasattr(lyr, "value_cache"):
                return lyr.key_cache, lyr.value_cache

    # Some builds: dc.caches
    if hasattr(dc, "caches"):
        if layer_idx < len(dc.caches) and dc.caches[layer_idx] is not None:
            lyr = dc.caches[layer_idx]
            if isinstance(lyr, tuple) and len(lyr) >= 2:
                return lyr[0], lyr[1]
            if hasattr(lyr, "key_cache") and hasattr(lyr, "value_cache"):
                return lyr.key_cache, lyr.value_cache

    raise RuntimeError("DynamicCache missing key/value for extraction on this HF build")


# ------------------------- Test -------------------------


@pytest.mark.unit
class TestQuestSparsityCorrectness_Upstream:
    def test_compare_outputs(
        self,
        original_attention: nn.Module,
        new_attention: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        position_embeddings,
        past_key_value: Optional[Cache],
        tolerance: float,
    ) -> bool:
        original_attention.eval()
        new_attention.eval()

        # sync parameters
        param_dict = original_attention.state_dict()
        new_attention.load_state_dict(param_dict)

        # sanity: params match
        new_dict = new_attention.state_dict()
        for k, v in original_attention.state_dict().items():
            assert k in new_dict, f"Missing key in new attention: {k}"
            assert torch.allclose(
                v, new_dict[k], atol=tolerance
            ), f"Param mismatch at {k}"

        with torch.no_grad():
            # ORIGINAL (upstream Quest) — pass a tuple so it concatenates past even with use_cache=False
            past_k, past_v = _extract_layer_kv(past_key_value, layer_idx=32)
            out_orig, attn_orig, _ = original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(past_k, past_v),
                output_attentions=True,
                use_cache=False,
            )

            # NEW (QuestTopKMasker) — pass the cache object directly
            out_new, attn_new = new_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_value,
                output_attentions=True,
                use_cache=False,
                sparse_meta_data={},
            )

        assert torch.allclose(
            out_orig, out_new, atol=tolerance
        ), "Attention outputs differ"
        assert torch.allclose(
            attn_orig, attn_new, atol=tolerance
        ), "Attention weights differ"
