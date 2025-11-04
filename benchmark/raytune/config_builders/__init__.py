"""Configuration builders for sparse attention configs."""

from .base import BaseConfigBuilder
from .factory import get_config_builder, get_all_config_builders, register_builder

# Import builders to trigger registration via decorators
from .dense import DenseConfigBuilder  # noqa: E402, F401
from .double_sparsity import DoubleSparsityConfigBuilder  # noqa: E402, F401
from .vattention_oracle import VAttentionOracleConfigBuilder  # noqa: E402, F401
from .vattention_hashattention import VAttentionHashAttentionConfigBuilder  # noqa: E402, F401
from .oracle_topk import OracleTopKConfigBuilder  # noqa: E402, F401
from .oracle_topp import OracleTopPConfigBuilder  # noqa: E402, F401
from .hashattention_topk import HashAttentionTopKConfigBuilder  # noqa: E402, F401
from .magicpig import MagicPigConfigBuilder  # noqa: E402, F401
from .quest_top_k import QuestTopKConfigBuilder  # noqa: E402, F401
from .random_sampling import RandomSamplingConfigBuilder  # noqa: E402, F401

__all__ = [
    "BaseConfigBuilder",
    "DenseConfigBuilder",
    "DoubleSparsityConfigBuilder",
    "VAttentionOracleConfigBuilder",
    "VAttentionHashAttentionConfigBuilder",
    "OracleTopKConfigBuilder",
    "OracleTopPConfigBuilder",
    "HashAttentionTopKConfigBuilder",
    "MagicPigConfigBuilder",
    "QuestTopKConfigBuilder",
    "RandomSamplingConfigBuilder",
    "get_config_builder",
    "get_all_config_builders",
    "register_builder",
]

