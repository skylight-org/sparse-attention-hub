"""Configuration builder for BucketMasker (bucket_top_k) configurations."""

from functools import partial
from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    BucketMaskerConfig,
    SinkMaskerConfig,
    LocalMaskerConfig,
)
from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


def _validity_check(config: ResearchAttentionConfig, sparsity_val: float) -> bool:
    """Check if the config meets the sparsity constraint."""
    # For BucketMasker, heavy_size should be <= sparsity_val
    return config.masker_configs[1].heavy_size <= sparsity_val


@register_builder("bucket_top_k")
class BucketTopKConfigBuilder(BaseConfigBuilder):
    """Builder for BucketMasker (bucket_top_k) sparse attention configurations."""

    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all BucketMasker attention configurations."""
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        for sparsity_objective in sparsity_objectives:
            sparsity_val: float = float(sparsity_objective) / 100.0
            heavy_size: float = sparsity_val
            classes = [SinkMaskerConfig, BucketMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "bucket_top_k", "sparsity_obj": sparsity_objective})

            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                BucketMaskerConfig(
                    K=7,
                    L=8,
                    top_t=8,
                    heavy_size=heavy_size
                ),
            ])

            # Set up search space for BucketMaskerConfig
            config.masker_configs[1].search_space = {
                "K": tune.grid_search([7]),
                "L": tune.grid_search([25, 30, 35, 40, 45, 60, 65, 70, 75, 80, 85, 90]),
                "top_t": tune.grid_search([8, 10, 12, 14, 16]),
                "heavy_size": tune.grid_search([heavy_size]),
            }

            config.validity_constraint = partial(_validity_check, sparsity_val=sparsity_val)
            config.objective = sparsity_objective

            to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs
