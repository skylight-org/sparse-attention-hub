"""Configuration builder for Quest TopK attention."""

from functools import partial
from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    QuestTopKMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


def _validity_check(config: ResearchAttentionConfig, mem_obj: int) -> bool:
    """Check if the config meets the memory objective constraint."""
    return mem_obj == 2 * (128 * config.masker_configs[2].label_bits) / config.masker_configs[2].page_size


@register_builder("quest_topk")
class QuestTopKConfigBuilder(BaseConfigBuilder):
    """Builder for Quest TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Quest TopK attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
            memory_objectives: List[int] - List of memory objectives to build the configurations.
        Ignores:
            model_config: Dict[str, str] - Model configuration
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        for sparsity_objective in sparsity_objectives:
            for memory_objective in memory_objectives:
                heavy_size: float = float(sparsity_objective) / 100.0 - (256.0 / 32768)
                aux_mem: int = memory_objective
     
                classes = [SinkMaskerConfig, LocalMaskerConfig, QuestTopKMaskerConfig]
                name: str = get_masker_list_name(classes, other_params={"builder": "quest_topk", "sparsity_obj": sparsity_objective, "memory_obj": memory_objective})

                config = ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=128),
                    LocalMaskerConfig(window_size=128),
                    QuestTopKMaskerConfig(
                        heavy_size=heavy_size - (256.0 / 32768),
                        page_size=128,
                        label_bits=16),
                ])
                
                config.masker_configs[2].search_space = {
                    "page_size": tune.grid_search([8, 16, 32, 64, 128]),
                    "label_bits": tune.grid_search([2, 4, 8, 16]),
                }
                # Set validity constraint to use the correct memory_objective for comparison
                config.validity_constraint = partial(_validity_check, mem_obj=aux_mem)
                # Set objective function
                config.objective = sparsity_objective
                
                to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs

