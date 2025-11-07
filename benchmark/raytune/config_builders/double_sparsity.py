"""Configuration builder for DoubleSparsity attention."""

from functools import partial
from typing import List, Optional, Tuple, Dict
import os
from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    DoubleSparsityTopKMaskerConfig,
    LocalMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name

from logging import getLogger
logger = getLogger(__name__)

def _validity_check(config: ResearchAttentionConfig, mem_obj: int) -> bool:
    """Check if the config meets the memory objective constraint."""
    return (128 // config.masker_configs[2].group_factor) * config.masker_configs[2].label_bits == mem_obj


@register_builder("double_sparsity")
class DoubleSparsityConfigBuilder(BaseConfigBuilder):
    """Builder for DoubleSparsity sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all double sparsity attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
            memory_objectives: List[int] - List of memory objectives to build the configurations.
            model_config: Dict[str, str] - Model configuration
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
    
        if model_config["double_sparsity_config_file"] is None or not os.path.exists(model_config["double_sparsity_config_file"]):
            logger.warning(f"Double sparsity config file {model_config['double_sparsity_config_file']} for model {model_config['model_name']} does not exist. Skipping Double Sparsity configurations.")
            return optimal_configs, to_optimize_configs

        for sparsity_objective in sparsity_objectives:
            for memory_objective in memory_objectives:
                heavy_size: float = float(sparsity_objective) / 100.0
                aux_mem: int = memory_objective
     
                classes = [SinkMaskerConfig, LocalMaskerConfig, DoubleSparsityTopKMaskerConfig]
                name: str = get_masker_list_name(classes, other_params={"builder": "double_sparsity", "sparsity_obj": sparsity_objective, "memory_obj": memory_objective})

                config = ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=128),
                    LocalMaskerConfig(window_size=128),
                    DoubleSparsityTopKMaskerConfig(
                        heavy_size=heavy_size - (256.0 / 32768),
                        group_factor=8,
                        label_bits=2,
                        sorted_channel_file=model_config["double_sparsity_config_file"],
                        channel_selection="q_proj"),
                ])
                
                config.masker_configs[2].search_space = {
                    "channel_selection": tune.grid_search(["q_proj"]),
                    "group_factor": tune.grid_search([2, 4, 8, 16]),
                    "label_bits": tune.grid_search([1, 2, 4, 8, 16]),
                }
                # Set validity constraint to use the correct memory_objective for comparison
                config.validity_constraint = partial(_validity_check, mem_obj=aux_mem)
                # Set objective function
                config.objective = sparsity_objective
                
                to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs

