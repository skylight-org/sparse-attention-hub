"""Configuration builder for MagicPig attention."""

from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    MagicPigConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("magicpig")
class MagicPigConfigBuilder(BaseConfigBuilder):
    """Builder for MagicPig sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all MagicPig attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
        Ignores:
            memory_objectives: List[int] - List of memory objectives
            model_config: Dict[str, str] - Model configuration
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        for sparsity_objective in sparsity_objectives:
            classes = [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "magicpig", "sparsity_obj": sparsity_objective})
            
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                MagicPigConfig(
                    lsh_l=8,  # Default value from search space
                    lsh_k=64   # Default value from search space
                )
            ])
            
            # Set up search space for LSH parameters
            config.masker_configs[2].search_space = {
                "lsh_l": tune.grid_search([16, 32, 64, 128]),
                "lsh_k": tune.grid_search([2, 4, 8, 16]),
            }
            
            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective
            
            to_optimize_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

