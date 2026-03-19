"""Configuration builder for Oracle TopP attention."""

from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopPMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("oracle_topp")
class OracleTopPConfigBuilder(BaseConfigBuilder):
    """Builder for Oracle TopP sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Oracle TopP attention configurations.

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
            classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopPMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "oracle_topp", "sparsity_obj": sparsity_objective})
            
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                OracleTopPMaskerConfig(top_p=0.7)  # Default middle value from search space
            ])
            
            # Set up search space for top_p parameter
            # Using the default search space from OracleTopPMaskerConfig
            config.masker_configs[2].search_space = {
                "top_p": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]),
            }
            
            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective
            
            to_optimize_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

