"""Configuration builder for Oracle TopP attention."""

from typing import List, Optional, Tuple

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
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Oracle TopP attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopPMaskerConfig]
        name: str = get_masker_list_name(classes, other_params={"objective": objective})
        
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
        
        to_optimize_configs.append((name, config, classes))
        return optimal_configs, to_optimize_configs

