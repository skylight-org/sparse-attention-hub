"""Configuration builder for Oracle TopK attention."""

from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("oracle_topk")
class OracleTopKConfigBuilder(BaseConfigBuilder):
    """Builder for Oracle TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Oracle TopK attention configurations based on the sparsity and memory objectives.

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
            heavy_size = float(sparsity_objective) / 100.0
            classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "oracle_topk", "sparsity_obj": sparsity_objective})
            
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                OracleTopKConfig(heavy_size=heavy_size - (256.0 / 32768)),
            ])
            # set validity to default
            config.validity_constraint = lambda config: True
            # set objective function
            config.objective = sparsity_objective

            optimal_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

