"""Configuration builder for PQCache attention."""

from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    PQCacheConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("pqcache_topk")
class PQCacheTopKConfigBuilder(BaseConfigBuilder):
    """Builder for PQCache TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all PQCache attention configurations.

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
            heavy_size: float = float(sparsity_objective) / 100.0
            classes = [SinkMaskerConfig, LocalMaskerConfig, PQCacheConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "pqcache_topk", "sparsity_obj": sparsity_objective})
            
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                PQCacheConfig(
                    heavy_size=heavy_size - (256.0 / 32768),
                    pq_group_factor=2,  # Default value: head_dim=128 // pq_sub_dim=64 = 2
                    pq_bits=6,  # Default value from search space
                    kmeans_iter=10,  # Default value from search space
                    init_offset=128,  # Matches sink_size
                    metric="euclidean",  # Default value from search space
                )
            ])
            
            # Set up search space for PQCache parameters
            # Note: pq_group_factor = head_dim // pq_sub_dim
            # Assuming head_dim=128: pq_sub_dim=64 -> pq_group_factor=2, pq_sub_dim=32 -> pq_group_factor=4
            config.masker_configs[2].search_space = {
                "pq_group_factor": tune.grid_search([2, 4]),  # Corresponds to pq_sub_dim=[64, 32] for head_dim=128
                "pq_bits": tune.grid_search([4, 6, 8]),
                "kmeans_iter": tune.grid_search([10]),
                "metric": tune.grid_search(["euclidean"]),
            }
            
            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective
            
            to_optimize_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

