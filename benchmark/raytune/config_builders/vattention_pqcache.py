"""Configuration builder for VAttention PQCache configurations."""

from functools import partial
from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    PQCacheConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


def _validity_check(config: ResearchAttentionConfig, sparsity_val: float) -> bool:
    """Check if the config meets the sparsity constraint.
    
    Args:
        config: ResearchAttentionConfig to validate.
        sparsity_val: Target sparsity value as a float.
        
    Returns:
        True if pqcache heavy_size + adaptive sampling base_rate_sampling <= sparsity_val, False otherwise.
    """
    return (config.masker_configs[2].heavy_size + config.masker_configs[3].base_rate_sampling) <= sparsity_val


@register_builder("vattention_pqcache_topk")
class VAttentionPQCacheTopKConfigBuilder(BaseConfigBuilder):
    """Builder for VAttention PQCache TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all VAttention PQCache attention configurations.

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
            sparsity_val: float = float(sparsity_objective) / 100.0
            heavy_size: float = float(sparsity_objective) / 100.0
            classes = [SinkMaskerConfig, LocalMaskerConfig, PQCacheConfig, AdaptiveSamplingMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "vattention_pqcache_topk", "sparsity_obj": sparsity_objective})
            
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
                ),
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.05,  # Middle value
                    epsilon=0.05,  # Middle value
                    delta=0.05,  # Middle value
                    init_offset=128,  # Middle value
                    local_offset=128  # Middle value
                )
            ])
            
            # Set up search space for PQCache parameters (from pqcache builder)
            # Note: pq_group_factor = head_dim // pq_sub_dim
            # Assuming head_dim=128: pq_sub_dim=64 -> pq_group_factor=2, pq_sub_dim=32 -> pq_group_factor=4
            config.masker_configs[2].search_space = {
                "pq_group_factor": tune.grid_search([2, 4]),  # Corresponds to pq_sub_dim=[64, 32] for head_dim=128
                "pq_bits": tune.grid_search([4, 8]),
                "kmeans_iter": tune.grid_search([10]),
                "metric": tune.grid_search(["euclidean"]),
            }
            
            # Set up search space for AdaptiveSamplingMaskerConfig (from vattention_hashattention builder)
            if sparsity_objective == 2:
                # Adaptive sampling with PQCache
                config.masker_configs[2].search_space["heavy_size"] = tune.grid_search([0.005, 0.01, 0.02 - (256.0 / 32768)])
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.005, 0.01]),
                    "epsilon": tune.grid_search([0.2, 0.4]),
                    "delta": tune.grid_search([0.2, 0.4])
                }

            elif sparsity_objective == 5:
                # Adaptive sampling with PQCache
                config.masker_configs[2].search_space["heavy_size"] = tune.grid_search([0.01, 0.025, 0.05])
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.01, 0.025]),
                    "epsilon": tune.grid_search([0.15, 0.25]),
                    "delta": tune.grid_search([0.15, 0.25])
                }

            elif sparsity_objective == 10:
                # Adaptive sampling with PQCache
                config.masker_configs[2].search_space["heavy_size"] = tune.grid_search([0.025, 0.05, 0.075])
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0.025, 0.05, 0.075]),
                    "epsilon": tune.grid_search([0.025, 0.05, 0.075]),
                    "delta": tune.grid_search([0.025, 0.05, 0.075])
                }
            elif sparsity_objective == 15:
                # Adaptive sampling with PQCache
                config.masker_configs[2].search_space["heavy_size"] = tune.grid_search([0.05, 0.1, 0.15])
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.05, 0.1]),
                    "epsilon": tune.grid_search([0.01, 0.04, 0.1]),
                    "delta": tune.grid_search([0.01, 0.04, 0.1])
                }

            elif sparsity_objective == 20:
                # Adaptive sampling with PQCache
                config.masker_configs[2].search_space["heavy_size"] = tune.grid_search([0.05, 0.1, 0.15])
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0.05, 0.1, 0.15]),
                    "epsilon": tune.grid_search([0.01, 0.04, 0.1]),
                    "delta": tune.grid_search([0.01, 0.04, 0.1])
                }
            else:
                raise ValueError(f"sparsity_objective not supported: {sparsity_objective}")
            
            # Set validity constraint to use the correct sparsity value for comparison
            config.validity_constraint = partial(_validity_check, sparsity_val=sparsity_val)
            # Set objective function
            config.objective = sparsity_objective

            to_optimize_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs


