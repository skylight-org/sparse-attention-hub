"""Configuration builder for VAttention Oracle TopK configurations."""

from functools import partial
from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


def _validity_check(config: ResearchAttentionConfig, sparsity_val: float) -> bool:
    """Check if the config meets the sparsity constraint."""
    return (config.masker_configs[2].heavy_size + config.masker_configs[3].base_rate_sampling) <= sparsity_val


@register_builder("vattention_oracle")
class VAttentionOracleConfigBuilder(BaseConfigBuilder):
    """Builder for VAttention Oracle TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all sparse attention configurations.

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
            classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig, AdaptiveSamplingMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "vattention_oracle", "sparsity_obj": sparsity_objective})
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                OracleTopKConfig(heavy_size=0.05),  # Middle value from search space
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.05,  # Middle value
                    epsilon=0.05,  # Middle value
                    delta=0.05,  # Middle value
                    init_offset=128,  # Middle value
                    local_offset=128  # Middle value
                )
            ])
            
            if sparsity_objective == 2:
                # Adaptive sampling with oracle top k
                config.masker_configs[2].search_space = {
                    "heavy_size": tune.grid_search([0.005, 0.01, 0.02 - (256.0 / 32768)]),
                }
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.005, 0.01]),
                    "epsilon": tune.grid_search([0.1, 0.2, 0.3, 0.4]),
                    "delta": tune.grid_search([0.1, 0.2, 0.3, 0.4])
                }

            elif sparsity_objective == 5:
                # Adaptive sampling with oracle top k
                config.masker_configs[2].search_space = {
                    "heavy_size": tune.grid_search([0.01, 0.025, 0.05]),
                }
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.01, 0.02, 0.03]),
                    "epsilon": tune.grid_search([0.05, 0.1, 0.2, 0.3]),
                    "delta": tune.grid_search([0.05, 0.1, 0.2, 0.3])
                }

            elif sparsity_objective == 10:
                # Adaptive sampling with oracle top k
                config.masker_configs[2].search_space = {
                    "heavy_size": tune.grid_search([0.025, 0.05, 0.075, 0.1]),
                }
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.025, 0.05, 0.075]),
                    "epsilon": tune.grid_search([0.025, 0.05, 0.075]),
                    "delta": tune.grid_search([0.025, 0.05, 0.075])
                }
            elif sparsity_objective == 15:
                # Adaptive sampling with oracle top k
                config.masker_configs[2].search_space = {
                    "heavy_size": tune.grid_search([0.05, 0.1, 0.15]),
                }
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0, 0.04, 0.06, 0.1]),
                    "epsilon": tune.grid_search([0.01, 0.025, 0.05, 0.1]),
                    "delta": tune.grid_search([0.01, 0.025, 0.05, 0.1])
                }

            elif sparsity_objective == 20:
                # Adaptive sampling with oracle top k
                config.masker_configs[2].search_space = {
                    "heavy_size": tune.grid_search([0.05, 0.1, 0.15]),
                }
                config.masker_configs[3].search_space = {
                    "base_rate_sampling": tune.grid_search([0.05, 0.1, 0.15]),
                    "epsilon": tune.grid_search([0.01, 0.025, 0.05, 0.1]),
                    "delta": tune.grid_search([0.01, 0.025, 0.05, 0.1])
                }
            else:
                raise ValueError(f"sparsity_objective not supported: {sparsity_objective}")
            
            # Set validity constraint to use the correct sparsity value for comparison
            config.validity_constraint = partial(_validity_check, sparsity_val=sparsity_val)
            # Set objective function
            config.objective = sparsity_objective

            to_optimize_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

