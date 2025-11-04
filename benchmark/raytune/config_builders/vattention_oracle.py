"""Configuration builder for VAttention Oracle TopK configurations."""

from typing import List, Optional, Tuple

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


@register_builder("vattention_oracle")
class VAttentionOracleConfigBuilder(BaseConfigBuilder):
    """Builder for VAttention Oracle TopK sparse attention configurations."""
    
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all sparse attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Args:
            weight_file: Path to weight file (required but not used for this config)
            objective: Objective function name (e.g., "sparsity_2", "sparsity_5", etc.)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        assert weight_file is not None, "Weight file is required for HashAttention Masker"
        
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig, AdaptiveSamplingMaskerConfig]
        name: str = get_masker_list_name(classes, other_params={"objective": objective})
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
        
        if objective == "sparsity_2":
            #1. Adaptive sampling with oracle top k
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search([0.005, 0.01, 0.02 - (256.0 / 32768)]),
            }
            config.masker_configs[3].search_space = {
                "base_rate_sampling": tune.grid_search([0, 0.005, 0.01]),
                "epsilon": tune.grid_search([0.1, 0.2, 0.3, 0.4]),
                "delta": tune.grid_search([0.1, 0.2, 0.3, 0.4])
            }

        elif objective == "sparsity_5":
            #1. Adaptive sampling with oracle top k
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search([0.01, 0.025, 0.05]),
            }
            config.masker_configs[3].search_space = {
                "base_rate_sampling": tune.grid_search([0, 0.01, 0.02, 0.03]),
                "epsilon": tune.grid_search([0.05, 0.1, 0.2, 0.3]),
                "delta": tune.grid_search([0.05, 0.1, 0.2, 0.3])
            }

        elif objective == "sparsity_10":
            #1. Adaptive sampling with oracle top k
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search([0.025, 0.05, 0.075, 0.1]),
            }
            config.masker_configs[3].search_space = {
                "base_rate_sampling": tune.grid_search([0, 0.025, 0.05, 0.075]),
                "epsilon": tune.grid_search([0.025, 0.05, 0.075]),
                "delta": tune.grid_search([0.025, 0.05, 0.075])
            }
        elif objective == "sparsity_15":
            #1. Adaptive sampling with oracle top k
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search([0.05, 0.1, 0.15]),
            }
            config.masker_configs[3].search_space = {
                "base_rate_sampling": tune.grid_search([0, 0.04, 0.06, 0.1]),
                "epsilon": tune.grid_search([0.01, 0.025, 0.05, 0.1]),
                "delta": tune.grid_search([0.01, 0.025, 0.05, 0.1])
            }

        elif objective == "sparsity_20":
            #1. Adaptive sampling with oracle top k
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search([0.05, 0.1, 0.15]),
            }
            config.masker_configs[3].search_space = {
                "base_rate_sampling": tune.grid_search([0.05, 0.1, 0.15]),
                "epsilon": tune.grid_search([0.01, 0.025, 0.05, 0.1]),
                "delta": tune.grid_search([0.01, 0.025, 0.05, 0.1])
            }
        else:
            raise ValueError(f"objective not supported: {objective}")
        
        sparsity = float(objective.split("_")[1]) / 100.0
        config.validity_constraint = lambda config: ((config.masker_configs[2].heavy_size + config.masker_configs[3].base_rate_sampling) <= sparsity )

        to_optimize_configs.append((name, config, classes))
        return optimal_configs, to_optimize_configs

