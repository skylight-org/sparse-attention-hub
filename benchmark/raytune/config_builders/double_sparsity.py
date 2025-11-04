"""Configuration builder for DoubleSparsity attention."""

from typing import List, Optional, Tuple

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


@register_builder("double_sparsity")
class DoubleSparsityConfigBuilder(BaseConfigBuilder):
    """Builder for DoubleSparsity sparse attention configurations."""
    
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        memory_objective: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all double sparsity attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Args:
            weight_file: Path to weight file (required but not used for DoubleSparsity)
            objective: Objective function name (e.g., "sparsity_5")
            memory_objective: Memory objective parameter (e.g., "32") - required
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        assert weight_file is not None, "Weight file is required for HashAttention Masker"
        assert memory_objective is not None, "memory_objective is required for get_double_sparsity_configs"
        
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        heavy_size: float = float(objective.split("_")[1]) / 100.0 - (256.0 / 32768)
        aux_mem: int = int(memory_objective)
 
        classes = [SinkMaskerConfig, LocalMaskerConfig, DoubleSparsityTopKMaskerConfig]
        name: str = get_masker_list_name(classes, other_params={"heavy_size": heavy_size, "aux_mem": aux_mem})

        config = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            DoubleSparsityTopKMaskerConfig(
                heavy_size=heavy_size,
                group_factor=8,
                label_bits=2,
                sorted_channel_file="/data/apdesai/code/DoubleSparse/config/meta-llama/Llama-3.1-8B-Instruct.json",
                channel_selection="q_proj"),
        ])
        
        config.masker_configs[2].search_space = {
            "channel_selection": tune.grid_search(["q_proj", "qk_proj"]),
            "group_factor": tune.grid_search([2, 4, 8, 16]),
            "label_bits": tune.grid_search([1, 2, 4, 8, 16]),
        }
        config.validity_constraint = lambda config: ((128 // config.masker_configs[2].group_factor) * config.masker_configs[2].label_bits == aux_mem)
        to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs

