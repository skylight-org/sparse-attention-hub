"""Configuration builder for Quest TopK attention."""

from typing import List, Optional, Tuple

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    QuestTopKMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("quest_top_k")
class QuestTopKConfigBuilder(BaseConfigBuilder):
    """Builder for Quest TopK sparse attention configurations."""
    
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        memory_objective: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Quest TopK attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Args:
            weight_file: Path to weight file (required but not used for QuestTopK)
            objective: Objective function name (e.g., "sparsity_5")
            memory_objective: Memory objective parameter (e.g., "32") - required
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        assert weight_file is not None, "Weight file is required for QuestTopK Masker"
        assert memory_objective is not None, "memory_objective is required for get_quest_top_k_configs"
        
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        heavy_size: float = float(objective.split("_")[1]) / 100.0 - (256.0 / 32768)
        aux_mem: int = int(memory_objective)
 
        classes = [SinkMaskerConfig, LocalMaskerConfig, QuestTopKMaskerConfig]
        name: str = get_masker_list_name(classes, other_params={"heavy_size": heavy_size, "aux_mem": aux_mem})

        config = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            QuestTopKMaskerConfig(
                heavy_size=heavy_size,
                page_size=128,
                label_bits=16),
        ])
        
        config.masker_configs[2].search_space = {
            "page_size": tune.grid_search([8, 16, 32, 64, 128]),
            "label_bits": tune.grid_search([2, 4, 8, 16]),
        }
        # Memory constraint: similar to double_sparsity pattern
        # For quest_top_k, memory usage depends on page_size and label_bits
        # Adjust this constraint based on actual memory requirements
        config.validity_constraint = lambda config: (aux_mem == 2 * (128 * config.masker_configs[2].label_bits) / config.masker_configs[2].page_size )
        to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs

