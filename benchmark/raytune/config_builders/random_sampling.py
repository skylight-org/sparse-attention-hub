"""Configuration builder for Random Sampling attention."""

from typing import List, Optional, Tuple, Dict

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    RandomSamplingMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("random_sampling")
class RandomSamplingConfigBuilder(BaseConfigBuilder):
    """Builder for Random Sampling sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Random Sampling attention configurations.

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

        classes = [SinkMaskerConfig, LocalMaskerConfig, RandomSamplingMaskerConfig]
        
        for sparsity_objective in sparsity_objectives:
            budget_size: float = float(sparsity_objective) / 100.0
            name: str = get_masker_list_name(classes, other_params={"builder": "random_sampling", "sparsity_obj": sparsity_objective})
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),  # Middle value from search space
                LocalMaskerConfig(window_size=128),  # Middle value from search space
                RandomSamplingMaskerConfig(sampling_rate=budget_size - (256.0 / 32768))  # Middle value from search space
            ])
            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective
            
            optimal_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

