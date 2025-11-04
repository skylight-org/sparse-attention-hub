"""Configuration builder for Random Sampling attention."""

from typing import List, Optional, Tuple

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
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all Random Sampling attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        classes = [SinkMaskerConfig, LocalMaskerConfig, RandomSamplingMaskerConfig]
        
        
        for budget_size in [0.02, 0.05, 0.1, 0.2]:
            name: str = get_masker_list_name(classes, other_params={"budget_size": budget_size})
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),  # Middle value from search space
                LocalMaskerConfig(window_size=128),  # Middle value from search space
                RandomSamplingMaskerConfig(sampling_rate=budget_size- (256.0 / 32768))  # Middle value from search space
            ])
            optimal_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

