"""Configuration builder for dense (no sparse attention) model."""

from typing import List, Optional, Tuple, Dict

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

from .base import BaseConfigBuilder
from .factory import register_builder


@register_builder("dense")
class DenseConfigBuilder(BaseConfigBuilder):
    """Builder for dense (no sparse attention) configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get dense baseline configuration.

        Ignores:
            sparsity_objectives: List[int] - List of sparsity objectives
            memory_objectives: List[int] - List of memory objectives
            model_config: Dict[str, str] - Model configuration
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        # Dense baseline: no sparse attention, so sparse_config and masker_classes are None
        # Since dense doesn't depend on sparsity or memory objectives, we just return a single config
        # with None values (no sparse attention configuration needed)
        optimal_configs.append(("dense", None, None))
        
        return optimal_configs, to_optimize_configs

