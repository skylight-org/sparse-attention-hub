"""Configuration builder for dense (no sparse attention) model."""

from typing import List, Optional, Tuple

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

from .base import BaseConfigBuilder
from .factory import register_builder


@register_builder("dense")
class DenseConfigBuilder(BaseConfigBuilder):
    """Builder for dense (no sparse attention) configurations."""
    
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get dense baseline configuration.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        For dense models, sparse_config and masker_classes are None to indicate
        no sparse attention is used.
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        # Dense baseline: no sparse attention, so sparse_config and masker_classes are None
        optimal_configs.append(("dense", None, None))
        
        return optimal_configs, to_optimize_configs

