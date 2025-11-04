"""Configuration builder for HashAttention TopK attention."""

from typing import List, Optional, Tuple

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("hashattention_topk")
class HashAttentionTopKConfigBuilder(BaseConfigBuilder):
    """Builder for HashAttention TopK sparse attention configurations."""
    
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all HashAttention TopK attention configurations.
        
        Returns list of (name, full_config, masker_classes) tuples.
        
        Note: The configs returned here are only used to determine which masker classes
        to use. The actual parameter values will be determined by Ray Tune search.
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        assert weight_file is not None, "Weight file is required for HashAttention Masker"
        
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        for heavy_size in [0.02, 0.05, 0.1, 0.2]:
            classes = [SinkMaskerConfig, LocalMaskerConfig, HashAttentionTopKMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"heavy_size": heavy_size})
            
            config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                HashAttentionTopKMaskerConfig(
                    heavy_size=heavy_size - (256.0 / 32768),
                    hat_bits=32,
                    hat_mlp_layers=3,
                    hat_mlp_hidden_size=128,
                    hat_mlp_activation="silu",
                    hat_weight_file=weight_file
                ),
            ])
            optimal_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

