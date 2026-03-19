"""Configuration builder for HashAttention TopK attention."""

from typing import List, Optional, Tuple, Dict

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name
import os
import logging

logger = logging.getLogger(__name__)


@register_builder("hashattention_topk")
class HashAttentionTopKConfigBuilder(BaseConfigBuilder):
    """Builder for HashAttention TopK sparse attention configurations."""
    
    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Get all HashAttention TopK attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
            model_config: Dict[str, str] - Model configuration (hash_attention_weight_file extracted from it)

        Ignores:
            memory_objectives: List[int] - List of memory objectives
        
        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        weight_file: str = model_config.get("hash_attention_weight_file")
        
        
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        if not weight_file or not os.path.isfile(weight_file):
            logger.warning(f"Weight file {weight_file} for model {model_config['model_name']} does not exist. Skipping HashAttention TopK configurations.")
            return optimal_configs, to_optimize_configs

        for sparsity_objective in sparsity_objectives:
            heavy_size: float = float(sparsity_objective) / 100.0
            classes = [SinkMaskerConfig, LocalMaskerConfig, HashAttentionTopKMaskerConfig]
            name: str = get_masker_list_name(classes, other_params={"builder": "hashattention_topk", "sparsity_obj": sparsity_objective})
            
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
            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective
            
            optimal_configs.append((name, config, classes))
        
        return optimal_configs, to_optimize_configs

