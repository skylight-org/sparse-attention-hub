"""Base class for configuration builders."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig


class BaseConfigBuilder(ABC):
    """Abstract base class for building sparse attention configurations.
    
    Each builder is responsible for creating configurations for a specific
    sparse attention method or combination of methods.
    """
    
    @abstractmethod
    def build_configs(
        self,
        weight_file: Optional[str] = None,
        objective: str = "default",
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Build sparse attention configurations.
        
        Args:
            weight_file: Path to weight file (required for some configs)
            objective: Objective function name (e.g., "sparsity_5", "default")
            **kwargs: Additional parameters specific to the builder
            
        Returns:
            Tuple of (optimal_configs, to_optimize_configs) where each is a list
            of (name, full_config, masker_classes) tuples.
            
            - optimal_configs: Configs that don't need hyperparameter search
            - to_optimize_configs: Configs that need Ray Tune optimization
        """
        pass

