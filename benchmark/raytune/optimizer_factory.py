"""
Optimizer Factory for Sparse Attention Configurations.

This module provides the core engine for creating optimizer objects that can
translate sparse attention masker configurations into Ray Tune search spaces.

The key design principle is that each masker's configuration class is responsible
for defining its own tunable parameters via a `get_search_space()` static method.
This factory then assembles these individual search spaces for optimization.
"""
import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Optional

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)

class SparseConfigOptimizer(ABC):
    """
    Abstract Base Class for sparse attention config optimizers.

    An optimizer's main responsibilities are to create a search space for Ray Tune
    and to instantiate a valid attention configuration from a set of parameters
    produced by a Ray Tune trial.
    """

    @abstractmethod
    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """Creates the Ray Tune search space for a given task."""
        pass

    @abstractmethod
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Creates an attention configuration instance from a dictionary of parameters."""
        pass

class CompositeConfigOptimizer(SparseConfigOptimizer):
    """Optimizer for a `ResearchAttentionConfig` composed of multiple maskers."""

    def __init__(self, research_attention_config: Optional[ResearchAttentionConfig] = None):
        self.research_attention_config = research_attention_config

    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """
        Creates a combined search space from all component masker configs.
        Each parameter is prefixed with its masker's name to prevent conflicts.
        """
        involved_masker_classes = [type(masker_config) for masker_config in self.research_attention_config.masker_configs]
        if len(set(involved_masker_classes)) != len(involved_masker_classes):
            raise ValueError("`research_attention_config` must have unique masker classes")
            
        combined_space = {}
        for masker_config in self.research_attention_config.masker_configs:
            masker_name = type(masker_config).__name__
            masker_space = masker_config.search_space
            for param_name, param_space in masker_space.items():
                combined_space[f"{masker_name}_{param_name}"] = param_space
        return combined_space

    def create_config_from_params(self, params: Dict[str, Any]) -> ResearchAttentionConfig:
        """Creates a ResearchAttentionConfig instance from the combined parameters."""
        masker_instances = []
        for masker_config in self.research_attention_config.masker_configs:
            masker_name = type(masker_config).__name__
            prefix = f"{masker_name}_"
            masker_params = {
                k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)
            }
            masker_config_copy = copy.deepcopy(masker_config)
            for key, value in masker_params.items():
                setattr(masker_config_copy, key, value)
            masker_instances.append(masker_config_copy)
            
        new_config =  ResearchAttentionConfig(masker_configs=masker_instances)
        if hasattr(self.research_attention_config, 'validity_constraint'):
            new_config.validity_constraint = self.research_attention_config.validity_constraint
        return new_config

def create_optimizer(research_attention_config: Optional[ResearchAttentionConfig] = None) -> SparseConfigOptimizer:
    """
    Factory function to create the appropriate optimizer.

    This function inspects the list of masker configurations and returns the
    correct optimizer type.
    
    Args:
        masker_configs: List of masker configuration classes to optimize
        template_config: Optional template configuration with fixed parameters
    """
    if not isinstance(research_attention_config, ResearchAttentionConfig) or not research_attention_config:
        raise ValueError("`masker_classes` must be a non-empty list of config classes and `research_attention_config` must be provided")

    logging.info(f"Creating optimizer for: {[type(c).__name__ for c in research_attention_config.masker_configs]}")

    return CompositeConfigOptimizer(research_attention_config)