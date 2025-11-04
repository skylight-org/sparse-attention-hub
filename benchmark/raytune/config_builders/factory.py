"""Factory for creating configuration builders."""

from typing import Dict, List, Optional, Tuple

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

from .base import BaseConfigBuilder

# Registry of available config builders
_BUILDER_REGISTRY: Dict[str, type[BaseConfigBuilder]] = {}


def register_builder(name: str):
    """Decorator to register a configuration builder.
    
    Usage:
        @register_builder("my_builder")
        class MyBuilder(BaseConfigBuilder):
            ...
    
    Args:
        name: Name to register the builder under
    """
    def decorator(builder_class: type[BaseConfigBuilder]) -> type[BaseConfigBuilder]:
        if not issubclass(builder_class, BaseConfigBuilder):
            raise TypeError(f"Builder class must inherit from BaseConfigBuilder")
        _BUILDER_REGISTRY[name] = builder_class
        return builder_class
    return decorator


def get_config_builder(builder_name: str) -> BaseConfigBuilder:
    """Get a configuration builder by name.
    
    Args:
        builder_name: Name of the builder (e.g., "double_sparsity", "vattention_oracle")
        
    Returns:
        Instance of the requested builder
        
    Raises:
        ValueError: If builder_name is not registered
    """
    if builder_name not in _BUILDER_REGISTRY:
        available = ", ".join(_BUILDER_REGISTRY.keys())
        raise ValueError(f"Unknown builder '{builder_name}'. Available builders: {available}")
    
    builder_class = _BUILDER_REGISTRY[builder_name]
    return builder_class()


def get_all_config_builders() -> Dict[str, BaseConfigBuilder]:
    """Get all registered configuration builders.
    
    Returns:
        Dictionary mapping builder names to builder instances
    """
    return {name: get_config_builder(name) for name in _BUILDER_REGISTRY.keys()}


def build_all_configs(
    weight_file: Optional[str] = None,
    objective: str = "default",
    builder_names: Optional[List[str]] = None,
    **kwargs
) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
           List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
    """Build configs using all specified builders.
    
    Args:
        weight_file: Path to weight file
        objective: Objective function name
        builder_names: List of builder names to use. If None, uses all builders.
        **kwargs: Additional parameters passed to each builder
        
    Returns:
        Tuple of (optimal_configs, to_optimize_configs) aggregated from all builders
    """
    if builder_names is None:
        builders = get_all_config_builders()
    else:
        builders = {name: get_config_builder(name) for name in builder_names}
    
    all_optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
    all_to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
    
    for builder_name, builder in builders.items():
        optimal_configs, to_optimize_configs = builder.build_configs(
            weight_file=weight_file,
            objective=objective,
            **kwargs
        )
        all_optimal_configs.extend(optimal_configs)
        all_to_optimize_configs.extend(to_optimize_configs)
    
    return all_optimal_configs, all_to_optimize_configs

