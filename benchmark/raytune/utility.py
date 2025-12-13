"""
Utility functions for the Ray Tune benchmark.
"""

from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
import logging


@dataclass
class OptimalConfig:
    """Stores optimal configuration found during search."""
    model: str
    task: str
    masker_name: str
    sparse_config: Optional[Any]  # ResearchAttentionConfig, but avoiding circular import
    masker_classes: Optional[List] = field(default=None)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    search_time: float = 0.0
    num_trials: int = 0


def get_masker_list_name(masker_classes: List, other_params: Dict = None) -> str:
    """Generate a name based on the masker classes being used."""
    if not masker_classes:
        return "dense"
    
    # Extract just the key part of each masker name
    parts = []
    for cls in masker_classes:
        name = cls.__name__.replace("MaskerConfig", "").replace("Config", "")
        # Convert camelCase to lowercase
        name = ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
        parts.append(name)
    
    if other_params:
        parts.append("other")
        for key, value in other_params.items():
            parts.append(f"{key}_{value}")
    
    return "_".join(parts)


def create_sparsity_objective(target_density: float, penalty_weight: float = 10.0):
    """Create an objective function that targets a specific sparsity level.
    
    Args:
        target_density: Target density level (e.g., 0.05 for 5% density)
        penalty_weight: Weight for penalty when density exceeds target
        
    Returns:
        Objective function that can be used for optimization
    """
    def objective(error: float, density: float) -> float:
        # Base objective: heavily weight error, lightly weight density
        base_score = 0.99 * error + 0.01 * density
        
        # Add penalty if density exceeds target
        penalty = penalty_weight * max(0, density - target_density)
        
        return base_score + penalty
    
    objective.__name__ = f"objective_sparsity_{int(target_density * 100)}_percent"
    return objective


# Pre-defined objective functions for common sparsity levels
OBJECTIVE_FUNCTIONS = {
    "sparsity_5": create_sparsity_objective(0.05),
    "sparsity_10": create_sparsity_objective(0.10),
    "sparsity_15": create_sparsity_objective(0.15),
    "sparsity_20": create_sparsity_objective(0.20),
    "sparsity_25": create_sparsity_objective(0.25),
    "default": lambda error, density: error + 0.1 * density + (5.0 if density > 0.5 else 0.0),
}


def get_all_masker_config_classes() -> Dict[str, Type]:
    """Dynamically discover all masker config classes.
    
    This function automatically discovers all masker config classes by importing
    the implementation modules and finding classes that end with 'Config'.
    
    Returns:
        Dictionary mapping class names to class types
    """
    import importlib
    import inspect
    
    config_classes = {}
    
    # Import all implementation modules
    implementation_modules = [
        "sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations",
        "sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations",
    ]
    
    for module_name in implementation_modules:
        try:
            module = importlib.import_module(module_name)
            
            # Find all classes in the module that end with 'Config'
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (name.endswith('Config') and 
                    hasattr(obj, '__module__') and 
                    obj.__module__.startswith(module_name)):
                    config_classes[name] = obj
                    
        except ImportError as e:
            logging.warning(f"Could not import {module_name}: {e}")
            continue
    
    return config_classes


def serialize_sparse_config(config: Optional[Any]) -> Optional[Dict]:
    """Convert ResearchAttentionConfig to JSON-serializable format.
    
    Args:
        config: ResearchAttentionConfig instance to serialize
        
    Returns:
        Dictionary representation of the config, or None if config is None
    """
    if config is None:
        return None
        
    # Serialize each masker config
    masker_configs = []
    for masker in config.masker_configs:
        masker_dict = {
            "type": type(masker).__name__,
            "params": {}
        }
        # Add all attributes
        for attr in dir(masker):
            if not attr.startswith("_") and hasattr(masker, attr):
                value = getattr(masker, attr)
                if isinstance(value, (int, float, str, bool, type(None))):
                    masker_dict["params"][attr] = value
        masker_configs.append(masker_dict)
    
    return {
        "type": "ResearchAttentionConfig",
        "masker_configs": masker_configs
    }


def deserialize_sparse_config(data: Optional[Dict]) -> Optional[Any]:
    """Reconstruct ResearchAttentionConfig from JSON data.
    
    Args:
        data: Dictionary representation of the config
        
    Returns:
        ResearchAttentionConfig instance, or None if data is None or invalid
    """
    if data is None:
        return None
        
    if data.get("type") != "ResearchAttentionConfig":
        return None
        
    # Dynamically discover all available masker config classes
    config_map = get_all_masker_config_classes()
    
    #reconstruct masker configs so we let errors propagate for critical failures
    #this ennsures missing files or invalid configurations cause immediate failues in stead of silent skipping maskers and producing misleading results
    masker_configs = []
    for masker_data in data.get("masker_configs", []):
        config_class = config_map.get(masker_data["type"])
        if config_class:
            #create instance with parameters so we let ValueError/FileNotFoundError propagate
            params = masker_data.get("params", {})
            masker_configs.append(config_class(**params))
        else:
            raise ValueError(
                f"Unknown masker config type: {masker_data['type']}. "
                f"Available types: {list(config_map.keys())}"
            )
    
    #import ResearchAttentionConfig here to avoid circular imports
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    return ResearchAttentionConfig(masker_configs=masker_configs) if masker_configs else None
