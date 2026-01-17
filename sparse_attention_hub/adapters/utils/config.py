"""Configuration classes for ModelServer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelServerConfig:
    """Configuration for ModelServer behavior.

    Attributes:
        delete_on_zero_reference: If True, models/tokenizers are deleted immediately when reference count reaches 0.
                                 If False, they remain in memory until explicit cleanup.
        enable_stats_logging: Whether to enable detailed statistics logging.

        model_registry_path: Optional path to a YAML file describing supported/verified models and default load kwargs.
        require_verified_models: If True and a registry is configured, model loading is blocked unless the model is
                                marked as verified.
        allow_unregistered_models: If False and a registry is configured, model loading is blocked unless the model is
                                  present in the registry.
    """

    delete_on_zero_reference: bool = False  # Lazy deletion by default
    enable_stats_logging: bool = True

    model_registry_path: Optional[str] = None
    require_verified_models: bool = False
    allow_unregistered_models: bool = True
