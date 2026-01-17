"""Model registry utilities.

This is intentionally opt-in: nothing will load/parse YAML unless
`ModelServerConfig.model_registry_path` is set.

The registry can:
- mark models as verified
- specify an explicit Transformers model class to use (vs AutoModelForCausalLM)
- supply default kwargs for `from_pretrained`
- construct config objects (e.g. quantization_config) from a constructor spec

Example YAML:

models:
  mistralai/Ministral-3-8B-Instruct-2512:
    verified: true
    model_class: Mistral3ForConditionalGeneration
    default_model_kwargs:
      device_map: auto
      quantization_config:
        constructor: FineGrainedFP8Config
        kwargs:
          dequantize: true
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class ModelRegistryError(RuntimeError):
    """Raised when the model registry cannot be loaded or resolved."""


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    verified: bool
    model_class: Optional[str]
    default_model_kwargs: Dict[str, Any]


def _import_symbol(symbol: str) -> Any:
    """Import a symbol by dotted path, or from transformers if not dotted."""

    if "." in symbol:
        module_path, _, attr = symbol.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, attr)

    module = importlib.import_module("transformers")
    return getattr(module, symbol)


def _maybe_construct(value: Any) -> Any:
    """Construct objects from a constructor spec dict.

    Supported forms:
      - {"constructor": "FineGrainedFP8Config", "kwargs": {...}}
      - {"type": "FineGrainedFP8Config", "args": {...}} (alias)

    Anything else is returned unchanged.
    """

    if not isinstance(value, Mapping):
        return value

    constructor = value.get("constructor") or value.get("type")
    if not constructor:
        return value

    kwargs = value.get("kwargs") or value.get("args") or {}
    if not isinstance(kwargs, Mapping):
        raise ModelRegistryError(
            f"Invalid constructor kwargs for {constructor}: expected mapping, got {type(kwargs)}"
        )

    cls = _import_symbol(str(constructor))
    return cls(**dict(kwargs))


def load_model_registry(path: str) -> Dict[str, RegistryEntry]:
    """Load a YAML model registry file."""

    if not path:
        return {}

    if not os.path.exists(path):
        raise ModelRegistryError(f"Model registry path does not exist: {path}")

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ModelRegistryError(
            "PyYAML is required to load a model registry; install with `pip install pyyaml`"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, Mapping):
        raise ModelRegistryError(f"Registry root must be a mapping, got {type(data)}")

    models = data.get("models") or {}
    if not isinstance(models, Mapping):
        raise ModelRegistryError(
            f"Registry 'models' must be a mapping, got {type(models)}"
        )

    registry: Dict[str, RegistryEntry] = {}
    for model_id, raw in models.items():
        if not isinstance(raw, Mapping):
            raise ModelRegistryError(
                f"Registry entry for {model_id} must be a mapping, got {type(raw)}"
            )

        default_kwargs = raw.get("default_model_kwargs") or {}
        if not isinstance(default_kwargs, Mapping):
            raise ModelRegistryError(
                f"default_model_kwargs for {model_id} must be a mapping, got {type(default_kwargs)}"
            )

        normalized_kwargs: Dict[str, Any] = {
            str(k): _maybe_construct(v) for k, v in default_kwargs.items()
        }

        registry[str(model_id)] = RegistryEntry(
            model_id=str(model_id),
            verified=bool(raw.get("verified", False)),
            model_class=(str(raw["model_class"]) if raw.get("model_class") else None),
            default_model_kwargs=normalized_kwargs,
        )

    return registry


def resolve_model_class(model_class: Optional[str]) -> Optional[Any]:
    """Resolve a Transformers model class by name/path."""

    if model_class is None:
        return None

    try:
        return _import_symbol(model_class)
    except Exception as e:
        raise ModelRegistryError(f"Failed to resolve model_class {model_class}: {e}") from e
