"""Unit tests for the model registry utilities."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any, Dict

import pytest

from sparse_attention_hub.adapters.model_servers import model_registry
from sparse_attention_hub.adapters.model_servers.model_registry import (
    ModelRegistryError,
    RegistryEntry,
    _maybe_construct,
    load_model_registry,
    resolve_model_class,
)


@pytest.mark.unit
class TestMaybeConstruct:
    def test_pass_through_for_non_mapping(self) -> None:
        value = [1, 2, 3]

        assert _maybe_construct(value) is value

    def test_returns_mapping_without_constructor(self) -> None:
        payload: Dict[str, Any] = {"alpha": 1, "beta": {"nested": True}}

        assert _maybe_construct(payload) == payload

    def test_constructs_with_constructor_and_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Dummy:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        monkeypatch.setattr(model_registry, "_import_symbol", lambda name: Dummy)

        spec = {"constructor": "Dummy", "kwargs": {"gamma": 3}}

        result = _maybe_construct(spec)

        assert isinstance(result, Dummy)
        assert result.kwargs == {"gamma": 3}

    def test_constructs_with_type_alias_and_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Dummy:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        monkeypatch.setattr(model_registry, "_import_symbol", lambda name: Dummy)

        spec = {"type": "Dummy", "args": {"delta": "v"}}

        result = _maybe_construct(spec)

        assert isinstance(result, Dummy)
        assert result.kwargs == {"delta": "v"}

    def test_invalid_kwargs_type_raises(self) -> None:
        spec = {"constructor": "Dummy", "kwargs": ["not", "a", "mapping"]}

        with pytest.raises(ModelRegistryError, match="Invalid constructor kwargs"):
            _maybe_construct(spec)


@pytest.mark.unit
class TestLoadModelRegistry:
    def test_empty_path_returns_empty_registry(self) -> None:
        assert load_model_registry("") == {}

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"

        with pytest.raises(ModelRegistryError, match="path does not exist"):
            load_model_registry(str(missing))

    def test_invalid_root_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text("- not a mapping\n", encoding="utf-8")

        with pytest.raises(ModelRegistryError, match="Registry root must be a mapping"):
            load_model_registry(str(path))

    def test_invalid_models_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text("models: [1]\n", encoding="utf-8")

        with pytest.raises(
            ModelRegistryError, match="Registry 'models' must be a mapping"
        ):
            load_model_registry(str(path))

    def test_entry_not_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text("models:\n  demo: value\n", encoding="utf-8")

        with pytest.raises(ModelRegistryError, match="must be a mapping"):
            load_model_registry(str(path))

    def test_default_kwargs_not_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text(
            """
models:
  demo:
    model_class: builtins.dict
    default_model_kwargs: [1]
""".lstrip(),
            encoding="utf-8",
        )

        with pytest.raises(
            ModelRegistryError, match="default_model_kwargs.*must be a mapping"
        ):
            load_model_registry(str(path))

    def test_loads_registry_and_constructs_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text(
            """
models:
  first:
    model_class: builtins.dict
    default_model_kwargs:
      config:
        constructor: builtins.dict
        kwargs:
          a: 1
      name: demo
  second:
    default_model_kwargs: {}
""".lstrip(),
            encoding="utf-8",
        )

        registry = load_model_registry(str(path))

        assert set(registry.keys()) == {"first", "second"}
        first_entry = registry["first"]
        second_entry = registry["second"]

        assert isinstance(first_entry, RegistryEntry)
        assert isinstance(second_entry, RegistryEntry)

        assert first_entry.model_class == "builtins.dict"
        assert first_entry.default_model_kwargs["config"] == {"a": 1}
        assert first_entry.default_model_kwargs["name"] == "demo"

        assert second_entry.model_class is None
        assert second_entry.default_model_kwargs == {}

    def test_missing_pyyaml_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text("models: {}\n", encoding="utf-8")

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any):
            if name == "yaml":
                raise ImportError("missing pyyaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ModelRegistryError, match="PyYAML is required"):
            load_model_registry(str(path))


@pytest.mark.unit
class TestResolveModelClass:
    def test_none_returns_none(self) -> None:
        assert resolve_model_class(None) is None

    def test_resolves_via_import_symbol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        calls: Dict[str, int] = {}

        def fake_import(symbol: str) -> Any:
            calls[symbol] = calls.get(symbol, 0) + 1
            return sentinel

        monkeypatch.setattr(model_registry, "_import_symbol", fake_import)

        assert resolve_model_class("custom.path.Class") is sentinel
        assert calls == {"custom.path.Class": 1}

    def test_import_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raising_import(symbol: str) -> Any:
            raise ImportError("boom")

        monkeypatch.setattr(model_registry, "_import_symbol", raising_import)

        with pytest.raises(ModelRegistryError, match="Failed to resolve model_class"):
            resolve_model_class("missing.module.Class")

    def test_non_dotted_import_failure_is_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raising_import(symbol: str) -> Any:
            raise AttributeError("no attr")

        monkeypatch.setattr(model_registry, "_import_symbol", raising_import)

        with pytest.raises(ModelRegistryError, match="Failed to resolve model_class"):
            resolve_model_class("AutoModel")
