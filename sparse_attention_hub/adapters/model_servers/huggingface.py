"""HuggingFace implementation of ModelServer for centralized model management."""

import gc
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import ModelServerConfig
from ..utils.exceptions import (
    ModelCreationError,
    ResourceCleanupError,
    TokenizerCreationError,
)
from ..utils.gpu_utils import cleanup_gpu_memory
from .base import ModelServer
from .model_registry import (
    ModelRegistryError,
    RegistryEntry,
    load_model_registry,
    resolve_model_class,
)


class ModelServerHF(ModelServer):
    """HuggingFace implementation of ModelServer.

    Manages HuggingFace models and tokenizers with centralized resource tracking,
    reference counting, and GPU memory management.
    """

    def __init__(self, config: Optional[ModelServerConfig] = None) -> None:
        """Initialize HuggingFace ModelServer.

        Args:
            config: Configuration for ModelServer behavior
        """
        super().__init__(config)

        # Cache registry to avoid reading/parsing YAML repeatedly.
        self._model_registry: Optional[Dict[str, RegistryEntry]] = None
        self._model_registry_path: Optional[str] = None

    def _get_model_registry(self) -> Dict[str, RegistryEntry]:
        registry_path = getattr(self.config, "model_registry_path", None)
        if not registry_path:
            return {}

        if (
            self._model_registry is not None
            and self._model_registry_path == registry_path
        ):
            return self._model_registry

        self._model_registry_path = registry_path
        self._model_registry = load_model_registry(registry_path)
        return self._model_registry

    def _create_model(
        self, model_name: str, gpu_id: Optional[int], model_kwargs: Dict[str, Any]
    ) -> Any:
        """Create a HuggingFace model instance.

        Args:
            model_name: Name of the HuggingFace model to create
            gpu_id: GPU ID to place the model on (None for CPU)
            model_kwargs: Additional model creation arguments

        Returns:
            Created HuggingFace model instance

        Raises:
            ModelCreationError: If model creation fails
        """
        try:
            # Create model using HuggingFace transformers
            self.logger.debug(
                f"Loading HuggingFace model: {model_name} with kwargs: {model_kwargs}"
            )

            effective_kwargs = dict(model_kwargs or {})
            model_cls = AutoModelForCausalLM

            # Optional registry support (no-op unless configured)
            registry_path = getattr(self.config, "model_registry_path", None)
            if registry_path:
                registry = self._get_model_registry()
                entry = registry.get(model_name)

                if entry is None:
                    self.logger.warning(
                        f"Model '{model_name}' is not registered in the model registry at "
                        f"{self._model_registry_path}. Proceeding only if allow_unregistered_models=True."
                    )
                    if not getattr(self.config, "allow_unregistered_models", True):
                        raise ModelRegistryError(
                            f"Model '{model_name}' is not registered and allow_unregistered_models=False"
                        )
                else:
                    # Start with registry defaults, then override with call-site kwargs
                    effective_kwargs = dict(entry.default_model_kwargs)
                    effective_kwargs.update(model_kwargs or {})

                    resolved = resolve_model_class(entry.model_class)
                    if resolved is not None:
                        model_cls = resolved

            try:
                model = model_cls.from_pretrained(model_name, **effective_kwargs)
            except ValueError as e:
                # Transformers may reject flex_attention for some architectures (e.g. Qwen3-Next).
                # If the caller requested flex_attention, fall back to eager so the model can load.
                requested_attn_impl = effective_kwargs.get(
                    "attn_implementation"
                ) or effective_kwargs.get("attention_implementation")
                message = str(e)
                if (
                    requested_attn_impl in {"flex_attention", "flex"}
                    and "flex_attention" in message
                    and "does not support" in message
                ):
                    self.logger.warning(
                        f"Model {model_name} does not support flex_attention. Falling back to attn_implementation='eager'."
                    )
                    effective_kwargs.pop("attention_implementation", None)
                    effective_kwargs["attn_implementation"] = "eager"
                    model = model_cls.from_pretrained(model_name, **effective_kwargs)
                else:
                    raise

            # Handle device placement
            # If device_map is set, placement may be handled by accelerate/sharding.
            if effective_kwargs.get("device_map") is not None:
                self.logger.debug(
                    f"device_map is set for {model_name}. Skipping explicit .to(device) placement"
                )
            elif gpu_id is not None:
                if torch.cuda.is_available():
                    if isinstance(gpu_id, str) and gpu_id.startswith("cuda"):
                        device = gpu_id
                    else:
                        device = "cuda:" + f"{gpu_id}"
                    self.logger.debug(f"Moving model {model_name} to device: {device}")
                    model = model.to(device)  # type: ignore[arg-type]
                else:
                    self.logger.warning(
                        f"CUDA not available, placing model {model_name} on CPU instead of GPU {gpu_id}"
                    )
                    model = model.to(torch.device("cpu"))  # type: ignore[arg-type]
            else:
                # Explicitly place on CPU
                model = model.to(torch.device("cpu"))  # type: ignore[arg-type]

            self.logger.info(
                f"Successfully created HuggingFace model: {model_name} on {'GPU' if gpu_id is not None else 'CPU'}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Failed to create HuggingFace model {model_name}: {e}")
            raise ModelCreationError(model_name, gpu_id, e)

    def _create_tokenizer(
        self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
    ) -> Any:
        """Create a HuggingFace tokenizer instance.

        Args:
            tokenizer_name: Name of the HuggingFace tokenizer to create
            tokenizer_kwargs: Additional tokenizer creation arguments

        Returns:
            Created HuggingFace tokenizer instance

        Raises:
            TokenizerCreationError: If tokenizer creation fails
        """
        try:
            self.logger.debug(
                f"Loading HuggingFace tokenizer: {tokenizer_name} with kwargs: {tokenizer_kwargs}"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, **tokenizer_kwargs
            )

            # Set pad token if not already set (common requirement for HuggingFace models)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.debug(
                    f"Set pad_token to eos_token for tokenizer: {tokenizer_name}"
                )

            self.logger.info(
                f"Successfully created HuggingFace tokenizer: {tokenizer_name}"
            )
            return tokenizer

        except Exception as e:
            self.logger.error(
                f"Failed to create HuggingFace tokenizer {tokenizer_name}: {e}"
            )
            raise TokenizerCreationError(tokenizer_name, e)

    def _delete_model(self, model: Any, gpu_id: Optional[int]) -> None:
        """Delete a HuggingFace model and clean up its resources.

        Args:
            model: HuggingFace model instance to delete
            gpu_id: GPU ID where the model is placed

        Raises:
            ResourceCleanupError: If cleanup fails
        """
        try:
            self.logger.debug(
                f"Deleting HuggingFace model on {'GPU' if gpu_id is not None else 'CPU'}"
            )

            # Move model to CPU before deletion to ensure GPU memory is freed
            if gpu_id is not None and torch.cuda.is_available():
                try:
                    model = model.to(torch.device("cpu"))
                    self.logger.debug("Moved model to CPU before deletion")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to move model to CPU before deletion: {e}"
                    )

            # Explicitly delete the model
            del model

            # Force garbage collection
            gc.collect()

            # Clean up GPU memory if applicable
            if gpu_id is not None and torch.cuda.is_available():
                try:
                    cleanup_gpu_memory(gpu_id)
                    self.logger.debug(f"Cleaned up GPU memory for GPU {gpu_id}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to cleanup GPU memory for GPU {gpu_id}: {e}"
                    )

            self.logger.debug(
                "Successfully deleted HuggingFace model and cleaned up resources"
            )

        except Exception as e:
            self.logger.error(f"Failed to delete HuggingFace model: {e}")
            raise ResourceCleanupError("model", "model", e)

    def _delete_tokenizer(self, tokenizer: Any) -> None:
        """Delete a HuggingFace tokenizer and clean up its resources.

        Args:
            tokenizer: HuggingFace tokenizer instance to delete

        Raises:
            ResourceCleanupError: If cleanup fails
        """
        try:
            self.logger.debug("Deleting HuggingFace tokenizer")

            # Explicitly delete the tokenizer
            del tokenizer

            # Force garbage collection
            gc.collect()

            self.logger.debug("Successfully deleted HuggingFace tokenizer")

        except Exception as e:
            self.logger.error(f"Failed to delete HuggingFace tokenizer: {e}")
            raise ResourceCleanupError("tokenizer", "tokenizer", e)

    def get_model_with_device_info(
        self, model_name: str, gpu: Optional[int], model_kwargs: Dict[str, Any]
    ) -> Tuple[Any, str]:
        """Get a model and return both the model and device information.

        This is a convenience method specific to HuggingFace that provides additional
        device information for debugging and monitoring.

        Args:
            model_name: Name of the model
            gpu: GPU ID to place the model on (None for CPU)
            model_kwargs: Additional model creation arguments

        Returns:
            Tuple of (model, device_info_string)
        """
        model = self.get_model(model_name, gpu, model_kwargs)

        # Determine actual device placement
        if hasattr(model, "device"):
            device_info = str(model.device)
        elif gpu is not None:
            device_info = "cuda:" + f"{gpu}" if torch.cuda.is_available() else "cpu"
        else:
            device_info = "cpu"

        return model, device_info

    def validate_gpu_availability(self, gpu_id: int) -> bool:
        """Validate that a specific GPU is available for model placement.

        Args:
            gpu_id: GPU ID to validate

        Returns:
            True if GPU is available, False otherwise
        """
        if not torch.cuda.is_available():
            return False

        try:
            if gpu_id >= torch.cuda.device_count():
                return False

            # Test GPU access
            with torch.cuda.device(gpu_id):
                test_tensor = torch.zeros(1, device="cuda:" + f"{gpu_id}")
                del test_tensor
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            self.logger.warning(f"GPU {gpu_id} validation failed: {e}")
            return False

    def get_gpu_memory_info(
        self, gpu_id: Optional[int] = None
    ) -> Dict[str, Union[float, str]]:
        """Get GPU memory information for monitoring purposes.

        Args:
            gpu_id: GPU ID to query (None for current device)

        Returns:
            Dictionary with memory information in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        try:
            device = gpu_id if gpu_id is not None else torch.cuda.current_device()

            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - allocated,
            }

        except Exception as e:
            self.logger.error(f"Failed to get GPU memory info for GPU {gpu_id}: {e}")
            return {"error": str(e)}
