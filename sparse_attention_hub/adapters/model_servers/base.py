"""Abstract base class for ModelServer implementations."""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Type

from ..utils.config import ModelServerConfig
from ..utils.exceptions import (
    ModelCreationError,
    ReferenceCountError,
    ResourceCleanupError,
    TokenizerCreationError,
)
from ..utils.model_utils import generate_model_key, generate_tokenizer_key


@dataclass
class ModelEntry:
    """Entry for tracking a loaded model.

    Attributes:
        model: The loaded model object
        reference_count: Number of active references to this model
        gpu_id: GPU ID where model is placed (None for CPU)
        model_kwargs: Model creation arguments
        creation_time: When the model was created
        last_access_time: When the model was last accessed
    """

    model: Any
    reference_count: int
    gpu_id: Optional[int]  # None implies CPU
    model_kwargs: Dict[str, Any]
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)


@dataclass
class TokenizerEntry:
    """Entry for tracking a loaded tokenizer.

    Attributes:
        tokenizer: The loaded tokenizer object
        reference_count: Number of active references to this tokenizer
        tokenizer_kwargs: Tokenizer creation arguments
        creation_time: When the tokenizer was created
        last_access_time: When the tokenizer was last accessed
    """

    tokenizer: Any
    reference_count: int
    tokenizer_kwargs: Dict[str, Any]
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)


class ModelServer(ABC):
    """Abstract singleton base class for centralized model and tokenizer management.

    This class provides a centralized registry for models and tokenizers to eliminate
    duplication and improve resource management. Concrete implementations must provide
    framework-specific creation and deletion methods.

    We can only have one kind of concrete class in the system at a time. For example,
    ModelServerHF or ModelServerVLLM. We can only have one of them.
    """

    _cls: Optional[Type["ModelServer"]] = None
    _instance: Optional["ModelServer"] = None
    _lock = threading.Lock()
    _initialized: bool = False  # Type annotation for instance attribute

    def __new__(cls, config: Optional[ModelServerConfig] = None) -> "ModelServer":
        """Singleton pattern implementation with thread safety."""
        with ModelServer._lock:
            if ModelServer._instance is None:
                ModelServer._instance = super().__new__(cls)
                ModelServer._instance._initialized = False
                ModelServer._cls = cls
            else:
                # check that the concrete class is the same and config matches exactly
                if ModelServer._cls != cls:
                    raise ValueError(
                        "Trying to create ModelServer with different concrete class when instance already exists"
                    )
                instance = ModelServer._instance
                if config and hasattr(instance, "config") and instance.config != config:
                    raise ValueError(
                        "Trying to create ModelServer with different config when instance already exists"
                    )

            return ModelServer._instance

    def __init__(self, config: Optional[ModelServerConfig] = None) -> None:
        """Initialize ModelServer (called only once due to singleton pattern).

        Args:
            config: Configuration for ModelServer behavior
        """
        if self._initialized:
            return

        self.config = config or ModelServerConfig()
        self._models: Dict[str, ModelEntry] = {}
        self._tokenizers: Dict[str, TokenizerEntry] = {}
        self._operation_lock = threading.RLock()  # Reentrant lock for nested operations

        # Setup logging
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        self._initialized = True

        if self.config.enable_stats_logging:
            self.logger.info(f"ModelServer initialized with config: {self.config}")

    def __del__(self) -> None:
        """Clean up all resources when ModelServer is destroyed."""
        try:
            model_stats = self.release_memory_and_clean_all(force_delete=True)
            tokenizer_stats = self.release_tokenizers_and_clean_all(force_delete=True)
            if self.config.enable_stats_logging:
                self.logger.info(
                    f"ModelServer cleanup - Models: {model_stats}, Tokenizers: {tokenizer_stats}"
                )
        except Exception as e:
            self.logger.error(f"Error during ModelServer cleanup: {e}")

    # Abstract methods that concrete implementations must provide

    @abstractmethod
    def _create_model(
        self, model_name: str, gpu_id: Optional[int], model_kwargs: Dict[str, Any]
    ) -> Any:
        """Create a model instance using framework-specific methods.

        Args:
            model_name: Name of the model to create
            gpu_id: GPU ID to place the model on (None for CPU)
            model_kwargs: Additional model creation arguments

        Returns:
            Created model instance

        Raises:
            ModelCreationError: If model creation fails
        """
        pass

    @abstractmethod
    def _create_tokenizer(
        self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
    ) -> Any:
        """Create a tokenizer instance using framework-specific methods.

        Args:
            tokenizer_name: Name of the tokenizer to create
            tokenizer_kwargs: Additional tokenizer creation arguments

        Returns:
            Created tokenizer instance

        Raises:
            TokenizerCreationError: If tokenizer creation fails
        """
        pass

    @abstractmethod
    def _delete_model(self, model: Any, gpu_id: Optional[int]) -> None:
        """Delete a model and clean up its resources.

        Args:
            model: Model instance to delete
            gpu_id: GPU ID where the model is placed

        Raises:
            ResourceCleanupError: If cleanup fails
        """
        pass

    @abstractmethod
    def _delete_tokenizer(self, tokenizer: Any) -> None:
        """Delete a tokenizer and clean up its resources.

        Args:
            tokenizer: Tokenizer instance to delete

        Raises:
            ResourceCleanupError: If cleanup fails
        """
        pass

    # Public model management methods

    def get_model(
        self, model_name: str, gpu: Optional[int], model_kwargs: Dict[str, Any]
    ) -> Any:
        """Get a model, creating it if necessary or returning existing instance.

        Args:
            model_name: Name of the model
            gpu: GPU ID to place the model on (None for CPU)
            model_kwargs: Additional model creation arguments

        Returns:
            Model instance

        Raises:
            ModelCreationError: If model creation fails
        """
        key = generate_model_key(model_name, gpu, model_kwargs)

        with self._operation_lock:
            if key in self._models:
                # Model exists, increment reference and return
                entry = self._models[key]
                entry.reference_count += 1
                entry.last_access_time = datetime.now()

                if self.config.enable_stats_logging:
                    self.logger.debug(
                        f"Returning existing model: {key}, ref_count: {entry.reference_count}"
                    )

                return entry.model
            else:
                # Model doesn't exist, create new one
                try:
                    if self.config.enable_stats_logging:
                        self.logger.info(f"Creating new model: {key}")

                    model = self._create_model(model_name, gpu, model_kwargs)

                    entry = ModelEntry(
                        model=model,
                        reference_count=1,
                        gpu_id=gpu,
                        model_kwargs=model_kwargs,
                    )

                    self._models[key] = entry

                    if self.config.enable_stats_logging:
                        self.logger.info(
                            f"Created model: {key}, total_models: {len(self._models)}"
                        )

                    return model

                except Exception as e:
                    raise ModelCreationError(model_name, gpu, e)

    def clear_model(
        self, model_name: str, gpu: Optional[int], model_kwargs: Dict[str, Any]
    ) -> bool:
        """Clear a model reference, potentially deleting it if no references remain.

        Args:
            model_name: Name of the model
            gpu: GPU ID where the model is placed
            model_kwargs: Model creation arguments

        Returns:
            True if model reference was cleared, False if model not found
        """
        key = generate_model_key(model_name, gpu, model_kwargs)

        with self._operation_lock:
            if key not in self._models:
                self.logger.warning(f"Attempted to clear non-existent model: {key}")
                return False

            entry = self._models[key]
            entry.reference_count -= 1

            if self.config.enable_stats_logging:
                self.logger.debug(
                    f"Cleared model reference: {key}, ref_count: {entry.reference_count}"
                )

            if entry.reference_count < 0:
                raise ReferenceCountError(key, "decrement", entry.reference_count)

            if entry.reference_count == 0 and self.config.delete_on_zero_reference:
                self._delete_model_entry(key, entry)

            return True

    def release_memory_and_clean(
        self,
        model_name: str,
        gpu: Optional[int],
        model_kwargs: Dict[str, Any],
        force_delete: bool = False,
    ) -> bool:
        """Release memory for a specific model.

        Args:
            model_name: Name of the model
            gpu: GPU ID where the model is placed
            model_kwargs: Model creation arguments
            force_delete: If True, delete even with active references

        Returns:
            True if model was deleted, False otherwise
        """
        key = generate_model_key(model_name, gpu, model_kwargs)

        with self._operation_lock:
            if key not in self._models:
                self.logger.warning(f"Attempted to clean non-existent model: {key}")
                return False

            entry = self._models[key]

            if entry.reference_count == 0:
                self._delete_model_entry(key, entry)
                return True
            elif force_delete:
                self.logger.warning(
                    f"Force deleting model with {entry.reference_count} active references: {key}"
                )
                self._delete_model_entry(key, entry)
                return True
            else:
                self.logger.warning(
                    f"Cannot delete model with {entry.reference_count} active references: {key}"
                )
                return False

    def release_memory_and_clean_all(
        self, force_delete: bool = False
    ) -> Dict[str, int]:
        """Clean up all models with zero references or force cleanup of all models.

        Args:
            force_delete: If True, delete all models regardless of reference count

        Returns:
            Dictionary with deletion statistics
        """
        deleted_count = 0
        skipped_count = 0

        with self._operation_lock:
            # Create a list of keys to avoid modifying dict during iteration
            model_keys = list(self._models.keys())

            for key in model_keys:
                if (
                    key not in self._models
                ):  # May have been deleted by another operation
                    continue

                entry = self._models[key]

                if entry.reference_count == 0:
                    self._delete_model_entry(key, entry)
                    deleted_count += 1
                elif force_delete:
                    self.logger.warning(
                        f"Force deleting model with {entry.reference_count} active references: {key}"
                    )
                    self._delete_model_entry(key, entry)
                    deleted_count += 1
                else:
                    if self.config.enable_stats_logging:
                        self.logger.debug(
                            f"Skipping model with {entry.reference_count} active references: {key}"
                        )
                    skipped_count += 1

        result = {"deleted": deleted_count, "skipped": skipped_count}
        if self.config.enable_stats_logging:
            self.logger.info(f"Model cleanup completed: {result}")

        return result

    # Public tokenizer management methods

    def get_tokenizer(
        self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
    ) -> Any:
        """Get a tokenizer, creating it if necessary or returning existing instance.

        Args:
            tokenizer_name: Name of the tokenizer
            tokenizer_kwargs: Additional tokenizer creation arguments

        Returns:
            Tokenizer instance

        Raises:
            TokenizerCreationError: If tokenizer creation fails
        """
        key = generate_tokenizer_key(tokenizer_name, tokenizer_kwargs)

        with self._operation_lock:
            if key in self._tokenizers:
                # Tokenizer exists, increment reference and return
                entry = self._tokenizers[key]
                entry.reference_count += 1
                entry.last_access_time = datetime.now()

                if self.config.enable_stats_logging:
                    self.logger.debug(
                        f"Returning existing tokenizer: {key}, ref_count: {entry.reference_count}"
                    )

                return entry.tokenizer
            else:
                # Tokenizer doesn't exist, create new one
                try:
                    if self.config.enable_stats_logging:
                        self.logger.info(f"Creating new tokenizer: {key}")

                    tokenizer = self._create_tokenizer(tokenizer_name, tokenizer_kwargs)

                    entry = TokenizerEntry(
                        tokenizer=tokenizer,
                        reference_count=1,
                        tokenizer_kwargs=tokenizer_kwargs,
                    )

                    self._tokenizers[key] = entry

                    if self.config.enable_stats_logging:
                        self.logger.info(
                            f"Created tokenizer: {key}, total_tokenizers: {len(self._tokenizers)}"
                        )

                    return tokenizer

                except Exception as e:
                    raise TokenizerCreationError(tokenizer_name, e)

    def clear_tokenizer(
        self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
    ) -> bool:
        """Clear a tokenizer reference, potentially deleting it if no references remain.

        Args:
            tokenizer_name: Name of the tokenizer
            tokenizer_kwargs: Tokenizer creation arguments

        Returns:
            True if tokenizer reference was cleared, False if tokenizer not found
        """
        key = generate_tokenizer_key(tokenizer_name, tokenizer_kwargs)

        with self._operation_lock:
            if key not in self._tokenizers:
                self.logger.warning(f"Attempted to clear non-existent tokenizer: {key}")
                return False

            entry = self._tokenizers[key]
            entry.reference_count -= 1

            if self.config.enable_stats_logging:
                self.logger.debug(
                    f"Cleared tokenizer reference: {key}, ref_count: {entry.reference_count}"
                )

            if entry.reference_count < 0:
                raise ReferenceCountError(key, "decrement", entry.reference_count)

            if entry.reference_count == 0 and self.config.delete_on_zero_reference:
                self._delete_tokenizer_entry(key, entry)

            return True

    def release_tokenizer_and_clean(
        self,
        tokenizer_name: str,
        tokenizer_kwargs: Dict[str, Any],
        force_delete: bool = False,
    ) -> bool:
        """Release memory for a specific tokenizer.

        Args:
            tokenizer_name: Name of the tokenizer
            tokenizer_kwargs: Tokenizer creation arguments
            force_delete: If True, delete even with active references

        Returns:
            True if tokenizer was deleted, False otherwise
        """
        key = generate_tokenizer_key(tokenizer_name, tokenizer_kwargs)

        with self._operation_lock:
            if key not in self._tokenizers:
                self.logger.warning(f"Attempted to clean non-existent tokenizer: {key}")
                return False

            entry = self._tokenizers[key]

            if entry.reference_count == 0:
                self._delete_tokenizer_entry(key, entry)
                return True
            elif force_delete:
                self.logger.warning(
                    f"Force deleting tokenizer with {entry.reference_count} active references: {key}"
                )
                self._delete_tokenizer_entry(key, entry)
                return True
            else:
                self.logger.warning(
                    f"Cannot delete tokenizer with {entry.reference_count} active references: {key}"
                )
                return False

    def release_tokenizers_and_clean_all(
        self, force_delete: bool = False
    ) -> Dict[str, int]:
        """Clean up all tokenizers with zero references or force cleanup of all tokenizers.

        Args:
            force_delete: If True, delete all tokenizers regardless of reference count

        Returns:
            Dictionary with deletion statistics
        """
        deleted_count = 0
        skipped_count = 0

        with self._operation_lock:
            # Create a list of keys to avoid modifying dict during iteration
            tokenizer_keys = list(self._tokenizers.keys())

            for key in tokenizer_keys:
                if (
                    key not in self._tokenizers
                ):  # May have been deleted by another operation
                    continue

                entry = self._tokenizers[key]

                if entry.reference_count == 0:
                    self._delete_tokenizer_entry(key, entry)
                    deleted_count += 1
                elif force_delete:
                    self.logger.warning(
                        f"Force deleting tokenizer with {entry.reference_count} active references: {key}"
                    )
                    self._delete_tokenizer_entry(key, entry)
                    deleted_count += 1
                else:
                    if self.config.enable_stats_logging:
                        self.logger.debug(
                            f"Skipping tokenizer with {entry.reference_count} active references: {key}"
                        )
                    skipped_count += 1

        result = {"deleted": deleted_count, "skipped": skipped_count}
        if self.config.enable_stats_logging:
            self.logger.info(f"Tokenizer cleanup completed: {result}")

        return result

    # Utility methods

    def get_model_stats(self) -> Dict[str, Any]:
        """Return statistics about loaded models.

        Returns:
            Dictionary containing model statistics
        """
        with self._operation_lock:
            total_models = len(self._models)
            total_references = sum(
                entry.reference_count for entry in self._models.values()
            )
            zero_ref_models = sum(
                1 for entry in self._models.values() if entry.reference_count == 0
            )

            return {
                "total_models": total_models,
                "total_references": total_references,
                "zero_reference_models": zero_ref_models,
                "model_keys": list(self._models.keys()),
            }

    def get_tokenizer_stats(self) -> Dict[str, Any]:
        """Return statistics about loaded tokenizers.

        Returns:
            Dictionary containing tokenizer statistics
        """
        with self._operation_lock:
            total_tokenizers = len(self._tokenizers)
            total_references = sum(
                entry.reference_count for entry in self._tokenizers.values()
            )
            zero_ref_tokenizers = sum(
                1 for entry in self._tokenizers.values() if entry.reference_count == 0
            )

            return {
                "total_tokenizers": total_tokenizers,
                "total_references": total_references,
                "zero_reference_tokenizers": zero_ref_tokenizers,
                "tokenizer_keys": list(self._tokenizers.keys()),
            }

    def cleanup_unused(self) -> Dict[str, Dict[str, int]]:
        """Clean up models and tokenizers with zero references.

        Returns:
            Dictionary with cleanup statistics
        """
        model_stats = {"deleted": 0, "skipped": 0}
        tokenizer_stats = {"deleted": 0, "skipped": 0}

        with self._operation_lock:
            # Cleanup models
            model_keys = list(self._models.keys())
            for key in model_keys:
                if key not in self._models:
                    continue

                entry = self._models[key]

                if entry.reference_count == 0:
                    self._delete_model_entry(key, entry)
                    model_stats["deleted"] += 1
                else:
                    model_stats["skipped"] += 1

            # Cleanup tokenizers
            tokenizer_keys = list(self._tokenizers.keys())
            for key in tokenizer_keys:
                if key not in self._tokenizers:
                    continue

                tokenizer_entry: TokenizerEntry = self._tokenizers[key]

                if tokenizer_entry.reference_count == 0:
                    self._delete_tokenizer_entry(key, tokenizer_entry)
                    tokenizer_stats["deleted"] += 1
                else:
                    tokenizer_stats["skipped"] += 1

        result = {"models": model_stats, "tokenizers": tokenizer_stats}
        if self.config.enable_stats_logging:
            self.logger.info(f"Cleanup unused completed: {result}")

        return result

    # Private helper methods

    def _delete_model_entry(self, key: str, entry: ModelEntry) -> None:
        """Delete a model entry and clean up resources.

        Args:
            key: Model key
            entry: Model entry to delete
        """
        try:
            self._delete_model(entry.model, entry.gpu_id)
            del self._models[key]

            if self.config.enable_stats_logging:
                self.logger.info(
                    f"Deleted model: {key}, remaining_models: {len(self._models)}"
                )

        except Exception as e:
            raise ResourceCleanupError(key, "model", e)

    def _delete_tokenizer_entry(self, key: str, entry: TokenizerEntry) -> None:
        """Delete a tokenizer entry and clean up resources.

        Args:
            key: Tokenizer key
            entry: Tokenizer entry to delete
        """
        try:
            self._delete_tokenizer(entry.tokenizer)
            del self._tokenizers[key]

            if self.config.enable_stats_logging:
                self.logger.info(
                    f"Deleted tokenizer: {key}, remaining_tokenizers: {len(self._tokenizers)}"
                )

        except Exception as e:
            raise ResourceCleanupError(key, "tokenizer", e)
