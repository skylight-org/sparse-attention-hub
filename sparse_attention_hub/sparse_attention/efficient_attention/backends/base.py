"""Base classes for sparse attention backends."""

from abc import ABC, abstractmethod
from typing import Any


class SparseBackend(ABC):
    """Abstract base class for sparse attention backends.

    This class defines the interface that all sparse attention backend implementations
    must follow. Subclasses should implement the three abstract methods to provide
    backend-specific functionality for attention computation, input transformation,
    and output transformation.

    Example:
        >>> class MyBackend(SparseBackend):
        ...     def attention_computation_backend(self, *args, **kwargs):
        ...         # Implementation here
        ...         pass
        ...     def pre_attention_transforms(self, *args, **kwargs):
        ...         # Implementation here
        ...         pass
        ...     def post_attention_transform(self, attention_output, **kwargs):
        ...         # Implementation here
        ...         pass
    """

    @abstractmethod
    def attention_computation_backend(self, *args: Any, **kwargs: Any) -> Any:
        """Perform attention computation using the backend.

        This method should implement the core attention computation logic
        specific to the backend implementation.

        Args:
            *args: Variable length positional arguments for attention computation.
            **kwargs: Variable length keyword arguments for attention computation.

        Returns:
            Result of the attention computation, type depends on implementation.
        """
        pass

    @abstractmethod
    def pre_attention_transforms(self, *args: Any, **kwargs: Any) -> Any:
        """Transform inputs for the backend.

        This method should transform input tensors or data structures
        into a format suitable for the backend's attention computation.
        This is for compatibility with sparse attention hub

        Args:
            *args: Variable length positional arguments for input transformation.
            **kwargs: Variable length keyword arguments for input transformation.

        Returns:
            Transformed inputs, type depends on implementation.
        """
        pass

    @abstractmethod
    def post_attention_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Transform attention output to ensure it's in the correct format.
        This method should transform the attention output tensor to ensure
        it's in the expected format (typically (B, H, Q, D)).
        This is for compatibility with sparse attention hub.

        Args:
            *args: Variable length positional arguments, typically containing:
                attention_output: Attention output tensor.
            **kwargs: Variable length keyword arguments for output transformation.

        Returns:
            Transformed attention output tensor, type depends on implementation.
        """
        pass

