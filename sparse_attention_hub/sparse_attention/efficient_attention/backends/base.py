"""Base classes for sparse attention backends."""

from abc import ABC, abstractmethod
from typing import Any

from ...utils.mask import Mask


class SparseBackend(ABC):
    """Abstract base class for sparse attention backends.

    This class defines the interface that all sparse attention backend implementations
    must follow. Subclasses should implement the abstract methods to provide
    backend-specific functionality for attention computation and output transformation.

    Example:
        >>> class MyBackend(SparseBackend):
        ...     def attention_computation_backend(self, *args, **kwargs):
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
    def convert_indexer_format(self, sparse_attention_mask: Mask) -> Any:
        """Convert sparse attention mask to backend-specific format.

        This method converts a Mask object to a format suitable for the backend's
        attention computation. Different backends may return different formats.

        Args:
            sparse_attention_mask: Mask object representing the sparse attention pattern.

        Returns:
            Backend-specific format for the sparse attention mask. Type depends on implementation.
        """
        pass

    def check_correctness_with_research_backend(self, other_sparse_attention_mask: Mask, *args) -> bool:
        """Check correctness with another backend.

        This method checks if the current backend's output matches the output of the research backend.

        Args:
            other_sparse_attention_mask: Mask object representing the sparse attention pattern of the other backend.
            *args: Variable length positional arguments for the other backend.

        Returns:
            True if the outputs match, False otherwise.
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

