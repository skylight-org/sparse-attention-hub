"""StreamingLLM implementation for efficient attention.

This package provides StreamingLLM implementations for both research and native backends.
StreamingLLM combines sink tokens and local attention patterns for efficient long-context inference.
"""

from .native import (
    StreamingLLMNativeBackend,
    StreamingLLMNativeBackendConfig,
)
from .research import (
    StreamingLLMResearchBackend,
    StreamingLLMResearchBackendConfig,
)

__all__ = [
    "StreamingLLMResearchBackend",
    "StreamingLLMResearchBackendConfig",
    "StreamingLLMNativeBackend",
    "StreamingLLMNativeBackendConfig",
]

