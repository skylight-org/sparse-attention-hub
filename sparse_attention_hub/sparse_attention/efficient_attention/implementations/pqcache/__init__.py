"""PQCache implementation for efficient attention.

This package provides PQCache implementations for the research backend.
PQCache combines sink tokens, local attention, and PQ-based top-K selection for efficient long-context inference.
"""

from .research import (
    PQCacheResearchBackend,
    PQCacheResearchBackendConfig,
)

__all__ = [
    "PQCacheResearchBackend",
    "PQCacheResearchBackendConfig",
]

