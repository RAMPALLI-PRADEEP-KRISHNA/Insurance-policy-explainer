"""Shim redirecting to database.qdrant_store."""

from database.qdrant_store import (
    QdrantStore,
    qdrant_store,
    get_or_create_qdrant_client,
    get_or_create_shared_client,
)

__all__ = [
    "QdrantStore",
    "qdrant_store",
    "get_or_create_qdrant_client",
    "get_or_create_shared_client",
]
