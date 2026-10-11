"""Shim redirecting to database.index_store."""

from database.index_store import IndexStore, index_store

__all__ = ["IndexStore", "index_store"]
