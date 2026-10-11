"""Persistent local Qdrant vector store for insurance policy chunks.

Enforces document isolation, deterministic point IDs, collection validation,
and seamless compatibility with existing IndexStore interfaces.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import config
from src.ingestion.embedder import Embedder, embedder as default_embedder
from src.models.data_models import Chunk, ChunkType, Document, IndexMetadata
from src.utils.exceptions import (
    CorruptedIndexError,
    IndexNotFoundError,
    IndexStoreError,
    StaleIndexError,
)
from src.utils.logging_utils import logger
from src.utils.timing import timed


import atexit

_QDRANT_CLIENT_REGISTRY: Dict[str, QdrantClient] = {}


def _cleanup_qdrant_clients():
    """Cleanly close all open QdrantClient instances before process exit."""
    for client in list(_QDRANT_CLIENT_REGISTRY.values()):
        try:
            client.close()
        except Exception:
            pass
    _QDRANT_CLIENT_REGISTRY.clear()


atexit.register(_cleanup_qdrant_clients)


def get_or_create_qdrant_client(storage_path: Path) -> QdrantClient:
    """Return shared QdrantClient for a path within the process to prevent file lock contention."""
    resolved = str(Path(storage_path).resolve())
    if resolved not in _QDRANT_CLIENT_REGISTRY:
        _QDRANT_CLIENT_REGISTRY[resolved] = QdrantClient(path=resolved)
    return _QDRANT_CLIENT_REGISTRY[resolved]


class QdrantStore:
    """
    Manages persistent local Qdrant vector storage with strict document isolation.
    """

    def __init__(
        self,
        storage_path: Path = config.QDRANT_STORAGE_PATH,
        indexes_dir: Path = config.INDEXES_DIR,
        collection_base_name: str = config.QDRANT_COLLECTION_NAME,
        embedder_instance: Optional[Embedder] = None,
        schema_version: str = config.INDEX_SCHEMA_VERSION,
        client_instance: Optional[Any] = None,
    ):
        self.storage_path = Path(storage_path)
        self.indexes_dir = Path(indexes_dir)
        self.collection_base_name = collection_base_name
        self.embedder = embedder_instance or default_embedder
        self.schema_version = schema_version
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self._client = client_instance
        self._cache: Dict[str, Tuple[Any, List[Chunk], IndexMetadata]] = {}

    @property
    def client(self) -> QdrantClient:
        """Lazily initialize persistent local QdrantClient via process registry."""
        if self._client is None:
            self._client = get_or_create_qdrant_client(self.storage_path)
        return self._client

    def close(self) -> None:
        """Close client and release file lock."""
        resolved = str(Path(self.storage_path).resolve())
        c = _QDRANT_CLIENT_REGISTRY.pop(resolved, None) or self._client
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self._client = None

    def get_collection_name(self, dimension: Optional[int] = None) -> str:
        """Return collection name keyed by embedding dimension to avoid vector size collision."""
        dim = dimension or self.embedder.dimension
        return f"{self.collection_base_name}_{dim}"

    def ensure_collection(self, dimension: Optional[int] = None) -> str:
        """Ensure that the Qdrant collection for the active dimension exists."""
        dim = dimension or self.embedder.dimension
        col_name = self.get_collection_name(dim)
        collections = [c.name for c in self.client.get_collections().collections]

        if col_name not in collections:
            logger.info("Creating persistent Qdrant collection '%s' (dim=%d, distance=Cosine)...", col_name, dim)
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
        return col_name

    def get_index_dir(self, document_id: str) -> Path:
        """Return the directory path for a specific document's metadata."""
        return self.indexes_dir / document_id

    def build_index(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
        force_rebuild: bool = False,
        progress_callback=None,
    ) -> IndexMetadata:
        """
        Embed document chunks, upsert into persistent Qdrant, and save metadata.

        Args:
            document: Canonical Document object.
            chunks: Optional pre-generated chunks; if None, chunks the document.
            force_rebuild: Force rebuild even if index exists.
            progress_callback: Optional progress reporter.

        Returns:
            IndexMetadata describing the stored index.
        """
        doc_id = document.doc_id
        from src.ingestion.chunker import chunk_document
        doc_chunks = chunks if chunks is not None else chunk_document(document)
        doc_chunks = [c for c in doc_chunks if c.doc_id == doc_id] if doc_chunks else []

        idx_dir = self.get_index_dir(doc_id)

        # Check if valid index already exists
        if not force_rebuild and self.index_exists(doc_id):
            try:
                existing_meta = self.get_metadata(doc_id)
                if self._is_index_current(existing_meta, document, len(doc_chunks)):
                    logger.info("Using existing valid Qdrant index for document '%s' (%d chunks)", doc_id, len(doc_chunks))
                    return existing_meta
            except Exception as e:
                logger.warning("Could not validate existing Qdrant index for '%s': %s", doc_id, e)

        idx_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # 1. Embed chunks if needed
        logger.info("Embedding %d chunks for document '%s'...", len(doc_chunks), doc_id)
        with timed("document embedding"):
            if progress_callback:
                vectors_list = []
                bs = self.embedder.batch_size
                for i in range(0, len(doc_chunks), bs):
                    batch = doc_chunks[i : i + bs]
                    vectors_list.append(self.embedder.embed_chunks(batch))
                    progress_callback(min(i + bs, len(doc_chunks)), len(doc_chunks))
                vectors = np.vstack(vectors_list) if vectors_list else np.empty((0, self.embedder.dimension), dtype="float32")
            else:
                vectors = self.embedder.embed_chunks(doc_chunks)

        dim = vectors.shape[1] if vectors.ndim == 2 and len(vectors) > 0 else self.embedder.dimension
        col_name = self.ensure_collection(dim)

        # 2. Build Qdrant Points with deterministic UUIDs
        with timed("Qdrant build + persist"):
            points = []
            chunks_data = []

            for idx, c in enumerate(doc_chunks):
                # Deterministic UUID5 based on doc_id and chunk_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{c.chunk_id}"))
                vector = vectors[idx].tolist()

                c_dict = c.model_dump()
                c_dict["embedding"] = None  # Don't duplicate in JSON
                chunks_data.append(c_dict)

                payload = {
                    "doc_id": doc_id,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "page_numbers": c.page_numbers,
                    "section": c.section,
                    "subsection": c.subsection,
                    "table_index": c.table_index,
                    "chunk_type": c.chunk_type.value,
                    "text": c.text,
                    "char_count": c.char_count,
                    "chunk_index": c.chunk_index,
                    "file_hash": document.file_hash,
                    "embedding_model": self.embedder.model_name,
                    "embedding_dimension": dim,
                    "schema_version": self.schema_version,
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # 3. Batch upsert points into Qdrant
            batch_size = 64
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=col_name,
                    points=batch_points,
                    wait=True,
                )

            # 4. Save metadata JSON and chunks JSON to disk
            chunks_path = idx_dir / "chunks.json"
            meta_path = idx_dir / "metadata.json"

            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2)

            metadata = IndexMetadata(
                document_id=doc_id,
                filename=document.filename,
                file_hash=document.file_hash,
                chunk_count=len(doc_chunks),
                embedding_model=self.embedder.model_name,
                embedding_dimension=dim,
                normalized=True,
                index_type="qdrant_cosine",
                backend="qdrant",
                created_at=datetime.now(timezone.utc),
                index_schema_version=self.schema_version,
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(metadata.model_dump_json(indent=2))

            # Update in-memory cache
            self._cache[doc_id] = (self.client, doc_chunks, metadata)
            document.index_path = str(idx_dir)

        elapsed = time.time() - t0
        logger.info(
            "Successfully indexed '%s' in Qdrant (points=%d, dim=%d) in %.2fs",
            doc_id,
            len(doc_chunks),
            dim,
            elapsed,
        )
        return metadata

    def load_index(
        self,
        document_id: str,
    ) -> Tuple[Any, List[Chunk], IndexMetadata]:
        """
        Load chunks and metadata for a document into memory.
        Uses in-memory cache if already loaded.
        """
        if not self.index_exists(document_id):
            self._cache.pop(document_id, None)
            raise IndexNotFoundError(f"Index files not found or incomplete for document '{document_id}'")

        if document_id in self._cache:
            return self._cache[document_id]

        idx_dir = self.get_index_dir(document_id)
        meta_path = idx_dir / "metadata.json"
        chunks_path = idx_dir / "chunks.json"

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = IndexMetadata.model_validate(json.load(f))
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted metadata for '{document_id}': {e}")

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_raw = json.load(f)
            chunks = [Chunk.model_validate(c) for c in chunks_raw]
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted chunks for '{document_id}': {e}")

        # Ensure collection exists
        self.ensure_collection(metadata.embedding_dimension)

        self._cache[document_id] = (self.client, chunks, metadata)
        return self.client, chunks, metadata

    def search(
        self,
        document_id: str,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform a Qdrant vector similarity search filtered strictly by document_id.

        Args:
            document_id: Foreign key to filter vectors.
            query_vector: 1D query vector.
            top_k: Maximum candidate count.

        Returns:
            List of (Chunk, similarity_score) sorted descending.
        """
        if not self.index_exists(document_id):
            raise IndexNotFoundError(f"No index found for document ID '{document_id}'")

        if isinstance(query_vector, np.ndarray):
            q_vec = query_vector.flatten().tolist()
        else:
            q_vec = list(query_vector)

        dim = len(q_vec)
        col_name = self.ensure_collection(dim)

        # Strict document-level filter
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

        try:
            response = self.client.query_points(
                collection_name=col_name,
                query=q_vec,
                query_filter=doc_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        except Exception as e:
            logger.error("Qdrant search failed for doc_id '%s': %s", document_id, e)
            raise IndexStoreError(f"Qdrant search failed: {e}") from e

        results: List[Tuple[Chunk, float]] = []
        for hit in hits:
            p = hit.payload or {}
            # Reconstruct chunk
            try:
                chunk = Chunk(
                    chunk_id=p.get("chunk_id", str(hit.id)),
                    doc_id=p.get("doc_id", document_id),
                    filename=p.get("filename", ""),
                    page_number=p.get("page_number", 1),
                    page_numbers=p.get("page_numbers", [1]),
                    section=p.get("section"),
                    subsection=p.get("subsection"),
                    table_index=p.get("table_index"),
                    chunk_type=ChunkType(p.get("chunk_type", "TEXT")),
                    text=p.get("text", ""),
                    char_count=p.get("char_count", len(p.get("text", ""))),
                    chunk_index=p.get("chunk_index", 0),
                )
                score = float(hit.score)
                results.append((chunk, score))
            except Exception as e:
                logger.warning("Error reconstructing chunk from Qdrant payload: %s", e)

        return results

    def index_exists(self, document_id: str) -> bool:
        """Check if metadata and chunks exist on disk for document_id."""
        idx_dir = self.get_index_dir(document_id)
        meta_file = idx_dir / "metadata.json"
        chunks_file = idx_dir / "chunks.json"
        return meta_file.is_file() and chunks_file.is_file()

    def get_metadata(self, document_id: str) -> IndexMetadata:
        """Load and return metadata for a document's index."""
        meta_file = self.get_index_dir(document_id) / "metadata.json"
        if not meta_file.is_file():
            raise IndexNotFoundError(f"Metadata not found for document {document_id}")
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return IndexMetadata.model_validate(data)
        except Exception as e:
            raise CorruptedIndexError(f"Failed to read metadata for {document_id}: {str(e)}")

    def _is_index_current(
        self,
        meta: IndexMetadata,
        document: Document,
        chunk_count: int,
    ) -> bool:
        """Check if an existing index matches current document content, model, and schema."""
        return (
            meta.document_id == document.doc_id
            and meta.file_hash == document.file_hash
            and meta.chunk_count == chunk_count
            and meta.embedding_model == self.embedder.model_name
            and meta.embedding_dimension == self.embedder.dimension
            and meta.index_schema_version == self.schema_version
        )

    def is_fresh(self, document_id: str, file_hash: str) -> bool:
        """
        Check if an index exists, matches file hash, matches embedding model & dim,
        and matches current schema version.
        """
        if not self.index_exists(document_id):
            return False
        try:
            meta = self.get_metadata(document_id)
            return (
                meta.file_hash == file_hash
                and meta.embedding_model == self.embedder.model_name
                and meta.embedding_dimension == self.embedder.dimension
                and meta.index_schema_version == self.schema_version
            )
        except Exception:
            return False

    def delete_document(self, document_id: str) -> None:
        """Delete all points for a document from Qdrant and disk."""
        dim = self.embedder.dimension
        col_name = self.get_collection_name(dim)
        try:
            self.client.delete(
                collection_name=col_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.warning("Failed to delete Qdrant points for doc '%s': %s", document_id, e)

        # Clear cache and disk files
        self._cache.pop(document_id, None)
        idx_dir = self.get_index_dir(document_id)
        if idx_dir.exists():
            import shutil
            shutil.rmtree(idx_dir, ignore_errors=True)

    def cleanup_invalid_folders(self) -> Tuple[int, int]:
        """Clean up non-hash named folders in indexes directory."""
        if not self.indexes_dir.exists():
            return 0, 0
        import re, shutil
        folders = list(self.indexes_dir.iterdir())
        before = len(folders)
        for item in folders:
            if item.is_dir() and not re.match(r"^[a-f0-9]{64}$", item.name):
                shutil.rmtree(item, ignore_errors=True)
        after = len(list(self.indexes_dir.iterdir()))
        return before, after

    def enforce_index_limit(self, max_indexes: int = 20) -> None:
        """Keep at most max_indexes most recently modified index folders."""
        if not self.indexes_dir.exists():
            return
        import shutil
        folders = [f for f in self.indexes_dir.iterdir() if f.is_dir()]
        if len(folders) <= max_indexes:
            return
        folders.sort(key=lambda p: p.stat().st_mtime)
        for f in folders[: len(folders) - max_indexes]:
            shutil.rmtree(f, ignore_errors=True)


# Singleton instance
qdrant_store = QdrantStore()
