from src.utils.timing import timed
"""Unified Vector Store facade supporting Qdrant and NumPy backends."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from src.config import config
from src.models.data_models import Document, Chunk, IndexMetadata
from src.ingestion.embedder import embedder as default_embedder, Embedder
from src.ingestion.chunker import chunk_document
from src.utils.exceptions import (
    IndexStoreError,
    IndexNotFoundError,
    StaleIndexError,
    CorruptedIndexError,
)
from src.utils.logging_utils import logger


class NumPyIndexStore:
    """
    Manages per-document NumPy Flat IP indexes.
    Enforces the fundamental architectural invariant:
      ONE DOCUMENT -> ONE NUMPY INDEX -> ONLY THAT DOCUMENT'S CHUNKS.
    """

    def __init__(
        self,
        indexes_dir: Path = config.INDEXES_DIR,
        embedder_instance: Optional[Embedder] = None,
        schema_version: str = config.INDEX_SCHEMA_VERSION,
    ):
        self.indexes_dir = Path(indexes_dir)
        self.embedder = embedder_instance or default_embedder
        self.schema_version = schema_version
        self._cache: Dict[str, Tuple[np.ndarray, List[Chunk], IndexMetadata]] = {}

    def get_index_dir(self, document_id: str) -> Path:
        """Return the directory path for a specific document's index."""
        return self.indexes_dir / document_id

    def cleanup_invalid_folders(self) -> tuple[int, int]:
        if not self.indexes_dir.exists():
            return 0, 0
        import re, shutil
        folders = list(self.indexes_dir.iterdir())
        before = len(folders)
        for item in folders:
            if item.is_dir() and not re.match(r'^[a-f0-9]{64}$', item.name):
                shutil.rmtree(item)
        after = len(list(self.indexes_dir.iterdir()))
        return before, after

    def enforce_index_limit(self, max_indexes: int = 20) -> None:
        if not self.indexes_dir.exists():
            return
        import re, shutil
        valid_folders = []
        for item in self.indexes_dir.iterdir():
            if item.is_dir() and re.match(r'^[a-f0-9]{64}$', item.name):
                meta_path = item / "metadata.json"
                if meta_path.exists():
                    valid_folders.append((meta_path.stat().st_mtime, item))
                else:
                    shutil.rmtree(item)
        valid_folders.sort()
        if len(valid_folders) > max_indexes:
            for _, item in valid_folders[:-max_indexes]:
                shutil.rmtree(item)

    def index_exists(self, document_id: str) -> bool:
        """Check if index files exist for a document."""
        idx_dir = self.get_index_dir(document_id)
        npy_file = idx_dir / "index.npy"
        chunks_file = idx_dir / "chunks.json"
        meta_file = idx_dir / "metadata.json"
        return npy_file.is_file() and chunks_file.is_file() and meta_file.is_file()

    def build_index(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
        force_rebuild: bool = False,
        progress_callback=None,
    ) -> IndexMetadata:
        """Build and persist a NumPy index for a Document."""
        doc_id = document.doc_id
        idx_dir = self.get_index_dir(doc_id)

        doc_chunks = chunks if chunks is not None else chunk_document(document)

        if not force_rebuild and self.index_exists(doc_id):
            try:
                existing_meta = self.get_metadata(doc_id)
                if self._is_index_current(existing_meta, document, len(doc_chunks)):
                    logger.info("Using existing valid NumPy index for document '%s' (%d chunks)", doc_id, len(doc_chunks))
                    return existing_meta
                else:
                    logger.info("Existing index for document '%s' is stale. Rebuilding...", doc_id)
            except Exception as e:
                logger.warning("Could not validate existing index for '%s' (%s). Rebuilding...", doc_id, e)

        logger.info("Building NumPy index for document '%s' (%d chunks)...", doc_id, len(doc_chunks))
        t0 = time.time()
        with timed("document embedding"):
            if progress_callback:
                vectors_list = []
                bs = self.embedder.batch_size
                for i in range(0, len(doc_chunks), bs):
                    batch = doc_chunks[i:i + bs]
                    vectors_list.append(self.embedder.embed_chunks(batch))
                    progress_callback(min(i + bs, len(doc_chunks)), len(doc_chunks))
                vectors = np.vstack(vectors_list) if vectors_list else np.empty((0, self.embedder.dimension), dtype="float32")
            else:
                vectors = self.embedder.embed_chunks(doc_chunks)
        dim = vectors.shape[1] if vectors.ndim == 2 and len(vectors) > 0 else self.embedder.dimension

        with timed("NumPy build + persist"):
            if len(doc_chunks) > 0:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms
                index = np.asarray(vectors, dtype=np.float32)
            else:
                index = np.empty((0, dim), dtype=np.float32)

            idx_dir.mkdir(parents=True, exist_ok=True)
            npy_path = idx_dir / "index.npy"
            chunks_path = idx_dir / "chunks.json"
            meta_path = idx_dir / "metadata.json"

            try:
                np.save(str(npy_path), index)
            except Exception as e:
                logger.error("Failed to write NumPy index to %s: %s", npy_path, e)
                raise IndexStoreError(f"NumPy index write failed: {str(e)}")

            try:
                chunks_data = []
                for c in doc_chunks:
                    c_dict = c.model_dump()
                    c_dict["embedding"] = None
                    chunks_data.append(c_dict)
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, indent=2)
            except Exception as e:
                logger.error("Failed to write chunks JSON to %s: %s", chunks_path, e)
                raise IndexStoreError(f"Chunks serialization failed: {str(e)}")

            metadata = IndexMetadata(
                document_id=doc_id,
                filename=document.filename,
                file_hash=document.file_hash,
                chunk_count=len(doc_chunks),
                embedding_model=self.embedder.model_name,
                embedding_dimension=dim,
                normalized=True,
                index_type="numpy_flat_ip",
                backend="numpy",
                created_at=datetime.now(timezone.utc),
                index_schema_version=self.schema_version,
            )

            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(metadata.model_dump_json(indent=2))
            except Exception as e:
                logger.error("Failed to write metadata JSON to %s: %s", meta_path, e)
                raise IndexStoreError(f"Metadata serialization failed: {str(e)}")

            self._cache[doc_id] = (index, doc_chunks, metadata)
            document.index_path = str(idx_dir)

        elapsed = time.time() - t0
        logger.info(
            "Successfully built and persisted NumPy index for '%s' in %.2fs (chunks=%d, dim=%d)",
            doc_id,
            elapsed,
            len(doc_chunks),
            dim,
        )
        return metadata

    def load_index(
        self,
        document_id: str,
    ) -> Tuple[np.ndarray, List[Chunk], IndexMetadata]:
        if not self.index_exists(document_id):
            self._cache.pop(document_id, None)
            raise IndexNotFoundError(f"No index found for document ID '{document_id}'")

        if document_id in self._cache:
            return self._cache[document_id]

        idx_dir = self.get_index_dir(document_id)
        npy_path = idx_dir / "index.npy"
        chunks_path = idx_dir / "chunks.json"
        meta_path = idx_dir / "metadata.json"

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            metadata = IndexMetadata.model_validate(meta_dict)
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted metadata in {meta_path}: {e}")

        try:
            index = np.load(str(npy_path))
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted NumPy index at {npy_path}: {e}")

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            chunks = [Chunk.model_validate(c) for c in chunks_data]
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted chunks JSON at {chunks_path}: {e}")

        self._cache[document_id] = (index, chunks, metadata)
        return index, chunks, metadata

    def search(
        self,
        document_id: str,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        index, chunks, metadata = self.load_index(document_id)

        if len(chunks) == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q_vec = np.ascontiguousarray(q_vec)

        if q_vec.shape[1] != index.shape[1]:
            raise IndexStoreError(
                f"Query vector dimension ({q_vec.shape[1]}) does not match index dimension ({index.shape[1]})"
            )

        norm = np.linalg.norm(q_vec)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
            q_vec = q_vec / norm

        k = min(top_k, index.shape[0])
        scores = index @ q_vec[0]

        if len(scores) <= k:
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -k)[-k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        results: List[Tuple[Chunk, float]] = []
        for idx in indices:
            if 0 <= idx < len(chunks):
                results.append((chunks[idx], float(scores[idx])))

        return results

    def get_metadata(self, document_id: str) -> IndexMetadata:
        meta_file = self.get_index_dir(document_id) / "metadata.json"
        if not meta_file.is_file():
            raise IndexNotFoundError(f"Metadata not found for document {document_id}")
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return IndexMetadata.model_validate(data)
        except Exception as e:
            raise CorruptedIndexError(f"Failed to read metadata for {document_id}: {str(e)}")

    def is_fresh(self, document_id: str, file_hash: str) -> bool:
        if not self.index_exists(document_id):
            return False
        try:
            meta = self.get_metadata(document_id)
            return (
                meta.file_hash == file_hash
                and meta.index_schema_version == self.schema_version
                and meta.embedding_dimension == self.embedder.dimension
                and meta.embedding_model == self.embedder.model_name
            )
        except Exception:
            return False

    def _is_index_current(
        self,
        meta: IndexMetadata,
        document: Document,
        chunk_count: int,
    ) -> bool:
        return (
            meta.document_id == document.doc_id
            and meta.file_hash == document.file_hash
            and meta.chunk_count == chunk_count
            and meta.embedding_model == self.embedder.model_name
            and meta.embedding_dimension == self.embedder.dimension
            and meta.index_schema_version == self.schema_version
        )


class IndexStore:
    """
    Unified Vector Store facade.
    Delegates to QdrantStore or NumPyIndexStore based on config.VECTOR_STORE_BACKEND.
    """

    def __init__(
        self,
        indexes_dir: Path = config.INDEXES_DIR,
        embedder_instance: Optional[Embedder] = None,
        schema_version: str = config.INDEX_SCHEMA_VERSION,
        backend: Optional[str] = None,
    ):
        self.backend = (backend or config.VECTOR_STORE_BACKEND).lower()
        self.indexes_dir = Path(indexes_dir)
        self.schema_version = schema_version
        emb = embedder_instance or default_embedder

        if self.backend == "qdrant":
            from src.ingestion.qdrant_store import QdrantStore
            qdrant_path = (
                self.indexes_dir.parent / "qdrant_db"
                if self.indexes_dir != config.INDEXES_DIR
                else config.QDRANT_STORAGE_PATH
            )
            self._impl = QdrantStore(
                storage_path=qdrant_path,
                indexes_dir=self.indexes_dir,
                embedder_instance=emb,
                schema_version=self.schema_version,
            )
        else:
            self._impl = NumPyIndexStore(
                indexes_dir=self.indexes_dir,
                embedder_instance=emb,
                schema_version=self.schema_version,
            )

    @property
    def embedder(self) -> Embedder:
        return self._impl.embedder

    @embedder.setter
    def embedder(self, value: Embedder) -> None:
        self._impl.embedder = value

    @property
    def _cache(self):
        return self._impl._cache

    def get_index_dir(self, document_id: str) -> Path:
        return self._impl.get_index_dir(document_id)

    def cleanup_invalid_folders(self) -> tuple[int, int]:
        return self._impl.cleanup_invalid_folders()

    def enforce_index_limit(self, max_indexes: int = 20) -> None:
        return self._impl.enforce_index_limit(max_indexes)

    def index_exists(self, document_id: str) -> bool:
        return self._impl.index_exists(document_id)

    def build_index(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
        force_rebuild: bool = False,
        progress_callback=None,
    ) -> IndexMetadata:
        return self._impl.build_index(
            document,
            chunks=chunks,
            force_rebuild=force_rebuild,
            progress_callback=progress_callback,
        )

    def load_index(
        self,
        document_id: str,
    ) -> Tuple[Any, List[Chunk], IndexMetadata]:
        return self._impl.load_index(document_id)

    def search(
        self,
        document_id: str,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        return self._impl.search(document_id, query_vector, top_k=top_k)

    def get_metadata(self, document_id: str) -> IndexMetadata:
        return self._impl.get_metadata(document_id)

    def is_fresh(self, document_id: str, file_hash: str) -> bool:
        return self._impl.is_fresh(document_id, file_hash)


# Default singleton instance
index_store = IndexStore()


def build_index(
    document: Document,
    chunks: Optional[List[Chunk]] = None,
    force_rebuild: bool = False,
    progress_callback=None,
) -> IndexMetadata:
    """Convenience functional interface for building an index."""
    return index_store.build_index(
        document,
        chunks=chunks,
        force_rebuild=force_rebuild,
        progress_callback=progress_callback,
    )
