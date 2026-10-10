from src.utils.timing import timed
"""Persistent, document-isolated FAISS IndexFlatIP store for insurance policy chunks."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.config import config
from src.models.data_models import Document, Chunk, IndexMetadata
from src.ingestion.embedder import embedder, Embedder
from src.ingestion.chunker import chunk_document
from src.utils.exceptions import (
    IndexStoreError,
    IndexNotFoundError,
    StaleIndexError,
    CorruptedIndexError,
)
from src.utils.logging_utils import logger


class IndexStore:
    """
    Manages per-document FAISS indexes.
    Enforces the fundamental architectural invariant:
      ONE DOCUMENT -> ONE FAISS INDEX -> ONLY THAT DOCUMENT'S CHUNKS.
    """

    def __init__(
        self,
        indexes_dir: Path = config.INDEXES_DIR,
        embedder_instance: Optional[Embedder] = None,
        schema_version: str = config.INDEX_SCHEMA_VERSION,
    ):
        self.indexes_dir = indexes_dir
        self.embedder = embedder_instance or embedder
        self.schema_version = schema_version
        # In-memory cache: doc_id -> (np.ndarray, List[Chunk], IndexMetadata)
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
        """
        Build and persist a NumPy index for a Document.

        Args:
            document: Canonical Document instance
            chunks: Optional pre-generated chunks; if None, chunks the document
            force_rebuild: Force rebuilding even if a valid index exists

        Returns:
            IndexMetadata describing the stored index.
        """
        doc_id = document.doc_id
        idx_dir = self.get_index_dir(doc_id)

        # 1. Resolve chunks
        doc_chunks = chunks if chunks is not None else chunk_document(document)

        # 2. Check if valid index already exists
        if not force_rebuild and self.index_exists(doc_id):
            try:
                existing_meta = self.get_metadata(doc_id)
                if self._is_index_current(existing_meta, document, len(doc_chunks)):
                    logger.info("Using existing valid index for document '%s' (%d chunks)", doc_id, len(doc_chunks))
                    return existing_meta
                else:
                    logger.info("Existing index for document '%s' is stale. Rebuilding...", doc_id)
            except Exception as e:
                logger.warning("Could not validate existing index for '%s' (%s). Rebuilding...", doc_id, e)

        # 3. Embed chunks
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
        dim = self.embedder.dimension

        with timed("NumPy build + persist"):
            # 4. Construct NumPy Index
            if len(doc_chunks) > 0:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms
                index = np.asarray(vectors, dtype=np.float32)
            else:
                index = np.empty((0, dim), dtype=np.float32)

            # 5. Persist index, chunks, and metadata
            idx_dir.mkdir(parents=True, exist_ok=True)
            npy_path = idx_dir / "index.npy"
            chunks_path = idx_dir / "chunks.json"
            meta_path = idx_dir / "metadata.json"

            # Save NumPy binary index
            try:
                np.save(str(npy_path), index)
            except Exception as e:
                logger.error("Failed to write NumPy index to %s: %s", npy_path, e)
                raise IndexStoreError(f"NumPy index write failed: {str(e)}")

            # Save chunks JSON
            try:
                chunks_data = []
                for c in doc_chunks:
                    c_dict = c.model_dump()
                    c_dict["embedding"] = None  # Stored in NumPy binary
                    chunks_data.append(c_dict)
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, indent=2)
            except Exception as e:
                logger.error("Failed to write chunks JSON to %s: %s", chunks_path, e)
                raise IndexStoreError(f"Chunks serialization failed: {str(e)}")

            # Save metadata JSON
            metadata = IndexMetadata(
                document_id=doc_id,
                filename=document.filename,
                file_hash=document.file_hash,
                chunk_count=len(doc_chunks),
                embedding_model=self.embedder.model_name,
                embedding_dimension=dim,
                normalized=True,
                index_type="numpy_flat_ip",
                created_at=datetime.now(timezone.utc),
                index_schema_version=self.schema_version,
            )

            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(metadata.model_dump_json(indent=2))
            except Exception as e:
                logger.error("Failed to write metadata JSON to %s: %s", meta_path, e)
                raise IndexStoreError(f"Metadata serialization failed: {str(e)}")

            # Update cache
            self._cache[doc_id] = (index, doc_chunks, metadata)
            document.index_path = str(idx_dir)

        elapsed = time.time() - t0
        logger.info(
            "Successfully built and persisted index for '%s' in %.2fs (chunks=%d, dim=%d)",
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
        """
        Load index, chunks, and metadata for a document into memory.
        Uses in-memory cache if already loaded.

        Raises:
            IndexNotFoundError, CorruptedIndexError
        """
        if document_id in self._cache:
            return self._cache[document_id]

        idx_dir = self.get_index_dir(document_id)
        if not self.index_exists(document_id):
            raise IndexNotFoundError(f"No FAISS index found for document ID '{document_id}' at {idx_dir}")

        npy_path = idx_dir / "index.npy"
        chunks_path = idx_dir / "chunks.json"
        meta_path = idx_dir / "metadata.json"

        # 1. Load metadata
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            metadata = IndexMetadata.model_validate(meta_dict)
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted metadata in {meta_path}: {str(e)}")

        # 2. Check schema compatibility
        if metadata.index_schema_version != self.schema_version:
            raise StaleIndexError(
                f"Index schema version mismatch for {document_id}: "
                f"expected {self.schema_version}, found {metadata.index_schema_version}"
            )

        # 3. Load NumPy index
        try:
            index = np.load(str(npy_path))
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted NumPy index at {npy_path}: {str(e)}")

        # 4. Load chunks
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            chunks = [Chunk.model_validate(c) for c in raw_chunks]
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted chunks JSON at {chunks_path}: {str(e)}")

        # Validate vector-chunk count alignment
        if index.shape[0] != len(chunks):
            raise CorruptedIndexError(
                f"Index vector count ({index.shape[0]}) does not match chunk count ({len(chunks)})"
            )

        self._cache[document_id] = (index, chunks, metadata)
        return index, chunks, metadata

    def search(
        self,
        document_id: str,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform a raw FAISS vector similarity search against a specific document's index.

        Args:
            document_id: Target document ID (enforces document isolation)
            query_vector: 1D or 2D float32 normalized vector
            top_k: Number of candidates to retrieve

        Returns:
            List of (Chunk, similarity_score) sorted descending by similarity.
        """
        index, chunks, metadata = self.load_index(document_id)

        if len(chunks) == 0 or index.shape[0] == 0:
            return []

        # Ensure query_vector is 2D float32 contiguous array
        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q_vec = np.ascontiguousarray(q_vec)

        # Verify dimension
        if q_vec.shape[1] != index.shape[1]:
            raise IndexStoreError(
                f"Query vector dimension ({q_vec.shape[1]}) does not match index dimension ({index.shape[1]})"
            )

        # Normalize query vector if not unit norm
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

    def is_fresh(self, document_id: str, file_hash: str) -> bool:
        """Check if an index exists, matches the provided file hash, and has valid schema."""
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
        """Check if an existing index matches current document content, model, and schema."""
        return (
            meta.document_id == document.doc_id
            and meta.file_hash == document.file_hash
            and meta.chunk_count == chunk_count
            and meta.embedding_model == self.embedder.model_name
            and meta.embedding_dimension == self.embedder.dimension
            and meta.index_schema_version == self.schema_version
        )


# Singleton index store
index_store = IndexStore()


def build_index(
    document: Document,
    chunks: Optional[List[Chunk]] = None,
    force_rebuild: bool = False,
        progress_callback=None,
) -> IndexMetadata:
    """Convenience functional interface for building a FAISS index."""
    return index_store.build_index(document, chunks=chunks, force_rebuild=force_rebuild)


