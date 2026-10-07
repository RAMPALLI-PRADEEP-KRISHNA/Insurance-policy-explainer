from src.utils.timing import timed
"""Persistent, document-isolated FAISS IndexFlatIP store for insurance policy chunks."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss

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
        # In-memory cache: doc_id -> (faiss.Index, List[Chunk], IndexMetadata)
        self._cache: Dict[str, Tuple[faiss.Index, List[Chunk], IndexMetadata]] = {}

    def get_index_dir(self, document_id: str) -> Path:
        """Return the directory path for a specific document's index."""
        return self.indexes_dir / document_id

    def index_exists(self, document_id: str) -> bool:
        """Check if index files exist for a document."""
        idx_dir = self.get_index_dir(document_id)
        faiss_file = idx_dir / "index.faiss"
        chunks_file = idx_dir / "chunks.json"
        meta_file = idx_dir / "metadata.json"
        return faiss_file.is_file() and chunks_file.is_file() and meta_file.is_file()

    def build_index(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
        force_rebuild: bool = False,
    ) -> IndexMetadata:
        """
        Build and persist a FAISS IndexFlatIP index for a Document.

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
        logger.info("Building FAISS index for document '%s' (%d chunks)...", doc_id, len(doc_chunks))
        t0 = time.time()
        with timed("document embedding"):
            vectors = self.embedder.embed_chunks(doc_chunks)
        dim = self.embedder.dimension

        with timed("FAISS build + persist"):
            # 4. Construct FAISS IndexFlatIP
                index = faiss.IndexFlatIP(dim)
                if len(doc_chunks) > 0:
                    index.add(vectors)

                # 5. Persist index, chunks, and metadata
                idx_dir.mkdir(parents=True, exist_ok=True)
                faiss_path = idx_dir / "index.faiss"
                chunks_path = idx_dir / "chunks.json"
                meta_path = idx_dir / "metadata.json"

                # Save FAISS binary index
                try:
                    faiss.write_index(index, str(faiss_path))
                except Exception as e:
                    logger.error("Failed to write FAISS index to %s: %s", faiss_path, e)
                    raise IndexStoreError(f"FAISS index write failed: {str(e)}")

                # Save chunks JSON (exclude embeddings from chunks.json to conserve disk space)
                try:
                    chunks_data = []
                    for c in doc_chunks:
                        c_dict = c.model_dump()
                        c_dict["embedding"] = None  # Stored in FAISS binary
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
                    index_type="IndexFlatIP",
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
    ) -> Tuple[faiss.Index, List[Chunk], IndexMetadata]:
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

        faiss_path = idx_dir / "index.faiss"
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

        # 3. Load FAISS index
        try:
            index = faiss.read_index(str(faiss_path))
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted FAISS index at {faiss_path}: {str(e)}")

        # 4. Load chunks
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            chunks = [Chunk.model_validate(c) for c in raw_chunks]
        except Exception as e:
            raise CorruptedIndexError(f"Corrupted chunks JSON at {chunks_path}: {str(e)}")

        # Validate vector-chunk count alignment
        if index.ntotal != len(chunks):
            raise CorruptedIndexError(
                f"Index vector count ({index.ntotal}) does not match chunk count ({len(chunks)})"
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

        if len(chunks) == 0 or index.ntotal == 0:
            return []

        # Ensure query_vector is 2D float32 contiguous array
        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q_vec = np.ascontiguousarray(q_vec)

        # Verify dimension
        if q_vec.shape[1] != index.d:
            raise IndexStoreError(
                f"Query vector dimension ({q_vec.shape[1]}) does not match index dimension ({index.d})"
            )

        # Normalize query vector if not unit norm
        norm = np.linalg.norm(q_vec)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
            q_vec = q_vec / norm

        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_vec, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(chunks):
                results.append((chunks[idx], float(score)))

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
) -> IndexMetadata:
    """Convenience functional interface for building a FAISS index."""
    return index_store.build_index(document, chunks=chunks, force_rebuild=force_rebuild)
