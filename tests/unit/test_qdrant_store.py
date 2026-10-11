"""Unit tests for persistent local QdrantStore, document isolation, and recovery."""

import uuid
from pathlib import Path
import numpy as np
import pytest

from database.qdrant_store import QdrantStore
from backend.ingestion.embedder import Embedder
from backend.models.data_models import Document, Chunk, ChunkType, IndexMetadata
from backend.utils.exceptions import IndexNotFoundError, CorruptedIndexError, IndexStoreError


class MockEmbedder:
    """Mock embedder providing deterministic 384-dimensional vectors."""
    def __init__(self, dimension=384, model_name="snowflake/snowflake-arctic-embed-xs"):
        self.dimension = dimension
        self.model_name = model_name
        self.batch_size = 32

    def embed_texts(self, texts, batch_size=None):
        np.random.seed(42)
        vecs = np.random.randn(len(texts), self.dimension).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def embed_chunks(self, chunks, batch_size=None):
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query):
        np.random.seed(42)
        v = np.random.randn(self.dimension).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimension=384)


@pytest.fixture
def temp_qdrant_store(tmp_path, mock_embedder):
    """Create a persistent QdrantStore instance using a temporary directory."""
    return QdrantStore(
        storage_path=tmp_path / "qdrant_db",
        indexes_dir=tmp_path / "indexes",
        collection_base_name="test_policy_chunks",
        embedder_instance=mock_embedder,
        schema_version="2",
    )


def make_doc(doc_id: str, filename: str = "test.pdf", file_hash: str = "hash123") -> Document:
    return Document(
        doc_id=doc_id,
        filename=filename,
        file_hash=file_hash,
        page_count=2,
    )


def make_chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    chunks = []
    for idx, t in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{idx:04d}",
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                page_number=1,
                page_numbers=[1],
                section="Section 1",
                chunk_type=ChunkType.CLAUSE,
                text=t,
                chunk_index=idx,
            )
        )
    return chunks


def test_qdrant_collection_creation_and_insertion(temp_qdrant_store):
    """Test collection creation, deterministic vector insertion, and search."""
    doc = make_doc("doc-001")
    chunks = make_chunks("doc-001", [
        "Pre-existing disease waiting period is 36 months.",
        "Room rent is capped at 1% of Sum Insured per day."
    ])

    meta = temp_qdrant_store.build_index(doc, chunks=chunks)
    assert meta.document_id == "doc-001"
    assert meta.chunk_count == 2
    assert meta.embedding_dimension == 384
    assert meta.backend == "qdrant"

    # Search
    q_vec = temp_qdrant_store.embedder.embed_query("What is the waiting period?")
    results = temp_qdrant_store.search("doc-001", q_vec, top_k=2)

    assert len(results) == 2
    assert results[0][0].doc_id == "doc-001"
    assert results[0][1] > 0.0


def test_qdrant_document_filtering_isolation(temp_qdrant_store):
    """Verify document-ID filtering strictly prevents cross-policy retrieval."""
    doc1 = make_doc("doc-A", file_hash="hashA")
    chunks1 = make_chunks("doc-A", ["Policy A: Cataract coverage limit is Rs. 40,000."])
    temp_qdrant_store.build_index(doc1, chunks=chunks1)

    doc2 = make_doc("doc-B", file_hash="hashB")
    chunks2 = make_chunks("doc-B", ["Policy B: Cataract coverage limit is Rs. 20,000."])
    temp_qdrant_store.build_index(doc2, chunks=chunks2)

    q_vec = temp_qdrant_store.embedder.embed_query("Cataract coverage limit")

    # Search in doc-A
    res_a = temp_qdrant_store.search("doc-A", q_vec, top_k=5)
    assert len(res_a) == 1
    assert res_a[0][0].doc_id == "doc-A"
    assert "Policy A" in res_a[0][0].text

    # Search in doc-B
    res_b = temp_qdrant_store.search("doc-B", q_vec, top_k=5)
    assert len(res_b) == 1
    assert res_b[0][0].doc_id == "doc-B"
    assert "Policy B" in res_b[0][0].text


def test_qdrant_persistence_and_client_restart(tmp_path, mock_embedder):
    """Verify that points and metadata survive client restart from disk."""
    storage_path = tmp_path / "qdrant_persist_db"
    indexes_dir = tmp_path / "indexes"

    store1 = QdrantStore(
        storage_path=storage_path,
        indexes_dir=indexes_dir,
        collection_base_name="restart_test",
        embedder_instance=mock_embedder,
    )

    doc = make_doc("doc-restart")
    chunks = make_chunks("doc-restart", ["Maternity coverage waiting period is 24 months."])
    store1.build_index(doc, chunks=chunks)

    # Simulate client restart by creating a new store instance with same storage path
    del store1

    store2 = QdrantStore(
        storage_path=storage_path,
        indexes_dir=indexes_dir,
        collection_base_name="restart_test",
        embedder_instance=mock_embedder,
    )

    assert store2.index_exists("doc-restart")
    assert store2.is_fresh("doc-restart", "hash123")

    q_vec = mock_embedder.embed_query("Maternity waiting period")
    results = store2.search("doc-restart", q_vec, top_k=1)
    assert len(results) == 1
    assert "Maternity coverage" in results[0][0].text


def test_qdrant_deterministic_ids_idempotency(temp_qdrant_store):
    """Verify upserting same document twice does not duplicate points."""
    doc = make_doc("doc-idempotent")
    chunks = make_chunks("doc-idempotent", ["Clause 1 text", "Clause 2 text"])

    temp_qdrant_store.build_index(doc, chunks=chunks, force_rebuild=True)
    # Upsert again
    temp_qdrant_store.build_index(doc, chunks=chunks, force_rebuild=True)

    q_vec = temp_qdrant_store.embedder.embed_query("Clause")
    results = temp_qdrant_store.search("doc-idempotent", q_vec, top_k=10)
    assert len(results) == 2


def test_qdrant_deletion(temp_qdrant_store):
    """Verify deletion removes document points and metadata."""
    doc = make_doc("doc-del")
    chunks = make_chunks("doc-del", ["To be deleted"])
    temp_qdrant_store.build_index(doc, chunks=chunks)

    assert temp_qdrant_store.index_exists("doc-del")
    temp_qdrant_store.delete_document("doc-del")

    assert not temp_qdrant_store.index_exists("doc-del")
    assert not temp_qdrant_store.is_fresh("doc-del", "hash123")


def test_qdrant_schema_mismatch_and_freshness(temp_qdrant_store):
    """Verify stale schema version triggers is_fresh=False."""
    doc = make_doc("doc-stale", file_hash="correct_hash")
    chunks = make_chunks("doc-stale", ["Content"])
    temp_qdrant_store.build_index(doc, chunks=chunks)

    # Different hash
    assert not temp_qdrant_store.is_fresh("doc-stale", "wrong_hash")

    # Correct hash matches
    assert temp_qdrant_store.is_fresh("doc-stale", "correct_hash")

    # Alter schema version
    temp_qdrant_store.schema_version = "999"
    assert not temp_qdrant_store.is_fresh("doc-stale", "correct_hash")
