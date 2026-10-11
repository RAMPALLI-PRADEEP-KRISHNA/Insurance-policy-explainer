"""Unit tests for per-document FAISS IndexStore, isolation, persistence, and recovery."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.ingestion.index_store import IndexStore
from src.ingestion.embedder import embedder
from src.models.data_models import Document, Chunk, ChunkType, IndexMetadata
from src.utils.exceptions import IndexNotFoundError, StaleIndexError, CorruptedIndexError


@pytest.fixture
def temp_index_store(tmp_path):
    """Create an IndexStore instance using a temporary directory with NumPy backend."""
    return IndexStore(indexes_dir=tmp_path / "indexes", embedder_instance=embedder, backend="numpy")


def make_dummy_document(doc_id: str, filename: str = "test.pdf", file_hash: str = "hash123") -> Document:
    """Helper to construct a valid Document model."""
    return Document(
        doc_id=doc_id,
        filename=filename,
        file_hash=file_hash,
        page_count=2,
    )


def make_dummy_chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    """Helper to construct valid Chunk models."""
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


def test_build_and_search_index(temp_index_store):
    """Verify building index, vector insertion, and low-level similarity search."""
    doc = make_dummy_document("doc-001")
    texts = [
        "Room rent limit is Rs. 5,000 per day.",
        "Waiting period for pre-existing disease is 36 months.",
        "Maternity coverage is excluded under this policy.",
    ]
    chunks = make_dummy_chunks("doc-001", texts)

    meta = temp_index_store.build_index(doc, chunks=chunks)
    assert meta.document_id == "doc-001"
    assert meta.chunk_count == 3
    assert meta.embedding_dimension == embedder.dimension
    assert meta.index_type == "numpy_flat_ip"

    # Search for waiting period
    q_vec = embedder.embed_query("How many months is the waiting period?")
    results = temp_index_store.search("doc-001", q_vec, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert "36 months" in top_chunk.text
    assert top_score > 0.4
    assert top_chunk.doc_id == "doc-001"


def test_index_persistence_and_reload(temp_index_store):
    """Verify that persisted index files reload cleanly and preserve search ranking."""
    doc = make_dummy_document("doc-002")
    chunks = make_dummy_chunks("doc-002", ["Organ donor expenses are payable up to Sum Insured."])

    temp_index_store.build_index(doc, chunks=chunks)

    # Verify physical files exist
    idx_dir = temp_index_store.get_index_dir("doc-002")
    assert (idx_dir / "index.npy").is_file()
    assert (idx_dir / "chunks.json").is_file()
    assert (idx_dir / "metadata.json").is_file()

    # Clear memory cache to force disk reload
    temp_index_store._cache.clear()

    q_vec = embedder.embed_query("Are organ donor expenses covered?")
    results = temp_index_store.search("doc-002", q_vec, top_k=1)

    assert len(results) == 1
    assert "Organ donor" in results[0][0].text


def test_document_isolation_invariant(temp_index_store):
    """
    CRITICAL ARCHITECTURAL TEST:
    Verify that Document A's index contains ONLY Document A's data and never leaks Document B.
    """
    doc_a = make_dummy_document("doc-alpha", filename="alpha.pdf", file_hash="hash_alpha")
    doc_b = make_dummy_document("doc-beta", filename="beta.pdf", file_hash="hash_beta")

    chunks_a = make_dummy_chunks("doc-alpha", ["Policy Alpha covers robotic surgery up to 10 Lakhs."])
    chunks_b = make_dummy_chunks("doc-beta", ["Policy Beta covers bariatric surgery with 48 months waiting."])

    temp_index_store.build_index(doc_a, chunks=chunks_a)
    temp_index_store.build_index(doc_b, chunks=chunks_b)

    # Query for Alpha's robotic surgery against Beta's index
    q_vec = embedder.embed_query("robotic surgery limit")
    results_in_beta = temp_index_store.search("doc-beta", q_vec, top_k=5)

    # Beta's index must NOT contain Alpha's chunk
    for chunk, _ in results_in_beta:
        assert chunk.doc_id == "doc-beta"
        assert "Policy Alpha" not in chunk.text
        assert "robotic" not in chunk.text


def test_stale_index_rebuild_detection(temp_index_store):
    """Verify that a modified document hash or chunk count triggers rebuild."""
    doc = make_dummy_document("doc-003", file_hash="original_hash")
    chunks_v1 = make_dummy_chunks("doc-003", ["Initial clause text."])
    temp_index_store.build_index(doc, chunks=chunks_v1)

    # Modify document hash (simulating updated PDF)
    doc_updated = make_dummy_document("doc-003", file_hash="modified_hash")
    chunks_v2 = make_dummy_chunks("doc-003", ["Updated clause 1.", "New clause 2."])

    new_meta = temp_index_store.build_index(doc_updated, chunks=chunks_v2)
    assert new_meta.file_hash == "modified_hash"
    assert new_meta.chunk_count == 2


def test_corrupted_index_handling(temp_index_store):
    """Verify CorruptedIndexError when index files are malformed."""
    doc = make_dummy_document("doc-corrupt")
    chunks = make_dummy_chunks("doc-corrupt", ["Some text"])
    temp_index_store.build_index(doc, chunks=chunks)

    # Corrupt metadata.json
    idx_dir = temp_index_store.get_index_dir("doc-corrupt")
    temp_index_store._cache.clear()
    (idx_dir / "metadata.json").write_text("NOT_VALID_JSON", encoding="utf-8")

    with pytest.raises(CorruptedIndexError):
        temp_index_store.load_index("doc-corrupt")


def test_index_not_found(temp_index_store):
    """Verify IndexNotFoundError when searching non-existent document ID."""
    q_vec = embedder.embed_query("sample")
    with pytest.raises(IndexNotFoundError):
        temp_index_store.search("non_existent_doc_id", q_vec)
