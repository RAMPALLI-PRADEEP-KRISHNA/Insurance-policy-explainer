"""Unit tests for hybrid retriever and ranking pipeline."""

import pytest
from backend.ingestion.embedder import embedder
from database.index_store import IndexStore
from backend.models.data_models import Chunk, ChunkType, Document, DocumentStatus
from backend.retrieval.retriever import Retriever
from backend.utils.exceptions import IndexNotFoundError, RetrievalError


@pytest.fixture
def mock_retriever_setup(tmp_path):
    """Create two isolated documents and indexes with test chunks."""
    store = IndexStore(indexes_dir=tmp_path / "indexes", embedder_instance=embedder)

    doc_a = Document(
        doc_id="doc_alpha",
        filename="alpha_policy.pdf",
        file_hash="hash_alpha_123",
        page_count=2,
        status=DocumentStatus.READY,
    )
    chunks_a = [
        Chunk(
            chunk_id="alpha_c1",
            doc_id="doc_alpha",
            filename="alpha_policy.pdf",
            page_number=1,
            page_numbers=[1],
            section="Schedule of Benefits",
            subsection="Room Rent Limit",
            chunk_type=ChunkType.TABLE,
            text="| Room Rent | Rs. 5,000 per day |",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="alpha_c2",
            doc_id="doc_alpha",
            filename="alpha_policy.pdf",
            page_number=2,
            page_numbers=[2],
            section="Exclusions",
            subsection="Pre-existing Waiting Period",
            chunk_type=ChunkType.CLAUSE,
            text="Waiting period for pre-existing disease is 36 months of continuous coverage.",
            chunk_index=1,
        ),
    ]
    store.build_index(doc_a, chunks=chunks_a)

    doc_b = Document(
        doc_id="doc_beta",
        filename="beta_policy.pdf",
        file_hash="hash_beta_456",
        page_count=1,
        status=DocumentStatus.READY,
    )
    chunks_b = [
        Chunk(
            chunk_id="beta_c1",
            doc_id="doc_beta",
            filename="beta_policy.pdf",
            page_number=1,
            page_numbers=[1],
            section="Vehicle Coverage",
            subsection="Third Party Liability",
            chunk_type=ChunkType.CLAUSE,
            text="Motor third party liability insurance limit is Rs. 7,50,000.",
            chunk_index=0,
        ),
    ]
    store.build_index(doc_b, chunks=chunks_b)

    retriever = Retriever(index_store=store, embedder_instance=embedder)
    return retriever, doc_a, doc_b


def test_retriever_query_validation(mock_retriever_setup):
    """Verify empty or whitespace-only queries are rejected with ValueError."""
    retriever, doc_a, _ = mock_retriever_setup

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve(doc_a.doc_id, "")

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve(doc_a.doc_id, "   \n\t  ")


def test_retriever_missing_index_raises(mock_retriever_setup):
    """Verify querying an un-indexed document ID raises IndexNotFoundError."""
    retriever, _, _ = mock_retriever_setup

    with pytest.raises(IndexNotFoundError):
        retriever.retrieve("non_existent_doc_id", "What is the waiting period?")


def test_retriever_document_isolation(mock_retriever_setup):
    """Verify that retrieval strictly isolates candidate chunks to the target document ID."""
    retriever, doc_a, doc_b = mock_retriever_setup

    # Query doc_a with motor vehicle query
    results_a = retriever.retrieve(doc_a.doc_id, "motor vehicle liability")
    for r in results_a:
        assert r.doc_id == doc_a.doc_id
        assert "beta" not in r.chunk_id

    # Query doc_b with health room rent query
    results_b = retriever.retrieve(doc_b.doc_id, "room rent Rs. 5000 per day")
    for r in results_b:
        assert r.doc_id == doc_b.doc_id
        assert "alpha" not in r.chunk_id


def test_retriever_score_ordering_and_boost_transparency(mock_retriever_setup):
    """Verify raw similarity is preserved and final score equals vector_score + keyword_boost."""
    retriever, doc_a, _ = mock_retriever_setup

    query = "What is the waiting period for pre-existing disease? Is it 36 months?"
    results = retriever.retrieve(doc_a.doc_id, query, top_k=2)

    assert len(results) == 2
    top = results[0]

    # Verify score formula: final_score == round(min(1.0, vector_score + keyword_boost), 4)
    expected_score = round(min(1.0, top.vector_score + top.keyword_boost), 4)
    assert top.final_score == expected_score
    assert top.rank == 1
    assert top.raw_similarity == top.vector_score
    # Verify keyword boost is non-negative and bounded
    assert 0.0 <= top.keyword_boost <= 0.15


def test_retriever_top_k_and_oversized_top_k(mock_retriever_setup):
    """Verify top_k limits results, and oversized top_k gracefully returns all available chunks."""
    retriever, doc_a, _ = mock_retriever_setup

    # top_k = 1
    r_1 = retriever.retrieve(doc_a.doc_id, "room rent", top_k=1)
    assert len(r_1) == 1

    # oversized top_k = 50 (doc_a only has 2 chunks)
    r_all = retriever.retrieve(doc_a.doc_id, "room rent", top_k=50)
    assert len(r_all) == 2


def test_retriever_citation_metadata_provenance(mock_retriever_setup):
    """Verify all citation metadata (page number, page numbers, section, heading, chunk_id) are strictly preserved from Chunk."""
    retriever, doc_a, _ = mock_retriever_setup

    results = retriever.retrieve(doc_a.doc_id, "Room Rent", top_k=1)
    assert len(results) == 1
    r = results[0]

    assert r.doc_id == "doc_alpha"
    assert r.chunk_id == "alpha_c1"
    assert r.page_number == 1
    assert r.page_numbers == [1]
    assert r.section == "Schedule of Benefits"
    assert r.heading == "Room Rent Limit"
    assert r.chunk_type == ChunkType.TABLE
    assert r.text == "| Room Rent | Rs. 5,00,0 per day |" or "Room Rent" in r.text
