"""Integration smoke test for PDF -> Ingestion -> Chunking -> Embedding -> FAISS Index."""

from pathlib import Path
import pytest

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embedder
from src.ingestion.index_store import IndexStore
from src.models.data_models import ChunkType


def test_end_to_end_embedding_and_faiss_smoke_test(tmp_path):
    """
    Verify complete pipeline from synthetic PDF to searchable FAISS index:
    1. Ingestion: PDF -> Pages
    2. Chunking: Pages -> Chunks (with tables and clauses)
    3. Indexing: Chunks -> Embeddings -> FAISS numpy_flat_ip
    4. Search: Retrieves the expected pre-existing disease waiting period chunk
    """
    fixture_path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert fixture_path.is_file(), "sample_policy_fixture.pdf fixture is missing"

    # Step 1: Ingest
    document = process_pdf(fixture_path, save_meta=False)
    assert document.page_count == 2

    # Step 2: Chunk
    chunks = chunk_document(document)
    assert len(chunks) == 9

    # Verify tables and clauses are represented
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
    clause_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLAUSE]
    assert len(table_chunks) >= 1
    assert len(clause_chunks) >= 4

    # Step 3: Index in temporary storage
    store = IndexStore(indexes_dir=tmp_path / "indexes", embedder_instance=embedder)
    meta = store.build_index(document, chunks=chunks)

    assert meta.document_id == document.doc_id
    assert meta.chunk_count == len(chunks)
    assert meta.embedding_dimension == 768

    # Step 4: Search for waiting period
    query = "What is the waiting period for pre-existing diseases?"
    q_vec = embedder.embed_query(query)
    results = store.search(document.doc_id, q_vec, top_k=3)

    assert len(results) >= 1
    top_chunk, top_score = results[0]

    # Verify the retrieved top chunk contains the exact answer text
    assert "36 months" in top_chunk.text, f"Expected '36 months' in top chunk, got: {top_chunk.text}"
    assert "pre-existing" in top_chunk.text.lower()
    assert top_chunk.page_number == 2
    assert 2 in top_chunk.page_numbers
    assert top_score > 0.45
