"""End-to-end integration test for the hybrid retrieval and evidence validation pipeline."""

import time
from pathlib import Path
import pytest

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embedder
from src.ingestion.index_store import IndexStore
from src.retrieval.retriever import Retriever
from src.retrieval.evidence_validator import EvidenceValidator
from src.models.data_models import ChunkType, EvidenceStatus


@pytest.fixture(scope="module")
def indexed_synthetic_fixture(tmp_path_factory):
    """Ingest, chunk, and index the real synthetic policy fixture once for module tests."""
    temp_dir = tmp_path_factory.mktemp("retrieval_indexes")
    fixture_path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert fixture_path.is_file(), "sample_policy_fixture.pdf missing"

    document = process_pdf(fixture_path, save_meta=False)
    chunks = chunk_document(document)
    assert len(chunks) == 9

    store = IndexStore(indexes_dir=temp_dir, embedder_instance=embedder)
    store.build_index(document, chunks=chunks)

    retriever = Retriever(index_store=store, embedder_instance=embedder)
    validator = EvidenceValidator()

    return document, retriever, validator


def test_query_a_pre_existing_disease_waiting_period(indexed_synthetic_fixture):
    """
    Query A: 'What is the waiting period for pre-existing diseases?'
    Expected: Retrieves 36 months pre-existing clause on Page 2 with SUFFICIENT status.
    """
    doc, retriever, validator = indexed_synthetic_fixture
    query = "What is the waiting period for pre-existing diseases?"

    results = retriever.retrieve(doc.doc_id, query, top_k=3)
    assert len(results) >= 1

    top = results[0]
    assert "36 months" in top.text
    assert top.page_number == 2
    assert "pre-existing" in top.text.lower()
    assert top.raw_similarity > 0.45
    assert top.final_score >= top.raw_similarity

    evidence = validator.validate(query, results, doc_id=doc.doc_id)
    assert evidence.status == EvidenceStatus.SUFFICIENT
    assert top.chunk_id in evidence.supporting_chunk_ids
    assert "36" in evidence.evidence_summary


def test_query_b_specific_illness_waiting_period(indexed_synthetic_fixture):
    """
    Query B: 'What is the waiting period for specific illnesses?'
    Expected: Retrieves clause 2.2 ('Specific Disease Waiting Period: 24 months for specified procedures like cataract')
    ranked higher than the 36-month pre-existing clause.
    """
    doc, retriever, validator = indexed_synthetic_fixture
    query = "What is the waiting period for specific illnesses?"

    results = retriever.retrieve(doc.doc_id, query, top_k=3)
    assert len(results) >= 2

    top = results[0]
    # Top chunk must be specific disease/illness clause, not pre-existing clause
    assert "specific disease" in top.text.lower() or "cataract" in top.text.lower()
    assert "24 months" in top.text
    assert "pre-existing disease" not in top.heading.lower() if top.heading else True


def test_query_c_room_rent_limit(indexed_synthetic_fixture):
    """
    Query C: 'What is the room rent limit?'
    Expected: Retrieves the table chunk containing 'Room Rent | Up to 1% of Sum Insured per day'.
    """
    doc, retriever, validator = indexed_synthetic_fixture
    query = "What is the room rent limit?"

    results = retriever.retrieve(doc.doc_id, query, top_k=3)
    assert len(results) >= 1

    top = results[0]
    assert top.chunk_type == ChunkType.TABLE
    assert "room rent" in top.text.lower()
    assert "1%" in top.text
    assert top.page_number == 1

    evidence = validator.validate(query, results, doc_id=doc.doc_id)
    assert evidence.status == EvidenceStatus.SUFFICIENT
    assert "1%" in evidence.evidence_summary


def test_query_d_co_pay_clause(indexed_synthetic_fixture):
    """
    Query D: 'Is there a co-pay?'
    Expected: Retrieves the chunk containing co-pay terms (10% co-pay for age 61 and above).
    """
    doc, retriever, validator = indexed_synthetic_fixture
    query = "Is there a co-pay?"

    results = retriever.retrieve(doc.doc_id, query, top_k=3)
    assert len(results) >= 1

    top = results[0]
    assert "co-pay" in top.text.lower() or "co-payment" in top.text.lower()
    assert "10%" in top.text or "61" in top.text

    evidence = validator.validate(query, results, doc_id=doc.doc_id)
    assert evidence.status in {EvidenceStatus.SUFFICIENT, EvidenceStatus.MARGINAL}


def test_query_e_tamil_waiting_period_retrieval(indexed_synthetic_fixture):
    """
    Query E: 'ஏற்கனவே உள்ள நோய்களுக்கான காத்திருப்பு காலம் என்ன?'
    Expected: Cross-lingual retrieval matches the 36-month pre-existing disease clause at Rank 1.
    """
    doc, retriever, validator = indexed_synthetic_fixture
    tamil_query = "ஏற்கனவே உள்ள நோய்களுக்கான காத்திருப்பு காலம் என்ன?"

    results = retriever.retrieve(doc.doc_id, tamil_query, top_k=3)
    assert len(results) >= 1

    top = results[0]
    # Check that rank 1 is clause 2.3 with 36 months
    assert "36 months" in top.text
    assert "pre-existing" in top.text.lower()
    assert top.page_number == 2
    assert top.final_score >= 0.65

    evidence = validator.validate(tamil_query, results, doc_id=doc.doc_id)
    assert evidence.status == EvidenceStatus.SUFFICIENT
    assert top.chunk_id in evidence.supporting_chunk_ids


@pytest.mark.perf
def test_retrieval_performance_timings(indexed_synthetic_fixture):
    """
    Measure and log execution latency for each stage of the retrieval pipeline:
    1. Query embedding time
    2. FAISS search time
    3. Signal boosting time
    4. Evidence validation time
    5. Total retrieval pipeline time
    """
    doc, retriever, validator = indexed_synthetic_fixture
    query = "What is the waiting period for pre-existing diseases?"

    # 1. Query Embedding
    t0 = time.perf_counter()
    q_vec = retriever.embedder.embed_query(query)
    t_embed = time.perf_counter() - t0

    # 2. FAISS search
    t0 = time.perf_counter()
    candidates = retriever.index_store.search(doc.doc_id, q_vec, top_k=10)
    t_faiss = time.perf_counter() - t0

    # 3. Full retrieval (includes boosting and ranking)
    t0 = time.perf_counter()
    results = retriever.retrieve(doc.doc_id, query, top_k=5)
    t_retrieval = time.perf_counter() - t0
    t_boost = max(0.0, t_retrieval - t_embed - t_faiss)

    # 4. Evidence validation
    t0 = time.perf_counter()
    evidence = validator.validate(query, results, doc_id=doc.doc_id)
    t_val = time.perf_counter() - t0

    total_time = t_retrieval + t_val

    # Print / assert performance within reasonable CPU bounds (< 500ms on CPU)
    print(
        f"\n[PERFORMANCE METRICS]\n"
        f"  - Query Embedding:     {t_embed*1000:.2f} ms\n"
        f"  - FAISS Vector Search: {t_faiss*1000:.2f} ms\n"
        f"  - Signal Boosting:     {t_boost*1000:.2f} ms\n"
        f"  - Evidence Validation: {t_val*1000:.2f} ms\n"
        f"  - Total Pipeline Time: {total_time*1000:.2f} ms\n"
    )

    assert t_embed < 0.25, f"Query embedding took too long: {t_embed:.2f}s"
    assert t_faiss < 0.05, f"FAISS search took too long: {t_faiss:.2f}s"
    assert t_val < 0.05, f"Evidence validation took too long: {t_val:.2f}s"
    assert total_time < 0.50, f"Total pipeline latency exceeded 500ms: {total_time:.2f}s"

