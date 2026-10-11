"""Live Gemini smoke test verifying grounded answer generation when API key is provided."""

from pathlib import Path
import pytest

pytestmark = pytest.mark.live
from backend.config import config
from backend.ingestion.pdf_extractor import process_pdf
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import embedder
from database.index_store import IndexStore
from backend.retrieval.retriever import Retriever
from backend.retrieval.evidence_validator import EvidenceValidator
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.gemini_client import GeminiClient
from backend.models.data_models import EvidenceStatus


@pytest.fixture(scope="module")
def end_to_end_system(tmp_path_factory):
    """Set up the complete RAG pipeline on the synthetic insurance policy."""
    temp_dir = tmp_path_factory.mktemp("live_smoke_indexes")
    fixture_path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert fixture_path.is_file(), "sample_policy_fixture.pdf missing"

    doc = process_pdf(fixture_path, save_meta=False)
    chunks = chunk_document(doc)

    store = IndexStore(indexes_dir=temp_dir, embedder_instance=embedder)
    store.build_index(doc, chunks=chunks)

    retriever = Retriever(index_store=store, embedder_instance=embedder)
    validator = EvidenceValidator()
    generator = AnswerGenerator()

    return doc, retriever, validator, generator


def test_live_gemini_smoke_or_skip(end_to_end_system):
    """
    Live smoke test for Gemini grounded generation:
    1. Query: 'What is the waiting period for pre-existing diseases?' -> 36 months, Page 2
    2. Query: 'Does this policy cover maternity expenses?' -> Refusal / Abstention
    """
    api_key = config.GEMINI_API_KEY
    if not api_key or api_key.startswith("test_") or "your_" in api_key.lower():
        pytest.skip("LIVE GEMINI SMOKE TEST SKIPPED — GEMINI_API_KEY not configured.")

    doc, retriever, validator, generator = end_to_end_system

    # Validate model availability
    client = GeminiClient(api_key=api_key)
    client.validate_model()

    # Query 1: Pre-existing diseases
    query_ped = "What is the waiting period for pre-existing diseases?"
    results_ped = retriever.retrieve(doc.doc_id, query_ped)
    evidence_ped = validator.validate(query_ped, results_ped, doc_id=doc.doc_id)

    assert evidence_ped.status == EvidenceStatus.SUFFICIENT

    answer_ped = generator.generate_answer(query_ped, evidence_ped, language="en")
    assert answer_ped.evidence_status == EvidenceStatus.SUFFICIENT
    assert "36" in answer_ped.answer
    assert len(answer_ped.citations) >= 1
    assert answer_ped.citations[0].page_number == 2
    assert "36 months" in answer_ped.citations[0].source_snippet

    # Query 2: Maternity expenses (absent from synthetic policy)
    query_mat = "Does this policy cover maternity expenses?"
    results_mat = retriever.retrieve(doc.doc_id, query_mat)
    evidence_mat = validator.validate(query_mat, results_mat, doc_id=doc.doc_id)

    # Must be INSUFFICIENT or MARGINAL without claiming coverage
    answer_mat = generator.generate_answer(query_mat, evidence_mat, language="en")
    assert answer_mat.evidence_status in {EvidenceStatus.INSUFFICIENT, EvidenceStatus.MARGINAL}
    if answer_mat.evidence_status == EvidenceStatus.INSUFFICIENT:
        assert answer_mat.citations == []
        assert "not find sufficient information" in answer_mat.answer



