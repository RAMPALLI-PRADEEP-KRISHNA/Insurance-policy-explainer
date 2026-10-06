"""Integration and unit tests for Streamlit UI service functions and pipeline integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.app import (
    process_uploaded_policy,
    answer_policy_question,
    render_citations,
    render_evidence_status_badge,
)
from src.ingestion.embedder import embedder
from src.ingestion.index_store import IndexStore
from src.models.data_models import (
    Citation,
    DocumentQuality,
    DocumentStatus,
    EvidenceStatus,
    GroundedAnswer,
)
from src.utils.exceptions import MissingAPIKeyError


@pytest.fixture
def sample_pdf_bytes():
    """Load real synthetic fixture PDF bytes."""
    path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert path.is_file(), "sample_policy_fixture.pdf missing"
    return path.read_bytes()


def test_app_module_imports_cleanly():
    """Verify that src.app imports cleanly without errors."""
    import src.app as app
    assert hasattr(app, "process_uploaded_policy")
    assert hasattr(app, "answer_policy_question")
    assert hasattr(app, "main")


def test_process_uploaded_policy_pipeline(tmp_path, sample_pdf_bytes):
    """
    Verify upload processing workflow:
    Uploaded PDF bytes -> PDF extraction -> Chunking -> IndexStore
    """
    custom_store = IndexStore(indexes_dir=tmp_path / "ui_indexes", embedder_instance=embedder)

    doc, chunks = process_uploaded_policy(
        file_bytes=sample_pdf_bytes,
        filename="test_policy.pdf",
        indexes_store=custom_store,
    )

    assert doc.doc_id is not None
    assert doc.status == DocumentStatus.READY
    assert doc.quality == DocumentQuality.GOOD
    assert doc.page_count == 2
    assert len(chunks) == 9

    # Verify index was built and is fresh
    assert custom_store.index_exists(doc.doc_id)
    assert custom_store.is_fresh(doc.doc_id, doc.file_hash)

    # Test index reuse (calling again with same doc hash skips rebuild)
    doc_reuse, chunks_reuse = process_uploaded_policy(
        file_bytes=sample_pdf_bytes,
        filename="test_policy.pdf",
        indexes_store=custom_store,
    )
    assert doc_reuse.file_hash == doc.file_hash
    assert len(chunks_reuse) == 9


def test_answer_policy_question_sufficient_mocked(tmp_path, sample_pdf_bytes):
    """
    Verify question answering workflow for SUFFICIENT evidence:
    Retriever -> EvidenceValidator -> AnswerGenerator -> GroundedAnswer
    """
    custom_store = IndexStore(indexes_dir=tmp_path / "ui_indexes", embedder_instance=embedder)
    doc, _ = process_uploaded_policy(sample_pdf_bytes, "test_policy.pdf", indexes_store=custom_store)

    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "The waiting period for pre-existing diseases is 36 months of continuous coverage."

    from src.generation.answer_generator import AnswerGenerator
    from src.retrieval.retriever import Retriever

    custom_retriever = Retriever(index_store=custom_store, embedder_instance=embedder)
    custom_generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    query = "What is the waiting period for pre-existing diseases?"
    grounded_ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=custom_retriever,
        generator_instance=custom_generator,
    )

    assert grounded_ans.evidence_status == EvidenceStatus.SUFFICIENT
    assert "36 months" in grounded_ans.answer
    assert len(grounded_ans.citations) >= 1

    # Verify citation metadata provenance
    top_cit = grounded_ans.citations[0]
    assert top_cit.page_number == 2
    assert "36 months" in top_cit.source_snippet
    assert top_cit.document_id == doc.doc_id


def test_answer_policy_question_insufficient_state(tmp_path, sample_pdf_bytes):
    """
    Verify question answering workflow for INSUFFICIENT evidence:
    Unevidenced query against synthetic policy -> controlled refusal, empty citations.
    """
    custom_store = IndexStore(indexes_dir=tmp_path / "ui_indexes", embedder_instance=embedder)
    doc, _ = process_uploaded_policy(sample_pdf_bytes, "test_policy.pdf", indexes_store=custom_store)

    from src.retrieval.retriever import Retriever
    custom_retriever = Retriever(index_store=custom_store, embedder_instance=embedder)

    query = "What are the rules regarding quantum mechanics and string theory equations?"
    grounded_ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=custom_retriever,
    )

    # Must be INSUFFICIENT
    assert grounded_ans.evidence_status == EvidenceStatus.INSUFFICIENT
    assert "not find sufficient information" in grounded_ans.answer.lower()
    # No citations generated
    assert grounded_ans.citations == []


def test_answer_policy_question_marginal_state(tmp_path, sample_pdf_bytes):
    """
    Verify question answering workflow for MARGINAL evidence:
    Maternity inquiry with general unverified condition -> cautious answer with general citations.
    """
    custom_store = IndexStore(indexes_dir=tmp_path / "ui_indexes", embedder_instance=embedder)
    doc, _ = process_uploaded_policy(sample_pdf_bytes, "test_policy.pdf", indexes_store=custom_store)

    from src.retrieval.retriever import Retriever
    custom_retriever = Retriever(index_store=custom_store, embedder_instance=embedder)

    query = "Does the policy cover maternity and delivery expenses?"
    grounded_ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=custom_retriever,
    )

    assert grounded_ans.evidence_status == EvidenceStatus.MARGINAL
    assert "do not clearly or definitively specify" in grounded_ans.answer.lower()
    assert len(grounded_ans.citations) >= 1


def test_render_ui_badges_and_citations_callables():
    """Verify UI badge and citation rendering helper functions execute without crashing."""
    citation = Citation(
        document_id="doc_1",
        chunk_id="chunk_1",
        filename="policy.pdf",
        page_number=2,
        page_numbers=[2],
        section="Exclusions",
        heading="Waiting Period",
        source_snippet="36 months waiting period",
    )

    with patch("streamlit.expander") as mock_exp, \
         patch("streamlit.markdown") as mock_md, \
         patch("streamlit.caption") as mock_cap, \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.success") as mock_success, \
         patch("streamlit.warning") as mock_warning, \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.columns", return_value=(MagicMock(), MagicMock())):

        # Should render citations cleanly
        render_citations([citation])
        render_citations([])  # Empty handled

        # Should render all 4 badge states cleanly
        render_evidence_status_badge(EvidenceStatus.SUFFICIENT, 0.95)
        render_evidence_status_badge(EvidenceStatus.MARGINAL, 0.55)
        render_evidence_status_badge(EvidenceStatus.INSUFFICIENT, 0.0)
        render_evidence_status_badge(EvidenceStatus.CONFLICTING, 0.50)


def test_answer_policy_question_missing_api_key():
    """Verify clean exception when Gemini API key is missing on SUFFICIENT question."""
    from src.models.data_models import Chunk, ChunkType, Evidence, RetrievalResult
    from src.generation.answer_generator import AnswerGenerator
    from src.generation.gemini_client import GeminiClient

    client_without_key = GeminiClient(api_key=None, app_config=MagicMock(GEMINI_API_KEY=None, GEMINI_MODEL="gemini-2.5-flash"))
    generator = AnswerGenerator(gemini_client_instance=client_without_key)

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        filename="p.pdf",
        page_number=1,
        page_numbers=[1],
        chunk_type=ChunkType.CLAUSE,
        text="36 months waiting period",
        chunk_index=0,
    )
    result = RetrievalResult(
        chunk=chunk,
        vector_score=0.9,
        keyword_boost=0.1,
        final_score=1.0,
        rank=1,
        passed_threshold=True,
    )
    evidence = Evidence(
        query="What is waiting period?",
        doc_id="d1",
        status=EvidenceStatus.SUFFICIENT,
        results=[result],
        supporting_chunk_ids=["c1"],
        confidence=1.0,
        evidence_summary="36 months waiting period",
    )

    with pytest.raises(MissingAPIKeyError):
        generator.generate_answer("What is waiting period?", evidence)


def test_document_isolation_switching_policies(tmp_path, sample_pdf_bytes, make_text_pdf):
    """Verify switching active documents isolates retrieval to the new document ID."""
    custom_store = IndexStore(indexes_dir=tmp_path / "ui_indexes", embedder_instance=embedder)

    # Ingest Document A (synthetic health policy with 36 months waiting period)
    doc_a, _ = process_uploaded_policy(sample_pdf_bytes, "health_policy.pdf", indexes_store=custom_store)

    # Ingest Document B (motor vehicle policy with zero health clauses)
    motor_bytes = make_text_pdf(["Motor Vehicle Third Party Insurance: Limit Rs. 7,50,000 for bodily injury."])
    doc_b, _ = process_uploaded_policy(motor_bytes, "motor_policy.pdf", indexes_store=custom_store)

    from src.retrieval.retriever import Retriever
    retriever_inst = Retriever(index_store=custom_store, embedder_instance=embedder)

    # Query Document A
    res_a = retriever_inst.retrieve(doc_a.doc_id, "waiting period for pre-existing disease")
    for r in res_a:
        assert r.doc_id == doc_a.doc_id
        assert "motor" not in r.filename.lower()

    # Query Document B with same question -> motor policy should not leak health chunks
    res_b = retriever_inst.retrieve(doc_b.doc_id, "waiting period for pre-existing disease")
    for r in res_b:
        assert r.doc_id == doc_b.doc_id
        assert "health" not in r.filename.lower()
