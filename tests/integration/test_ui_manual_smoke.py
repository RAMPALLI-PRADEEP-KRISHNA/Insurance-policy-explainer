"""Integration tests verifying the 6 manual smoke test scenarios specified in Phase 5."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.app import process_uploaded_policy, answer_policy_question
from src.ingestion.embedder import embedder
from src.ingestion.index_store import IndexStore
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.retriever import Retriever
from src.retrieval.evidence_validator import EvidenceValidator
from src.models.data_models import DocumentQuality, DocumentStatus, EvidenceStatus


@pytest.fixture(scope="module")
def smoke_environment(tmp_path_factory):
    """Set up the isolated index directory and synthetic fixture bytes."""
    temp_dir = tmp_path_factory.mktemp("manual_smoke_indexes")
    fixture_path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert fixture_path.is_file(), "sample_policy_fixture.pdf missing"
    health_bytes = fixture_path.read_bytes()

    custom_store = IndexStore(indexes_dir=temp_dir, embedder_instance=embedder)
    custom_retriever = Retriever(index_store=custom_store, embedder_instance=embedder)
    custom_validator = EvidenceValidator()

    # Mock Gemini to return realistic grounded responses for SUFFICIENT questions
    mock_gemini = MagicMock()
    def _mock_gen(prompt, system_instruction=None, **kwargs):
        p_lower = prompt.lower()
        if "pre-existing" in p_lower or "36 months" in p_lower:
            return "The waiting period for pre-existing diseases is 36 months of continuous coverage."
        if "room rent" in p_lower:
            return "Room rent is limited up to 1% of the Sum Insured per day."
        return "Grounded policy explanation."
    mock_gemini.generate.side_effect = _mock_gen

    custom_generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    return {
        "store": custom_store,
        "retriever": custom_retriever,
        "validator": custom_validator,
        "generator": custom_generator,
        "health_bytes": health_bytes,
    }


def test_smoke_1_upload_synthetic_policy(smoke_environment):
    """Smoke Test 1: Upload synthetic policy -> document processed, index available, no crash."""
    env = smoke_environment
    doc, chunks = process_uploaded_policy(
        file_bytes=env["health_bytes"],
        filename="sample_policy.pdf",
        indexes_store=env["store"],
    )

    assert doc.status == DocumentStatus.READY
    assert doc.quality == DocumentQuality.GOOD
    assert doc.page_count == 2
    assert len(chunks) == 9
    assert env["store"].index_exists(doc.doc_id)
    env["doc_health"] = doc


def test_smoke_2_ask_waiting_period_pre_existing(smoke_environment):
    """Smoke Test 2: Ask pre-existing waiting period -> 36 months, SUFFICIENT, Page 2 citation."""
    env = smoke_environment
    doc = env["doc_health"]
    query = "What is the waiting period for pre-existing diseases?"

    ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    assert ans.evidence_status == EvidenceStatus.SUFFICIENT
    assert "36 months" in ans.answer
    assert len(ans.citations) >= 1
    assert ans.citations[0].page_number == 2
    assert "36 months" in ans.citations[0].source_snippet


def test_smoke_3_ask_room_rent_limit(smoke_environment):
    """Smoke Test 3: Ask room rent limit -> references table, Page 1 citation."""
    env = smoke_environment
    doc = env["doc_health"]
    query = "What is the room rent limit?"

    ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    assert ans.evidence_status == EvidenceStatus.SUFFICIENT
    assert "1%" in ans.answer or "room rent" in ans.answer.lower()
    assert len(ans.citations) >= 1
    assert ans.citations[0].page_number == 1
    assert "1%" in ans.citations[0].source_snippet


def test_smoke_4_ask_maternity_expenses(smoke_environment):
    """Smoke Test 4: Ask maternity expenses -> MARGINAL/INSUFFICIENT, cautious/refusal, no fabricated claim."""
    env = smoke_environment
    doc = env["doc_health"]
    query = "Does this policy cover maternity expenses?"

    ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    assert ans.evidence_status in {EvidenceStatus.MARGINAL, EvidenceStatus.INSUFFICIENT}
    # Invariant: Never claims "maternity is fully covered" or "maternity is permanently excluded"
    assert "maternity is covered" not in ans.answer.lower()
    assert "maternity is excluded" not in ans.answer.lower()


def test_smoke_5_ask_pre_existing_disease(smoke_environment):
    """Smoke Test 5: Ask pre-existing disease waiting period -> English answer, 36-month fact preserved, Page 2 citation."""
    env = smoke_environment
    doc = env["doc_health"]
    query = "What is the waiting period for pre-existing diseases?"

    ans = answer_policy_question(
        doc_id=doc.doc_id,
        query=query,
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    assert ans.language == "en"
    assert ans.evidence_status == EvidenceStatus.SUFFICIENT
    assert "36" in ans.answer
    assert len(ans.citations) >= 1
    assert ans.citations[0].page_number == 2
    assert "36 months" in ans.citations[0].source_snippet


def test_smoke_6_upload_second_policy_and_isolation(smoke_environment, make_text_pdf):
    """Smoke Test 6: Upload second policy -> active document changes, queries isolated, no leakage."""
    env = smoke_environment
    motor_bytes = make_text_pdf(["Motor Vehicle Third Party Insurance Limit: Rs. 7,50,000 for third-party injury."])

    doc_motor, chunks_motor = process_uploaded_policy(
        file_bytes=motor_bytes,
        filename="motor_policy.pdf",
        indexes_store=env["store"],
    )

    assert doc_motor.doc_id != env["doc_health"].doc_id
    assert doc_motor.filename == "motor_policy.pdf"

    # Query motor policy for third party limit
    ans_motor = answer_policy_question(
        doc_id=doc_motor.doc_id,
        query="What is the motor third party limit?",
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    for c in ans_motor.citations:
        assert c.document_id == doc_motor.doc_id
        assert "health" not in c.filename.lower()

    # Query motor policy for health waiting period -> must NOT leak health policy chunks!
    ans_leak_check = answer_policy_question(
        doc_id=doc_motor.doc_id,
        query="What is the waiting period for pre-existing diseases?",
        language="en",
        retriever_instance=env["retriever"],
        validator_instance=env["validator"],
        generator_instance=env["generator"],
    )

    for c in ans_leak_check.citations:
        assert c.document_id == doc_motor.doc_id
        assert "sample_policy" not in c.filename


