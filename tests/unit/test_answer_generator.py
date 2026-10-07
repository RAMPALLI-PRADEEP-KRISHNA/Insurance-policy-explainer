"""Unit tests for grounded answer generator under all evidence states."""

from unittest.mock import MagicMock
import pytest
from src.generation.answer_generator import AnswerGenerator
from src.models.data_models import (
    Chunk,
    ChunkType,
    Evidence,
    EvidenceStatus,
    RetrievalResult,
)


def _make_chunk_result(
    chunk_id: str,
    text: str,
    page_number: int = 1,
    section: str = "Test Section",
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc_policy_01",
        filename="health_policy.pdf",
        page_number=page_number,
        page_numbers=[page_number],
        section=section,
        chunk_type=ChunkType.CLAUSE,
        text=text,
        chunk_index=0,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=0.88,
        keyword_boost=0.10,
        final_score=0.98,
        rank=1,
        passed_threshold=True,
    )


def test_answer_generator_sufficient_state():
    """Verify SUFFICIENT evidence generates grounded answer with authoritative citations."""
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "The waiting period for pre-existing diseases is 36 months of continuous coverage."

    generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    query = "What is the waiting period for pre-existing diseases?"
    r1 = _make_chunk_result(
        "c_ped_36",
        "2.3 Pre-existing Disease Waiting Period: 36 months of continuous coverage.",
        page_number=2,
    )
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1],
        supporting_chunk_ids=["c_ped_36"],
        confidence=0.98,
        evidence_summary="[Page 2] Exclusions:\n2.3 Pre-existing Disease Waiting Period: 36 months of continuous coverage.",
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert answer.evidence_status == EvidenceStatus.SUFFICIENT
    assert "36 months" in answer.answer
    assert answer.confidence == 0.98
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 2
    assert answer.citations[0].chunk_id == "c_ped_36"
    assert "36 months" in answer.citations[0].source_snippet
    # Verify Gemini was invoked once
    mock_gemini.generate.assert_called_once()


def test_answer_generator_marginal_state():
    """Verify MARGINAL evidence returns a cautious answer without inventing facts."""
    mock_gemini = MagicMock()
    generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    query = "What is the waiting period for cataract surgery?"
    r1 = _make_chunk_result(
        "c_gen_wait",
        "General waiting period is 30 days. Specific procedures have waiting periods.",
        page_number=2,
    )
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.MARGINAL,
        results=[r1],
        supporting_chunk_ids=["c_gen_wait"],
        confidence=0.55,
        evidence_summary="[Page 2] General waiting period is 30 days.",
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert answer.evidence_status == EvidenceStatus.MARGINAL
    # Must use cautious wording
    assert "do not clearly or definitively specify" in answer.answer or "discuss related terms" in answer.answer
    # Citations still provided from the partial evidence
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 2


def test_answer_generator_insufficient_state_abstains():
    """Verify INSUFFICIENT evidence abstains without calling LLM and without claiming absence."""
    mock_gemini = MagicMock()
    generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    query = "Does the policy cover maternity expenses?"
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.INSUFFICIENT,
        results=[],
        supporting_chunk_ids=[],
        confidence=0.0,
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert answer.evidence_status == EvidenceStatus.INSUFFICIENT
    assert "could not find sufficient information" in answer.answer
    # Invariant: Never claims "the policy does not cover it"
    assert "policy does not cover" not in answer.answer.lower()
    # Citations must be empty
    assert answer.citations == []
    # LLM must NOT be called on INSUFFICIENT
    mock_gemini.generate.assert_not_called()


def test_answer_generator_conflicting_state():
    """Verify CONFLICTING evidence reports the contradiction and cites both sources."""
    mock_gemini = MagicMock()
    generator = AnswerGenerator(gemini_client_instance=mock_gemini)

    query = "What is the co-payment percentage?"
    r1 = _make_chunk_result("c_co_10", "Co-payment is 10% for all claims.", page_number=1)
    r2 = _make_chunk_result("c_co_20", "Co-payment is 20% for all claims.", page_number=3)
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.CONFLICTING,
        results=[r1, r2],
        supporting_chunk_ids=["c_co_10", "c_co_20"],
        confidence=0.50,
        detected_conflicts=["Conflicting percentage: 10% in chunk c_co_10 vs 20% in chunk c_co_20."],
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert answer.evidence_status == EvidenceStatus.CONFLICTING
    assert "conflicting information" in answer.answer.lower()
    # Citations for both conflicting chunks must be retained
    assert len(answer.citations) == 2
    cited_pages = {c.page_number for c in answer.citations}
    assert cited_pages == {1, 3}
    # LLM not called to resolve the contradiction arbitrarily
    mock_gemini.generate.assert_not_called()


def test_numeric_preservation_and_conditions():
    """Verify numeric values and conditions (e.g. age 61, 10% co-pay) are preserved in generation output."""
    mock_gemini = MagicMock()
    expected_ans = "There is a 10% co-pay applicable specifically for insured persons of age 61 and above."
    mock_gemini.generate.return_value = expected_ans

    generator = AnswerGenerator(gemini_client_instance=mock_gemini)
    query = "Is there a co-pay and does it depend on age?"
    r1 = _make_chunk_result("c_co_age", "10% co-pay for age 61 and above", page_number=1)
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1],
        supporting_chunk_ids=["c_co_age"],
        confidence=0.95,
        evidence_summary="10% co-pay for age 61 and above",
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert "10%" in answer.answer
    assert "age 61" in answer.answer


def test_negation_preservation():
    """Verify negative coverage clauses ('excluded', 'not covered') are preserved."""
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "Hazardous sports and aesthetic surgery are permanently excluded from coverage."

    generator = AnswerGenerator(gemini_client_instance=mock_gemini)
    query = "Are hazardous sports covered?"
    r1 = _make_chunk_result("c_excl", "2.4 Permanent Exclusions: Hazardous sports, self-inflicted injury.", page_number=2)
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1],
        supporting_chunk_ids=["c_excl"],
        confidence=0.90,
        evidence_summary="Hazardous sports are permanently excluded.",
    )

    answer = generator.generate_answer(query, evidence, language="en")

    assert "excluded" in answer.answer.lower() or "not covered" in answer.answer.lower()


def test_citation_integrity_cannot_be_overwritten_by_llm():
    """Verify that even if the LLM output claims 'Page 99', the citation metadata remains strictly Page 2."""
    mock_gemini = MagicMock()
    # LLM hallucinates an invalid page number in its prose
    mock_gemini.generate.return_value = "According to Page 99, the waiting period is 36 months."

    generator = AnswerGenerator(gemini_client_instance=mock_gemini)
    query = "What is the waiting period?"
    r1 = _make_chunk_result("c_ped_real", "36 months waiting period", page_number=2)
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1],
        supporting_chunk_ids=["c_ped_real"],
        confidence=0.92,
        evidence_summary="36 months waiting period",
    )

    answer = generator.generate_answer(query, evidence, language="en")

    # Authoritative citation remains strictly Page 2
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 2
    assert answer.citations[0].chunk_id == "c_ped_real"


def test_tamil_answer_generation():
    """Verify Tamil answer generation preserves numbers, conditions, and authoritative citations."""
    mock_gemini = MagicMock()
    mock_gemini.generate.side_effect = [
        "The waiting period for pre-existing diseases is 36 months of continuous coverage.",
        "முன்பே இருக்கும் நோய்களுக்கான காத்திருப்பு காலம் 36 months (முப்பத்தாறு மாதங்கள்) ஆகும்.",
    ]

    generator = AnswerGenerator(gemini_client_instance=mock_gemini)
    query = "ஏற்கனவே உள்ள நோய்களுக்கான காத்திருப்பு காலம் என்ன?"
    r1 = _make_chunk_result("c_ped_tamil", "2.3 Pre-existing Disease Waiting Period: 36 months", page_number=2)
    evidence = Evidence(
        query=query,
        doc_id="doc_policy_01",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1],
        supporting_chunk_ids=["c_ped_tamil"],
        confidence=0.89,
        evidence_summary="36 months waiting period",
    )

    answer = generator.generate_answer(query, evidence, language="ta")

    assert answer.language == "ta"
    assert "36" in answer.answer
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 2
    assert answer.citations[0].chunk_id == "c_ped_tamil"
import pytest
from unittest.mock import MagicMock
from src.generation.answer_generator import AnswerGenerator
from src.models.data_models import Evidence, EvidenceStatus
from src.utils.exceptions import GeminiRateLimitError, MissingAPIKeyError, GenerationTimeoutError

def test_generator_429_llm_unavailable():
    mock_client = MagicMock()
    mock_client.generate.side_effect = GeminiRateLimitError("Quota exceeded")
    
    generator = AnswerGenerator(gemini_client_instance=mock_client)
    evidence = Evidence(
        query="What is the waiting period?",
        doc_id="d1",
        status=EvidenceStatus.SUFFICIENT,
        results=[],
        supporting_chunk_ids=["c1"],
        confidence=0.9,
        evidence_summary="36 months waiting period"
    )
    
    ans = generator.generate_answer("What is the waiting period?", evidence)
    
    assert ans.llm_unavailable is True
    assert "Quota exceeded" in ans.llm_unavailable_reason
    assert ans.answer == "36 months waiting period"
    assert mock_client.generate.call_count == 1

def test_generator_missing_key_raises():
    mock_client = MagicMock()
    mock_client.generate.side_effect = MissingAPIKeyError("No API key")
    
    generator = AnswerGenerator(gemini_client_instance=mock_client)
    evidence = Evidence(
        query="What is the waiting period?",
        doc_id="d1",
        status=EvidenceStatus.SUFFICIENT,
        results=[],
        supporting_chunk_ids=[],
        confidence=0.9,
        evidence_summary="36 months"
    )
    
    with pytest.raises(MissingAPIKeyError):
        generator.generate_answer("What is the waiting period?", evidence)

def test_generator_no_evidence_no_llm_call():
    mock_client = MagicMock()
    
    generator = AnswerGenerator(gemini_client_instance=mock_client)
    evidence = Evidence(
        query="What is the meaning of life?",
        doc_id="d1",
        status=EvidenceStatus.INSUFFICIENT,
        results=[],
        supporting_chunk_ids=[],
        confidence=0.1,
        evidence_summary=""
    )
    
    ans = generator.generate_answer("What is the meaning of life?", evidence)
    
    assert ans.llm_unavailable is False
    assert mock_client.generate.call_count == 0
    assert "I could not find sufficient information" in ans.answer
