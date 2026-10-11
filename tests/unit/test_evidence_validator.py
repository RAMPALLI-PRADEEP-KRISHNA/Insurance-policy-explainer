"""Unit tests for four-state evidence validator."""

from backend.models.data_models import (
    Chunk,
    ChunkType,
    EvidenceStatus,
    RetrievalResult,
)
from backend.retrieval.evidence_validator import EvidenceValidator


def _make_result(
    chunk_id: str,
    text: str,
    final_score: float,
    page_number: int = 1,
    passed_threshold: bool = True,
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="test_doc",
        filename="policy.pdf",
        page_number=page_number,
        page_numbers=[page_number],
        section="Test Section",
        chunk_type=ChunkType.CLAUSE,
        text=text,
        chunk_index=0,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=final_score,
        keyword_boost=0.0,
        final_score=final_score,
        rank=1,
        passed_threshold=passed_threshold,
    )


def test_validate_sufficient_state():
    """Verify SUFFICIENT when evidence directly and with high confidence answers the query."""
    validator = EvidenceValidator()
    query = "What is the waiting period for pre-existing diseases?"
    results = [
        _make_result(
            "c1",
            "Coverage for pre-existing diseases commences after thirty-six (36) months of continuous coverage.",
            final_score=0.78,
            page_number=2,
        ),
        _make_result(
            "c2",
            "Initial waiting period is 30 days.",
            final_score=0.66,
            page_number=2,
        ),
    ]

    evidence = validator.validate(query, results, doc_id="test_doc")

    assert evidence.status == EvidenceStatus.SUFFICIENT
    assert evidence.confidence >= 0.75
    assert "c1" in evidence.supporting_chunk_ids
    assert evidence.evidence_summary is not None
    assert "36" in evidence.evidence_summary


def test_validate_marginal_low_threshold_state():
    """Verify MARGINAL when evidence score is between LOW_THRESHOLD (0.40) and HIGH_THRESHOLD (0.65)."""
    validator = EvidenceValidator()
    query = "What are the outpatient consultation limits?"
    results = [
        _make_result(
            "c1",
            "Outpatient consultation is subject to specified network clinic charges.",
            final_score=0.52,  # Between 0.40 and 0.65
        )
    ]

    evidence = validator.validate(query, results, doc_id="test_doc")

    assert evidence.status == EvidenceStatus.MARGINAL
    assert evidence.confidence == 0.52
    assert "c1" in evidence.supporting_chunk_ids


def test_validate_marginal_partial_subject_state():
    """Verify MARGINAL when query asks for a specific condition (e.g. cataract) but evidence only discusses general terms."""
    validator = EvidenceValidator()
    query = "What is the waiting period for cataract surgery?"
    results = [
        _make_result(
            "c1",
            "General waiting period for specified procedures is twenty-four months.",
            final_score=0.72,  # high score but lacks 'cataract'
        )
    ]

    evidence = validator.validate(query, results, doc_id="test_doc")

    assert evidence.status == EvidenceStatus.MARGINAL
    assert "cataract" not in results[0].text.lower()
    assert "incomplete" in evidence.validator_notes.lower() or "does not clearly identify" in evidence.validator_notes.lower()


def test_validate_insufficient_below_threshold_or_empty():
    """Verify INSUFFICIENT when no results or all results below LOW_THRESHOLD (0.40)."""
    validator = EvidenceValidator()
    query = "Does the policy cover maternity expenses?"

    # Case A: Empty results
    ev_empty = validator.validate(query, [], doc_id="test_doc")
    assert ev_empty.status == EvidenceStatus.INSUFFICIENT
    assert ev_empty.confidence == 0.0
    assert ev_empty.supporting_chunk_ids == []

    # Case B: Below-threshold score
    below_thresh_results = [
        _make_result(
            "c1",
            "Room rent limit is 1% of Sum Insured per day.",
            final_score=0.28,
            passed_threshold=False,
        )
    ]
    ev_low = validator.validate(query, below_thresh_results, doc_id="test_doc")
    assert ev_low.status == EvidenceStatus.INSUFFICIENT
    assert ev_low.supporting_chunk_ids == []


def test_insufficient_does_not_convert_to_explicitly_absent():
    """
    CRITICAL ARCHITECTURAL INVARIANT:
    The validator must never report 'the policy explicitly lacks or excludes this coverage'
    when evidence is INSUFFICIENT. It must only report insufficient retrieved evidence.
    """
    validator = EvidenceValidator()
    query = "Is robotic surgery covered?"
    ev = validator.validate(query, [], doc_id="test_doc")

    assert ev.status == EvidenceStatus.INSUFFICIENT
    notes = ev.validator_notes.lower()
    assert "explicitly absent" not in notes
    assert "does not contain" in notes or "insufficient" in notes


def test_validate_conflicting_numeric_statements():
    """Verify CONFLICTING when two candidate chunks state different numeric values for the same attribute."""
    validator = EvidenceValidator()
    query = "What is the co-payment percentage?"
    results = [
        _make_result(
            "c_copay_10",
            "Co-payment is 10% of admissible claim amount for all hospitalizations.",
            final_score=0.75,
            page_number=1,
        ),
        _make_result(
            "c_copay_20",
            "Co-payment is 20% of admissible claim amount for all hospitalizations.",
            final_score=0.74,
            page_number=3,
        ),
    ]

    evidence = validator.validate(query, results, doc_id="test_doc")

    assert evidence.status == EvidenceStatus.CONFLICTING
    assert len(evidence.detected_conflicts) >= 1
    assert "c_copay_10" in evidence.supporting_chunk_ids
    assert "c_copay_20" in evidence.supporting_chunk_ids
    assert "Conflicting percentage" in evidence.detected_conflicts[0]


def test_validate_conflicting_coverage_statements():
    """Verify CONFLICTING when one chunk states a procedure is covered while another states excluded."""
    validator = EvidenceValidator()
    query = "Does the policy cover ayurvedic treatment?"
    results = [
        _make_result(
            "c_ayurveda_incl",
            "Ayurvedic treatment is covered up to the full sum insured in recognized AYUSH hospitals.",
            final_score=0.72,
            page_number=2,
        ),
        _make_result(
            "c_ayurveda_excl",
            "Ayurvedic treatment is excluded from coverage under standard exclusions.",
            final_score=0.70,
            page_number=4,
        ),
    ]

    evidence = validator.validate(query, results, doc_id="test_doc")

    assert evidence.status == EvidenceStatus.CONFLICTING
    assert len(evidence.detected_conflicts) >= 1
    assert "ayurvedic" in evidence.detected_conflicts[0].lower()
