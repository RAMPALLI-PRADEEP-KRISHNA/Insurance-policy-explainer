"""Unit tests for the two-policy comparison engine, normalizer, and evidence extractor."""

import uuid
import pytest

from src.comparison.comparison_engine import (
    ComparisonEngine,
    comparison_engine,
    compare_policies,
)
from src.comparison.evidence_extractor import (
    ComparisonEvidenceExtractor,
    comparison_evidence_extractor,
)
from src.comparison.normalizer import (
    ComparisonNormalizer,
    comparison_normalizer,
    generate_difference_description,
)
from src.models.data_models import (
    Chunk,
    ChunkType,
    Citation,
    ComparisonCategory,
    ComparisonCellStatus,
    ComparisonFact,
    Document,
    DocumentQuality,
    DocumentStatus,
)


def _make_dummy_doc(doc_id: str = "doc-1", filename: str = "policy1.pdf") -> Document:
    return Document(
        doc_id=doc_id,
        filename=filename,
        file_hash="hash-" + doc_id,
        page_count=2,
        status=DocumentStatus.READY,
        quality=DocumentQuality.GOOD,
    )


def _make_chunk(
    doc_id: str,
    text: str,
    page_number: int = 1,
    chunk_type: ChunkType = ChunkType.CLAUSE,
    section: str = "Benefits",
    filename: str = "test.pdf",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        page_numbers=[page_number],
        chunk_type=chunk_type,
        char_count=len(text),
        token_count=len(text.split()),
        section=section,
    )


# ---------------------------------------------------------------------------
# Test 1 — Numeric comparison (36 months vs 24 months)
# ---------------------------------------------------------------------------
def test_1_numeric_duration_comparison():
    """Verify numeric duration normalization and objective difference description."""
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    chunks_a = [_make_chunk(doc_a.doc_id, "Pre-existing diseases waiting period: 36 months of continuous coverage.")]
    chunks_b = [_make_chunk(doc_b.doc_id, "Pre-existing disease waiting period: 24 months of continuous coverage.")]

    fact_a = comparison_evidence_extractor.extract_fact(doc_a, ComparisonCategory.WAITING_PERIOD_PRE_EXISTING, chunks_a)
    fact_b = comparison_evidence_extractor.extract_fact(doc_b, ComparisonCategory.WAITING_PERIOD_PRE_EXISTING, chunks_b)

    assert fact_a.status == ComparisonCellStatus.FOUND
    assert fact_a.value == 36.0
    assert fact_a.unit == "months"

    assert fact_b.status == ComparisonCellStatus.FOUND
    assert fact_b.value == 24.0
    assert fact_b.unit == "months"

    diff = generate_difference_description(fact_a, fact_b)
    assert "24" in diff and "36" in diff
    # Must be objective and never state winner/better
    assert "better" not in diff.lower()
    assert "winner" not in diff.lower()


# ---------------------------------------------------------------------------
# Test 2 — Different units/basis (1%/day vs Rs. 5,000/day)
# ---------------------------------------------------------------------------
def test_2_different_units_and_basis_preserved():
    """Verify that different units (percent vs INR) are not blindly compared as raw numbers."""
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    chunks_a = [_make_chunk(doc_a.doc_id, "Room Rent: Up to 1% of Sum Insured per day.")]
    chunks_b = [_make_chunk(doc_b.doc_id, "Room Rent: Rs. 5,000 per day.")]

    fact_a = comparison_evidence_extractor.extract_fact(doc_a, ComparisonCategory.ROOM_RENT_LIMIT, chunks_a)
    fact_b = comparison_evidence_extractor.extract_fact(doc_b, ComparisonCategory.ROOM_RENT_LIMIT, chunks_b)

    assert fact_a.status == ComparisonCellStatus.FOUND
    assert fact_a.unit == "percent"
    assert fact_a.value == 1.0

    assert fact_b.status == ComparisonCellStatus.FOUND
    assert fact_b.unit == "INR"
    assert fact_b.value == 5000.0

    diff = generate_difference_description(fact_a, fact_b)
    assert "different limit basis" in diff.lower()
    assert "1%" in diff
    assert "Rs. 5,000" in diff


# ---------------------------------------------------------------------------
# Test 3 — Conditional co-pay (10% age 61+ vs 20%)
# ---------------------------------------------------------------------------
def test_3_conditional_copay_preserved():
    """Verify that conditional qualifiers (age 61+) are retained in facts and differences."""
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    chunks_a = [_make_chunk(doc_a.doc_id, "Co-payment: 10% co-pay for age 61 and above.")]
    chunks_b = [_make_chunk(doc_b.doc_id, "Co-payment: 20% co-pay applicable on all admissible claims.")]

    fact_a = comparison_evidence_extractor.extract_fact(doc_a, ComparisonCategory.CO_PAYMENT, chunks_a)
    fact_b = comparison_evidence_extractor.extract_fact(doc_b, ComparisonCategory.CO_PAYMENT, chunks_b)

    assert fact_a.status == ComparisonCellStatus.FOUND
    assert fact_a.value == 10.0
    assert "61" in (fact_a.conditions or "")

    assert fact_b.status == ComparisonCellStatus.FOUND
    assert fact_b.value == 20.0

    diff = generate_difference_description(fact_a, fact_b)
    assert "conditions differ" in diff.lower()
    assert "61" in diff


# ---------------------------------------------------------------------------
# Test 4 — NOT_MENTIONED != EXPLICITLY_ABSENT (Critical Invariant)
# ---------------------------------------------------------------------------
def test_4_not_mentioned_never_converted_to_explicitly_absent():
    """
    CRITICAL INVARIANT TEST:
    Verify that absence of evidence yields NOT_MENTIONED and is NEVER converted to EXPLICITLY_ABSENT.
    """
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    # Policy A has no deductible statement at all
    chunks_a = [_make_chunk(doc_a.doc_id, "Hospitalization benefits up to Rs. 10,00,000.")]
    # Policy B explicitly has Rs. 10,000 deductible
    chunks_b = [_make_chunk(doc_b.doc_id, "Deductible: A compulsory deductible of Rs. 10,000 applies per policy year.")]

    fact_a = comparison_evidence_extractor.extract_fact(doc_a, ComparisonCategory.DEDUCTIBLE, chunks_a)
    fact_b = comparison_evidence_extractor.extract_fact(doc_b, ComparisonCategory.DEDUCTIBLE, chunks_b)

    assert fact_a.status == ComparisonCellStatus.NOT_MENTIONED
    assert fact_a.status != ComparisonCellStatus.EXPLICITLY_ABSENT
    assert fact_b.status == ComparisonCellStatus.FOUND
    assert fact_b.value == 10000.0

    diff = generate_difference_description(fact_a, fact_b)
    assert "no corresponding deductible clause was identified in policy a" in diff.lower()
    # Must NOT state that Policy A has no deductible
    assert "policy a has no deductible" not in diff.lower()


# ---------------------------------------------------------------------------
# Test 5 — Explicit absence (EXPLICITLY_ABSENT)
# ---------------------------------------------------------------------------
def test_5_explicit_absence_detected():
    """Verify that explicit negative statements yield EXPLICITLY_ABSENT with citations."""
    doc = _make_dummy_doc("doc-absent", "policy_absent.pdf")
    chunks = [
        _make_chunk(doc.doc_id, "There is no co-payment applicable under this policy."),
        _make_chunk(doc.doc_id, "Nil deductible applies to all claims."),
        _make_chunk(doc.doc_id, "No room rent cap is imposed on any room category."),
    ]

    fact_copay = comparison_evidence_extractor.extract_fact(doc, ComparisonCategory.CO_PAYMENT, chunks)
    assert fact_copay.status == ComparisonCellStatus.EXPLICITLY_ABSENT
    assert len(fact_copay.citations) == 1

    fact_ded = comparison_evidence_extractor.extract_fact(doc, ComparisonCategory.DEDUCTIBLE, chunks)
    assert fact_ded.status == ComparisonCellStatus.EXPLICITLY_ABSENT
    assert len(fact_ded.citations) == 1

    fact_rr = comparison_evidence_extractor.extract_fact(doc, ComparisonCategory.ROOM_RENT_LIMIT, chunks)
    assert fact_rr.status == ComparisonCellStatus.EXPLICITLY_ABSENT
    assert len(fact_rr.citations) == 1


# ---------------------------------------------------------------------------
# Test 6 — Ambiguous evidence (AMBIGUOUS)
# ---------------------------------------------------------------------------
def test_6_ambiguous_evidence_conflicting_clauses():
    """Verify that contradictory non-conditional clauses trigger AMBIGUOUS status."""
    doc = _make_dummy_doc("doc-conflict", "policy_conflict.pdf")
    chunks = [
        _make_chunk(doc.doc_id, "Co-payment is 10% on all network claims."),
        _make_chunk(doc.doc_id, "Co-payment is 20% on all network claims."),
    ]

    fact = comparison_evidence_extractor.extract_fact(doc, ComparisonCategory.CO_PAYMENT, chunks)
    assert fact.status == ComparisonCellStatus.AMBIGUOUS
    assert "contradictory" in fact.description.lower()
    assert len(fact.citations) >= 2


# ---------------------------------------------------------------------------
# Test 7 — Exclusion vs Coverage
# ---------------------------------------------------------------------------
def test_7_exclusion_vs_coverage():
    """Verify comparison between permanently excluded clause and conditionally covered clause."""
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    chunks_a = [_make_chunk(doc_a.doc_id, "Permanent exclusions: Aesthetic surgery is excluded permanently.")]
    chunks_b = [_make_chunk(doc_b.doc_id, "Aesthetic surgery is covered subject to policy conditions following injury.")]

    fact_a = comparison_evidence_extractor.extract_fact(doc_a, ComparisonCategory.PERMANENT_EXCLUSIONS, chunks_a)
    fact_b = comparison_evidence_extractor.extract_fact(doc_b, ComparisonCategory.PERMANENT_EXCLUSIONS, chunks_b)

    assert fact_a.status == ComparisonCellStatus.FOUND
    assert fact_b.status == ComparisonCellStatus.EXPLICITLY_ABSENT

    diff = generate_difference_description(fact_a, fact_b)
    assert "permanently excludes aesthetic surgery" in diff.lower()
    assert "covers aesthetic surgery" in diff.lower()


# ---------------------------------------------------------------------------
# Test 8 — No cross-policy leakage & Document isolation
# ---------------------------------------------------------------------------
def test_8_no_cross_policy_leakage():
    """Verify strict document isolation: facts and citations never mix between Policy A and B."""
    doc_a = _make_dummy_doc("doc-a", "policy_a.pdf")
    doc_b = _make_dummy_doc("doc-b", "policy_b.pdf")

    chunks_a = [_make_chunk(doc_a.doc_id, "Pre-existing diseases waiting period: 36 months.", filename="policy_a.pdf")]
    chunks_b = [_make_chunk(doc_b.doc_id, "Pre-existing diseases waiting period: 24 months.", filename="policy_b.pdf")]

    report = compare_policies(doc_a, doc_b, chunks_a=chunks_a, chunks_b=chunks_b)

    for row in report.rows:
        # Policy A fact check
        assert row.policy_a_fact.policy_id == doc_a.doc_id
        for cit in row.policy_a_fact.citations:
            assert cit.document_id == doc_a.doc_id
            assert cit.filename == doc_a.filename

        # Policy B fact check
        assert row.policy_b_fact.policy_id == doc_b.doc_id
        for cit in row.policy_b_fact.citations:
            assert cit.document_id == doc_b.doc_id
            assert cit.filename == doc_b.filename


# ---------------------------------------------------------------------------
# Test 9 — Self-comparison rejected
# ---------------------------------------------------------------------------
def test_9_self_comparison_rejected():
    """Verify comparing a document with itself is rejected with ValueError."""
    doc = _make_dummy_doc("doc-same", "same.pdf")
    with pytest.raises(ValueError, match="Cannot compare document.*with itself"):
        compare_policies(doc, doc)
