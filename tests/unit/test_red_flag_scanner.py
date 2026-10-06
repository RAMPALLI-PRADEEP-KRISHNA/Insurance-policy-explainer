"""Unit tests for the evidence-grounded Red-Flag Scanner."""

import uuid
import pytest

from src.models.data_models import (
    Chunk,
    ChunkType,
    Document,
    DocumentQuality,
    DocumentStatus,
    RiskCategory,
    RiskSeverity,
)
from src.risk_scanner.red_flag_scanner import (
    RedFlagScanner,
    red_flag_scanner,
    scan_policy,
)


def _make_dummy_doc(doc_id: str = "doc-123", filename: str = "sample_test.pdf") -> Document:
    return Document(
        doc_id=doc_id,
        filename=filename,
        file_hash="abcdef123456",
        page_count=2,
        status=DocumentStatus.READY,
        quality=DocumentQuality.GOOD,
    )


def _make_chunk(
    doc_id: str,
    text: str,
    page_number: int = 1,
    chunk_type: ChunkType = ChunkType.CLAUSE,
    section: str = "Coverage Terms",
    subsection: str = "Policy Clauses",
    filename: str = "sample_test.pdf",
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
        subsection=subsection,
    )


def test_scanner_waiting_period_severities():
    """Verify waiting period detection and severity graduation (HIGH, MEDIUM, LOW)."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Pre-existing disease waiting period: 36 months of continuous coverage.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "Specific disease waiting period: 24 months for specified procedures like cataract.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "Initial waiting period: 30 days from inception for all illnesses except accidents.",
            page_number=2,
        ),
    ]

    report = red_flag_scanner.scan(doc, chunks)
    wp_flags = [f for f in report.findings if f.category == RiskCategory.WAITING_PERIOD]

    assert len(wp_flags) == 3
    # Check severity assignments
    ped_flag = next(f for f in wp_flags if "Pre-Existing" in f.title or "36" in f.source_snippet)
    assert ped_flag.severity == RiskSeverity.HIGH
    assert ped_flag.page_number == 1

    spec_flag = next(f for f in wp_flags if "Specific" in f.title or "24" in f.source_snippet)
    assert spec_flag.severity == RiskSeverity.MEDIUM
    assert spec_flag.page_number == 1

    init_flag = next(f for f in wp_flags if "Initial" in f.title or "30" in f.source_snippet)
    assert init_flag.severity == RiskSeverity.LOW
    assert init_flag.page_number == 2


def test_scanner_permanent_exclusions_high_severity():
    """Verify permanent exclusion detection receives HIGH review priority."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Permanent exclusions: The company shall not be liable for hazardous sports, cosmetic surgery, and self-inflicted injury.",
            page_number=2,
        )
    ]

    report = red_flag_scanner.scan(doc, chunks)
    excl_flags = [f for f in report.findings if f.category == RiskCategory.PERMANENT_EXCLUSION]

    assert len(excl_flags) == 1
    assert excl_flags[0].severity == RiskSeverity.HIGH
    assert "Permanent Exclusions" in excl_flags[0].title
    assert excl_flags[0].page_number == 2


def test_scanner_copayment_and_negative_handling():
    """Verify co-payment percentage detection and negation filtering."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "A mandatory 20% co-payment applies to all claims across network hospitals.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "10% co-pay for age 61 and above.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "No co-payment is applicable for Tier 1 cities under plan Gold.",
            page_number=2,
        ),
    ]

    report = red_flag_scanner.scan(doc, chunks)
    copay_flags = [f for f in report.findings if f.category == RiskCategory.CO_PAYMENT]

    # The negative "No co-payment" should be filtered out
    assert len(copay_flags) == 2

    high_copay = next(f for f in copay_flags if "20" in f.source_snippet)
    assert high_copay.severity == RiskSeverity.HIGH

    med_copay = next(f for f in copay_flags if "10" in f.source_snippet)
    assert med_copay.severity == RiskSeverity.MEDIUM
    assert "61" in med_copay.description


def test_scanner_room_rent_limit_high_severity():
    """Verify room rent limits receive HIGH severity due to proportionate deduction."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Room rent is limited up to 1% of Sum Insured per day. ICU charges up to 2%.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "No room rent cap applies to deluxe room category.",
            page_number=2,
        ),
    ]

    report = red_flag_scanner.scan(doc, chunks)
    rr_flags = [f for f in report.findings if f.category == RiskCategory.ROOM_RENT_LIMIT]

    assert len(rr_flags) == 1
    assert rr_flags[0].severity == RiskSeverity.HIGH
    assert "proportionate" in rr_flags[0].description.lower()


def test_scanner_sublimits_and_deductibles():
    """Verify sub-limits and deductibles are captured with exact citations."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Cataract surgery is capped at a sub-limit of Rs. 40,000 per eye.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "An aggregate deductible of Rs. 25,000 applies per policy year.",
            page_number=2,
        ),
    ]

    report = red_flag_scanner.scan(doc, chunks)
    assert any(f.category == RiskCategory.SUB_LIMIT for f in report.findings)
    assert any(f.category == RiskCategory.DEDUCTIBLE for f in report.findings)


def test_scanner_claim_conditions_and_restoration():
    """Verify procedural claim conditions and restoration limits."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Claim intimation must be submitted within 24 hours of hospitalization.",
            page_number=1,
        ),
        _make_chunk(
            doc.doc_id,
            "Restoration of sum insured applies once in a policy year for unrelated illness only.",
            page_number=2,
        ),
    ]

    report = red_flag_scanner.scan(doc, chunks)
    assert any(f.category == RiskCategory.CLAIM_CONDITION for f in report.findings)
    assert any(f.category == RiskCategory.RESTORATION_RESTRICTION for f in report.findings)


def test_scanner_table_chunk_parsing():
    """Verify tabular chunks are parsed row-by-row for granular findings."""
    doc = _make_dummy_doc()
    table_text = (
        "| Benefit Category | Policy Limit / Co-Pay |\n"
        "| --- | --- |\n"
        "| Room Rent | Up to 1% of Sum Insured per day |\n"
        "| Co-payment | 10% co-pay for age 61 and above |"
    )
    chunks = [
        _make_chunk(
            doc.doc_id,
            table_text,
            page_number=1,
            chunk_type=ChunkType.TABLE,
        )
    ]

    report = red_flag_scanner.scan(doc, chunks)
    categories = {f.category for f in report.findings}

    assert RiskCategory.ROOM_RENT_LIMIT in categories
    assert RiskCategory.CO_PAYMENT in categories


def test_scanner_citation_metadata_integrity():
    """Verify citations in RedFlag objects are populated strictly from chunk metadata."""
    doc = _make_dummy_doc()
    c = _make_chunk(
        doc.doc_id,
        "Entry Age: 18 years to 65 years. Lifelong renewability.",
        page_number=1,
        section="Section 1: Eligibility",
        subsection="Entry Limits",
    )
    report = red_flag_scanner.scan(doc, [c])

    assert len(report.findings) >= 1
    flag = report.findings[0]
    assert flag.document_id == doc.doc_id
    assert flag.chunk_id == c.chunk_id
    assert flag.page_number == 1
    assert flag.section == "Section 1: Eligibility"
    assert len(flag.citations) == 1

    cit = flag.citations[0]
    assert cit.document_id == doc.doc_id
    assert cit.chunk_id == c.chunk_id
    assert cit.page_number == 1
    assert cit.section == "Section 1: Eligibility"
    assert "18 years to 65 years" in cit.source_snippet


def test_scanner_scan_risk_findings_data_model_conformance():
    """Verify scan_risk_findings returns all 12 categories with found=True/False flags."""
    doc = _make_dummy_doc()
    chunks = [
        _make_chunk(
            doc.doc_id,
            "Permanent exclusions: Cosmetic surgery, self-inflicted injury.",
            page_number=1,
        )
    ]

    findings = red_flag_scanner.scan_risk_findings(doc, chunks)
    assert len(findings) == 12

    # PERMANENT_EXCLUSION must be found
    perm = next(f for f in findings if f.category == RiskCategory.PERMANENT_EXCLUSION)
    assert perm.found is True
    assert perm.severity == RiskSeverity.HIGH
    assert len(perm.source_chunks) == 1
    assert len(perm.citations) == 1

    # DEDUCTIBLE was not in chunks, must be found=False
    ded = next(f for f in findings if f.category == RiskCategory.DEDUCTIBLE)
    assert ded.found is False
    assert ded.severity is None
    assert len(ded.source_chunks) == 0
    assert len(ded.citations) == 0
    assert "no explicit deductible" in ded.summary.lower()


def test_scanner_empty_chunks():
    """Verify scanning with empty chunks returns an empty report safely."""
    doc = _make_dummy_doc()
    report = red_flag_scanner.scan(doc, chunks=[])
    assert report.total_flags == 0
    assert len(report.findings) == 0
    assert len(report.categories_found) == 0
