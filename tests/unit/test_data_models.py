"""Unit tests verifying canonical data models and structural invariants."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.models.data_models import (
    Document,
    Page,
    Chunk,
    TableData,
    DocumentStatus,
    DocumentQuality,
    ExtractionMethod,
    ChunkType,
)


def test_page_indexing_invariants():
    """Verify that page_number must be 1-indexed and equal page_index + 1."""
    # Valid: page 1, index 0
    p1 = Page(
        doc_id="doc-123",
        page_number=1,
        page_index=0,
        text="Sample policy clause.",
    )
    assert p1.page_number == 1
    assert p1.page_index == 0
    assert p1.char_count == len("Sample policy clause.")

    # Invalid: page_number != page_index + 1
    with pytest.raises(ValidationError) as exc:
        Page(
            doc_id="doc-123",
            page_number=2,
            page_index=0,
            text="Mismatch.",
        )
    assert "page_number" in str(exc.value)

    # Invalid: 0-indexed page number
    with pytest.raises(ValidationError):
        Page(
            doc_id="doc-123",
            page_number=0,
            page_index=0,
            text="Invalid zero page.",
        )


def test_chunk_invariants():
    """Verify that chunk metadata preserves doc_id, 1-indexed page_number, and page_numbers."""
    c = Chunk(
        chunk_id="chunk-abc",
        doc_id="doc-123",
        filename="policy.pdf",
        page_number=3,
        page_numbers=[3, 4],
        chunk_type=ChunkType.PARAGRAPH,
        text="Waiting period shall be 30 days from inception.",
        chunk_index=0,
    )
    assert c.page_number == 3
    assert 3 in c.page_numbers
    assert c.char_count == len("Waiting period shall be 30 days from inception.")

    # Automatically includes page_number in page_numbers if missing
    c2 = Chunk(
        chunk_id="chunk-def",
        doc_id="doc-123",
        filename="policy.pdf",
        page_number=5,
        page_numbers=[6],
        chunk_type=ChunkType.SECTION_HEADING,
        text="Section 5: Exclusions",
        chunk_index=1,
    )
    assert 5 in c2.page_numbers


def test_document_validation_and_ocr_page_ids():
    """Verify Document invariants and 1-indexed ocr_page_ids."""
    doc = Document(
        doc_id="doc-uuid-123",
        filename="policy.pdf",
        file_hash="abcdef1234567890",
        page_count=5,
        ocr_page_ids=[1, 4],
        status=DocumentStatus.READY,
        quality=DocumentQuality.GOOD,
    )
    assert doc.page_count == 5
    assert doc.ocr_page_ids == [1, 4]

    # Verify that ocr_page_ids < 1 is rejected
    with pytest.raises(ValidationError):
        Document(
            doc_id="doc-uuid-123",
            filename="policy.pdf",
            file_hash="abcdef1234567890",
            page_count=5,
            ocr_page_ids=[0, 2],  # 0 is invalid
            status=DocumentStatus.READY,
            quality=DocumentQuality.GOOD,
        )


def test_table_data_model():
    """Verify TableData structure and validation."""
    t = TableData(
        page_number=2,
        table_index=0,
        row_count=2,
        col_count=2,
        headers=["Coverage", "Limit"],
        rows=[["Cataract", "Rs. 40,000"], ["Hernia", "Rs. 50,000"]],
        markdown="| Coverage | Limit |\n|---|---|\n| Cataract | Rs. 40,000 |",
    )
    assert t.page_number == 2
    assert t.row_count == 2
    assert len(t.headers) == 2


def test_risk_scanner_data_models():
    """Verify RiskCategory, RiskSeverity, RedFlag, and RedFlagReport data models."""
    from src.models.data_models import (
        RiskCategory,
        RiskSeverity,
        RedFlag,
        RiskFinding,
        RedFlagReport,
        Citation,
    )

    cit = Citation(
        document_id="doc-1",
        chunk_id="chunk-1",
        filename="test.pdf",
        page_number=1,
        page_numbers=[1],
        source_snippet="Sample snippet",
    )

    flag = RedFlag(
        category=RiskCategory.WAITING_PERIOD,
        title="36-Month PED Waiting Period",
        description="Pre-existing diseases require 36 months of waiting.",
        severity=RiskSeverity.HIGH,
        document_id="doc-1",
        chunk_id="chunk-1",
        page_number=1,
        source_snippet="Pre-existing diseases waiting period: 36 months.",
        citations=[cit],
    )
    assert flag.severity == RiskSeverity.HIGH
    assert flag.page_number == 1

    report = RedFlagReport(
        document_id="doc-1",
        filename="test.pdf",
        total_flags=1,
        findings=[flag],
        categories_found=[RiskCategory.WAITING_PERIOD],
    )
    assert report.total_flags == 1
    assert "disclaimer" in report.model_dump()
    assert len(report.disclaimer) > 10

    finding = RiskFinding(
        doc_id="doc-1",
        category=RiskCategory.WAITING_PERIOD,
        found=True,
        summary="36-Month PED Waiting Period",
        citations=[cit],
        severity=RiskSeverity.HIGH,
    )
    assert finding.found is True
    assert finding.severity == RiskSeverity.HIGH


def test_comparison_data_models():
    """Verify ComparisonCategory, ComparisonCellStatus, ComparisonFact, ComparisonRow, and ComparisonReport."""
    from src.models.data_models import (
        ComparisonCategory,
        ComparisonCellStatus,
        ComparisonFact,
        ComparisonRow,
        ComparisonReport,
        Citation,
    )

    cit = Citation(
        document_id="doc-a",
        chunk_id="chunk-a",
        filename="policy_a.pdf",
        page_number=1,
        page_numbers=[1],
        source_snippet="Waiting period 36 months",
    )

    fact_a = ComparisonFact(
        policy_id="doc-a",
        filename="policy_a.pdf",
        category=ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
        status=ComparisonCellStatus.FOUND,
        value=36.0,
        unit="months",
        original_value="36 months",
        description="Pre-existing diseases waiting period: 36 months",
        citations=[cit],
    )
    assert fact_a.status == ComparisonCellStatus.FOUND
    assert fact_a.value == 36.0

    fact_b = ComparisonFact(
        policy_id="doc-b",
        filename="policy_b.pdf",
        category=ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
        status=ComparisonCellStatus.NOT_MENTIONED,
        description="No clause identified",
    )
    assert fact_b.status == ComparisonCellStatus.NOT_MENTIONED
    assert fact_b.status != ComparisonCellStatus.EXPLICITLY_ABSENT

    row = ComparisonRow(
        category=ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
        label="Pre-Existing Disease Waiting Period",
        policy_a_fact=fact_a,
        policy_b_fact=fact_b,
        difference="Policy A specifies 36 months; no corresponding clause identified in Policy B.",
    )
    assert row.category == ComparisonCategory.WAITING_PERIOD_PRE_EXISTING

    report = ComparisonReport(
        doc_id_a="doc-a",
        doc_id_b="doc-b",
        filename_a="policy_a.pdf",
        filename_b="policy_b.pdf",
        rows=[row],
    )
    assert report.doc_id_a == "doc-a"
    assert report.doc_id_b == "doc-b"
    assert len(report.rows) == 1
    assert "disclaimer" in report.model_dump()
