"""Integration tests for the Red-Flag Scanner on real fixture PDFs."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.app import scan_policy_red_flags, render_red_flags
from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.models.data_models import RiskCategory, RiskSeverity, DocumentQuality, DocumentStatus
from src.risk_scanner.red_flag_scanner import red_flag_scanner, RedFlagScanner


@pytest.fixture(scope="module")
def sample_policy_doc_and_chunks():
    """Ingest and chunk the real synthetic fixture PDF."""
    pdf_path = Path("tests/fixtures/sample_policy_fixture.pdf")
    assert pdf_path.is_file(), "sample_policy_fixture.pdf missing"

    doc = process_pdf(pdf_path, save_meta=False)
    assert doc.status == DocumentStatus.READY
    assert doc.quality == DocumentQuality.GOOD

    chunks = chunk_document(doc)
    assert len(chunks) > 0
    return doc, chunks


def test_red_flag_pipeline_on_sample_fixture(sample_policy_doc_and_chunks):
    """
    End-to-end integration test verifying that the scanner detects all expected
    clauses from the synthetic insurance policy fixture:
    - 36-month PED waiting period (HIGH)
    - 24-month specific illness waiting period (MEDIUM)
    - 30-day initial waiting period (LOW)
    - Permanent exclusions (HIGH)
    - 10% co-payment (MEDIUM)
    - 1% room rent limit (HIGH)
    - Entry age 18-65 (MEDIUM)
    """
    doc, chunks = sample_policy_doc_and_chunks
    report = red_flag_scanner.scan(doc, chunks)

    assert report.document_id == doc.doc_id
    assert report.filename == doc.filename
    assert report.total_flags >= 6

    # Verify categories detected
    detected_cats = {f.category for f in report.findings}
    assert RiskCategory.WAITING_PERIOD in detected_cats
    assert RiskCategory.PERMANENT_EXCLUSION in detected_cats
    assert RiskCategory.ROOM_RENT_LIMIT in detected_cats
    assert RiskCategory.CO_PAYMENT in detected_cats
    assert RiskCategory.AGE_RESTRICTION in detected_cats

    # 1. 36-month PED
    ped_flag = next(f for f in report.findings if f.category == RiskCategory.WAITING_PERIOD and f.severity == RiskSeverity.HIGH)
    assert "36" in ped_flag.source_snippet
    assert ped_flag.page_number == 2

    # 2. 24-month specific disease
    spec_flag = next(f for f in report.findings if f.category == RiskCategory.WAITING_PERIOD and "24 months" in f.title)
    assert "24 months" in spec_flag.source_snippet
    assert spec_flag.page_number == 2
    assert spec_flag.severity == RiskSeverity.MEDIUM

    # 3. 30-day initial waiting period
    init_flag = next(f for f in report.findings if f.category == RiskCategory.WAITING_PERIOD and f.severity == RiskSeverity.LOW)
    assert "30 days" in init_flag.source_snippet
    assert init_flag.page_number == 2

    # 4. Permanent exclusions
    perm_flag = next(f for f in report.findings if f.category == RiskCategory.PERMANENT_EXCLUSION)
    assert perm_flag.severity == RiskSeverity.HIGH
    assert "Hazardous sports" in perm_flag.source_snippet
    assert perm_flag.page_number == 2

    # 5. Room rent limit
    rr_flag = next(f for f in report.findings if f.category == RiskCategory.ROOM_RENT_LIMIT)
    assert rr_flag.severity == RiskSeverity.HIGH
    assert "1%" in rr_flag.source_snippet
    assert rr_flag.page_number == 1

    # 6. Co-payment
    copay_flag = next(f for f in report.findings if f.category == RiskCategory.CO_PAYMENT)
    assert copay_flag.severity == RiskSeverity.MEDIUM
    assert "10%" in copay_flag.source_snippet
    assert copay_flag.page_number == 1


def test_app_service_scan_policy_red_flags(sample_policy_doc_and_chunks):
    """Verify the decoupled service function in src.app works identically."""
    doc, chunks = sample_policy_doc_and_chunks
    report = scan_policy_red_flags(document=doc, chunks=chunks)

    assert report.total_flags >= 6
    assert report.disclaimer is not None
    assert len(report.categories_found) >= 4


def test_render_red_flags_streamlit_mock(sample_policy_doc_and_chunks):
    """Verify render_red_flags executes cleanly without UI exceptions."""
    doc, chunks = sample_policy_doc_and_chunks
    report = scan_policy_red_flags(document=doc, chunks=chunks)

    col_mocks = [MagicMock() for _ in range(4)]
    with patch("streamlit.columns", return_value=col_mocks) as mock_cols, \
         patch("streamlit.selectbox", return_value="All Severities") as mock_select, \
         patch("streamlit.markdown") as mock_md, \
         patch("streamlit.write") as mock_write, \
         patch("streamlit.container") as mock_container, \
         patch("streamlit.expander") as mock_expander, \
         patch("streamlit.caption") as mock_caption:

        mock_cols.side_effect = lambda spec: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]
        mock_container.return_value.__enter__ = MagicMock()
        mock_container.return_value.__exit__ = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_red_flags(report)
        assert mock_md.call_count >= 5
        assert mock_caption.call_count >= 1


