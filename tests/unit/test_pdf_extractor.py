"""Unit tests for PDFExtractor and process_pdf function."""

import pytest
from pathlib import Path
from unittest.mock import patch

from backend.ingestion.pdf_extractor import PDFExtractor, process_pdf
from backend.models.data_models import DocumentStatus, DocumentQuality, ExtractionMethod
from backend.utils.exceptions import (
    InvalidPDFError,
    PDFValidationError,
    PageLimitExceededError,
)


def test_valid_text_pdf(make_text_pdf):
    """Test extracting a standard single-page text PDF."""
    sample_text = (
        "POLICY CLAUSE 1: HOSPITALIZATION BENEFIT\n"
        "The company agrees to pay reasonable and customary medical expenses incurred during hospitalization."
    )
    pdf_bytes = make_text_pdf([sample_text])

    extractor = PDFExtractor()
    doc = extractor.extract(pdf_bytes, filename="sample_policy.pdf", save_meta=False)

    assert doc.doc_id is not None
    assert doc.filename == "sample_policy.pdf"
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert doc.status == DocumentStatus.READY
    assert doc.quality in [DocumentQuality.GOOD, DocumentQuality.WARNING]

    page = doc.pages[0]
    assert page.page_number == 1
    assert page.page_index == 0
    assert "HOSPITALIZATION BENEFIT" in page.text
    assert page.extraction_method == ExtractionMethod.NATIVE


def test_multi_page_pdf_association_and_1_indexed_numbers(make_text_pdf):
    """Test multi-page PDF preserves page text association and 1-indexed page numbers."""
    page_texts = [
        "Page One: Definitions and Scope of Coverage.",
        "Page Two: Waiting Periods and Exclusions. 30 days initial waiting period applies.",
        "Page Three: Claims Procedure and Required Documents for cashless hospitalization.",
    ]
    pdf_bytes = make_text_pdf(page_texts)

    doc = process_pdf(pdf_bytes, filename="multi_policy.pdf", save_meta=False)

    assert doc.page_count == 3
    assert len(doc.pages) == 3

    for idx, page in enumerate(doc.pages):
        expected_page_num = idx + 1
        assert page.page_number == expected_page_num
        assert page.page_index == idx
        assert page.doc_id == doc.doc_id

    assert "Page One" in doc.pages[0].text
    assert "Waiting Periods" in doc.pages[1].text
    assert "Claims Procedure" in doc.pages[2].text


def test_empty_or_zero_byte_pdf():
    """Test handling of 0-byte or empty inputs."""
    extractor = PDFExtractor()

    # Zero bytes
    with pytest.raises(InvalidPDFError):
        extractor.extract(b"", filename="empty.pdf", save_meta=False)


def test_non_pdf_input():
    """Test handling of non-PDF bytes (missing %PDF- header)."""
    extractor = PDFExtractor()
    non_pdf_data = b"This is just a plain text string, not a PDF document at all."

    with pytest.raises(InvalidPDFError) as exc:
        extractor.extract(non_pdf_data, filename="fake.pdf", save_meta=False)
    assert "missing '%PDF-' header" in str(exc.value)


def test_unique_document_id(make_text_pdf):
    """Test that two extractions of the same PDF produce distinct, unique document IDs."""
    pdf_bytes = make_text_pdf(["Sample policy text"])
    extractor = PDFExtractor()

    doc1 = extractor.extract(pdf_bytes, filename="doc1.pdf", save_meta=False)
    doc2 = extractor.extract(pdf_bytes, filename="doc2.pdf", save_meta=False)

    assert doc1.doc_id != doc2.doc_id
    assert doc1.file_hash == doc2.file_hash  # Same content hash, distinct doc_id


def test_ocr_required_page_detection_without_tesseract(make_scanned_image_pdf):
    """Test detection of an image-only page and graceful handling when OCR is unavailable."""
    pdf_bytes = make_scanned_image_pdf("SCANNED POLICY CONTENT")

    # Force ocr_engine.is_available to return False
    with patch("backend.ingestion.ocr_engine.ocr_engine.is_available", return_value=False):
        doc = process_pdf(pdf_bytes, filename="scanned.pdf", save_meta=False)

    assert doc.page_count == 1
    page = doc.pages[0]
    assert page.page_number == 1
    assert page.extraction_method == ExtractionMethod.EMPTY
    # Warning about OCR availability is recorded
    assert any("Tesseract OCR is not available" in w or "OCR required" in w for w in page.warnings)


def test_page_limit_enforcement(make_text_pdf):
    """Test that exceeding MAX_PDF_PAGES raises PageLimitExceededError."""
    pdf_bytes = make_text_pdf(["Page 1", "Page 2", "Page 3"])
    extractor = PDFExtractor(max_pages=2)

    with pytest.raises(PageLimitExceededError):
        extractor.extract(pdf_bytes, filename="long_policy.pdf", save_meta=False)


def test_large_page_text_handling(make_text_pdf):
    """Test handling a page with extensive dense clause text."""
    long_clause_text = "Clause Section: " + ("The insurer covers emergency treatment subject to limits. " * 100)
    pdf_bytes = make_text_pdf([long_clause_text])

    doc = process_pdf(pdf_bytes, filename="dense_policy.pdf", save_meta=False)
    assert doc.page_count == 1
    assert len(doc.pages[0].text) > 1000
    assert doc.pages[0].quality_score >= 0.70
