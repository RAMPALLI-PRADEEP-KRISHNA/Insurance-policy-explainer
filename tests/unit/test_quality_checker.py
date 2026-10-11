"""Unit tests for text quality assessment and document quality derivation."""

import pytest
import pymupdf

from backend.ingestion.quality_checker import QualityChecker, QualityThresholds
from backend.models.data_models import DocumentQuality


def test_assess_good_quality_page():
    """Verify high-quality prose is scored >= 0.70 with recommended state GOOD."""
    checker = QualityChecker()
    doc = pymupdf.open()
    page = doc.new_page()

    sample_text = (
        "Section 1: Scope of Cover. The Company hereby agrees, subject to the terms, conditions, "
        "and exclusions herein contained, to pay hospital expenses reasonably and necessarily incurred "
        "by the Insured Person for medical treatment of illness or accidental injury during the policy period. "
        "The maximum liability of the Company under this Benefit shall not exceed the Sum Insured."
    )
    score, state, warnings = checker.assess_page_quality(sample_text, page)
    doc.close()

    assert score >= 0.70
    assert state == "GOOD"
    assert len(warnings) == 0


def test_assess_empty_page_without_images():
    """Verify completely empty page without images is categorized as EMPTY."""
    checker = QualityChecker()
    doc = pymupdf.open()
    page = doc.new_page()

    score, state, warnings = checker.assess_page_quality("", page)
    doc.close()

    assert state == "EMPTY"
    assert "empty" in warnings[0].lower()


def test_assess_corrupted_garbage_text():
    """Verify text with high replacement / control characters is detected as degraded."""
    checker = QualityChecker()
    doc = pymupdf.open()
    page = doc.new_page()

    garbage_text = "\ufffd\ufffd\ufffd\x01\x02\x03\x04\x00\x05\x06%^&!@#" * 10
    score, state, warnings = checker.assess_page_quality(garbage_text, page)
    doc.close()

    assert score < 0.40
    assert any("corrupted" in w.lower() for w in warnings)


def test_derive_document_quality():
    """Verify deterministic derivation of document-level quality states."""
    checker = QualityChecker()

    # All high quality -> GOOD
    doc_q = checker.derive_document_quality([0.85, 0.90, 0.78], ocr_page_count=0, total_pages=3)
    assert doc_q == DocumentQuality.GOOD

    # Low mean quality -> POOR
    doc_q_poor = checker.derive_document_quality([0.15, 0.20, 0.10], ocr_page_count=0, total_pages=3)
    assert doc_q_poor == DocumentQuality.POOR

    # High OCR usage (>= 30%) -> OCR_REQUIRED
    doc_q_ocr = checker.derive_document_quality([0.65, 0.70, 0.60], ocr_page_count=2, total_pages=4)
    assert doc_q_ocr == DocumentQuality.OCR_REQUIRED

    # Failure flag or empty -> FAILED
    doc_q_failed = checker.derive_document_quality([], ocr_page_count=0, total_pages=0, failed=True)
    assert doc_q_failed == DocumentQuality.FAILED
