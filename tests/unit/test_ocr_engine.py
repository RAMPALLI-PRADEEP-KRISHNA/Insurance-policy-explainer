"""Unit tests for OCR engine detection, status reporting, and fallback execution."""

import pytest
import pymupdf
from unittest.mock import patch, MagicMock

from src.ingestion.ocr_engine import OCREngine


def test_tesseract_detection_when_missing():
    """Verify that when Tesseract is missing, availability is False and status includes instructions."""
    engine = OCREngine(custom_cmd="non_existent_tesseract_binary.exe")
    assert engine.is_available() is False

    status = engine.get_status()
    assert status["available"] is False
    assert "winget install UB-Mannheim.TesseractOCR" in status["install_instructions"]


def test_ocr_page_fallback_when_unavailable():
    """Verify that calling OCR when Tesseract is unavailable returns a clear warning without crashing."""
    engine = OCREngine(custom_cmd="non_existent_binary.exe")
    doc = pymupdf.open()
    page = doc.new_page()

    text, success, warning = engine.ocr_page(page)
    doc.close()

    assert text == ""
    assert success is False
    assert warning is not None
    assert "Tesseract OCR is not available" in warning
    assert "winget install" in warning


def test_ocr_page_fallback_mocked_success():
    """Verify OCR execution flow when Tesseract returns recognized text."""
    engine = OCREngine()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test")

    # Mock is_available to True and pytesseract.image_to_string
    with patch.object(engine, "is_available", return_value=True), \
         patch("src.ingestion.ocr_engine.pytesseract.image_to_string", return_value="Recognized Scanned Text"):
        text, success, err = engine.ocr_page(page)

    doc.close()
    assert success is True
    assert text == "Recognized Scanned Text"
    assert err is None
