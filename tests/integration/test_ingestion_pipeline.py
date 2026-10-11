"""Integration test for end-to-end PDF ingestion and metadata persistence."""

import json
from pathlib import Path
import pytest
import pymupdf

from backend.ingestion.pdf_extractor import process_pdf
from backend.models.data_models import Document, DocumentStatus, DocumentQuality


def test_end_to_end_ingestion_and_persistence(tmp_path):
    """Verify that a multi-page PDF is ingested, structured, and saved with complete metadata."""
    # Build a multi-page realistic synthetic policy fixture
    doc_pymupdf = pymupdf.open()

    # Page 1: Policy Schedule & Table
    p1 = doc_pymupdf.new_page(width=595, height=842)
    p1.insert_text((50, 50), "HEALTH SHIELD INSURANCE POLICY - SCHEDULE", fontsize=14)
    p1.insert_text((50, 75), "Policyholder: John Doe | Sum Insured: Rs. 10,00,000", fontsize=11)

    # Simple table grid on page 1
    cols = [50, 250, 450]
    rows = [100, 130, 160]
    for y in rows:
        p1.draw_line(pymupdf.Point(cols[0], y), pymupdf.Point(cols[-1], y), color=(0, 0, 0), width=1)
    for x in cols:
        p1.draw_line(pymupdf.Point(x, rows[0]), pymupdf.Point(x, rows[-1]), color=(0, 0, 0), width=1)
    p1.insert_text((60, 120), "Benefit Item", fontsize=10)
    p1.insert_text((260, 120), "Coverage Limit", fontsize=10)
    p1.insert_text((60, 150), "Room Rent", fontsize=10)
    p1.insert_text((260, 150), "Rs. 5,000 per day", fontsize=10)

    # Page 2: Exclusions & Conditions
    p2 = doc_pymupdf.new_page(width=595, height=842)
    p2.insert_text((50, 50), "SECTION 4: EXCLUSIONS AND WAITING PERIODS", fontsize=14)
    exclusions_text = (
        "4.1 Initial Waiting Period: A waiting period of 30 days from policy commencement applies to all claims "
        "except those arising out of accidental bodily injury.\n"
        "4.2 Pre-existing Diseases (PED): Covered after continuous coverage of 36 months.\n"
        "4.3 Permanent Exclusions: Cost of spectacles, cosmetic surgery, and unproven treatments."
    )
    p2.insert_textbox(pymupdf.Rect(50, 80, 545, 400), exclusions_text, fontsize=10)

    pdf_bytes = doc_pymupdf.tobytes()
    doc_pymupdf.close()

    # Ingest document with persistence enabled
    doc = process_pdf(pdf_bytes, filename="Health_Shield_Policy.pdf", save_meta=True)

    assert doc.status == DocumentStatus.READY
    assert doc.quality in [DocumentQuality.GOOD, DocumentQuality.WARNING]
    assert doc.page_count == 2
    assert len(doc.pages) == 2

    # Check Page 1
    page1 = doc.pages[0]
    assert page1.page_number == 1
    assert "HEALTH SHIELD INSURANCE" in page1.text

    # Check Page 2
    page2 = doc.pages[1]
    assert page2.page_number == 2
    assert "Initial Waiting Period" in page2.text
    assert "30 days" in page2.text

    # Verify persisted metadata
    assert doc.meta_path != ""
    meta_file = Path(doc.meta_path)
    assert meta_file.is_file()

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    # Validate that stored JSON can be parsed back into a canonical Document model
    reloaded_doc = Document.model_validate(meta_json)
    assert reloaded_doc.doc_id == doc.doc_id
    assert reloaded_doc.file_hash == doc.file_hash
    assert reloaded_doc.page_count == 2
    assert len(reloaded_doc.pages) == 2
    assert reloaded_doc.pages[1].page_number == 2
