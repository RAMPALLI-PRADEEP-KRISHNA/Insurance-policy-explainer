import sys
import types
fast = types.ModuleType('sklearn.metrics.cluster._expected_mutual_info_fast')
fast.expected_mutual_information = lambda *args: 0.0
sys.modules['sklearn.metrics.cluster._expected_mutual_info_fast'] = fast

"""Pytest configuration and synthetic PDF fixture generators."""

import io
import pytest
from pathlib import Path
from typing import List
import pymupdf
from PIL import Image, ImageDraw


@pytest.fixture
def make_text_pdf():
    """Factory fixture to create a valid multi-page PDF with specific text per page."""
    def _create(pages_text: List[str]) -> bytes:
        doc = pymupdf.open()
        for text in pages_text:
            page = doc.new_page(width=595, height=842)  # Standard A4
            rect = pymupdf.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, text, fontsize=9)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    return _create


@pytest.fixture
def make_table_pdf():
    """Factory fixture to create a PDF containing a structured table with borders."""
    def _create() -> bytes:
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "Schedule of Benefits", fontsize=14)

        # Draw a table grid with text inside cells
        # Columns: Category (x=50 to 200), Limit (x=200 to 350), Co-pay (x=350 to 500)
        # Rows: y=100 (Header), y=140 (Row 1), y=180 (Row 2), y=220 (Row 3)
        cols = [50, 200, 350, 500]
        rows = [100, 140, 180, 220]

        data = [
            ["Category", "Limit", "Co-payment"],
            ["Room Rent", "Rs. 5,000 per day", "10%"],
            ["ICU Charges", "Rs. 10,000 per day", "None"],
        ]

        # Draw horizontal lines
        for y in rows:
            page.draw_line(pymupdf.Point(cols[0], y), pymupdf.Point(cols[-1], y), color=(0, 0, 0), width=1)

        # Draw vertical lines
        for x in cols:
            page.draw_line(pymupdf.Point(x, rows[0]), pymupdf.Point(x, rows[-1]), color=(0, 0, 0), width=1)

        # Insert cell text
        for r_idx, row in enumerate(data):
            for c_idx, val in enumerate(row):
                x = cols[c_idx] + 10
                y = rows[r_idx] + 25
                page.insert_text((x, y), val, fontsize=10)

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    return _create


@pytest.fixture
def make_scanned_image_pdf():
    """Factory fixture to create a PDF page with an image and no native text layer."""
    def _create(text_to_draw: str = "SCANNED EXCLUSIONS CLAUSE 12A") -> bytes:
        # Create an image using Pillow
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), text_to_draw, fill=(0, 0, 0))

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Insert image into PyMuPDF page (no text layer)
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        rect = pymupdf.Rect(50, 50, 545, 792)
        page.insert_image(rect, stream=img_bytes)

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    return _create

