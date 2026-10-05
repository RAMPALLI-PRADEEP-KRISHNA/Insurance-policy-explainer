"""Unit tests for table extraction and markdown conversion."""

import pytest
import pymupdf

from src.ingestion.table_extractor import TableExtractor, render_markdown_table


def test_render_markdown_table():
    """Verify markdown table rendering with sanitization."""
    headers = ["Benefit", "Limit", "Notes"]
    rows = [
        ["Room Rent", "Rs. 5,000", "Per day limit"],
        ["ICU", "Rs. 10,000 | Max", "Actuals"],
    ]
    md = render_markdown_table(headers, rows)

    assert "| Benefit | Limit | Notes |" in md
    assert "| --- | --- | --- |" in md
    assert "| Room Rent | Rs. 5,000 | Per day limit |" in md
    # Pipe character inside cell is escaped
    assert r"Rs. 10,000 \| Max" in md


def test_table_extractor_on_plain_text_page():
    """Verify table extraction returns empty list on text page without tables."""
    extractor = TableExtractor()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is plain prose text with no table borders or grid structure.")

    tables = extractor.extract_tables_from_page(page)
    doc.close()

    assert isinstance(tables, list)
    assert len(tables) == 0


def test_table_extractor_on_structured_table(make_table_pdf):
    """Verify table extraction on synthetic PDF with table grid lines."""
    pdf_bytes = make_table_pdf()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    extractor = TableExtractor()
    tables = extractor.extract_tables_from_page(page)
    doc.close()

    assert len(tables) >= 1
    table = tables[0]
    assert table.page_number == 1
    assert table.row_count >= 1
    assert "Category" in table.headers or any("Category" in row for row in table.rows)
