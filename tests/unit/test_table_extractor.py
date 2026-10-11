"""Unit tests for table extraction and markdown conversion."""

import pytest
import pymupdf

from backend.ingestion.table_extractor import TableExtractor, render_markdown_table


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


def test_table_header_deduplication_regression():
    """
    Regression test for Phase 1 table fixture issue:
    Ensure header row is not duplicated in markdown when PyMuPDF tab.extract() returns it as row 0.
    """
    extractor = TableExtractor()
    doc = pymupdf.open("tests/fixtures/sample_policy_fixture.pdf")
    page = doc[0]

    tables = extractor.extract_tables_from_page(page)
    doc.close()

    assert len(tables) == 1
    table = tables[0]
    # Check that headers are extracted
    assert "Benefit Category" in table.headers
    assert "Policy Limit / Co-Pay" in table.headers

    # Check that row 0 is NOT a duplicate of headers
    assert table.rows[0] != table.headers
    assert table.rows[0][0] == "Room Rent"
    assert table.rows[1][0] == "Co-payment"

    # Verify markdown table only has ONE occurrence of the header text
    md_lines = table.markdown.strip().splitlines()
    assert len(md_lines) == 4  # Header line, Separator line, 2 Data rows
    assert md_lines[0].startswith("| Benefit Category |")
    assert md_lines[1].startswith("| --- |")
    assert md_lines[2].startswith("| Room Rent |")
    assert md_lines[3].startswith("| Co-payment |")


def test_table_first_row_preserved_when_distinct():
    """Verify that when first row contains legitimate data distinct from headers, it is NOT dropped."""
    headers = ["Treatment", "Cost"]
    rows = [
        ["Cataract Surgery", "Rs. 35,000"],
        ["Knee Replacement", "Rs. 1,50,000"],
    ]
    md = render_markdown_table(headers, rows)
    lines = md.splitlines()
    assert len(lines) == 4
    assert "Cataract Surgery" in lines[2]

