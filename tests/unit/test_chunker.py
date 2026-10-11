"""Comprehensive tests for structure-aware document chunking."""

import pytest
import pymupdf

from backend.ingestion.pdf_extractor import process_pdf
from backend.ingestion.chunker import Chunker, chunk_document
from backend.models.data_models import Document, Page, Chunk, ChunkType, TableData


def test_normal_paragraphs_and_headings(make_text_pdf):
    """Test chunking standard prose under section headings."""
    text_p1 = (
        "SECTION 1: SCOPE OF COVERAGE\n"
        "The Company hereby agrees to pay reasonable and customary medical expenses.\n"
        "This coverage applies to all in-patient hospitalizations exceeding 24 hours."
    )
    pdf_bytes = make_text_pdf([text_p1])
    doc = process_pdf(pdf_bytes, filename="policy_p1.pdf", save_meta=False)

    chunks = chunk_document(doc)
    assert len(chunks) >= 2

    # Check heading chunk
    heading_chunks = [c for c in chunks if c.chunk_type == ChunkType.HEADING]
    assert len(heading_chunks) >= 1
    assert "SECTION 1" in heading_chunks[0].text

    # Check clause chunk
    clause_chunks = [c for c in chunks if c.chunk_type in [ChunkType.CLAUSE, ChunkType.TEXT]]
    assert len(clause_chunks) >= 1
    assert "reasonable and customary" in clause_chunks[0].text
    assert clause_chunks[0].section == "SECTION 1: SCOPE OF COVERAGE"
    assert "[SECTION 1: SCOPE OF COVERAGE]" in clause_chunks[0].text


def test_exclusion_bullet_lists_grouped(make_text_pdf):
    """Test that bullet lists of exclusions remain grouped under their governing heading."""
    text_p1 = (
        "SECTION 4: PERMANENT EXCLUSIONS\n"
        "The following treatments and expenses are specifically excluded from coverage:\n"
        "- (a) Investigation and evaluation diagnostics only\n"
        "- (b) Rest cure, rehabilitation, and respite care\n"
        "- (c) Obesity and weight control treatments\n"
        "- (d) Cosmetic or plastic surgery unless necessitated by burns\n"
        "- (e) Hazardous or adventure sports participation"
    )
    pdf_bytes = make_text_pdf([text_p1])
    doc = process_pdf(pdf_bytes, filename="exclusions.pdf", save_meta=False)

    chunks = chunk_document(doc)
    list_chunks = [c for c in chunks if c.chunk_type in [ChunkType.LIST_ITEM, ChunkType.CLAUSE]]
    assert len(list_chunks) >= 1

    # Ensure list items are grouped together rather than 1 chunk per bullet
    combined_list_text = "\n".join(c.text for c in list_chunks)
    assert "Investigation and evaluation" in combined_list_text
    assert "Cosmetic or plastic surgery" in combined_list_text
    assert list_chunks[0].section == "SECTION 4: PERMANENT EXCLUSIONS"


def test_condition_bound_to_numeric_clause_regression(make_text_pdf):
    """
    Regression test: Ensure conditional qualification is never separated from its numeric clause.
    Example: 'Co-payment: 20%' must remain with 'Applicable only to treatment at non-network hospitals.'
    """
    text_p1 = (
        "SECTION 3: COST SHARING\n"
        "Co-payment: 20% on every admissible claim.\n"
        "Applicable only to treatment availed at non-network hospitals."
    )
    pdf_bytes = make_text_pdf([text_p1])
    doc = process_pdf(pdf_bytes, filename="copay.pdf", save_meta=False)

    # Use small max_chars to test that condition stays bound
    chunker = Chunker(max_chars=300)
    chunks = chunker.chunk_document(doc)

    content_chunks = [c for c in chunks if c.chunk_type != ChunkType.HEADING]
    assert len(content_chunks) >= 1

    # Verify both the 20% co-payment AND the non-network condition exist in the SAME chunk
    target_chunk = None
    for c in content_chunks:
        if "20%" in c.text:
            target_chunk = c
            break

    assert target_chunk is not None
    assert "non-network hospitals" in target_chunk.text, "Condition was detached from numeric co-pay value!"


def test_waiting_period_clauses_intact(make_text_pdf):
    """Test waiting period clauses preserve numeric timeframes and qualification clauses."""
    text_p1 = (
        "SECTION 2: WAITING PERIODS\n"
        "2.1 Initial Waiting Period: 30 days initial waiting period applies to all illnesses.\n"
        "Provided that this waiting period shall not apply to accidental hospitalization."
    )
    pdf_bytes = make_text_pdf([text_p1])
    doc = process_pdf(pdf_bytes, filename="waiting.pdf", save_meta=False)

    chunks = chunk_document(doc)
    clause_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLAUSE]
    assert len(clause_chunks) >= 1
    c = clause_chunks[0]
    assert "30 days" in c.text
    assert "Provided that" in c.text
    assert c.section == "SECTION 2: WAITING PERIODS"


def test_table_preserved_as_atomic_chunk(make_table_pdf):
    """Test that a detected table is emitted as a single atomic TABLE chunk."""
    pdf_bytes = make_table_pdf()
    doc = process_pdf(pdf_bytes, filename="table_policy.pdf", save_meta=False)

    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

    assert len(table_chunks) == 1
    t_chunk = table_chunks[0]
    assert t_chunk.page_number == 1
    assert t_chunk.page_numbers == [1]
    assert "| Category | Limit | Co-payment |" in t_chunk.text
    assert "| Room Rent | Rs. 5,000 per day | 10% |" in t_chunk.text
    assert "| ICU Charges | Rs. 10,000 per day | None |" in t_chunk.text


def test_large_table_split_preserves_headers(make_text_pdf):
    """Test that when a table exceeds max_chars, it is split by complete rows with repeated headers."""
    # Manually create a TableData with 10 rows
    headers = ["Service", "Daily Limit", "Copay"]
    rows = [[f"Procedure {i}", f"Rs. {i * 1000}", f"{i}%"] for i in range(1, 15)]
    from backend.ingestion.table_extractor import render_markdown_table
    table_md = render_markdown_table(headers, rows)

    table_data = TableData(
        page_number=1,
        table_index=0,
        row_count=len(rows),
        col_count=len(headers),
        headers=headers,
        rows=rows,
        markdown=table_md,
    )

    page = Page(
        doc_id="doc-test-table",
        page_number=1,
        page_index=0,
        text="",
        tables=[table_data],
    )

    doc = Document(
        doc_id="doc-test-table",
        filename="big_table.pdf",
        file_hash="dummyhash",
        page_count=1,
        pages=[page],
    )

    # Use small max_chars (250) to force table split
    chunker = Chunker(max_chars=250)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for tc in chunks:
        assert tc.chunk_type == ChunkType.TABLE
        assert "| Service | Daily Limit | Copay |" in tc.text
        assert "| --- | --- | --- |" in tc.text
        # Never broken mid-cell
        assert tc.text.endswith("|")


def test_cross_page_clause_continuity(make_text_pdf):
    """
    Test that a clause starting on Page 1 and ending on Page 2
    is joined and preserves page_numbers=[1, 2].
    """
    # Page 1 ends in mid-sentence without terminal punctuation
    p1 = "SECTION 5: CLAIMS\nThe Insured Person shall immediately notify the Company within 24 hours of admission,"
    # Page 2 continues the clause
    p2 = "provided that delay in notification may be condoned where justified by medical emergency."

    pdf_bytes = make_text_pdf([p1, p2])
    doc = process_pdf(pdf_bytes, filename="crosspage.pdf", save_meta=False)

    chunks = chunk_document(doc)
    # Find the cross-page chunk
    cross_chunks = [c for c in chunks if len(c.page_numbers) > 1]

    assert len(cross_chunks) >= 1
    c = cross_chunks[0]
    assert c.page_number == 1
    assert c.page_numbers == [1, 2]
    assert "within 24 hours" in c.text
    assert "medical emergency" in c.text


def test_document_isolation_between_policies(make_text_pdf):
    """Test that chunks generated from Doc A never receive Doc B's identifier or data."""
    pdf_a = make_text_pdf(["Policy A Scope of Cover Rs. 5 Lakhs."])
    pdf_b = make_text_pdf(["Policy B Scope of Cover Rs. 15 Lakhs."])

    doc_a = process_pdf(pdf_a, filename="policy_A.pdf", save_meta=False)
    doc_b = process_pdf(pdf_b, filename="policy_B.pdf", save_meta=False)

    assert doc_a.doc_id != doc_b.doc_id

    chunks_a = chunk_document(doc_a)
    chunks_b = chunk_document(doc_b)

    for ca in chunks_a:
        assert ca.doc_id == doc_a.doc_id
        assert ca.filename == "policy_A.pdf"
        assert "Policy B" not in ca.text

    for cb in chunks_b:
        assert cb.doc_id == doc_b.doc_id
        assert cb.filename == "policy_B.pdf"
        assert "Policy A" not in cb.text


def test_deterministic_chunk_ordering_and_ids(make_text_pdf):
    """Test that chunk IDs and indices are deterministic and monotonic."""
    text = (
        "SECTION 1: DEFINITIONS\n"
        "Clause 1: Accident means a sudden, unforeseen and involuntary event.\n"
        "Clause 2: Co-payment means a cost sharing requirement."
    )
    pdf_bytes = make_text_pdf([text])
    doc = process_pdf(pdf_bytes, filename="defs.pdf", save_meta=False)

    chunks = chunk_document(doc)
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.chunk_id == f"{doc.doc_id}_c{idx:04d}"
        assert c.char_count == len(c.text)


def test_empty_page_handling(make_text_pdf):
    """Test handling of an empty page without crashing or generating empty chunks."""
    doc_pymupdf = pymupdf.open()
    # Blank page 1
    doc_pymupdf.new_page()
    # Text page 2
    p2 = doc_pymupdf.new_page()
    p2.insert_text((50, 50), "SECTION 1: VALID CONTENT ON PAGE TWO")
    pdf_bytes = doc_pymupdf.tobytes()
    doc_pymupdf.close()

    doc = process_pdf(pdf_bytes, filename="empty_p1.pdf", save_meta=False)
    chunks = chunk_document(doc)

    assert len(chunks) >= 1
    # All chunks must belong to page 2 (page 1 had no content)
    for c in chunks:
        assert 2 in c.page_numbers
        assert len(c.text.strip()) > 0
