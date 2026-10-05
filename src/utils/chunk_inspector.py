"""Developer inspection utility to inspect and audit generated chunks."""

import sys
from pathlib import Path
from typing import List, Optional

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.models.data_models import Document, Chunk


def format_chunk_summary(chunk: Chunk, preview_chars: int = 140) -> str:
    """Format a single chunk into an audit string."""
    preview = repr(chunk.text[:preview_chars])[1:-1]
    if len(chunk.text) > preview_chars:
        preview += "..."

    sec_info = chunk.section or "None"
    if chunk.subsection:
        sec_info += f" > {chunk.subsection}"

    pages_str = str(chunk.page_numbers) if len(chunk.page_numbers) > 1 else str(chunk.page_number)

    lines = [
        f"Chunk ID:       {chunk.chunk_id}",
        f"Type:           {chunk.chunk_type.value}",
        f"Page(s):        {pages_str}",
        f"Section:        {sec_info}",
        f"Char Count:     {chunk.char_count}",
        f"Text Preview:   {preview}",
        "-" * 70,
    ]
    return "\n".join(lines)


def inspect_chunks(
    pdf_path_or_doc: Path | str | Document,
    max_chars: int = 800,
    overlap_chars: int = 160,
    print_output: bool = True,
) -> List[Chunk]:
    """
    Ingest a PDF, generate chunks, and format a human-readable audit report.

    Returns:
        List of generated Chunk objects.
    """
    if isinstance(pdf_path_or_doc, Document):
        doc = pdf_path_or_doc
    else:
        path = Path(pdf_path_or_doc)
        doc = process_pdf(path, save_meta=False)

    chunks = chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)

    if print_output:
        print("=" * 70)
        print(f"CHUNK INSPECTION REPORT: {doc.filename}")
        print(f"Doc ID: {doc.doc_id} | Total Pages: {doc.page_count} | Total Chunks: {len(chunks)}")
        print("=" * 70)
        for c in chunks:
            print(format_chunk_summary(c))

    return chunks


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample_policy_fixture.pdf"
    inspect_chunks(target)
