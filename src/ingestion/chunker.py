"""Structure-aware document chunking preserving clauses, headings, tables, and page metadata."""

import re
from typing import List, Optional, Tuple, Dict, Any
from src.config import config
from src.models.data_models import Document, Page, Chunk, ChunkType, TableData
from src.utils.logging_utils import logger


# Condition and qualification markers in insurance policies
CONDITION_MARKERS = [
    "subject to",
    "provided that",
    "provided always that",
    "except",
    "unless",
    "only if",
    "notwithstanding",
    "shall not",
    "will not be covered",
    "is excluded",
    "waiting period",
    "co-payment",
    "sub-limit",
    "deductible",
    "applicable only to",
]


class Chunker:
    """
    Structure-aware chunker for insurance policy documents.
    Preserves tables intact, binds conditions to clauses, maintains section headings,
    groups bullet lists, and tracks cross-page continuations.
    """

    def __init__(
        self,
        max_chars: int = config.MAX_CHUNK_CHARS,
        overlap_chars: int = config.OVERLAP_CHARS,
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Transform a Document and its Pages into an ordered list of structured Chunks.

        Args:
            document: Canonical Document instance with pages populated.

        Returns:
            List of Chunk objects adhering to canonical schema.
        """
        all_chunks: List[Chunk] = []
        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        chunk_counter = 0

        # Buffer for cross-page text continuation
        pending_cross_page_text: Optional[str] = None
        pending_cross_page_start_page: Optional[int] = None

        for page in document.pages:
            page_num = page.page_number

            # 1. Process and emit tables on this page first
            table_cells: set = set()
            for t_idx, table in enumerate(page.tables):
                # Collect cells so they are not duplicated in prose
                for h in table.headers:
                    if h:
                        table_cells.add(h.strip().lower())
                for row in table.rows:
                    for cell in row:
                        if cell:
                            table_cells.add(cell.strip().lower())

                # Emit table chunks
                table_chunks = self._chunk_table(
                    table=table,
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page_number=page_num,
                    section=current_section,
                    subsection=current_subsection,
                    start_index=chunk_counter,
                )
                for tc in table_chunks:
                    all_chunks.append(tc)
                    chunk_counter += 1

            # 2. Extract and clean prose lines (excluding text that came from tables)
            raw_lines = page.text.splitlines()
            prose_lines: List[str] = []
            for line in raw_lines:
                s = line.strip()
                if not s:
                    continue
                # Skip lines that are exact table cell contents if table was already extracted
                if table_cells and s.lower() in table_cells:
                    continue
                prose_lines.append(s)

            if not prose_lines:
                continue

            # Handle cross-page clause continuation from previous page
            if pending_cross_page_text:
                prose_lines.insert(0, pending_cross_page_text)
                clause_start_page = pending_cross_page_start_page or page_num
                pending_cross_page_text = None
                pending_cross_page_start_page = None
            else:
                clause_start_page = page_num

            # Check if this page ends in the middle of a sentence (no terminal punctuation)
            # If so, detach the last line to be joined with next page
            last_line = prose_lines[-1] if prose_lines else ""
            if (
                len(prose_lines) > 1
                and not last_line.endswith((".", ":", ";", "?", "!"))
                and not self._is_heading(last_line)
                and page_num < document.page_count
            ):
                pending_cross_page_text = prose_lines.pop()
                pending_cross_page_start_page = clause_start_page

            # 3. Structure prose into logical blocks (sections, clauses, bullet lists)
            blocks = self._parse_prose_blocks(prose_lines)

            for block in blocks:
                b_type = block["type"]
                b_text = block["text"]

                if b_type == "SECTION":
                    current_section = b_text
                    current_subsection = None
                    # We can record the heading as a HEADING chunk
                    chunk_id = f"{document.doc_id}_c{chunk_counter:04d}"
                    all_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            filename=document.filename,
                            page_number=clause_start_page,
                            page_numbers=[page_num] if clause_start_page == page_num else [clause_start_page, page_num],
                            section=current_section,
                            subsection=current_subsection,
                            chunk_type=ChunkType.HEADING,
                            text=b_text,
                            chunk_index=chunk_counter,
                        )
                    )
                    chunk_counter += 1

                elif b_type == "SUBSECTION":
                    current_subsection = b_text
                    chunk_id = f"{document.doc_id}_c{chunk_counter:04d}"
                    all_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            filename=document.filename,
                            page_number=clause_start_page,
                            page_numbers=[page_num] if clause_start_page == page_num else [clause_start_page, page_num],
                            section=current_section,
                            subsection=current_subsection,
                            chunk_type=ChunkType.HEADING,
                            text=b_text,
                            chunk_index=chunk_counter,
                        )
                    )
                    chunk_counter += 1

                elif b_type == "CLAUSE_WITH_TITLE":
                    clause_sub = block.get("sub_label") or current_subsection
                    clause_chunks = self._chunk_text_block(
                        text=b_text,
                        block_type="CLAUSE",
                        doc_id=document.doc_id,
                        filename=document.filename,
                        page_number=clause_start_page,
                        current_page=page_num,
                        section=current_section,
                        subsection=clause_sub,
                        start_index=chunk_counter,
                    )
                    for c in clause_chunks:
                        all_chunks.append(c)
                        chunk_counter += 1

                else:
                    # Content clause, paragraph, or list
                    clause_chunks = self._chunk_text_block(
                        text=b_text,
                        block_type=b_type,
                        doc_id=document.doc_id,
                        filename=document.filename,
                        page_number=clause_start_page,
                        current_page=page_num,
                        section=current_section,
                        subsection=current_subsection,
                        start_index=chunk_counter,
                    )
                    for c in clause_chunks:
                        all_chunks.append(c)
                        chunk_counter += 1

        # If any pending cross-page text was left at EOF, emit it
        if pending_cross_page_text:
            chunk_id = f"{document.doc_id}_c{chunk_counter:04d}"
            p_start = pending_cross_page_start_page or document.page_count
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page_number=p_start,
                    page_numbers=[p_start, document.page_count] if p_start != document.page_count else [p_start],
                    section=current_section,
                    subsection=current_subsection,
                    chunk_type=ChunkType.TEXT,
                    text=pending_cross_page_text,
                    chunk_index=chunk_counter,
                )
            )
            chunk_counter += 1

        document.chunk_count = len(all_chunks)
        logger.info(
            "Chunked document %s into %d structure-aware chunks",
            document.doc_id,
            len(all_chunks),
        )
        return all_chunks

    def _chunk_table(
        self,
        table: TableData,
        doc_id: str,
        filename: str,
        page_number: int,
        section: Optional[str],
        subsection: Optional[str],
        start_index: int,
    ) -> List[Chunk]:
        """
        Emit a table as one or more atomic TABLE chunks without splitting mid-row.
        """
        table_md = table.markdown.strip()
        if not table_md:
            return []

        # If the table markdown fits within max_chars, emit as single atomic chunk
        if len(table_md) <= self.max_chars or len(table.rows) <= 1:
            chunk_id = f"{doc_id}_c{start_index:04d}"
            # Prefix with section context if present
            context_header = f"[{section}]\n" if section else ""
            return [
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_numbers=[page_number],
                    section=section,
                    subsection=subsection,
                    table_index=table.table_index,
                    chunk_type=ChunkType.TABLE,
                    text=context_header + table_md,
                    chunk_index=start_index,
                )
            ]

        # If table is large, split strictly by whole rows while repeating the header line
        chunks: List[Chunk] = []
        md_lines = table_md.splitlines()
        if len(md_lines) < 3:
            # Fallback if markdown doesn't have standard structure
            chunk_id = f"{doc_id}_c{start_index:04d}"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_numbers=[page_number],
                    section=section,
                    subsection=subsection,
                    table_index=table.table_index,
                    chunk_type=ChunkType.TABLE,
                    text=table_md,
                    chunk_index=start_index,
                )
            ]

        header_lines = md_lines[:2]  # Header and separator
        data_rows = md_lines[2:]
        current_rows: List[str] = []
        c_idx = start_index

        for row in data_rows:
            current_rows.append(row)
            sub_table_text = "\n".join(header_lines + current_rows)
            if len(sub_table_text) >= (self.max_chars - 100):
                chunk_id = f"{doc_id}_c{c_idx:04d}"
                context_header = f"[{section} (Table Part {len(chunks)+1})]\n" if section else ""
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_number,
                        page_numbers=[page_number],
                        section=section,
                        subsection=subsection,
                        table_index=table.table_index,
                        chunk_type=ChunkType.TABLE,
                        text=context_header + sub_table_text,
                        chunk_index=c_idx,
                    )
                )
                c_idx += 1
                current_rows = []

        if current_rows:
            chunk_id = f"{doc_id}_c{c_idx:04d}"
            context_header = f"[{section} (Table Part {len(chunks)+1})]\n" if section else ""
            sub_table_text = "\n".join(header_lines + current_rows)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_numbers=[page_number],
                    section=section,
                    subsection=subsection,
                    table_index=table.table_index,
                    chunk_type=ChunkType.TABLE,
                    text=context_header + sub_table_text,
                    chunk_index=c_idx,
                )
            )

        return chunks

    def _parse_prose_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Group lines into semantic blocks: SECTION, SUBSECTION, CLAUSE_WITH_TITLE, LIST, or CLAUSE.
        """
        blocks: List[Dict[str, Any]] = []
        current_type = "CLAUSE"
        current_lines: List[str] = []
        current_sub_label: Optional[str] = None

        for line in lines:
            # Check if this line is an operative qualification or condition modifying the preceding clause
            is_condition_line = any(
                line.lower().startswith(marker) or f" {marker}" in line.lower()[:30]
                for marker in CONDITION_MARKERS
            )
            if is_condition_line:
                if current_lines:
                    current_lines.append(line)
                    continue
                elif blocks and blocks[-1]["type"] not in ["SECTION", "SUBSECTION"]:
                    blocks[-1]["text"] += "\n" + line
                    continue

            if self._is_section_heading(line):
                if current_lines:
                    blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})
                    current_lines = []
                    current_sub_label = None
                blocks.append({"type": "SECTION", "text": line, "sub_label": None})
                current_type = "CLAUSE"

            elif self._is_subsection_heading(line):
                if current_lines:
                    blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})
                    current_lines = []
                    current_sub_label = None
                blocks.append({"type": "SUBSECTION", "text": line, "sub_label": line})
                current_type = "CLAUSE"

            elif self._is_clause_with_title(line):
                if current_lines:
                    blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})
                    current_lines = []
                sub_label, _ = self._extract_clause_title(line)
                blocks.append({"type": "CLAUSE_WITH_TITLE", "text": line, "sub_label": sub_label})
                current_type = "CLAUSE"
                current_sub_label = None

            elif self._is_list_item(line):
                if current_type != "LIST" and current_lines:
                    # Check if the preceding line was an intro to this list
                    if current_lines[-1].endswith(":"):
                        intro = current_lines.pop()
                        if current_lines:
                            blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})
                        current_lines = [intro, line]
                    else:
                        blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})
                        current_lines = [line]
                else:
                    current_lines.append(line)
                current_type = "LIST"

            else:
                if current_type == "LIST":
                    # Check if line is an indented continuation of list
                    if line.startswith("   ") or line.startswith("\t"):
                        current_lines.append(line)
                        continue
                    else:
                        blocks.append({"type": "LIST", "text": "\n".join(current_lines), "sub_label": current_sub_label})
                        current_lines = [line]
                        current_type = "CLAUSE"
                        current_sub_label = None
                else:
                    current_lines.append(line)

        if current_lines:
            blocks.append({"type": current_type, "text": "\n".join(current_lines), "sub_label": current_sub_label})

        return blocks

    def _chunk_text_block(
        self,
        text: str,
        block_type: str,
        doc_id: str,
        filename: str,
        page_number: int,
        current_page: int,
        section: Optional[str],
        subsection: Optional[str],
        start_index: int,
    ) -> List[Chunk]:
        """
        Split a content text block into chunks respecting sentence and condition boundaries.
        """
        chunks: List[Chunk] = []
        p_nums = [page_number] if page_number == current_page else [page_number, current_page]
        chunk_t = ChunkType.LIST_ITEM if block_type == "LIST" else ChunkType.CLAUSE

        # Context breadcrumb to prepend
        prefix = ""
        if section and subsection:
            prefix = f"[{section} > {subsection}]\n"
        elif section:
            prefix = f"[{section}]\n"

        # If block fits in max_chars, return directly
        full_text = prefix + text if prefix and not text.startswith("[") else text
        if len(full_text) <= self.max_chars:
            chunk_id = f"{doc_id}_c{start_index:04d}"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_numbers=p_nums,
                    section=section,
                    subsection=subsection,
                    chunk_type=chunk_t,
                    text=full_text,
                    chunk_index=start_index,
                )
            ]

        # Break text into clauses/sentences without breaking condition bindings
        units = self._split_preserving_conditions(text)
        current_unit_group: List[str] = []
        c_idx = start_index

        for unit in units:
            candidate = prefix + "\n".join(current_unit_group + [unit])
            if len(candidate) <= self.max_chars:
                current_unit_group.append(unit)
            else:
                if current_unit_group:
                    chunk_id = f"{doc_id}_c{c_idx:04d}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_number,
                            page_numbers=p_nums,
                            section=section,
                            subsection=subsection,
                            chunk_type=chunk_t,
                            text=prefix + "\n".join(current_unit_group),
                            chunk_index=c_idx,
                        )
                    )
                    c_idx += 1
                    # With overlap: keep the last unit if overlap requested
                    if self.overlap_chars > 0 and len(current_unit_group[-1]) <= self.overlap_chars:
                        current_unit_group = [current_unit_group[-1], unit]
                    else:
                        current_unit_group = [unit]
                else:
                    # Single unit exceeds max_chars: hard-chunk it safely
                    chunk_id = f"{doc_id}_c{c_idx:04d}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_number,
                            page_numbers=p_nums,
                            section=section,
                            subsection=subsection,
                            chunk_type=chunk_t,
                            text=prefix + unit[: self.max_chars],
                            chunk_index=c_idx,
                        )
                    )
                    c_idx += 1
                    current_unit_group = []

        if current_unit_group:
            chunk_id = f"{doc_id}_c{c_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_numbers=p_nums,
                    section=section,
                    subsection=subsection,
                    chunk_type=chunk_t,
                    text=prefix + "\n".join(current_unit_group),
                    chunk_index=c_idx,
                )
            )

        return chunks

    def _split_preserving_conditions(self, text: str) -> List[str]:
        """
        Split prose or list into logical units, ensuring qualifying conditions
        stay bound to the clauses they modify.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return []

        # If it's already a list of lines/bullets, treat each line as a candidate unit
        # but bind lines that start with condition markers to previous line
        units: List[str] = []
        for line in lines:
            is_condition = any(
                line.lower().startswith(marker) or f" {marker}" in line.lower()[:30]
                for marker in CONDITION_MARKERS
            )
            if is_condition and units:
                # Merge condition with immediately preceding clause
                units[-1] = units[-1] + " " + line
            else:
                units.append(line)

        return units

    def _is_section_heading(self, line: str) -> bool:
        """Identify major section headings (e.g., 'SECTION 1', 'PART A', 'EXCLUSIONS')."""
        s = line.strip()
        if not s or len(s) > 100:
            return False
        if re.match(r"^(SECTION|PART|SCHEDULE|POLICY SCHEDULE)\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\d+\.\s+[A-Z\s]{3,}", s) and not s.endswith("."):
            return True
        if s.isupper() and 3 < len(s) < 80 and not s.endswith("."):
            return True
    def _is_clause_with_title(self, line: str) -> bool:
        """Check if line starts with a numbered/lettered title followed by colon and content text."""
        s = line.strip()
        pattern = r"^(\d+\.\d+(\.\d+)?|\([a-zA-Z0-9]+\))\s+([A-Za-z\s\-]+):\s*(.+)$"
        return bool(re.match(pattern, s))

    def _extract_clause_title(self, line: str) -> Tuple[str, str]:
        """Extract the clause identifier/title and body text from a clause line."""
        s = line.strip()
        pattern = r"^(\d+\.\d+(\.\d+)?|\([a-zA-Z0-9]+\))\s+([A-Za-z\s\-]+):\s*(.+)$"
        m = re.match(pattern, s)
        if m:
            title = f"{m.group(1)} {m.group(3).strip()}"
            body = m.group(4).strip()
            return title, body
        return "", s

    def _is_subsection_heading(self, line: str) -> bool:
        """Identify standalone subsections and numbered clause titles without body text."""
        s = line.strip()
        if not s or len(s) > 100:
            return False
        if self._is_clause_with_title(s):
            return False
        if re.match(r"^(\d+\.\d+(\.\d+)?|\([a-zA-Z0-9]+\))\s+([A-Za-z\s\-]+)[:]?$", s):
            return True
        if re.match(r"^(Clause|Condition)\s+\d+[:.]?$", s, re.IGNORECASE):
            return True
        return False

    def _is_heading(self, line: str) -> bool:
        """Check if line is any type of heading."""
        return self._is_section_heading(line) or self._is_subsection_heading(line)

    def _is_list_item(self, line: str) -> bool:
        """Check if line starts with bullet or list item notation."""
        s = line.strip()
        return bool(re.match(r"^([-•*]|\([a-zA-Z0-9]+\)|\d+\.)\s+", s))


chunker = Chunker()


def chunk_document(document: Document, max_chars: int = config.MAX_CHUNK_CHARS, overlap_chars: int = config.OVERLAP_CHARS) -> List[Chunk]:
    """Convenience functional interface for chunking a Document."""
    c = Chunker(max_chars=max_chars, overlap_chars=overlap_chars)
    return c.chunk_document(document)
