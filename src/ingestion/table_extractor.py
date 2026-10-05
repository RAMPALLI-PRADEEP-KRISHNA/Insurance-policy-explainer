"""Table extraction module preserving tabular structures from PDF pages."""

from typing import List, Optional
import pymupdf

from src.models.data_models import TableData
from src.utils.logging_utils import logger


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render table headers and rows into clean GitHub-flavored markdown."""
    if not headers and not rows:
        return ""

    # Sanitize newlines and pipe characters in cells
    def clean_cell(cell: Optional[str]) -> str:
        if cell is None:
            return ""
        val = str(cell).replace("\n", " ").replace("|", "\\|").strip()
        return val

    # If headers are missing, create default column names
    col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
    if col_count == 0:
        return ""

    clean_headers = [clean_cell(h) or f"Column {i + 1}" for i, h in enumerate(headers)] if headers else [f"Col {i+1}" for i in range(col_count)]
    clean_rows = [[clean_cell(c) for c in row] for row in rows]

    # Ensure all rows match column count
    for r in clean_rows:
        while len(r) < len(clean_headers):
            r.append("")
        if len(r) > len(clean_headers):
            r = r[:len(clean_headers)]

    header_line = "| " + " | ".join(clean_headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(clean_headers)) + " |"
    row_lines = ["| " + " | ".join(r) + " |" for r in clean_rows]

    return "\n".join([header_line, sep_line] + row_lines)


class TableExtractor:
    """Detects and extracts structured tables from PyMuPDF pages."""

    def extract_tables_from_page(self, page: pymupdf.Page) -> List[TableData]:
        """
        Detect and extract all tables on a page.

        Returns:
            List of TableData objects. If no tables are detected or table extraction
            is unsupported, returns an empty list without failing.
        """
        tables: List[TableData] = []
        if not hasattr(page, "find_tables"):
            logger.debug("Page object does not have 'find_tables' method.")
            return tables

        try:
            detected_tables = page.find_tables()
            if not detected_tables or len(detected_tables.tables) == 0:
                return tables

            for idx, tab in enumerate(detected_tables.tables):
                try:
                    # Extract header names and data rows
                    headers = [str(h).strip() if h is not None else "" for h in (tab.header.names or [])]
                    raw_data = tab.extract() or []

                    # If first row was consumed as headers by PyMuPDF, raw_data contains data rows
                    # If not, split first row as headers if appropriate
                    rows = [[str(c).strip() if c is not None else "" for c in row] for row in raw_data]

                    # Filter out completely empty rows
                    rows = [r for r in rows if any(cell.strip() for cell in r)]
                    if not rows and not any(headers):
                        continue

                    # If headers are all empty but we have rows, take row 0 as header
                    if (not any(headers) or not headers) and rows:
                        headers = rows[0]
                        rows = rows[1:]

                    col_count = max(len(headers), max((len(r) for r in rows), default=0))
                    row_count = len(rows)

                    md = render_markdown_table(headers, rows)
                    bbox_tuple = tuple(tab.bbox) if hasattr(tab, "bbox") and tab.bbox else None

                    table_data = TableData(
                        page_number=page.number + 1,
                        table_index=idx,
                        row_count=row_count,
                        col_count=col_count,
                        headers=headers,
                        rows=rows,
                        markdown=md,
                        bbox=bbox_tuple,
                    )
                    tables.append(table_data)
                    logger.debug("Extracted table %d on page %d with %d rows and %d cols", idx + 1, page.number + 1, row_count, col_count)

                except Exception as table_err:
                    logger.warning("Failed to parse table %d on page %d: %s", idx + 1, page.number + 1, table_err)
                    continue

        except Exception as e:
            # Table detection failed gracefully; normal text extraction continues unaffected
            logger.warning("find_tables failed on page %d: %s", page.number + 1, e)

        return tables


table_extractor = TableExtractor()
