"""PDF text and table extraction pipeline converting insurance PDFs to structured Document objects."""

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Optional, List
import pymupdf

from src.config import config
from src.models.data_models import (
    Document,
    Page,
    DocumentStatus,
    DocumentQuality,
    ExtractionMethod,
)
from src.ingestion.quality_checker import quality_checker
from src.ingestion.ocr_engine import ocr_engine
from src.ingestion.table_extractor import table_extractor
from src.utils.file_utils import calculate_sha256, validate_pdf_content
from src.utils.exceptions import (
    InvalidPDFError,
    UnreadablePDFError,
    PageLimitExceededError,
    ExtractionError,
)
from src.utils.logging_utils import logger


class PDFExtractor:
    """Extracts text, tables, and page metadata from insurance policy PDFs."""

    def __init__(self, max_pages: int = config.MAX_PDF_PAGES):
        self.max_pages = max_pages

    def extract(
        self,
        data_or_path: Union[str, Path, bytes],
        filename: Optional[str] = None,
        save_meta: bool = True,
    ) -> Document:
        """
        Validate, open, and ingest a PDF document.

        Args:
            data_or_path: Path to PDF or raw PDF bytes
            filename: Optional display name of the file
            save_meta: Whether to serialize metadata to data/documents/<doc_id>/

        Returns:
            Fully populated canonical Document instance with extracted Page objects.

        Raises:
            InvalidPDFError, UnreadablePDFError, PageLimitExceededError, ExtractionError
        """
        # Step 1: Validate file content & size
        pdf_bytes, detected_filename = validate_pdf_content(
            data_or_path, max_size_mb=config.MAX_FILE_SIZE_MB
        )
        final_filename = filename or detected_filename
        file_hash = calculate_sha256(pdf_bytes)
        doc_id = str(uuid.uuid4())

        logger.info("Ingesting document [doc_id=%s, file=%s, hash=%s]", doc_id, final_filename, file_hash[:8])

        # Step 2: Open with PyMuPDF safely
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error("Failed to parse PDF stream: %s", e)
            raise InvalidPDFError(f"Could not parse file as PDF: {str(e)}")

        try:
            if doc.is_encrypted:
                logger.error("PDF is password-protected or encrypted.")
                raise UnreadablePDFError(
                    f"Document '{final_filename}' is password-protected or encrypted and cannot be processed."
                )

            page_count = len(doc)
            if page_count == 0:
                raise InvalidPDFError(f"Document '{final_filename}' has 0 pages.")

            if page_count > self.max_pages:
                raise PageLimitExceededError(
                    f"Document has {page_count} pages, which exceeds the limit of {self.max_pages} pages."
                )

            # Step 3: Process pages
            pages: List[Page] = []
            ocr_page_ids: List[int] = []
            page_qualities: List[float] = []

            for page_idx in range(page_count):
                page_num = page_idx + 1  # 1-indexed page number
                pymupdf_page = doc[page_idx]
                page_warnings: List[str] = []

                # Extract native text
                raw_text = pymupdf_page.get_text("text") or ""
                cleaned_text = raw_text.strip()
                extraction_method = ExtractionMethod.NATIVE

                # Extract tables
                tables = table_extractor.extract_tables_from_page(pymupdf_page)

                # Assess native extraction quality
                quality_score, quality_state, q_warnings = quality_checker.assess_page_quality(
                    cleaned_text, pymupdf_page
                )
                page_warnings.extend(q_warnings)

                # Trigger OCR fallback if native text is insufficient and page needs OCR
                if quality_state == "OCR_REQUIRED" or quality_score < config.OCR_THRESHOLD:
                    logger.debug("Page %d triggered OCR fallback check (score=%.2f, state=%s)", page_num, quality_score, quality_state)
                    ocr_text, ocr_success, ocr_msg = ocr_engine.ocr_page(
                        pymupdf_page, dpi=config.OCR_DPI
                    )
                    if ocr_msg:
                        page_warnings.append(ocr_msg)

                    if ocr_success and len(ocr_text.strip()) > len(cleaned_text):
                        cleaned_text = ocr_text.strip()
                        extraction_method = ExtractionMethod.OCR
                        ocr_page_ids.append(page_num)
                        # Re-assess quality on OCR output
                        quality_score, _, _ = quality_checker.assess_page_quality(
                            cleaned_text, pymupdf_page
                        )
                    elif not cleaned_text:
                        extraction_method = ExtractionMethod.EMPTY

                page_qualities.append(quality_score)

                page_obj = Page(
                    doc_id=doc_id,
                    page_number=page_num,
                    page_index=page_idx,
                    text=cleaned_text,
                    extraction_method=extraction_method,
                    char_count=len(cleaned_text),
                    quality_score=quality_score,
                    tables=tables,
                    warnings=page_warnings,
                )
                pages.append(page_obj)

            # Step 4: Derive Document-Level Quality
            doc_quality = quality_checker.derive_document_quality(
                page_qualities=page_qualities,
                ocr_page_count=len(ocr_page_ids),
                total_pages=page_count,
            )

            status = DocumentStatus.READY if doc_quality != DocumentQuality.FAILED else DocumentStatus.FAILED

            # Step 5: Construct Document
            meta_dir = config.DOCUMENTS_DIR / doc_id
            meta_path = str(meta_dir / "meta.json") if save_meta else ""
            index_path = str(config.INDEXES_DIR / doc_id)

            document = Document(
                doc_id=doc_id,
                filename=final_filename,
                file_hash=file_hash,
                uploaded_at=datetime.now(timezone.utc),
                page_count=page_count,
                ocr_page_ids=ocr_page_ids,
                chunk_count=0,
                status=status,
                quality=doc_quality,
                pages=pages,
                meta_path=meta_path,
                index_path=index_path,
            )

            # Step 6: Persist metadata if requested
            if save_meta and status == DocumentStatus.READY:
                meta_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(document.model_dump_json(indent=2))
                logger.info("Persisted document metadata to %s", meta_path)

            logger.info(
                "Document ingestion complete: [doc_id=%s, pages=%d, quality=%s, ocr_pages=%d]",
                doc_id,
                page_count,
                doc_quality.value,
                len(ocr_page_ids),
            )
            return document

        finally:
            doc.close()


pdf_extractor = PDFExtractor()


def process_pdf(
    data_or_path: Union[str, Path, bytes],
    filename: Optional[str] = None,
    save_meta: bool = True,
) -> Document:
    """Convenience functional interface for PDF extraction."""
    return pdf_extractor.extract(data_or_path, filename=filename, save_meta=save_meta)
