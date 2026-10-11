"""Document ingestion and extraction package."""

from backend.ingestion.pdf_extractor import PDFExtractor, process_pdf
from backend.ingestion.ocr_engine import OCREngine, ocr_engine
from backend.ingestion.quality_checker import QualityChecker, quality_checker
from backend.ingestion.table_extractor import TableExtractor, table_extractor
from backend.ingestion.header_footer import HeaderFooterDetector, header_footer_detector
from backend.ingestion.chunker import Chunker, chunk_document
from backend.ingestion.embedder import Embedder, embedder
from database.index_store import IndexStore, index_store, build_index

__all__ = [
    "PDFExtractor",
    "process_pdf",
    "OCREngine",
    "ocr_engine",
    "QualityChecker",
    "quality_checker",
    "TableExtractor",
    "table_extractor",
    "HeaderFooterDetector",
    "header_footer_detector",
    "Chunker",
    "chunk_document",
    "Embedder",
    "embedder",
    "IndexStore",
    "index_store",
    "build_index",
]
