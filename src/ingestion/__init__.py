"""Document ingestion and extraction package."""

from src.ingestion.pdf_extractor import PDFExtractor, process_pdf
from src.ingestion.ocr_engine import OCREngine, ocr_engine
from src.ingestion.quality_checker import QualityChecker, quality_checker
from src.ingestion.table_extractor import TableExtractor, table_extractor
from src.ingestion.header_footer import HeaderFooterDetector, header_footer_detector

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
]
