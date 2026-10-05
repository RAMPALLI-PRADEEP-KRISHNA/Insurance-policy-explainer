"""Custom typed exceptions for Insurance Policy Explainer."""


class InsuranceExplainerError(Exception):
    """Base exception for all Insurance Policy Explainer errors."""
    pass


class PDFValidationError(InsuranceExplainerError):
    """Raised when an uploaded file fails basic PDF validation."""
    pass


class InvalidPDFError(PDFValidationError):
    """Raised when a file does not have a valid PDF header or format."""
    pass


class UnreadablePDFError(PDFValidationError):
    """Raised when a PDF cannot be read, parsed, or is encrypted/password-protected."""
    pass


class FileSizeExceededError(PDFValidationError):
    """Raised when a PDF exceeds the maximum allowed file size."""
    pass


class PageLimitExceededError(PDFValidationError):
    """Raised when a PDF exceeds the maximum allowed page count."""
    pass


class ExtractionError(InsuranceExplainerError):
    """Raised when text or content extraction fails unexpectedly."""
    pass


class OCRError(InsuranceExplainerError):
    """Base class for OCR-related errors."""
    pass


class TesseractNotFoundError(OCRError):
    """Raised when Tesseract OCR binary is required but not found on the system."""
    def __init__(self, message: str = "Tesseract OCR binary not found."):
        install_hint = (
            "\nTo install Tesseract OCR on Windows, run in an administrative shell:\n"
            "  winget install UB-Mannheim.TesseractOCR\n"
            "Or set the TESSERACT_CMD environment variable to the tesseract.exe path."
        )
        super().__init__(f"{message}{install_hint}")


class EmbeddingError(InsuranceExplainerError):
    """Raised when text embedding fails."""
    pass


class IndexStoreError(InsuranceExplainerError):
    """Base exception for FAISS vector index store operations."""
    pass


class IndexNotFoundError(IndexStoreError):
    """Raised when an index for a specific document ID does not exist."""
    pass


class StaleIndexError(IndexStoreError):
    """Raised when an existing index is stale (mismatched file hash, chunk count, model, or schema)."""
    pass


class CorruptedIndexError(IndexStoreError):
    """Raised when an index file or chunk metadata cannot be read or deserialized."""
    pass
