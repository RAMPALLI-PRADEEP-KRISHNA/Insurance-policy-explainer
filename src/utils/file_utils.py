"""File utility functions for hashing, validation, and safe path handling."""

import hashlib
from pathlib import Path
from typing import Union, Tuple
from src.utils.exceptions import (
    InvalidPDFError,
    FileSizeExceededError,
    PDFValidationError,
)

PDF_MAGIC_BYTES = b"%PDF-"


def calculate_sha256(data_or_path: Union[str, Path, bytes]) -> str:
    """Calculate SHA-256 hash of bytes or a file path."""
    hasher = hashlib.sha256()
    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif isinstance(data_or_path, bytes):
        hasher.update(data_or_path)
    else:
        raise ValueError("Expected file path (str/Path) or bytes.")
    return hasher.hexdigest()


def validate_pdf_content(
    data_or_path: Union[str, Path, bytes],
    max_size_mb: int = 50,
) -> Tuple[bytes, str]:
    """
    Validate that input data or file is a valid PDF within size limits.

    Returns:
        Tuple of (pdf_bytes, filename_or_identifier)

    Raises:
        PDFValidationError, InvalidPDFError, FileSizeExceededError
    """
    filename = "document.pdf"
    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        filename = path.name
        if not path.is_file():
            raise PDFValidationError(f"Specified PDF file does not exist: {path}")
        file_size = path.stat().st_size
        max_bytes = max_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise FileSizeExceededError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed {max_size_mb} MB."
            )
        if file_size == 0:
            raise InvalidPDFError("PDF file is completely empty (0 bytes).")
        pdf_bytes = path.read_bytes()
    elif isinstance(data_or_path, bytes):
        pdf_bytes = data_or_path
        if len(pdf_bytes) == 0:
            raise InvalidPDFError("Provided PDF byte buffer is empty.")
        max_bytes = max_size_mb * 1024 * 1024
        if len(pdf_bytes) > max_bytes:
            raise FileSizeExceededError(
                f"Byte data size ({len(pdf_bytes) / (1024 * 1024):.1f} MB) exceeds maximum {max_size_mb} MB."
            )
    else:
        raise PDFValidationError(f"Invalid PDF input type: {type(data_or_path)}")

    # Check magic header bytes
    if not pdf_bytes.startswith(PDF_MAGIC_BYTES):
        # Scan first 1024 bytes in case of leading whitespace or BOM
        header_sample = pdf_bytes[:1024]
        if PDF_MAGIC_BYTES not in header_sample:
            raise InvalidPDFError(
                "File header does not match PDF format (missing '%PDF-' header)."
            )

    return pdf_bytes, filename
