"""Unit tests for file utility functions."""

from pathlib import Path
import pytest

from backend.utils.file_utils import calculate_sha256, validate_pdf_content
from backend.utils.exceptions import InvalidPDFError, FileSizeExceededError, PDFValidationError


def test_calculate_sha256_bytes_and_file(tmp_path):
    """Verify SHA-256 calculation for both raw bytes and files on disk."""
    sample_bytes = b"%PDF-1.4 test document content"
    hash_from_bytes = calculate_sha256(sample_bytes)

    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(sample_bytes)
    hash_from_file = calculate_sha256(test_file)

    assert hash_from_bytes == hash_from_file
    assert len(hash_from_bytes) == 64


def test_validate_pdf_content_from_file(tmp_path):
    """Verify validating a PDF file path."""
    valid_pdf_file = tmp_path / "valid.pdf"
    valid_pdf_file.write_bytes(b"%PDF-1.5\nSample text")

    content, filename = validate_pdf_content(valid_pdf_file)
    assert filename == "valid.pdf"
    assert content.startswith(b"%PDF-")


def test_validate_pdf_file_not_found():
    """Verify FileNotFoundError or PDFValidationError when file path is invalid."""
    with pytest.raises(PDFValidationError):
        validate_pdf_content(Path("non_existent_folder/missing.pdf"))


def test_file_size_limit_exceeded(tmp_path):
    """Verify FileSizeExceededError when file exceeds configured limit."""
    oversized_file = tmp_path / "large.pdf"
    # Create 2 MB file and test with 1 MB limit
    oversized_file.write_bytes(b"%PDF-1.4 " + (b"X" * (2 * 1024 * 1024)))

    with pytest.raises(FileSizeExceededError):
        validate_pdf_content(oversized_file, max_size_mb=1)
