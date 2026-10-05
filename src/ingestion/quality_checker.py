"""Deterministic text extraction quality evaluation for pages and documents."""

import re
import string
from typing import List, Tuple, Dict, Any
import pymupdf

from src.config import config
from src.models.data_models import DocumentQuality, ExtractionMethod
from src.utils.logging_utils import logger


class QualityThresholds:
    """Configurable thresholds for heuristic quality assessment."""
    MIN_EXPECTED_CHARS_PER_PAGE: int = 200
    OCR_TRIGGER_CHAR_THRESHOLD: int = 80
    MIN_ALPHANUMERIC_RATIO: float = 0.45
    MAX_WHITESPACE_RATIO: float = 0.65
    MAX_GARBAGE_CHAR_RATIO: float = 0.15


class QualityChecker:
    """Evaluates text extraction quality on page and document levels."""

    def __init__(self, thresholds: QualityThresholds = QualityThresholds()):
        self.t = thresholds

    def assess_page_quality(
        self,
        text: str,
        page: pymupdf.Page,
    ) -> Tuple[float, str, List[str]]:
        """
        Assess extraction quality of raw text extracted from a PyMuPDF page.

        Returns:
            Tuple of (quality_score [0.0 - 1.0], recommended_state, warnings_list)
            where recommended_state is one of: "GOOD", "WARNING", "OCR_REQUIRED", "EMPTY"
        """
        warnings: List[str] = []
        raw_len = len(text)
        stripped = text.strip()

        # Check page visual content (images or graphics)
        has_images = len(page.get_images()) > 0

        # Case 1: Page has zero or minimal text
        if raw_len == 0 or len(stripped) == 0:
            if has_images:
                warnings.append("Page contains image(s) with no extractable native text layer. OCR required.")
                return 0.0, "OCR_REQUIRED", warnings
            else:
                warnings.append("Page appears visually and textually empty.")
                return 1.0, "EMPTY", warnings

        # Calculate character composition metrics
        alpha_count = sum(1 for c in text if c.isalnum())
        whitespace_count = sum(1 for c in text if c.isspace())
        printable_count = sum(1 for c in text if c in string.printable or ord(c) > 127)
        non_printable_count = raw_len - printable_count
        replacement_char_count = text.count("\ufffd")

        alpha_ratio = alpha_count / max(raw_len, 1)
        whitespace_ratio = whitespace_count / max(raw_len, 1)
        garbage_ratio = (non_printable_count + replacement_char_count) / max(raw_len, 1)

        # Check for garbled / corrupted font decoding (e.g. lots of  or unprintable symbols)
        if garbage_ratio > self.t.MAX_GARBAGE_CHAR_RATIO:
            warnings.append(
                f"High proportion of corrupted/replacement characters ({garbage_ratio:.1%}). Text may be decoded incorrectly."
            )
            if has_images:
                return 0.1, "OCR_REQUIRED", warnings
            return 0.2, "POOR", warnings

        # Case 2: Very short text on an image-bearing page (e.g., scanned stamp or scanned document with few letters)
        if raw_len < self.t.OCR_TRIGGER_CHAR_THRESHOLD and has_images:
            warnings.append(
                f"Very low character count ({raw_len}) on page with images; native text is insufficient."
            )
            return 0.1, "OCR_REQUIRED", warnings

        # Case 3: Very low alphanumeric content
        if alpha_ratio < self.t.MIN_ALPHANUMERIC_RATIO and raw_len > 30:
            warnings.append(
                f"Low ratio of alphanumeric characters ({alpha_ratio:.1%}). Text contains predominantly symbols or formatting artifact."
            )
            score = max(0.2, alpha_ratio)
            return score, "WARNING", warnings

        # Case 4: High whitespace ratio
        if whitespace_ratio > self.t.MAX_WHITESPACE_RATIO:
            warnings.append(f"Excessive whitespace ratio ({whitespace_ratio:.1%}).")

        # Compute continuous quality score
        length_factor = min(1.0, raw_len / self.t.MIN_EXPECTED_CHARS_PER_PAGE)
        score = (
            0.5 * length_factor +
            0.3 * alpha_ratio +
            0.2 * (1.0 - min(1.0, garbage_ratio * 4))
        )
        score = max(0.0, min(1.0, round(score, 3)))

        if score >= 0.70 and not warnings:
            return score, "GOOD", warnings
        elif score >= 0.40:
            return score, "WARNING", warnings
        else:
            if has_images:
                warnings.append("Low text quality score; page may benefit from OCR fallback.")
                return score, "OCR_REQUIRED", warnings
            return score, "WARNING", warnings

    def derive_document_quality(
        self,
        page_qualities: List[float],
        ocr_page_count: int,
        total_pages: int,
        failed: bool = False,
    ) -> DocumentQuality:
        """
        Derive document-level extraction quality deterministically from page scores.

        Rules:
        - FAILED: Unreadable document, zero pages, or fatal extraction failure.
        - GOOD: >= 90% pages with quality_score >= 0.70 and mean >= 0.70.
        - WARNING: 70-89% pages good or minor warnings.
        - OCR_REQUIRED: >= 30% of pages required OCR fallback.
        - POOR: Average quality score < 0.30 or high failure rate.
        """
        if failed or total_pages == 0:
            return DocumentQuality.FAILED

        if not page_qualities:
            return DocumentQuality.FAILED

        mean_quality = sum(page_qualities) / len(page_qualities)
        good_page_count = sum(1 for q in page_qualities if q >= 0.70)
        good_ratio = good_page_count / total_pages
        ocr_ratio = ocr_page_count / total_pages

        if mean_quality < 0.30:
            return DocumentQuality.POOR

        if ocr_ratio >= 0.30:
            return DocumentQuality.OCR_REQUIRED

        if good_ratio >= 0.90 and mean_quality >= 0.70:
            return DocumentQuality.GOOD

        if good_ratio >= 0.70 and mean_quality >= 0.50:
            return DocumentQuality.WARNING

        return DocumentQuality.WARNING


quality_checker = QualityChecker()
