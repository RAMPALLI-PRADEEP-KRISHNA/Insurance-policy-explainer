"""Unit tests for conservative header/footer detection."""

import pytest
from backend.ingestion.header_footer import HeaderFooterDetector


def test_detect_repeated_headers_and_footers():
    """Verify that repeated titles across multi-page documents are detected as headers."""
    detector = HeaderFooterDetector()

    pages = [
        "STAR COMPREHENSIVE INSURANCE POLICY UIN-12345\nPage 1 of 3\nThis is content on page one.\nFooter: Star Health",
        "STAR COMPREHENSIVE INSURANCE POLICY UIN-12345\nPage 2 of 3\nThis is content on page two.\nFooter: Star Health",
        "STAR COMPREHENSIVE INSURANCE POLICY UIN-12345\nPage 3 of 3\nThis is content on page three.\nFooter: Star Health",
    ]

    patterns = detector.detect_patterns(pages)
    headers = patterns["headers"]
    footers = patterns["footers"]

    assert "STAR COMPREHENSIVE INSURANCE POLICY UIN-12345" in headers
    assert "Footer: Star Health" in footers


def test_preserves_substantive_clauses():
    """Verify that operative insurance clauses are never categorized as disposable headers/footers."""
    detector = HeaderFooterDetector()

    # Even if repeated, an exclusion or waiting period clause must be preserved
    pages = [
        "Exclusion: 30 days initial waiting period applies to all claims.\nContent 1",
        "Exclusion: 30 days initial waiting period applies to all claims.\nContent 2",
        "Exclusion: 30 days initial waiting period applies to all claims.\nContent 3",
    ]

    patterns = detector.detect_patterns(pages)
    assert len(patterns["headers"]) == 0


def test_page_number_pattern_matching():
    """Verify regex matching for common page number representations."""
    detector = HeaderFooterDetector()

    assert detector.is_page_number_line("Page 4") is True
    assert detector.is_page_number_line("Page 12 of 45") is True
    assert detector.is_page_number_line("3 / 10") is True
    assert detector.is_page_number_line("- 5 -") is True
    assert detector.is_page_number_line("42") is True
    assert detector.is_page_number_line("Section 4: Exclusions") is False
