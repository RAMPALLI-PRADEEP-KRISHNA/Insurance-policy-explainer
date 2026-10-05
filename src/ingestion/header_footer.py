"""Conservative header and footer detection without destructive text deletion."""

import re
from typing import List, Dict, Set, Optional


class HeaderFooterDetector:
    """
    Identifies repeated headers and footers across multi-page documents conservatively.
    Preserves original text and avoids deleting potentially meaningful insurance clauses.
    """

    MIN_PAGES_FOR_DETECTION: int = 3
    REPETITION_THRESHOLD: float = 0.60  # Line must appear on at least 60% of pages

    PAGE_NUMBER_PATTERNS = [
        re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE),
        re.compile(r"^\d+\s*/\s*\d+$"),
        re.compile(r"^-\s*\d+\s*-$"),
        re.compile(r"^\d+$"),
    ]

    def detect_patterns(self, pages_text: List[str]) -> Dict[str, Set[str]]:
        """
        Analyze page texts and identify recurring header/footer candidates.

        Returns:
            Dict with 'headers' and 'footers' sets of detected repeated strings.
        """
        if len(pages_text) < self.MIN_PAGES_FOR_DETECTION:
            return {"headers": set(), "footers": set()}

        header_candidates: Dict[str, int] = {}
        footer_candidates: Dict[str, int] = {}
        total_pages = len(pages_text)

        for text in pages_text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue

            # Check top 2 lines for headers
            for line in lines[:2]:
                if len(line) < 120:  # Headers are typically short lines
                    header_candidates[line] = header_candidates.get(line, 0) + 1

            # Check bottom 2 lines for footers
            for line in lines[-2:]:
                if len(line) < 120:  # Footers are typically short lines or page numbers
                    footer_candidates[line] = footer_candidates.get(line, 0) + 1

        min_occurrences = max(2, int(total_pages * self.REPETITION_THRESHOLD))

        repeated_headers = {
            line for line, count in header_candidates.items()
            if count >= min_occurrences and not self._is_substantive_clause(line)
        }

        repeated_footers = {
            line for line, count in footer_candidates.items()
            if count >= min_occurrences and not self._is_substantive_clause(line)
        }

        return {"headers": repeated_headers, "footers": repeated_footers}

    def _is_substantive_clause(self, line: str) -> bool:
        """Check whether a line looks like an operative insurance clause that must not be stripped."""
        lower = line.lower()
        substantive_keywords = [
            "exclusion", "waiting period", "co-pay", "sub-limit",
            "deductible", "claim", "coverage", "insured", "sum insured",
            "rupees", "incurred", "hospitalization", "benefit",
        ]
        return any(kw in lower for kw in substantive_keywords)

    def is_page_number_line(self, line: str) -> bool:
        """Check if a line matches standalone page number patterns."""
        stripped = line.strip()
        return any(pat.match(stripped) for pat in self.PAGE_NUMBER_PATTERNS)


header_footer_detector = HeaderFooterDetector()
