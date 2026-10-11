"""Numeric entity extraction and preservation utilities for insurance retrieval."""

import re
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass(frozen=True)
class NumericEntity:
    """Represents an extracted numeric entity from query or chunk text."""
    entity_type: str  # percentage, currency, duration_months, duration_days, duration_years, age, number
    raw_text: str
    value: Optional[float]
    unit: Optional[str] = None


# Regex patterns for safety-critical insurance numbers
PERCENT_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\b|(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
CURRENCY_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakh|crore)?\b|\b([\d,]+(?:\.\d+)?)\s*(lakh|crore)\b",
    re.IGNORECASE,
)
MONTHS_PATTERN = re.compile(r"\b(\d+)\s*(?:months?|month)\b", re.IGNORECASE)
DAYS_PATTERN = re.compile(r"\b(\d+)\s*(?:days?|day)\b", re.IGNORECASE)
YEARS_PATTERN = re.compile(r"\b(\d+)\s*(?:years?|yrs?|yr)\b", re.IGNORECASE)
AGE_PATTERN = re.compile(
    r"\b(?:age\s*(?:of)?\s*(\d+)|(\d+)\s*(?:years?\s*(?:old|of\s*age)))\b|\bage\s*(\d+)\b",
    re.IGNORECASE,
)


def _clean_number(val_str: str) -> Optional[float]:
    """Parse comma-separated or plain numbers into float."""
    try:
        return float(val_str.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_numeric_entities(text: str) -> List[NumericEntity]:
    """
    Extract structured numeric entities preserving critical policy numbers:
    - percentages (e.g. '20%', '10%')
    - currency / rupee amounts (e.g. '₹5,00,000', 'Rs. 5,000', '5 lakh')
    - durations in months, days, years (e.g. '36 months', '90 days')
    - age constraints (e.g. 'age 61', '65 years')
    """
    if not text:
        return []

    entities: List[NumericEntity] = []
    seen_spans: Set[str] = set()

    # 1. Percentages
    for match in PERCENT_PATTERN.finditer(text):
        raw = match.group(0).strip()
        num_str = match.group(1) or match.group(2)
        val = _clean_number(num_str)
        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="percentage", raw_text=raw, value=val, unit="%"))
            seen_spans.add(raw.lower())

    # 2. Currency
    for match in CURRENCY_PATTERN.finditer(text):
        raw = match.group(0).strip()
        num_str = match.group(1) or match.group(3)
        unit = (match.group(2) or match.group(4) or "INR").upper()
        val = _clean_number(num_str)
        if val is not None and "LAKH" in unit:
            val = val * 100_000
        elif val is not None and "CRORE" in unit:
            val = val * 10_000_000

        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="currency", raw_text=raw, value=val, unit=unit))
            seen_spans.add(raw.lower())

    # 3. Durations: Months
    for match in MONTHS_PATTERN.finditer(text):
        raw = match.group(0).strip()
        val = _clean_number(match.group(1))
        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="duration_months", raw_text=raw, value=val, unit="months"))
            seen_spans.add(raw.lower())

    # 4. Durations: Days
    for match in DAYS_PATTERN.finditer(text):
        raw = match.group(0).strip()
        val = _clean_number(match.group(1))
        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="duration_days", raw_text=raw, value=val, unit="days"))
            seen_spans.add(raw.lower())

    # 5. Durations: Years
    for match in YEARS_PATTERN.finditer(text):
        raw = match.group(0).strip()
        val = _clean_number(match.group(1))
        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="duration_years", raw_text=raw, value=val, unit="years"))
            seen_spans.add(raw.lower())

    # 6. Age
    for match in AGE_PATTERN.finditer(text):
        raw = match.group(0).strip()
        num_str = match.group(1) or match.group(2) or match.group(3)
        val = _clean_number(num_str)
        if raw.lower() not in seen_spans:
            entities.append(NumericEntity(entity_type="age", raw_text=raw, value=val, unit="age"))
            seen_spans.add(raw.lower())

    return entities


def matches_numeric_context(query: str, chunk_text: str) -> bool:
    """
    Check if chunk text contains matching numbers or units referenced in the query.
    Returns True if an exact or contextual numeric match is detected.
    """
    q_entities = extract_numeric_entities(query)
    c_entities = extract_numeric_entities(chunk_text)

    if not q_entities:
        return False

    c_values_by_type = {}
    for ce in c_entities:
        c_values_by_type.setdefault(ce.entity_type, set()).add(ce.value)

    chunk_lower = chunk_text.lower()

    for qe in q_entities:
        # Direct raw token presence (e.g. '20%', '36 months', 'Rs. 5,000')
        if qe.raw_text.lower() in chunk_lower:
            return True

        # Matching numeric value in same semantic unit
        if qe.value is not None:
            if qe.entity_type in c_values_by_type and qe.value in c_values_by_type[qe.entity_type]:
                return True

            # Matching number as raw digits
            raw_digit = str(int(qe.value)) if qe.value.is_integer() else str(qe.value)
            if re.search(rf"\b{re.escape(raw_digit)}\b", chunk_text):
                # Also verify unit affinity if present
                if qe.unit and qe.unit.lower() in chunk_lower:
                    return True

    return False
