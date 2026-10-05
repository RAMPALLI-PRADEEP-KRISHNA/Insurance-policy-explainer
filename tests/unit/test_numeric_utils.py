"""Unit tests for numeric entity extraction and preservation utilities."""

from src.retrieval.numeric_utils import (
    NumericEntity,
    extract_numeric_entities,
    matches_numeric_context,
)


def test_extract_percentages():
    """Verify extraction of percentages in various formats."""
    text = "Co-payment is 20% for senior citizens and 10 percent for network hospitals."
    entities = extract_numeric_entities(text)
    types = [e.entity_type for e in entities]
    values = [e.value for e in entities]

    assert "percentage" in types
    assert 20.0 in values
    assert 10.0 in values


def test_extract_currency():
    """Verify extraction of rupee amounts and lakh notation."""
    text = "Sum Insured: Rs. 5,00,000 with ICU room rent capped at ₹5,000 per day or 5 lakh."
    entities = extract_numeric_entities(text)
    curr_entities = [e for e in entities if e.entity_type == "currency"]

    assert len(curr_entities) >= 2
    raw_texts = [e.raw_text for e in curr_entities]
    assert any("5,00,000" in t for t in raw_texts)
    assert any("5,000" in t for t in raw_texts)


def test_extract_durations_months_days_years():
    """Verify extraction of safety-critical duration units (months, days, years)."""
    text = "Waiting periods: 30 days initial, 24 months specific disease, 36 months pre-existing, and 3 years renewal."
    entities = extract_numeric_entities(text)

    months = [e for e in entities if e.entity_type == "duration_months"]
    days = [e for e in entities if e.entity_type == "duration_days"]
    years = [e for e in entities if e.entity_type == "duration_years"]

    assert len(months) == 2
    month_vals = {e.value for e in months}
    assert 24.0 in month_vals
    assert 36.0 in month_vals

    assert len(days) == 1
    assert days[0].value == 30.0

    assert len(years) == 1
    assert years[0].value == 3.0


def test_extract_ages():
    """Verify age entity extraction."""
    text = "Entry age is 18 years to 65 years. Co-pay applies to age 61 and above."
    entities = extract_numeric_entities(text)
    age_entities = [e for e in entities if e.entity_type == "age"]

    assert len(age_entities) >= 1
    age_vals = {e.value for e in age_entities}
    assert 61.0 in age_vals or 18.0 in age_vals


def test_matches_numeric_context():
    """Verify matching numeric context between query and candidate text."""
    # Percentage match
    q1 = "Is there a 20% co-pay?"
    c1 = "A 20% co-pay applies to all non-network claims."
    c1_diff = "A 10% co-pay applies."
    assert matches_numeric_context(q1, c1) is True

    # Month duration match
    q2 = "Is the waiting period 36 months?"
    c2 = "Pre-existing disease coverage starts after 36 months."
    assert matches_numeric_context(q2, c2) is True

    # Day duration match
    q3 = "What is covered after 90 days?"
    c3 = "Initial waiting period is 30 days from policy inception."
    assert matches_numeric_context(q3, c3) is False  # 90 vs 30 days


def test_empty_and_no_numeric_handling():
    """Verify empty input returns empty list without error."""
    assert extract_numeric_entities("") == []
    assert extract_numeric_entities("No numbers in this sentence.") == []
    assert matches_numeric_context("No numbers in query", "No numbers in chunk") is False
