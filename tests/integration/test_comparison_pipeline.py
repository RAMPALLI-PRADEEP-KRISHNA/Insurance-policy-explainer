"""Integration tests for the two-policy comparison pipeline using synthetic PDF fixtures."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.app import compare_uploaded_policies, render_comparison_report
from src.comparison.comparison_engine import compare_policies
from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.models.data_models import (
    ComparisonCategory,
    ComparisonCellStatus,
    DocumentQuality,
    DocumentStatus,
)


@pytest.fixture(scope="module")
def ingested_comparison_fixtures():
    """Ingest and chunk real synthetic policy fixtures A and B."""
    path_a = Path("tests/fixtures/policy_a_fixture.pdf")
    path_b = Path("tests/fixtures/policy_b_fixture.pdf")

    assert path_a.is_file(), "policy_a_fixture.pdf missing"
    assert path_b.is_file(), "policy_b_fixture.pdf missing"

    doc_a = process_pdf(path_a, save_meta=False)
    assert doc_a.status == DocumentStatus.READY
    chunks_a = chunk_document(doc_a)

    doc_b = process_pdf(path_b, save_meta=False)
    assert doc_b.status == DocumentStatus.READY
    chunks_b = chunk_document(doc_b)

    return doc_a, chunks_a, doc_b, chunks_b


def test_comparison_pipeline_on_synthetic_fixtures(ingested_comparison_fixtures):
    """
    End-to-end integration test comparing Policy A vs Policy B:
    - PED waiting: 36m vs 24m
    - Specific illness: 24m vs 12m
    - Room rent: 1%/day vs Rs. 5,000/day (different basis)
    - Co-pay: 10% (age 61+) vs 20%
    - Deductible: NOT_MENTIONED in A vs FOUND in B
    - Permanent exclusions: A excludes aesthetic surgery vs B covers
    """
    doc_a, chunks_a, doc_b, chunks_b = ingested_comparison_fixtures
    report = compare_policies(doc_a, doc_b, chunks_a=chunks_a, chunks_b=chunks_b)

    assert report.doc_id_a == doc_a.doc_id
    assert report.doc_id_b == doc_b.doc_id
    assert len(report.rows) == 12

    # 1. PED Waiting Period
    row_ped = next(r for r in report.rows if r.category == ComparisonCategory.WAITING_PERIOD_PRE_EXISTING)
    assert row_ped.policy_a_fact.status == ComparisonCellStatus.FOUND
    assert row_ped.policy_a_fact.value == 36.0
    assert row_ped.policy_b_fact.status == ComparisonCellStatus.FOUND
    assert row_ped.policy_b_fact.value == 24.0
    assert "24" in row_ped.difference and "36" in row_ped.difference

    # 2. Specific Illness Waiting Period
    row_spec = next(r for r in report.rows if r.category == ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS)
    assert row_spec.policy_a_fact.status == ComparisonCellStatus.FOUND
    assert row_spec.policy_a_fact.value == 24.0
    assert row_spec.policy_b_fact.status == ComparisonCellStatus.FOUND
    assert row_spec.policy_b_fact.value == 12.0

    # 3. Room Rent Limit (Different limit basis)
    row_rr = next(r for r in report.rows if r.category == ComparisonCategory.ROOM_RENT_LIMIT)
    assert row_rr.policy_a_fact.status == ComparisonCellStatus.FOUND
    assert row_rr.policy_a_fact.unit == "percent"
    assert row_rr.policy_b_fact.status == ComparisonCellStatus.FOUND
    assert row_rr.policy_b_fact.unit == "INR"
    assert "different limit basis" in row_rr.difference.lower()

    # 4. Co-Payment (Conditions differ)
    row_copay = next(r for r in report.rows if r.category == ComparisonCategory.CO_PAYMENT)
    assert row_copay.policy_a_fact.status == ComparisonCellStatus.FOUND
    assert "61" in (row_copay.policy_a_fact.conditions or "")
    assert row_copay.policy_b_fact.status == ComparisonCellStatus.FOUND
    assert "conditions differ" in row_copay.difference.lower()

    # 5. Deductible (Critical invariant: NOT_MENTIONED != EXPLICITLY_ABSENT)
    row_ded = next(r for r in report.rows if r.category == ComparisonCategory.DEDUCTIBLE)
    assert row_ded.policy_a_fact.status == ComparisonCellStatus.NOT_MENTIONED
    assert row_ded.policy_a_fact.status != ComparisonCellStatus.EXPLICITLY_ABSENT
    assert row_ded.policy_b_fact.status == ComparisonCellStatus.FOUND
    assert row_ded.policy_b_fact.value == 10000.0
    assert "no corresponding deductible clause was identified in policy a" in row_ded.difference.lower()

    # 6. Permanent Exclusions
    row_excl = next(r for r in report.rows if r.category == ComparisonCategory.PERMANENT_EXCLUSIONS)
    assert row_excl.policy_a_fact.status == ComparisonCellStatus.FOUND
    assert row_excl.policy_b_fact.status == ComparisonCellStatus.EXPLICITLY_ABSENT
    assert "permanently excludes aesthetic surgery" in row_excl.difference.lower()


def test_ui_service_compare_uploaded_policies(ingested_comparison_fixtures):
    """Verify the decoupled service function in src.app works cleanly."""
    doc_a, chunks_a, doc_b, chunks_b = ingested_comparison_fixtures
    report = compare_uploaded_policies(doc_a, doc_b, chunks_a, chunks_b)

    assert report.doc_id_a == doc_a.doc_id
    assert report.doc_id_b == doc_b.doc_id
    assert len(report.rows) == 12
    assert report.disclaimer is not None


def test_render_comparison_report_mock(ingested_comparison_fixtures):
    """Verify render_comparison_report executes in Streamlit without exceptions."""
    doc_a, chunks_a, doc_b, chunks_b = ingested_comparison_fixtures
    report = compare_uploaded_policies(doc_a, doc_b, chunks_a, chunks_b)

    with patch("streamlit.columns") as mock_cols, \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.markdown") as mock_md, \
         patch("streamlit.caption") as mock_caption, \
         patch("streamlit.write") as mock_write, \
         patch("streamlit.expander") as mock_expander:

        mock_cols.side_effect = lambda spec: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]
        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_comparison_report(report)
        assert mock_md.call_count >= 10
        assert mock_info.call_count >= 2


