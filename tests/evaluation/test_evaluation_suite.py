"""Automated test suite verifying Phase 7 deterministic evaluation metrics."""

from pathlib import Path
import pytest


from tests.evaluation.evaluation_runner import EvaluationRunner


@pytest.fixture(scope="module")
def evaluation_results():
    from backend.config import config
    config.LLM_PROVIDER = "fake"
    import os
    os.environ.pop("GEMINI_API_KEY", None)
    """Run full deterministic evaluation suite."""
    cases_file = Path("tests/evaluation/evaluation_cases.json")
    assert cases_file.is_file(), "evaluation_cases.json missing"
    runner = EvaluationRunner(cases_file)
    return runner.run_all()


def test_retrieval_recall_metrics(evaluation_results):
    """Verify Recall@1, Recall@3, and Recall@5 meet high reliability thresholds."""
    summary = evaluation_results["summary"]
    assert summary["recall_at_1"] >= 90.0, f"Recall@1 too low: {summary['recall_at_1']}%"
    assert summary["recall_at_3"] >= 95.0, f"Recall@3 too low: {summary['recall_at_3']}%"
    assert summary["recall_at_5"] >= 95.0, f"Recall@5 too low: {summary['recall_at_5']}%"


def test_evidence_sufficiency_accuracy(evaluation_results):
    """Verify four-state evidence sufficiency classification accuracy."""
    summary = evaluation_results["summary"]
    assert summary["evidence_sufficiency_accuracy"] >= 90.0


def test_citation_correctness_and_provenance(evaluation_results):
    """Verify that citations are 100% grounded and derived from supporting chunks."""
    summary = evaluation_results["summary"]
    assert summary["citation_correctness"] >= 95.0


def test_numeric_accuracy(evaluation_results):
    """Verify exact preservation of critical policy numbers."""
    summary = evaluation_results["summary"]
    assert summary["numeric_accuracy"] == 100.0


def test_controlled_abstention_accuracy(evaluation_results):
    """Verify system strictly abstains without hallucinating on unsupported queries."""
    summary = evaluation_results["summary"]
    assert summary["abstention_accuracy"] == 100.0


def test_dataset_scale_and_balance(evaluation_results):
    """Verify dataset scale meets minimum of 30 answerable and 15 unanswerable English cases."""
    summary = evaluation_results["summary"]
    assert summary["qa_cases_count"] >= 45, f"Total QA cases too low: {summary['qa_cases_count']}"
    assert summary["answerable_cases_count"] >= 30, f"Answerable cases too low: {summary['answerable_cases_count']}"
    assert summary["unanswerable_cases_count"] >= 15, f"Unanswerable cases too low: {summary['unanswerable_cases_count']}"


def test_two_policy_comparison_accuracy(evaluation_results):
    """Verify comparison dimensions, isolation, and absence invariants."""
    summary = evaluation_results["summary"]
    assert summary["comparison_dimension_accuracy"] >= 90.0
    assert summary["comparison_isolation_verified"] is True


def test_risk_scanner_coverage(evaluation_results):
    """Verify Red-Flag Scanner identifies major risk categories with valid citations."""
    summary = evaluation_results["summary"]
    assert summary["risk_scanner_status"] == "PASSED"


def test_ocr_failure_path_controlled(evaluation_results):
    """Verify scanned PDF failure path is graceful and non-silent."""
    summary = evaluation_results["summary"]
    assert summary["ocr_failure_controlled"] is True





