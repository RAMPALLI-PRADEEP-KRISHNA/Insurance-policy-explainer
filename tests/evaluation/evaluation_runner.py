"""Deterministic Evaluation Runner for Insurance Policy Explainer (Phase 7).

Evaluates:
1. Retrieval Recall@1, Recall@3, Recall@5
2. Evidence Sufficiency Accuracy (SUFFICIENT, MARGINAL, INSUFFICIENT, CONFLICTING)
3. Citation Correctness & Provenance
4. Numeric Accuracy
5. Controlled Abstention Accuracy
6. Tamil Query & Numeric Preservation
7. Two-Policy Comparison Engine Invariants
8. Red-Flag Scanner Category & Severity Coverage
9. OCR / Scanned PDF Graceful Failure Path
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.index_store import index_store
from src.retrieval.retriever import Retriever
from src.retrieval.evidence_validator import EvidenceValidator
from src.generation.answer_generator import AnswerGenerator
from src.generation.gemini_client import GeminiClient
from src.comparison.comparison_engine import compare_policies
from src.risk_scanner.red_flag_scanner import red_flag_scanner
from src.models.data_models import (
    Document,
    Chunk,
    EvidenceStatus,
    ComparisonCategory,
    ComparisonCellStatus,
    DocumentQuality,
    DocumentStatus,
    GroundedAnswer,
)


class EvaluationRunner:
    """Orchestrates deterministic evaluation across all pipeline layers."""

    def __init__(self, cases_path: Path):
        self.cases_path = cases_path
        self.retriever = Retriever()
        self.validator = EvidenceValidator()
        self.doc_cache: Dict[str, Document] = {}
        self.chunks_cache: Dict[str, List[Chunk]] = {}

        # Set up generator with mock if live key not available
        self.has_live_gemini = bool(os.getenv("GEMINI_API_KEY"))
        if self.has_live_gemini:
            self.generator = AnswerGenerator()
        else:
            mock_client = MagicMock(spec=GeminiClient)
            mock_client.generate.return_value = (
                "Based on the policy terms, the applicable limit is clearly specified in Section 2."
            )
            self.generator = AnswerGenerator(gemini_client_instance=mock_client)

    def load_document(self, fixture_path: str) -> Tuple[Document, List[Chunk]]:
        """Ingest, chunk, and index a PDF fixture if not already cached."""
        resolved = Path(fixture_path).resolve()
        assert resolved.is_file(), f"Fixture file not found: {resolved}"

        key = str(resolved)
        if key in self.doc_cache:
            return self.doc_cache[key], self.chunks_cache[key]

        doc = process_pdf(resolved, save_meta=False)
        chunks = chunk_document(doc)
        index_store.build_index(doc, chunks=chunks, force_rebuild=True)

        self.doc_cache[key] = doc
        self.chunks_cache[key] = chunks
        return doc, chunks

    def evaluate_qa_cases(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate direct factual, numeric, conditional, negative, unanswerable, conflicting, and Tamil cases."""
        metrics = {
            "total_qa_cases": len(cases),
            "recall_at_1_hits": 0,
            "recall_at_3_hits": 0,
            "recall_at_5_hits": 0,
            "recall_eval_count": 0,
            "sufficiency_matches": 0,
            "citation_valid_count": 0,
            "citation_eval_count": 0,
            "numeric_matches": 0,
            "numeric_eval_count": 0,
            "abstention_matches": 0,
            "abstention_eval_count": 0,
            "tamil_hits": 0,
            "tamil_eval_count": 0,
            "details": [],
        }

        for case in cases:
            case_id = case["id"]
            fixture = case["fixture"]
            query = case["query"]
            lang = case.get("language", "en")
            expected_suff = case["expected_sufficiency"]
            expected_pages = case.get("expected_target_pages", [])
            expected_numbers = case.get("expected_numbers", [])
            should_abstain = case.get("should_abstain", False)

            doc, chunks = self.load_document(fixture)
            results = self.retriever.retrieve(doc_id=doc.doc_id, query=query, top_k=5)
            evidence = self.validator.validate(query, results, doc_id=doc.doc_id)

            # 1. Retrieval Recall@K (only for cases with expected pages)
            r1_hit, r3_hit, r5_hit = False, False, False
            if expected_pages:
                metrics["recall_eval_count"] += 1
                retrieved_pages = [r.page_number for r in results]
                if retrieved_pages and retrieved_pages[0] in expected_pages:
                    r1_hit = True
                    metrics["recall_at_1_hits"] += 1
                if any(p in expected_pages for p in retrieved_pages[:3]):
                    r3_hit = True
                    metrics["recall_at_3_hits"] += 1
                if any(p in expected_pages for p in retrieved_pages[:5]):
                    r5_hit = True
                    metrics["recall_at_5_hits"] += 1

            # 2. Sufficiency Accuracy
            suff_match = (evidence.status.value == expected_suff)
            if suff_match:
                metrics["sufficiency_matches"] += 1

            # 3. Answer Generation & Citations
            answer: GroundedAnswer = self.generator.generate_answer(query, evidence, language=lang)
            citation_valid = True
            if answer.citations:
                metrics["citation_eval_count"] += 1
                chunk_id_set = {c.chunk_id for c in chunks}
                for cit in answer.citations:
                    if cit.document_id != doc.doc_id or cit.chunk_id not in chunk_id_set:
                        citation_valid = False
                if expected_pages and not any(cit.page_number in expected_pages for cit in answer.citations):
                    citation_valid = False
                if citation_valid:
                    metrics["citation_valid_count"] += 1

            # 4. Numeric Accuracy
            num_match = True
            if expected_numbers:
                metrics["numeric_eval_count"] += 1
                combined_text = " ".join([r.text for r in results])
                for exp_num in expected_numbers:
                    val_str = str(exp_num["value"])
                    if val_str not in combined_text:
                        num_match = False
                if num_match:
                    metrics["numeric_matches"] += 1

            # 5. Abstention Accuracy
            abstain_match = True
            if should_abstain:
                metrics["abstention_eval_count"] += 1
                # Must have INSUFFICIENT or MARGINAL evidence and controlled abstention answer
                if evidence.status not in {EvidenceStatus.INSUFFICIENT, EvidenceStatus.MARGINAL} or not answer.answer:
                    abstain_match = False
                if abstain_match:
                    metrics["abstention_matches"] += 1

            # 6. Tamil Evaluation
            tamil_match = True
            if lang == "ta":
                metrics["tamil_eval_count"] += 1
                if not r5_hit or evidence.status != EvidenceStatus.SUFFICIENT:
                    tamil_match = False
                if tamil_match:
                    metrics["tamil_hits"] += 1

            case_passed = suff_match and (r5_hit if expected_pages else True) and (abstain_match if should_abstain else True)
            metrics["details"].append({
                "id": case_id,
                "passed": case_passed,
                "sufficiency": evidence.status.value,
                "expected_sufficiency": expected_suff,
                "recall_at_1": r1_hit,
                "recall_at_5": r5_hit,
                "numeric_match": num_match,
                "abstention_match": abstain_match,
            })

        return metrics

    def evaluate_comparison_cases(self, comp_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate two-policy comparison matrix, normalized values, and isolation."""
        metrics = {
            "total_comparison_cases": len(comp_cases),
            "total_dimensions_evaluated": 0,
            "dimension_matches": 0,
            "isolation_verified": True,
            "details": [],
        }

        for comp in comp_cases:
            doc_a, chunks_a = self.load_document(comp["fixture_a"])
            doc_b, chunks_b = self.load_document(comp["fixture_b"])

            report = compare_policies(doc_a, doc_b, chunks_a=chunks_a, chunks_b=chunks_b)

            # Check strict document isolation in citations
            for row in report.rows:
                for cit in row.policy_a_fact.citations:
                    if cit.document_id != doc_a.doc_id:
                        metrics["isolation_verified"] = False
                for cit in row.policy_b_fact.citations:
                    if cit.document_id != doc_b.doc_id:
                        metrics["isolation_verified"] = False

            # Check expected dimension assertions
            expected_rows = comp.get("expected_rows", {})
            for cat_str, exp in expected_rows.items():
                metrics["total_dimensions_evaluated"] += 1
                cat = ComparisonCategory(cat_str)
                row = next((r for r in report.rows if r.category == cat), None)
                if not row:
                    continue

                dim_match = True
                if "status_a" in exp and row.policy_a_fact.status.value != exp["status_a"]:
                    dim_match = False
                if "val_a" in exp and row.policy_a_fact.value != exp["val_a"]:
                    dim_match = False
                if "status_b" in exp and row.policy_b_fact.status.value != exp["status_b"]:
                    dim_match = False
                if "val_b" in exp and row.policy_b_fact.value != exp["val_b"]:
                    dim_match = False
                if "unit_a" in exp and row.policy_a_fact.unit != exp["unit_a"]:
                    dim_match = False
                if "unit_b" in exp and row.policy_b_fact.unit != exp["unit_b"]:
                    dim_match = False
                if "diff_contains" in exp and exp["diff_contains"].lower() not in row.difference.lower():
                    dim_match = False

                if dim_match:
                    metrics["dimension_matches"] += 1

                metrics["details"].append({
                    "category": cat_str,
                    "matched": dim_match,
                    "row_diff": row.difference,
                })

        return metrics

    def evaluate_risk_scanner(self) -> Dict[str, Any]:
        """Evaluate red-flag scanner categories, severities, and citation integrity."""
        doc, chunks = self.load_document("tests/fixtures/sample_policy_fixture.pdf")
        report = red_flag_scanner.scan(doc, chunks=chunks)
        findings = report.findings

        categories_found = {f.category for f in findings}
        has_waiting = any("WAITING" in c.value for c in categories_found)
        has_room_rent = any("ROOM_RENT" in c.value for c in categories_found)
        has_copay = any("CO_PAYMENT" in c.value for c in categories_found)
        has_exclusions = any("PERMANENT_EXCLUSION" in c.value for c in categories_found)

        all_citations_valid = True
        for f in findings:
            for cit in f.citations:
                if cit.document_id != doc.doc_id or cit.page_number not in [1, 2]:
                    all_citations_valid = False

        return {
            "total_findings": len(findings),
            "categories_covered": len(categories_found),
            "waiting_period_found": has_waiting,
            "room_rent_found": has_room_rent,
            "copay_found": has_copay,
            "exclusions_found": has_exclusions,
            "citations_valid": all_citations_valid,
            "status": "PASSED" if (has_waiting and has_room_rent and has_copay and all_citations_valid) else "FAILED",
        }

    def evaluate_ocr_failure_path(self) -> Dict[str, Any]:
        """Evaluate graceful handling of scanned PDFs with zero extractable native text."""
        scanned_fixture = Path("tests/evaluation/fixtures/scanned_image_policy_fixture.pdf")
        doc = process_pdf(scanned_fixture, save_meta=False)

        # Scanned page has 0 native text
        native_empty = doc.pages[0].char_count == 0 or doc.pages[0].extraction_method.value != "native"
        has_warning_or_failed = len(doc.pages[0].warnings) > 0 or doc.quality == DocumentQuality.FAILED

        return {
            "scanned_pdf": str(scanned_fixture),
            "pages": doc.page_count,
            "char_count": doc.pages[0].char_count,
            "quality": doc.quality.value,
            "status": doc.status.value,
            "native_empty": native_empty,
            "controlled_handling": has_warning_or_failed,
        }

    def run_all(self) -> Dict[str, Any]:
        """Execute full evaluation suite and return consolidated summary."""
        with open(self.cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_metrics = self.evaluate_qa_cases(data.get("cases", []))
        comp_metrics = self.evaluate_comparison_cases(data.get("comparison_cases", []))
        scanner_metrics = self.evaluate_risk_scanner()
        ocr_metrics = self.evaluate_ocr_failure_path()

        # Consolidate rates
        recall_at_1_rate = qa_metrics["recall_at_1_hits"] / max(qa_metrics["recall_eval_count"], 1)
        recall_at_3_rate = qa_metrics["recall_at_3_hits"] / max(qa_metrics["recall_eval_count"], 1)
        recall_at_5_rate = qa_metrics["recall_at_5_hits"] / max(qa_metrics["recall_eval_count"], 1)
        sufficiency_acc = qa_metrics["sufficiency_matches"] / max(qa_metrics["total_qa_cases"], 1)
        citation_acc = qa_metrics["citation_valid_count"] / max(qa_metrics["citation_eval_count"], 1)
        numeric_acc = qa_metrics["numeric_matches"] / max(qa_metrics["numeric_eval_count"], 1)
        abstention_acc = qa_metrics["abstention_matches"] / max(qa_metrics["abstention_eval_count"], 1)
        tamil_acc = qa_metrics["tamil_hits"] / max(qa_metrics["tamil_eval_count"], 1)
        comp_acc = comp_metrics["dimension_matches"] / max(comp_metrics["total_dimensions_evaluated"], 1)

        summary = {
            "qa_cases_count": qa_metrics["total_qa_cases"],
            "recall_at_1": round(recall_at_1_rate * 100, 2),
            "recall_at_3": round(recall_at_3_rate * 100, 2),
            "recall_at_5": round(recall_at_5_rate * 100, 2),
            "evidence_sufficiency_accuracy": round(sufficiency_acc * 100, 2),
            "citation_correctness": round(citation_acc * 100, 2),
            "numeric_accuracy": round(numeric_acc * 100, 2),
            "abstention_accuracy": round(abstention_acc * 100, 2),
            "tamil_accuracy": round(tamil_acc * 100, 2),
            "comparison_dimension_accuracy": round(comp_acc * 100, 2),
            "comparison_isolation_verified": comp_metrics["isolation_verified"],
            "risk_scanner_status": scanner_metrics["status"],
            "ocr_failure_controlled": ocr_metrics["controlled_handling"],
            "live_gemini_tested": self.has_live_gemini,
        }

        report = {
            "summary": summary,
            "qa_metrics": qa_metrics,
            "comparison_metrics": comp_metrics,
            "risk_scanner_metrics": scanner_metrics,
            "ocr_metrics": ocr_metrics,
        }

        # Write to evaluation_report.json
        out_report_path = self.cases_path.parent / "evaluation_report.json"
        with open(out_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    cases_file = Path(__file__).resolve().parent / "evaluation_cases.json"
    runner = EvaluationRunner(cases_file)
    result = runner.run_all()
    print("\n=======================================================")
    print("PHASE 7 DETERMINISTIC EVALUATION RUNNER RESULTS")
    print("=======================================================")
    for k, v in result["summary"].items():
        print(f"  {k:35}: {v}")
    print("=======================================================\n")
