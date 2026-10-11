"""
Two-policy comparison engine for Insurance Policy Explainer.
Constructs a grounded, structured comparison matrix across normalized policy facts,
enforcing strict document isolation and preserving fact uncertainty.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone

from backend.comparison.evidence_extractor import (
    ComparisonEvidenceExtractor,
    comparison_evidence_extractor,
)
from backend.comparison.normalizer import generate_difference_description
from backend.models.data_models import (
    Chunk,
    ComparisonCategory,
    ComparisonReport,
    ComparisonRow,
    Document,
)
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

CATEGORY_LABELS: Dict[ComparisonCategory, str] = {
    ComparisonCategory.WAITING_PERIOD_PRE_EXISTING: "Pre-Existing Disease Waiting Period",
    ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS: "Specific Illness Waiting Period",
    ComparisonCategory.INITIAL_WAITING_PERIOD: "Initial Waiting Period",
    ComparisonCategory.ROOM_RENT_LIMIT: "Room Rent / ICU Capping Limit",
    ComparisonCategory.CO_PAYMENT: "Co-Payment Requirement",
    ComparisonCategory.DEDUCTIBLE: "Deductible Requirement",
    ComparisonCategory.PERMANENT_EXCLUSIONS: "Permanent Exclusions",
    ComparisonCategory.SPECIFIC_EXCLUSIONS: "Specific Exclusions",
    ComparisonCategory.AGE_LIMIT: "Entry & Renewal Age Limits",
    ComparisonCategory.DISEASE_PROCEDURE_LIMIT: "Specific Disease / Procedure Sub-Limits",
    ComparisonCategory.RESTORATION: "Sum Insured Restoration",
    ComparisonCategory.CLAIM_CONDITIONS: "Claim Intimation & Conditions",
}


class ComparisonEngine:
    """
    Compares two insurance policies across canonical dimensions.
    
    Guarantees:
    - Invariant: NOT_MENTIONED != EXPLICITLY_ABSENT.
    - Zero cross-policy leakage: chunks and citations are strictly isolated.
    - Objective difference explanations without competitive ranking ('better', 'winner').
    """

    def __init__(self, extractor_instance: Optional[ComparisonEvidenceExtractor] = None):
        self.extractor = extractor_instance or comparison_evidence_extractor

    def compare(
        self,
        doc_a: Document,
        doc_b: Document,
        chunks_a: Optional[List[Chunk]] = None,
        chunks_b: Optional[List[Chunk]] = None,
    ) -> ComparisonReport:
        """
        Execute comprehensive side-by-side comparison between Policy A and Policy B.

        Args:
            doc_a: Policy A Document.
            doc_b: Policy B Document.
            chunks_a: Optional pre-loaded chunks for Policy A.
            chunks_b: Optional pre-loaded chunks for Policy B.

        Returns:
            ComparisonReport containing structured rows with facts and differences.
        """
        if doc_a.doc_id == doc_b.doc_id:
            raise ValueError(f"Cannot compare document '{doc_a.doc_id}' with itself. Two distinct policies required.")

        rows: List[ComparisonRow] = []

        categories = [
            ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
            ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS,
            ComparisonCategory.INITIAL_WAITING_PERIOD,
            ComparisonCategory.ROOM_RENT_LIMIT,
            ComparisonCategory.CO_PAYMENT,
            ComparisonCategory.DEDUCTIBLE,
            ComparisonCategory.PERMANENT_EXCLUSIONS,
            ComparisonCategory.SPECIFIC_EXCLUSIONS,
            ComparisonCategory.AGE_LIMIT,
            ComparisonCategory.DISEASE_PROCEDURE_LIMIT,
            ComparisonCategory.RESTORATION,
            ComparisonCategory.CLAIM_CONDITIONS,
        ]

        for cat in categories:
            # 1. Extract isolated fact for Policy A
            fact_a = self.extractor.extract_fact(document=doc_a, category=cat, chunks=chunks_a)
            # Verify no cross-policy leakage
            assert fact_a.policy_id == doc_a.doc_id, "Policy A fact must carry Policy A document ID"
            assert all(c.document_id == doc_a.doc_id for c in fact_a.citations), "Policy A citations cannot contain foreign document IDs"

            # 2. Extract isolated fact for Policy B
            fact_b = self.extractor.extract_fact(document=doc_b, category=cat, chunks=chunks_b)
            # Verify no cross-policy leakage
            assert fact_b.policy_id == doc_b.doc_id, "Policy B fact must carry Policy B document ID"
            assert all(c.document_id == doc_b.doc_id for c in fact_b.citations), "Policy B citations cannot contain foreign document IDs"

            # 3. Generate grounded difference description
            diff_text = generate_difference_description(fact_a, fact_b)

            rows.append(
                ComparisonRow(
                    category=cat,
                    label=CATEGORY_LABELS[cat],
                    policy_a_fact=fact_a,
                    policy_b_fact=fact_b,
                    difference=diff_text,
                )
            )

        return ComparisonReport(
            doc_id_a=doc_a.doc_id,
            doc_id_b=doc_b.doc_id,
            filename_a=doc_a.filename,
            filename_b=doc_b.filename,
            rows=rows,
            generated_at=datetime.now(timezone.utc),
        )


comparison_engine = ComparisonEngine()


def compare_policies(
    doc_a: Document,
    doc_b: Document,
    chunks_a: Optional[List[Chunk]] = None,
    chunks_b: Optional[List[Chunk]] = None,
    engine_instance: Optional[ComparisonEngine] = None,
) -> ComparisonReport:
    """Convenience function to compare two policies using the singleton comparison engine."""
    engine = engine_instance or comparison_engine
    return engine.compare(doc_a=doc_a, doc_b=doc_b, chunks_a=chunks_a, chunks_b=chunks_b)
