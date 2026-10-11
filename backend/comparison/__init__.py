"""Policy comparison engine package."""

from backend.comparison.comparison_engine import (
    ComparisonEngine,
    comparison_engine,
    compare_policies,
)
from backend.comparison.evidence_extractor import (
    ComparisonEvidenceExtractor,
    comparison_evidence_extractor,
)
from backend.comparison.normalizer import (
    ComparisonNormalizer,
    comparison_normalizer,
    generate_difference_description,
)

__all__ = [
    "ComparisonEngine",
    "comparison_engine",
    "compare_policies",
    "ComparisonEvidenceExtractor",
    "comparison_evidence_extractor",
    "ComparisonNormalizer",
    "comparison_normalizer",
    "generate_difference_description",
]
