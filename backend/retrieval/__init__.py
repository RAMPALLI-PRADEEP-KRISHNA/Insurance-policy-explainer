"""Retrieval package for hybrid dense-lexical search and evidence validation."""

from backend.retrieval.retriever import Retriever, retriever
from backend.retrieval.evidence_validator import EvidenceValidator, evidence_validator
from backend.retrieval.numeric_utils import (
    NumericEntity,
    extract_numeric_entities,
    matches_numeric_context,
)

__all__ = [
    "Retriever",
    "retriever",
    "EvidenceValidator",
    "evidence_validator",
    "NumericEntity",
    "extract_numeric_entities",
    "matches_numeric_context",
]
