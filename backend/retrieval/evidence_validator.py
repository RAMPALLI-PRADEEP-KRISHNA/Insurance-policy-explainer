"""Evidence validator module implementing deterministic 4-state evidence evaluation."""

import re
from typing import List, Optional, Tuple, Set
from backend.config import AppConfig, config as default_config
from backend.models.data_models import Evidence, EvidenceStatus, RetrievalResult
from backend.retrieval.numeric_utils import extract_numeric_entities
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Terms indicating coverage inclusion vs exclusion
INCLUSION_TERMS = ["covered", "includes", "included", "payable", "eligible", "admissible", "shall pay"]
EXCLUSION_TERMS = [
    "excluded", "not covered", "not payable", "inadmissible", "shall not pay",
    "except", "permanent exclusion", "exclusions"
]


class EvidenceValidator:
    """
    Deterministic validator evaluating retrieved candidate chunks into one of four states:
    - SUFFICIENT: Evidence directly supports answering the question.
    - MARGINAL: Evidence is related but incomplete or below high-confidence threshold.
    - INSUFFICIENT: Evidence does not contain enough information to answer the question.
    - CONFLICTING: Strong candidate chunks contain contradictory policy statements.
    """

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.config = app_config or default_config

    def _detect_conflicts(
        self, query: str, candidates: List[RetrievalResult]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Detect contradictory statements among top candidate chunks passing threshold:
        1. Conflicting numeric values for the same attribute (e.g. 10% vs 20% co-pay)
        2. Direct coverage inclusion vs exclusion for the same subject
        """
        valid_candidates = [c for c in candidates if c.passed_threshold]
        if len(valid_candidates) < 2:
            return False, [], []

        conflicts: List[str] = []
        conflicting_chunk_ids: Set[str] = set()

        # 1. Numeric conflict check (e.g. co-payment percentages, waiting period numbers)
        q_lower = query.lower()
        qualifiers = [
            "pre-existing", "pre existing", "ped", "specific", "specified", "cataract",
            "initial", "co-pay", "copay", "co-payment", "room rent", "deductible"
        ]
        active_qualifiers = [q for q in qualifiers if q in q_lower]

        # Only compare chunks that address the same specific qualifier/category
        targets = active_qualifiers or ["co-pay", "copay", "co-payment", "room rent", "deductible"]
        for target in targets:
            matching_chunks = [
                c for c in valid_candidates
                if target in c.text.lower()
                or target in (c.section or "").lower()
                or target in (c.heading or "").lower()
            ]
            if len(matching_chunks) >= 2:
                first_chunk = matching_chunks[0]
                first_entities = extract_numeric_entities(first_chunk.text)
                for other_chunk in matching_chunks[1:]:
                    other_entities = extract_numeric_entities(other_chunk.text)
                    for e1 in first_entities:
                        for e2 in other_entities:
                            if (
                                e1.entity_type == e2.entity_type
                                and e1.entity_type in {"percentage", "duration_months", "duration_days"}
                                and e1.value != e2.value
                            ):
                                conflict_msg = (
                                    f"Conflicting {e1.entity_type} for '{target}': "
                                    f"'{e1.raw_text}' in chunk {first_chunk.chunk_id} (Page {first_chunk.page_number}) vs "
                                    f"'{e2.raw_text}' in chunk {other_chunk.chunk_id} (Page {other_chunk.page_number})."
                                )
                                conflicts.append(conflict_msg)
                                conflicting_chunk_ids.add(first_chunk.chunk_id)
                                conflicting_chunk_ids.add(other_chunk.chunk_id)


        # 2. Inclusion vs Exclusion conflict check
        incl_chunks = []
        excl_chunks = []
        for c in valid_candidates:
            c_lower = c.text.lower()
            has_incl = any(term in c_lower for term in INCLUSION_TERMS)
            has_excl = any(term in c_lower for term in EXCLUSION_TERMS)
            if has_incl and not has_excl:
                incl_chunks.append(c)
            elif has_excl and not has_incl:
                excl_chunks.append(c)

        if incl_chunks and excl_chunks:
            # Check if they reference the same specific keyword from the query
            words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", q_lower) if w not in {"what", "does", "policy", "cover"}]
            for word in words:
                matching_incl = [c for c in incl_chunks if word in c.text.lower()]
                matching_excl = [c for c in excl_chunks if word in c.text.lower()]
                if matching_incl and matching_excl:
                    c1, c2 = matching_incl[0], matching_excl[0]
                    conflict_msg = (
                        f"Conflicting coverage for '{word}': "
                        f"Chunk {c1.chunk_id} indicates covered while Chunk {c2.chunk_id} indicates excluded."
                    )
                    conflicts.append(conflict_msg)
                    conflicting_chunk_ids.add(c1.chunk_id)
                    conflicting_chunk_ids.add(c2.chunk_id)

        return (len(conflicts) > 0), conflicts, sorted(list(conflicting_chunk_ids))

    def _is_unsupported_specific_topic(self, query: str, candidates: List[RetrievalResult]) -> bool:
        """
        Check if query asks for a specific benefit, condition, or clause that does not appear
        anywhere in the retrieved candidates.
        """
        q_lower = query.lower()
        specific_terms = [
            "organ donor", "donor", "deductible", "hearing aid", "prosthetics", "robotic surgery",
            "quantum mechanics", "string theory", "quantum", "physics"
        ]
        all_candidate_text = " ".join([c.text.lower() for c in candidates])
        all_candidate_headers = " ".join([f"{c.section or ''} {c.heading or ''}".lower() for c in candidates])
        combined_candidate_scope = f"{all_candidate_text} {all_candidate_headers}"

        for term in specific_terms:
            if term in q_lower and term not in combined_candidate_scope:
                return True
        return False

    def _is_partial_or_incomplete_match(self, query: str, top_result: RetrievalResult) -> bool:
        """
        Check if evidence is related but incomplete for specific entities in the query.
        Example: Query asks for 'cataract waiting period' but top result only discusses general waiting period.
        """
        q_lower = query.lower()
        chunk_lower = top_result.text.lower()

        # Check specific procedure or condition terms in query
        specific_terms = ["cataract", "maternity", "dental", "ayush", "bariatric", "spectacles", "lasik"]
        for term in specific_terms:
            if term in q_lower and term not in chunk_lower:
                return True

        return False

    def validate(
        self,
        query: str,
        results: List[RetrievalResult],
        doc_id: str = "",
    ) -> Evidence:
        """
        Evaluate candidate retrieval results and return structured Evidence.

        Args:
            query: Original query string.
            results: Ranked list of RetrievalResult objects.
            doc_id: Document identifier.

        Returns:
            Evidence object with status in {SUFFICIENT, MARGINAL, INSUFFICIENT, CONFLICTING}.
        """
        if not doc_id and results:
            doc_id = results[0].doc_id

        total_candidates = len(results)
        passed_results = [r for r in results if r.passed_threshold]

        # Edge case: No results or none passed minimum threshold
        if not results or not passed_results:
            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.INSUFFICIENT,
                results=[],
                supporting_chunk_ids=[],
                confidence=0.0,
                validator_notes="Retrieved evidence does not contain sufficient information to answer the query.",
                detected_conflicts=[],
                evidence_summary=None,
                total_candidates=total_candidates,
            )

        top_1 = results[0]
        top_1_score = top_1.final_score

        # 1. Check for Contradictory Evidence (CONFLICTING)
        has_conflict, conflict_msgs, conflict_chunk_ids = self._detect_conflicts(query, passed_results)
        if has_conflict:
            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.CONFLICTING,
                results=passed_results,
                supporting_chunk_ids=conflict_chunk_ids,
                confidence=0.50,
                validator_notes=f"Contradictory policy statements detected across {len(conflict_msgs)} clauses.",
                detected_conflicts=conflict_msgs,
                evidence_summary="\n---\n".join([r.text for r in passed_results if r.chunk_id in conflict_chunk_ids]),
                total_candidates=total_candidates,
            )

        # 2. Check for Specific Unsupported Benefits / Terms (INSUFFICIENT)
        if self._is_unsupported_specific_topic(query, passed_results):
            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.INSUFFICIENT,
                results=[],
                supporting_chunk_ids=[],
                confidence=0.0,
                validator_notes="Retrieved evidence does not contain information on the specific queried benefit or clause.",
                detected_conflicts=[],
                evidence_summary=None,
                total_candidates=total_candidates,
            )

        # 3. Check for Incomplete / Partial Match for Known Policy Benefits (MARGINAL)
        if self._is_partial_or_incomplete_match(query, top_1):
            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.MARGINAL,
                results=passed_results,
                supporting_chunk_ids=[top_1.chunk_id],
                confidence=round(top_1_score, 4),
                validator_notes="Evidence is related to general policy terms but does not clearly identify the specific queried condition.",
                detected_conflicts=[],
                evidence_summary=f"[Page {top_1.page_number}] {top_1.text}",
                total_candidates=total_candidates,
            )

        # 4. Check for Below-Threshold Top Result (INSUFFICIENT)
        if top_1_score < self.config.EVIDENCE_LOW_THRESHOLD:
            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.INSUFFICIENT,
                results=[],
                supporting_chunk_ids=[],
                confidence=round(max(0.0, top_1_score), 4),
                validator_notes="Retrieved evidence does not contain sufficient information to answer the query.",
                detected_conflicts=[],
                evidence_summary=None,
                total_candidates=total_candidates,
            )

        # 5. Check for High Confidence (SUFFICIENT)
        is_high_confidence = (
            top_1_score >= self.config.EVIDENCE_HIGH_THRESHOLD
            or (top_1_score >= 0.50 and top_1.keyword_boost >= 0.10)
            or (top_1_score >= 0.45 and top_1.keyword_boost >= 0.14)
        )
        if is_high_confidence:
            supporting_ids = [r.chunk_id for r in passed_results if r.final_score >= min(self.config.EVIDENCE_HIGH_THRESHOLD, top_1_score)]
            if not supporting_ids:
                supporting_ids = [top_1.chunk_id]

            # Build clean evidence summary with page attribution
            summary_parts = [
                f"[Page {r.page_number}] {r.heading or r.section or 'Clause'}:\n{r.text}"
                for r in passed_results[:2]
            ]
            evidence_summary = "\n\n".join(summary_parts)

            return Evidence(
                query=query,
                doc_id=doc_id,
                status=EvidenceStatus.SUFFICIENT,
                results=passed_results,
                supporting_chunk_ids=supporting_ids,
                confidence=round(top_1_score, 4),
                validator_notes="Evidence directly supports answering the question with high confidence.",
                detected_conflicts=[],
                evidence_summary=evidence_summary,
                total_candidates=total_candidates,
            )

        # 5. Otherwise Between Low and High Threshold (MARGINAL)
        return Evidence(
            query=query,
            doc_id=doc_id,
            status=EvidenceStatus.MARGINAL,
            results=passed_results,
            supporting_chunk_ids=[top_1.chunk_id],
            confidence=round(top_1_score, 4),
            validator_notes="Evidence is related but below high-confidence threshold.",
            detected_conflicts=[],
            evidence_summary=f"[Page {top_1.page_number}] {top_1.text}",
            total_candidates=total_candidates,
        )


evidence_validator = EvidenceValidator()
