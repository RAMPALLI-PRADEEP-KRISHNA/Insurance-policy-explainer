"""
Targeted evidence extractor for two-policy comparison.
Extracts grounded facts from isolated document chunks, validating status
(FOUND, AMBIGUOUS, NOT_MENTIONED, EXPLICITLY_ABSENT) and assembling citations strictly from Chunk metadata.
"""

import re
from typing import Dict, List, Optional, Tuple

from backend.comparison.normalizer import comparison_normalizer
from backend.ingestion.chunker import chunk_document
from database.index_store import index_store, IndexStore
from backend.models.data_models import (
    Citation,
    Chunk,
    ChunkType,
    ComparisonCategory,
    ComparisonCellStatus,
    ComparisonFact,
    Document,
)
from backend.retrieval.numeric_utils import extract_numeric_entities
from backend.retrieval.retriever import retriever, Retriever
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Category query templates for dense / targeted retrieval
CATEGORY_QUERIES: Dict[ComparisonCategory, str] = {
    ComparisonCategory.WAITING_PERIOD_PRE_EXISTING: "pre-existing disease waiting period continuous coverage PED",
    ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS: "specific illness disease waiting period specified cataract hernia",
    ComparisonCategory.INITIAL_WAITING_PERIOD: "initial waiting period 30 days from inception",
    ComparisonCategory.ROOM_RENT_LIMIT: "room rent limit percentage sum insured per day ICU charges",
    ComparisonCategory.CO_PAYMENT: "co-payment copay percentage insured share age",
    ComparisonCategory.DEDUCTIBLE: "deductible aggregate voluntary compulsory amount",
    ComparisonCategory.PERMANENT_EXCLUSIONS: "permanent exclusions not covered shall not be liable aesthetic surgery",
    ComparisonCategory.SPECIFIC_EXCLUSIONS: "specific exclusions investigation evaluation dietary rest cure",
    ComparisonCategory.AGE_LIMIT: "entry age maximum renewal age eligibility limits",
    ComparisonCategory.DISEASE_PROCEDURE_LIMIT: "sub-limit modern treatments cataract surgery cap AYUSH",
    ComparisonCategory.RESTORATION: "restoration recharge reinstatement sum insured refill",
    ComparisonCategory.CLAIM_CONDITIONS: "claim intimation notice within hours submission documents",
}


class ComparisonEvidenceExtractor:
    """Extracts isolated, grounded comparison facts for a single policy document."""

    def __init__(
        self,
        retriever_instance: Optional[Retriever] = None,
        index_store_instance: Optional[IndexStore] = None,
    ):
        self.retriever = retriever_instance or retriever
        self.index_store = index_store_instance or index_store

    def extract_fact(
        self,
        document: Document,
        category: ComparisonCategory,
        chunks: Optional[List[Chunk]] = None,
    ) -> ComparisonFact:
        """
        Extract a structured ComparisonFact for a specific category from the document.
        Maintains strict document isolation.
        """
        doc_chunks = self._resolve_chunks(document, chunks)
        if not doc_chunks:
            return self._make_not_mentioned_fact(document, category)

        # 1. Filter candidate chunks relevant to this category
        candidate_chunks = self._find_candidate_chunks(document, category, doc_chunks)
        if not candidate_chunks:
            return self._make_not_mentioned_fact(document, category)

        # 2. Check for Ambiguity / Contradiction (e.g. 10% vs 20% co-pay)
        ambiguous_fact = self._check_ambiguity(document, category, candidate_chunks)
        if ambiguous_fact:
            return ambiguous_fact

        # 3. Check for Explicit Absence (e.g. "no co-payment", "nil deductible", "no room rent limit")
        absent_fact = self._check_explicit_absence(document, category, candidate_chunks)
        if absent_fact:
            return absent_fact

        # 4. Extract Positive Fact (FOUND)
        return self._extract_positive_fact(document, category, candidate_chunks)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _resolve_chunks(self, document: Document, chunks: Optional[List[Chunk]]) -> List[Chunk]:
        """Load chunks ensuring document isolation."""
        if chunks is not None:
            return [c for c in chunks if c.doc_id == document.doc_id]

        if self.index_store.index_exists(document.doc_id):
            try:
                _, loaded_chunks, _ = self.index_store.load_index(document.doc_id)
                return [c for c in loaded_chunks if c.doc_id == document.doc_id]
            except Exception as e:
                logger.warning("Error loading chunks from index store: %s", e)

        fresh_chunks = chunk_document(document)
        return [c for c in fresh_chunks if c.doc_id == document.doc_id]

    def _find_candidate_chunks(
        self,
        document: Document,
        category: ComparisonCategory,
        all_chunks: List[Chunk],
    ) -> List[Chunk]:
        """Find candidate chunks matching the category via keyword filtering and hybrid retrieval."""
        # 1. Direct keyword match
        kw_chunks = [c for c in all_chunks if self._matches_category_keywords(category, c.text)]
        if kw_chunks:
            return kw_chunks

        # 2. Dense retrieval fallback if index exists
        if self.index_store.index_exists(document.doc_id):
            try:
                query = CATEGORY_QUERIES.get(category, category.value.replace("_", " "))
                results = self.retriever.retrieve(doc_id=document.doc_id, query=query, top_k=5)
                # Keep chunks where score passes threshold AND matches category keywords
                matched_ids = {
                    r.chunk_id for r in results
                    if r.final_score >= 0.50 and self._matches_category_keywords(category, r.text)
                }
                if matched_ids:
                    return [c for c in all_chunks if c.chunk_id in matched_ids]
            except Exception as e:
                logger.warning("Retriever search fallback in comparison extractor: %s", e)

        return []

    def _matches_category_keywords(self, category: ComparisonCategory, text: str) -> bool:
        """Heuristic keyword matching for category identification."""
        t = text.lower()
        if category == ComparisonCategory.WAITING_PERIOD_PRE_EXISTING:
            return ("pre-existing" in t or "ped" in t) and ("waiting" in t or "coverage" in t)
        elif category == ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS:
            return ("specific" in t or "specified" in t or "cataract" in t) and ("waiting" in t or "months" in t)
        elif category == ComparisonCategory.INITIAL_WAITING_PERIOD:
            return "initial waiting" in t or ("30 days" in t and "waiting" in t)
        elif category == ComparisonCategory.ROOM_RENT_LIMIT:
            return "room rent" in t or "icu" in t or "room nursing" in t
        elif category == ComparisonCategory.CO_PAYMENT:
            return "co-pay" in t or "co-payment" in t or "copay" in t
        elif category == ComparisonCategory.DEDUCTIBLE:
            return "deductible" in t
        elif category == ComparisonCategory.PERMANENT_EXCLUSIONS:
            return "permanent exclusion" in t or "shall not be liable" in t or "aesthetic surgery" in t or "hazardous" in t
        elif category == ComparisonCategory.SPECIFIC_EXCLUSIONS:
            return "specific exclusion" in t or "investigation" in t or "rest cure" in t
        elif category == ComparisonCategory.AGE_LIMIT:
            return "entry age" in t or "renewal age" in t or "age limit" in t
        elif category == ComparisonCategory.DISEASE_PROCEDURE_LIMIT:
            return "sub-limit" in t or "cataract limit" in t or "bariatric" in t or "ayush" in t
        elif category == ComparisonCategory.RESTORATION:
            return "restoration" in t or "reinstatement" in t or "refill" in t
        elif category == ComparisonCategory.CLAIM_CONDITIONS:
            return "claim intimation" in t or "notice of claim" in t or "within 24 hours" in t
        return False

    def _check_ambiguity(
        self,
        document: Document,
        category: ComparisonCategory,
        candidates: List[Chunk],
    ) -> Optional[ComparisonFact]:
        """Detect contradictory or ambiguous clauses within the same policy."""
        if category == ComparisonCategory.CO_PAYMENT:
            # Check for multiple distinct non-conditional co-pay percentages
            percentages = set()
            pct_chunks = []
            for c in candidates:
                if re.search(r"\bco[- ]?pay(?:ment)?\b", c.text, re.IGNORECASE):
                    matches = re.findall(r"(\d{1,2})\s*%", c.text)
                    if matches:
                        percentages.update(matches)
                        pct_chunks.append(c)

            # If contradictory percentages exist without age conditioning (e.g. 10% and 20%)
            if len(percentages) > 1 and not any("age" in c.text.lower() for c in pct_chunks):
                citations = [self._make_citation(document, c, c.text[:200]) for c in pct_chunks[:2]]
                return ComparisonFact(
                    policy_id=document.doc_id,
                    filename=document.filename,
                    category=category,
                    status=ComparisonCellStatus.AMBIGUOUS,
                    description=f"Contradictory co-payment values ({', '.join(sorted(percentages))}%) detected across policy clauses.",
                    source_chunk_ids=[c.chunk_id for c in pct_chunks],
                    citations=citations,
                    confidence=0.5,
                )

        return None

    def _check_explicit_absence(
        self,
        document: Document,
        category: ComparisonCategory,
        candidates: List[Chunk],
    ) -> Optional[ComparisonFact]:
        """Detect explicit negative statements (EXPLICITLY_ABSENT)."""
        for chunk in candidates:
            text = chunk.text
            t_lower = text.lower()

            if category == ComparisonCategory.CO_PAYMENT:
                if re.search(r"\b(?:no|zero|nil|0%)\s*co[- ]?pay", t_lower):
                    return self._make_absent_fact(document, category, chunk, "No co-payment is applicable under policy terms.")

            elif category == ComparisonCategory.DEDUCTIBLE:
                if re.search(r"\b(?:no|nil|zero)\s*deductible\b", t_lower):
                    return self._make_absent_fact(document, category, chunk, "No deductible is applicable under policy terms.")

            elif category == ComparisonCategory.ROOM_RENT_LIMIT:
                if re.search(r"\b(?:no\s+room\s+rent\s+cap|no\s+sub[- ]?limit\s+on\s+room|any\s+room\s+category)\b", t_lower):
                    return self._make_absent_fact(document, category, chunk, "No room rent capping limit applies (any room category covered).")

            elif category == ComparisonCategory.PERMANENT_EXCLUSIONS:
                # If someone asks if an exclusion applies or aesthetic surgery is covered:
                if "aesthetic surgery is covered" in t_lower or "cosmetic surgery is covered" in t_lower:
                    # In exclusions category, stating that it is covered means it is not permanently excluded
                    return self._make_absent_fact(document, category, chunk, "Aesthetic surgery is covered subject to policy conditions.")

        return None

    def _extract_positive_fact(
        self,
        document: Document,
        category: ComparisonCategory,
        candidates: List[Chunk],
    ) -> ComparisonFact:
        """Extract normalized values and build FOUND fact."""
        # Inspect individual table rows or chunk lines
        best_chunk = candidates[0]
        best_snippet = best_chunk.text
        val, unit, orig_val, cond = None, None, None, None

        # Search for exact line or row in candidate chunks
        for chunk in candidates:
            # Check individual lines
            lines = [l.strip() for l in chunk.text.split("\n") if l.strip()]
            for line in lines:
                if not self._matches_category_keywords(category, line):
                    continue

                best_chunk = chunk
                best_snippet = line

                if category in {
                    ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
                    ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS,
                    ComparisonCategory.INITIAL_WAITING_PERIOD,
                }:
                    v, u, o, c = comparison_normalizer.normalize_duration(line)
                    if v is not None:
                        val, unit, orig_val, cond = v, u, o, c
                        break

                elif category == ComparisonCategory.ROOM_RENT_LIMIT:
                    # Check percent first, then currency
                    v, u, o, c = comparison_normalizer.normalize_percentage(line)
                    if v is not None:
                        val, unit, orig_val, cond = v, u, o, c
                        break
                    v, u, o, c = comparison_normalizer.normalize_currency(line)
                    if v is not None:
                        val, unit, orig_val, cond = v, u, o, c
                        break

                elif category == ComparisonCategory.CO_PAYMENT:
                    v, u, o, c = comparison_normalizer.normalize_percentage(line)
                    if v is not None:
                        val, unit, orig_val, cond = v, u, o, c
                        break

                elif category == ComparisonCategory.DEDUCTIBLE:
                    v, u, o, c = comparison_normalizer.normalize_currency(line)
                    if v is not None:
                        val, unit, orig_val, cond = v, u, o, c
                        break

                elif category == ComparisonCategory.AGE_LIMIT:
                    entities = extract_numeric_entities(line)
                    ages = [e for e in entities if e.entity_type in {"age", "duration_years"} and e.value is not None]
                    if ages:
                        max_age = max(a.value for a in ages)
                        val = float(max_age)
                        unit = "years"
                        orig_val = f"{int(max_age)} years"
                        cond = "entry age limit"
                        break

            if val is not None:
                break

        # If numeric category has no value and no matching clause, preserve uncertainty
        if val is None and category in {
            ComparisonCategory.WAITING_PERIOD_PRE_EXISTING,
            ComparisonCategory.WAITING_PERIOD_SPECIFIC_ILLNESS,
            ComparisonCategory.INITIAL_WAITING_PERIOD,
            ComparisonCategory.ROOM_RENT_LIMIT,
            ComparisonCategory.CO_PAYMENT,
            ComparisonCategory.DEDUCTIBLE,
            ComparisonCategory.AGE_LIMIT,
        }:
            return self._make_not_mentioned_fact(document, category)

        # Description text
        desc = best_snippet.strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."

        citations = [self._make_citation(document, best_chunk, best_snippet)]

        return ComparisonFact(
            policy_id=document.doc_id,
            filename=document.filename,
            category=category,
            status=ComparisonCellStatus.FOUND,
            value=val,
            unit=unit,
            original_value=orig_val,
            conditions=cond,
            description=desc,
            source_chunk_ids=[best_chunk.chunk_id],
            citations=citations,
            confidence=0.9,
        )

    def _make_absent_fact(
        self,
        document: Document,
        category: ComparisonCategory,
        chunk: Chunk,
        description: str,
    ) -> ComparisonFact:
        """Create an EXPLICITLY_ABSENT fact with supporting citation."""
        citation = self._make_citation(document, chunk, chunk.text[:200])
        return ComparisonFact(
            policy_id=document.doc_id,
            filename=document.filename,
            category=category,
            status=ComparisonCellStatus.EXPLICITLY_ABSENT,
            value=None,
            unit=None,
            original_value=None,
            conditions=None,
            description=description,
            source_chunk_ids=[chunk.chunk_id],
            citations=[citation],
            confidence=0.95,
        )

    def _make_not_mentioned_fact(
        self,
        document: Document,
        category: ComparisonCategory,
    ) -> ComparisonFact:
        """Create a NOT_MENTIONED fact. Preserves uncertainty (never claims explicit absence)."""
        cat_name = category.value.replace("_", " ").lower()
        return ComparisonFact(
            policy_id=document.doc_id,
            filename=document.filename,
            category=category,
            status=ComparisonCellStatus.NOT_MENTIONED,
            value=None,
            unit=None,
            original_value=None,
            conditions=None,
            description=f"No explicit {cat_name} clause identified in the policy.",
            source_chunk_ids=[],
            citations=[],
            confidence=0.0,
        )

    def _make_citation(self, document: Document, chunk: Chunk, snippet: str) -> Citation:
        """Build an authoritative Citation strictly from chunk metadata."""
        clean_snippet = snippet.strip().replace("\n", " ")
        if len(clean_snippet) > 250:
            clean_snippet = clean_snippet[:247] + "..."

        return Citation(
            document_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            filename=document.filename,
            page_number=chunk.page_number,
            page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
            section=chunk.section,
            heading=chunk.subsection or chunk.section,
            source_snippet=clean_snippet,
        )


comparison_evidence_extractor = ComparisonEvidenceExtractor()
