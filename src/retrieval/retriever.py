from src.utils.timing import timed
"""Retriever module orchestrating dense vector search and keyword signal boosting."""

import re
from typing import List, Optional, Tuple, Set
from src.config import AppConfig, config as default_config
from src.ingestion.embedder import Embedder, embedder as default_embedder
from src.ingestion.index_store import IndexStore, index_store as default_index_store
from src.models.data_models import Chunk, ChunkType, RetrievalResult
from src.retrieval.numeric_utils import matches_numeric_context
from src.utils.exceptions import IndexNotFoundError, RetrievalError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Specific concept keyword families: distinguishing terms for specific clauses/conditions
SPECIFIC_CONCEPT_FAMILIES = {
    "pre_existing": [
        "pre-existing", "pre existing", "ped", "ஏற்கனவே உள்ள", "முந்தைய நோய்"
    ],
    "specific_illness": [
        "specific disease", "specific illness", "specific illnesses", "specific",
        "specified procedures", "cataract", "குறிப்பிட்ட நோய்", "குறிப்பிட்ட நோய்கள்"
    ],
    "initial_waiting": [
        "initial waiting", "30 days", "ஆரம்ப காத்திருப்பு"
    ],
    "co_pay": [
        "co-pay", "copay", "co-payment", "copayment", "coinsurance", "இணை கட்டணம்", "இணைக்கட்டணம்"
    ],
    "room_rent": [
        "room rent", "icu charges", "icu", "அறை வாடகை", "அறை கட்டணம்"
    ],
    "sub_limit": [
        "sub-limit", "sublimit", "capped at", "limit per", "உள் வரம்பு"
    ],
    "deductible": [
        "deductible", "விலக்குத் தொகை"
    ],
    "maternity": [
        "maternity", "newborn", "pregnancy", "பிரசவம்"
    ],
    "sum_insured": [
        "sum insured", "coverage amount", "policy limit", "காப்பீட்டுத் தொகை"
    ],
    "age_limit": [
        "entry age", "maximum age", "minimum age", "age limit", "age criteria", "18 years", "65 years", "வயது வரம்பு", "வயது"
    ],
}

# General domain terms
GENERAL_DOMAIN_FAMILIES = {
    "waiting_period": [
        "waiting period", "initial waiting", "waiting", "காத்திருப்பு காலம்", "காத்திருப்புக் காலம்"
    ],
    "exclusion": [
        "exclusion", "exclusions", "excluded", "permanent exclusion", "விலக்கு", "விலக்குகள்"
    ],
}

# Table relevance query signals
TABLE_QUERY_SIGNALS = [
    "room rent", "rent", "limit", "amount", "co-pay", "copay", "copayment",
    "percentage", "percent", "%", "₹", "rs", "inr", "charges", "share",
    "deductible", "schedule", "benefit", "table", "rate", "fee",
    "வரம்பு", "வாடகை", "கட்டணம்", "அட்டவணை", "விழுக்காடு", "சதவீதம்"
]


class Retriever:
    """
    Production-grade hybrid retriever combining:
    1. Local multilingual MPNet embeddings (English & Tamil)
    2. Exact FAISS IndexFlatIP cosine similarity
    3. Transparent, bounded keyword and numeric signal boosting
    """

    def __init__(
        self,
        index_store: Optional[IndexStore] = None,
        embedder_instance: Optional[Embedder] = None,
        app_config: Optional[AppConfig] = None,
    ):
        self.index_store = index_store or default_index_store
        self.embedder = embedder_instance or default_embedder
        self.config = app_config or default_config

    def _compute_keyword_boost(self, query: str, chunk: Chunk) -> Tuple[float, dict]:
        """
        Compute transparent, bounded keyword and numeric signal boosts:
        - Numeric match: +0.10 if chunk contains numbers/units referenced in query
        - Specific concept match: +0.10 if chunk matches distinguishing query concept
        - General domain term match: +0.05 if chunk matches general insurance domain terms
        - Table boost: +0.05 if chunk is a TABLE and query seeks numeric limits
        Total boost is capped at MAX_KEYWORD_BOOST (0.15).
        """
        q_lower = query.lower()
        chunk_text_lower = chunk.text.lower()
        chunk_meta_lower = f"{chunk.section or ''} {chunk.subsection or ''}".lower()
        full_chunk_scope = f"{chunk_meta_lower} {chunk_text_lower}"

        q_norm = q_lower.replace("-", " ")
        scope_norm = full_chunk_scope.replace("-", " ")

        signals = {"numeric": 0.0, "specific_concept": 0.0, "general_domain": 0.0, "table": 0.0}

        # 1. Numeric Match
        if matches_numeric_context(query, chunk.text):
            signals["numeric"] = self.config.KEYWORD_BOOST_NUMERIC

        # 2. Specific Concept Match (Differentiator)
        for _, keywords in SPECIFIC_CONCEPT_FAMILIES.items():
            if any(kw in q_norm or kw in q_lower for kw in keywords):
                if any(kw in scope_norm or kw in full_chunk_scope for kw in keywords):
                    signals["specific_concept"] = self.config.KEYWORD_BOOST_RISK
                    break

        # 3. General Domain Match
        for _, keywords in GENERAL_DOMAIN_FAMILIES.items():
            if any(kw in q_norm or kw in q_lower for kw in keywords):
                if any(kw in scope_norm or kw in full_chunk_scope for kw in keywords):
                    signals["general_domain"] = 0.05
                    break

        # 4. Table Signal
        if chunk.chunk_type == ChunkType.TABLE:
            if any(signal in q_norm or signal in q_lower for signal in TABLE_QUERY_SIGNALS):
                signals["table"] = self.config.KEYWORD_BOOST_TABLE

        raw_boost = signals["numeric"] + signals["specific_concept"] + signals["general_domain"] + signals["table"]
        final_boost = min(raw_boost, self.config.MAX_KEYWORD_BOOST)
        return round(final_boost, 4), signals

    def retrieve(
        self,
        doc_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve and rank candidate chunks for a query from a specific document.

        Args:
            doc_id: Foreign key to document FAISS index.
            query: User question in English or Tamil.
            top_k: Number of final results to return (defaults to RETRIEVAL_FINAL_K).

        Returns:
            List of RetrievalResult objects sorted by final_score descending.

        Raises:
            ValueError: If query is empty or whitespace-only.
            IndexNotFoundError: If index for doc_id does not exist.
            RetrievalError: If embedding or search fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        clean_query = query.strip()
        final_k = top_k or self.config.RETRIEVAL_FINAL_K

        # 1. Verify index existence (strict document isolation)
        if not self.index_store.index_exists(doc_id):
            raise IndexNotFoundError(f"No index found for document '{doc_id}'.")

        # 2. Embed Query (using same multilingual model & normalization)
        try:
            with timed("query embedding"):
                q_vec = self.embedder.embed_query(clean_query)
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            raise RetrievalError(f"Query embedding failed: {str(e)}") from e

        # 3. Retrieve Candidate Pool from FAISS
        with timed("FAISS retrieval"):
            candidate_k = max(self.config.RETRIEVAL_CANDIDATE_K, final_k * 2)
            candidates = self.index_store.search(doc_id, q_vec, top_k=candidate_k)

        if not candidates:
            logger.warning("No candidate chunks retrieved for doc_id '%s'", doc_id)
            return []

        # 4. Apply Signal Boosting & Build RetrievalResults
        scored_results: List[RetrievalResult] = []
        seen_chunk_ids: Set[str] = set()

        for chunk, raw_score in candidates:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)

            vector_score = round(float(raw_score), 4)
            keyword_boost, _ = self._compute_keyword_boost(clean_query, chunk)
            final_score = round(min(1.0, vector_score + keyword_boost), 4)
            passed = final_score >= self.config.EVIDENCE_LOW_THRESHOLD

            scored_results.append(
                RetrievalResult(
                    chunk=chunk,
                    vector_score=vector_score,
                    keyword_boost=keyword_boost,
                    final_score=final_score,
                    rank=1,  # will be assigned after sort
                    passed_threshold=passed,
                )
            )

        # 5. Sort by final_score descending (tie-breaker: vector_score descending)
        scored_results.sort(key=lambda r: (r.final_score, r.vector_score), reverse=True)

        # 6. Assign ordinal ranks and truncate to top_k
        final_results = scored_results[:final_k]
        for rank_idx, result in enumerate(final_results, start=1):
            result.rank = rank_idx

        return final_results


retriever = Retriever()

