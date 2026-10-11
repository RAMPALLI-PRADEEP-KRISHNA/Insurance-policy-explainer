"""Citation assembly module guaranteeing provenance and preventing hallucinated citations."""

from typing import List, Set
from backend.models.data_models import Citation, Evidence, EvidenceStatus


def assemble_citations(evidence: Evidence, max_citations: int = 5) -> List[Citation]:
    """
    Assemble authoritative citations directly from stored Chunk metadata.

    CRITICAL SAFETY RULES:
    1. The LLM must NEVER generate or overwrite page numbers.
    2. The LLM must NEVER generate or overwrite chunk IDs.
    3. Citations are derived exclusively from verified Chunk objects.
    4. When evidence is INSUFFICIENT, citations are strictly empty.

    Args:
        evidence: Validated Evidence object from the retrieval/validation pipeline.
        max_citations: Maximum number of citations to include to prevent citation spam.

    Returns:
        List of Citation objects with exact page numbers, section headers, and source snippets.
    """
    if evidence.status == EvidenceStatus.INSUFFICIENT or not evidence.results:
        return []

    citations: List[Citation] = []
    seen_chunk_ids: Set[str] = set()

    # Prioritize chunks explicitly flagged as supporting by the validator
    supporting_ids = set(evidence.supporting_chunk_ids)

    candidate_results = sorted(
        evidence.results,
        key=lambda r: (r.chunk_id in supporting_ids, r.final_score),
        reverse=True,
    )

    for res in candidate_results:
        if res.chunk_id in seen_chunk_ids:
            continue
        if len(citations) >= max_citations:
            break

        seen_chunk_ids.add(res.chunk_id)

        # Snippet: clean excerpt from the chunk text (up to 300 characters)
        raw_text = res.text.strip()
        snippet = raw_text if len(raw_text) <= 300 else raw_text[:297].rstrip() + "..."

        citations.append(
            Citation(
                document_id=res.doc_id,
                chunk_id=res.chunk_id,
                filename=res.filename,
                page_number=res.page_number,
                page_numbers=res.page_numbers,
                section=res.section,
                heading=res.heading,
                source_snippet=snippet,
            )
        )

    return citations
