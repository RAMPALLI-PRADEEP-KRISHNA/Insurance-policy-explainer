"""Unit tests for citation assembler ensuring strict provenance and safety."""

from backend.generation.citation_assembler import assemble_citations
from backend.models.data_models import (
    Chunk,
    ChunkType,
    Evidence,
    EvidenceStatus,
    RetrievalResult,
)


def _make_result(chunk_id: str, page_number: int, section: str, text: str) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc_xyz",
        filename="policy_doc.pdf",
        page_number=page_number,
        page_numbers=[page_number],
        section=section,
        chunk_type=ChunkType.CLAUSE,
        text=text,
        chunk_index=0,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=0.85,
        keyword_boost=0.10,
        final_score=0.95,
        rank=1,
        passed_threshold=True,
    )


def test_assemble_citations_sufficient_evidence():
    """Verify citations preserve exact chunk metadata without LLM mutation."""
    r1 = _make_result("c_ped_001", 2, "Exclusions", "36 months waiting period for pre-existing diseases.")
    r2 = _make_result("c_init_002", 2, "Exclusions", "30 days initial waiting period.")

    ev = Evidence(
        query="What is the waiting period for pre-existing diseases?",
        doc_id="doc_xyz",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1, r2],
        supporting_chunk_ids=["c_ped_001"],
        confidence=0.95,
    )

    citations = assemble_citations(ev)

    assert len(citations) == 2
    # Verify authoritative provenance
    c1 = citations[0]
    assert c1.document_id == "doc_xyz"
    assert c1.chunk_id == "c_ped_001"
    assert c1.page_number == 2
    assert c1.page_numbers == [2]
    assert c1.section == "Exclusions"
    assert "36 months" in c1.source_snippet


def test_assemble_citations_empty_on_insufficient():
    """Verify citations are strictly empty when evidence status is INSUFFICIENT."""
    r_irrelevant = _make_result("c_irr_001", 1, "Preamble", "General definitions.")

    ev = Evidence(
        query="Does policy cover cosmetic surgery?",
        doc_id="doc_xyz",
        status=EvidenceStatus.INSUFFICIENT,
        results=[r_irrelevant],
        supporting_chunk_ids=[],
        confidence=0.0,
    )

    citations = assemble_citations(ev)
    assert citations == []


def test_assemble_citations_deduplication_and_limit():
    """Verify duplicate chunks are deduplicated and max_citations cap is respected."""
    r1 = _make_result("c_dup_001", 1, "Sec 1", "Clause text 1")
    r2 = _make_result("c_dup_001", 1, "Sec 1", "Clause text 1")  # duplicate ID
    r3 = _make_result("c_unique_002", 2, "Sec 2", "Clause text 2")

    ev = Evidence(
        query="Any query",
        doc_id="doc_xyz",
        status=EvidenceStatus.SUFFICIENT,
        results=[r1, r2, r3],
        supporting_chunk_ids=["c_dup_001", "c_unique_002"],
        confidence=0.90,
    )

    citations = assemble_citations(ev, max_citations=1)
    assert len(citations) == 1
    assert citations[0].chunk_id == "c_dup_001"
