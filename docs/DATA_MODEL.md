# Insurance Policy Explainer — Data Model

> **Version:** 2.0 (Post-Adversarial Review)
> **Date:** 2026-10-05

All structures are defined using Pydantic v2. This document is the canonical reference
for all data classes used in `models/data_models.py`.

---

## 1. Document

Represents a successfully ingested insurance policy PDF.

```python
class DocumentStatus(str, Enum):
    PENDING    = "PENDING"     # Uploaded, not yet processed
    INGESTING  = "INGESTING"   # Extraction/chunking/indexing in progress
    READY      = "READY"       # Fully indexed and queryable
    FAILED     = "FAILED"      # Ingestion failed; see error_message

class DocumentQuality(str, Enum):
    GOOD         = "GOOD"          # ≥ 90% pages with native text; mean quality_score ≥ 0.7
    WARNING      = "WARNING"       # 70–89% pages good; user warned before querying
    OCR_REQUIRED = "OCR_REQUIRED"  # < 70% pages had native text; OCR was applied
    POOR         = "POOR"          # OCR applied but mean quality_score < 0.3
    FAILED       = "FAILED"        # Cannot extract usable text; ingestion blocked

class Document(BaseModel):
    doc_id:           str             # UUID4, assigned at upload time
    filename:         str             # Original filename as uploaded
    file_hash:        str             # SHA-256 of the PDF bytes (deduplication)
    uploaded_at:      datetime        # UTC timestamp
    page_count:       int             # Total pages (from PyMuPDF)
    ocr_page_ids:     List[int]       # 1-INDEXED pages where OCR fallback was used
    chunk_count:      int             # Total chunks produced
    status:           DocumentStatus
    quality:          DocumentQuality # Computed after extraction
    error_message:    Optional[str]   # Populated only on FAILED
    meta_path:        str             # Absolute path to data/documents/<doc_id>/meta.json
    index_path:       str             # Absolute path to data/indexes/<doc_id>/
```

**Invariants:**
- `doc_id` is immutable once assigned.
- A `Document` with `status != READY` must not be used in retrieval or generation.
- `file_hash` is used to warn the user if the same file is uploaded twice, but a new `doc_id` is always assigned so isolation is maintained.
- `ocr_page_ids` stores **1-indexed** page numbers, consistent with all other user-facing page references.
- `quality == POOR` does not block ingestion but triggers a persistent UI warning banner.
- `quality == FAILED` blocks ingestion and shows an error; no index is written.

---

## 2. Page

Represents the extracted content of a single PDF page.

```python
class ExtractionMethod(str, Enum):
    NATIVE = "NATIVE"   # PyMuPDF text extraction succeeded
    OCR    = "OCR"      # OCR fallback used
    EMPTY  = "EMPTY"    # No text found even after OCR

class Page(BaseModel):
    doc_id:            str
    page_number:       int              # 1-indexed (human-facing)
    page_index:        int              # 0-indexed (internal PyMuPDF index)
    text:              str              # Extracted text (may be empty string)
    extraction_method: ExtractionMethod
    char_count:        int              # len(text)
    quality_score:     float            # 0.0–1.0; heuristic confidence in extraction quality
    image_path:        Optional[str]    # Path to rendered page image (used by OCR; temp file)
```

**Invariants:**
- `page_number` is always `page_index + 1`.
- `quality_score` is computed by `pdf_extractor` and used by `ocr_engine` to decide fallback.
- After OCR completes, `image_path` may be cleaned up (configurable).

---

## 3. Chunk

The fundamental unit of retrieval. A chunk is a contiguous text segment produced by
the chunker from one or more consecutive pages.

```python
class ChunkType(str, Enum):
    SECTION_HEADING = "SECTION_HEADING"
    PARAGRAPH       = "PARAGRAPH"
    LIST_ITEM       = "LIST_ITEM"
    TABLE           = "TABLE"           # Whole table (never split mid-row)
    UNKNOWN         = "UNKNOWN"

class Chunk(BaseModel):
    chunk_id:       str             # UUID4
    doc_id:         str             # Foreign key → Document.doc_id
    filename:       str             # Copied from Document.filename (redundancy for display)
    page_number:    int             # 1-indexed; start page of this chunk
    page_numbers:   List[int]       # All pages this chunk spans (for cross-page chunks)
    section:        Optional[str]   # Nearest preceding section heading, if detected
    chunk_type:     ChunkType
    text:           str             # The actual chunk text
    char_count:     int             # len(text)
    chunk_index:    int             # Position within the document's chunk list (0-based)
    embedding:      Optional[List[float]]  # Set after embedding; not persisted to JSON (stored in FAISS)
```

**Invariants:**
- `doc_id` must match the document that was indexed.
- `page_number` is sourced from `Page.page_number` — never invented.
- `text` is never truncated for storage purposes; the full chunk text is always preserved.
- `embedding` is not serialized to disk as part of the chunk JSON; it is stored in the FAISS index only.
- Minimum and maximum chunk character counts are enforced at chunking time (configurable).
- `chunk_type == TABLE`: the entire detected table is one chunk. Tables are **never split mid-row**.
- `chunk_type == LIST_ITEM`: list items are grouped with their immediately preceding heading.
- Cross-page continuations: `page_number` = start page; `page_numbers` = all spanned pages.

---

## 4. RetrievalResult

The output of the retrieval engine for a single candidate chunk.

```python
class RetrievalResult(BaseModel):
    chunk:            Chunk
    vector_score:     float          # Raw FAISS similarity score (lower L2 = more similar)
    rerank_score:     Optional[float]  # Score from reranker (higher = more relevant); None if reranking skipped
    final_score:      float          # Combined score used for ranking and threshold check
    rank:             int            # Position in ranked results (1-based)
    passed_threshold: bool           # Whether this result exceeds the evidence threshold
```

**Notes:**
- `final_score` is computed by the reranker or falls back to a normalized `vector_score`.
- The reranker is cross-encoder based (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Only `RetrievalResult` objects with `passed_threshold == True` proceed to generation.

---

## 5. Evidence

A collection of retrieval results that has passed validation and will be passed to the LLM.

```python
class EvidenceStatus(str, Enum):
    SUFFICIENT   = "SUFFICIENT"    # Proceed to generation
    MARGINAL     = "MARGINAL"      # Proceed with explicit low-confidence flag
    INSUFFICIENT = "INSUFFICIENT"  # Abstain; do not call LLM
    CONFLICTING  = "CONFLICTING"   # Contradictory clauses found; use conflict-aware prompt

class Evidence(BaseModel):
    query:            str
    doc_id:           str
    results:          List[RetrievalResult]  # Only passed-threshold results
    status:           EvidenceStatus
    total_candidates: int                    # Total candidates before threshold filter
    validator_notes:  Optional[str]          # Human-readable validator decision explanation
```

**Invariants:**
- All results in `evidence.results` share the same `doc_id`.
- If `status == INSUFFICIENT`, `results` may be empty or contain only low-quality results — surfaced for diagnostics only. They must **never** be passed to the LLM.
- If `status == CONFLICTING`, all results are passed to the LLM but a conflict-aware prompt is used. The LLM is instructed to present both clauses and describe their relationship.
- The `EvidenceValidator` is responsible for populating this object.

---

## 6. Citation

A verifiable reference to a specific chunk used to generate an answer.

```python
class Citation(BaseModel):
    citation_id:    str             # UUID4
    answer_id:      str             # Foreign key → Answer.answer_id
    chunk_id:       str             # Foreign key → Chunk.chunk_id
    doc_id:         str             # Foreign key → Document.doc_id
    filename:       str             # Display filename
    page_number:    int             # From Chunk.page_number — never from LLM
    page_numbers:   List[int]       # All pages the chunk spans
    section:        Optional[str]   # Section label from chunk
    source_text:    str             # The exact chunk text used as evidence
    rerank_score:   Optional[float] # Score for display (indicates confidence)
```

**Invariants:**
- `page_number` is always copied from the `Chunk` object, never from the LLM response text.
- `source_text` is the exact verbatim text of the chunk — not a summary.
- `citation_id` is generated by `citation_assembler.py` independently of the LLM.

---

## 7. Answer

The complete response to a user query.

```python
class AnswerType(str, Enum):
    GROUNDED    = "GROUNDED"     # LLM answer with verified citations
    ABSTENTION  = "ABSTENTION"   # Refused to answer due to insufficient evidence
    ERROR       = "ERROR"        # System error prevented answering

class LanguageCode(str, Enum):
    EN = "en"
    TA = "ta"

class ExplanationMode(str, Enum):
    STANDARD          = "STANDARD"
    BEGINNER_FRIENDLY = "BEGINNER_FRIENDLY"

class Answer(BaseModel):
    answer_id:         str               # UUID4
    doc_id:            str
    query:             str
    answer_type:       AnswerType
    text:              str               # The main answer text (English or Tamil)
    english_text:      Optional[str]     # Always stored even for Tamil answers
    language:          LanguageCode
    explanation_mode:  ExplanationMode
    citations:         List[Citation]    # Empty for ABSTENTION and ERROR
    source_chunks:     List[Chunk]       # Chunks passed to LLM (for display)
    evidence_status:   EvidenceStatus    # Copied from Evidence.status
    disclaimer:        str               # Fixed disclaimer text
    generated_at:      datetime
    latency_ms:        Optional[int]     # End-to-end latency for diagnostics
    error_detail:      Optional[str]     # Populated for AnswerType.ERROR
```

**Invariants:**
- `citations` are populated exclusively from `source_chunks` metadata — never from parsing `text`.
- For `AnswerType.ABSTENTION`, `text` contains the abstention message; `citations` is empty.
- `disclaimer` is always appended by `answer_builder.py`; it cannot be omitted.
- `english_text` stores the pre-translation English answer for evaluation and debugging.

---

## 8. RiskFinding

A single detected risk clause within a document.

```python
class RiskCategory(str, Enum):
    EXCLUSION          = "EXCLUSION"
    WAITING_PERIOD     = "WAITING_PERIOD"
    CO_PAYMENT         = "CO_PAYMENT"
    SUB_LIMIT          = "SUB_LIMIT"
    RESTRICTION        = "RESTRICTION"

class RiskFinding(BaseModel):
    finding_id:     str
    doc_id:         str
    category:       RiskCategory
    found:          bool                 # False if not found in document
    summary:        Optional[str]        # Brief description of finding
    source_chunks:  List[Chunk]          # Evidence chunks
    citations:      List[Citation]
    severity:       Optional[str]        # "HIGH" / "MEDIUM" / "LOW" — heuristic
```

---

## 9. ComparisonResult

The structured output of a two-policy comparison.

```python
class ComparisonCellStatus(str, Enum):
    FOUND              = "FOUND"               # Value extracted with evidence
    AMBIGUOUS          = "AMBIGUOUS"           # Topic found; specific value unclear
    NOT_MENTIONED      = "NOT_MENTIONED"       # No relevant chunks found; topic may be absent
    EXPLICITLY_ABSENT  = "EXPLICITLY_ABSENT"   # Chunk explicitly states not covered/not applicable

class ComparisonCell(BaseModel):
    doc_id:      str
    filename:    str
    status:      ComparisonCellStatus
    value:       Optional[str]       # Extracted attribute value
    evidence:    List[RetrievalResult]
    citations:   List[Citation]

class ComparisonRow(BaseModel):
    attribute:   str                 # e.g., "waiting_period_months"
    label:       str                 # Human-readable: "Waiting Period"
    policy_a:    ComparisonCell
    policy_b:    ComparisonCell
    note:        Optional[str]       # Qualitative difference note

class ComparisonResult(BaseModel):
    comparison_id:  str
    doc_id_a:       str
    doc_id_b:       str
    filename_a:     str
    filename_b:     str
    rows:           List[ComparisonRow]
    generated_at:   datetime
    disclaimer:     str
```

**Invariants:**
- `doc_id_a != doc_id_b` is enforced by the comparison engine.
- Evidence for `policy_a` cells comes exclusively from the index for `doc_id_a`.
- `NOT_MENTIONED` means no relevant evidence was found — it does NOT mean the attribute is excluded.
- `EXPLICITLY_ABSENT` means a chunk was found that explicitly states the benefit is not available.
- The UI must display `NOT_MENTIONED` and `EXPLICITLY_ABSENT` with clearly different labels and tooltips.
- The system must **never** convert `NOT_MENTIONED` into `EXPLICITLY_ABSENT` automatically.

---

## 10. EvaluationResult

Records the outcome of running a single evaluation question against the system.

```python
class EvaluationResult(BaseModel):
    eval_id:               str
    question_id:           str              # From eval dataset
    question_text:         str
    question_category:     str
    doc_id:                str
    answer:                Answer
    expected_answer:       str             # From eval dataset
    expected_page_numbers: List[int]       # From eval dataset
    expected_has_answer:   bool            # False for "should abstain" questions

    # Scoring dimensions (see EVALUATION_DESIGN.md)
    answer_correct:        Optional[bool]
    citation_correct:      Optional[bool]
    abstention_correct:    Optional[bool]
    evidence_sufficient:   Optional[bool]
    manual_review_needed:  bool

    # Computed metrics
    answer_score:          Optional[float]  # 0.0–1.0
    citation_f1:           Optional[float]
    notes:                 Optional[str]
    evaluated_at:          datetime
```

---

## Relationships Summary

```
Document ──< Page
Document ──< Chunk
Chunk ──< RetrievalResult
[RetrievalResult]+ ──> Evidence
Evidence ──> Answer
Answer ──< Citation
Citation → Chunk
Answer ──> [Chunk] (source_chunks)
Document ──< RiskFinding
RiskFinding ──< Citation
(Document, Document) ──> ComparisonResult
ComparisonResult ──< ComparisonRow
ComparisonRow ──> (ComparisonCell, ComparisonCell)
ComparisonCell ──< Citation
Answer ──> EvaluationResult
```
