"""Canonical Pydantic data models for Insurance Policy Explainer."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentStatus(str, Enum):
    """Lifecycle status of a document."""
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    READY = "READY"
    FAILED = "FAILED"


class DocumentQuality(str, Enum):
    """Aggregate text extraction quality level of an ingested document."""
    GOOD = "GOOD"                    # >= 90% pages with native text; mean quality_score >= 0.7
    WARNING = "WARNING"              # 70-89% pages good; user warned before querying
    OCR_REQUIRED = "OCR_REQUIRED"    # Significant pages had insufficient native text; OCR applied
    POOR = "POOR"                    # OCR applied or low-yield text; mean quality_score < 0.3
    FAILED = "FAILED"                # Unusable or completely corrupted text


class ExtractionMethod(str, Enum):
    """Method used to extract text from a single page."""
    NATIVE = "NATIVE"    # Native PDF text layer extracted successfully
    OCR = "OCR"          # OCR engine was triggered and extracted text
    EMPTY = "EMPTY"      # Page contains no extractable text even after checks


class ChunkType(str, Enum):
    """Structural classification of a document chunk."""
    HEADING = "HEADING"
    SECTION_HEADING = "SECTION_HEADING"  # Backward-compatible alias
    CLAUSE = "CLAUSE"
    PARAGRAPH = "PARAGRAPH"
    TEXT = "TEXT"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"      # Complete table kept intact without mid-row splitting
    UNKNOWN = "UNKNOWN"


class TableData(BaseModel):
    """Structured representation of a detected table on a page."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    table_index: int = Field(..., ge=0, description="0-indexed table position on this page")
    row_count: int = Field(..., ge=0)
    col_count: int = Field(..., ge=0)
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    markdown: str = Field(default="", description="Rendered markdown representation of table")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="Bounding box (x0, y0, x1, y1)"
    )


class Page(BaseModel):
    """Extracted content and diagnostic quality metadata of a single PDF page."""
    doc_id: str
    page_number: int = Field(..., ge=1, description="1-indexed human-facing page number")
    page_index: int = Field(..., ge=0, description="0-indexed internal page index")
    text: str = Field(default="", description="Cleaned extracted text of the page")
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE
    char_count: int = Field(default=0, ge=0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    tables: List[TableData] = Field(default_factory=list)
    image_path: Optional[str] = Field(default=None, description="Path to rendered image if OCR was used")
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_indices_and_counts(self) -> "Page":
        """Enforce indexing invariants: page_number = page_index + 1."""
        if self.page_number != self.page_index + 1:
            raise ValueError(
                f"page_number ({self.page_number}) must equal page_index + 1 ({self.page_index + 1})"
            )
        if self.char_count != len(self.text):
            self.char_count = len(self.text)
        return self


class Chunk(BaseModel):
    """The fundamental unit of retrieval in the RAG pipeline."""
    chunk_id: str = Field(..., description="Deterministic or unique identifier")
    doc_id: str = Field(..., description="Foreign key to Document.doc_id")
    filename: str = Field(..., description="Source filename for display traceability")
    page_number: int = Field(..., ge=1, description="1-indexed primary/start page of chunk")
    page_numbers: List[int] = Field(..., min_length=1, description="All pages this chunk spans")
    section: Optional[str] = Field(default=None, description="Associated section heading if detected")
    subsection: Optional[str] = Field(default=None, description="Associated subsection heading if detected")
    table_index: Optional[int] = Field(default=None, description="Table index on page if chunk is a TABLE")
    chunk_type: ChunkType = ChunkType.UNKNOWN
    text: str = Field(..., min_length=1, description="Chunk content text")
    char_count: int = Field(default=0, ge=0)
    chunk_index: int = Field(..., ge=0, description="Ordinal position in document")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding")

    @model_validator(mode="after")
    def validate_chunk_invariants(self) -> "Chunk":
        """Enforce chunk invariants."""
        if self.page_number not in self.page_numbers:
            self.page_numbers.insert(0, self.page_number)
        self.char_count = len(self.text)
        return self


class Document(BaseModel):
    """Represents a validated, ingested insurance policy PDF."""
    doc_id: str = Field(..., description="UUID4 document identifier")
    filename: str = Field(..., description="Original uploaded filename")
    file_hash: str = Field(..., description="SHA-256 hash of PDF bytes")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    page_count: int = Field(..., ge=1, description="Total pages in PDF")
    ocr_page_ids: List[int] = Field(
        default_factory=list, description="1-indexed page numbers where OCR was applied"
    )
    chunk_count: int = Field(default=0, ge=0)
    status: DocumentStatus = DocumentStatus.PENDING
    quality: DocumentQuality = DocumentQuality.GOOD
    pages: List[Page] = Field(default_factory=list, description="Extracted pages")
    error_message: Optional[str] = None
    meta_path: str = ""
    index_path: str = ""

    @field_validator("ocr_page_ids")
    @classmethod
    def validate_ocr_page_ids(cls, v: List[int]) -> List[int]:
        """Ensure all OCR page numbers are 1-indexed (>= 1)."""
        for page_num in v:
            if page_num < 1:
                raise ValueError(f"OCR page numbers must be >= 1 (got {page_num})")
        return sorted(list(set(v)))

    @model_validator(mode="after")
    def validate_document_status(self) -> "Document":
        """Validate consistency between status and error message."""
        if self.status == DocumentStatus.FAILED and not self.error_message:
            self.error_message = "Ingestion failed with unspecified error."
        return self


class IndexMetadata(BaseModel):
    """Metadata describing a persistent, document-isolated FAISS vector index."""
    document_id: str = Field(..., description="Foreign key to Document.doc_id")
    filename: str = Field(..., description="Source policy filename")
    file_hash: str = Field(..., description="SHA-256 hash of the policy document")
    chunk_count: int = Field(..., ge=0, description="Total vectors indexed")
    embedding_model: str = Field(..., description="Model identifier used for embedding")
    embedding_dimension: int = Field(..., ge=1, description="Vector dimension (e.g. 768)")
    normalized: bool = Field(default=True, description="Whether vectors are L2-normalized for cosine similarity")
    index_type: str = Field(default="IndexFlatIP", description="FAISS index class identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    index_schema_version: str = Field(default="1", description="Index schema version")


class EvidenceStatus(str, Enum):
    """The four-state validation status for retrieved evidence."""
    SUFFICIENT = "SUFFICIENT"
    MARGINAL = "MARGINAL"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"


class RetrievalResult(BaseModel):
    """Represents a scored, ranked candidate chunk retrieved from a document index."""
    chunk: Chunk = Field(..., description="Underlying retrieved Chunk object")
    vector_score: float = Field(..., description="Raw FAISS cosine similarity score")
    keyword_boost: float = Field(default=0.0, description="Additive keyword/numeric signal boost")
    final_score: float = Field(..., description="Combined retrieval score (vector_score + keyword_boost)")
    rank: int = Field(default=1, ge=1, description="1-based ordinal rank after sorting")
    passed_threshold: bool = Field(default=True, description="Whether this result meets or exceeds the minimum threshold")

    # Authoritative properties directly delegated to Chunk to guarantee metadata provenance
    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def page_numbers(self) -> List[int]:
        return self.chunk.page_numbers

    @property
    def section(self) -> Optional[str]:
        return self.chunk.section

    @property
    def heading(self) -> Optional[str]:
        return self.chunk.subsection or self.chunk.section

    @property
    def chunk_type(self) -> ChunkType:
        return self.chunk.chunk_type

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def raw_similarity(self) -> float:
        return self.vector_score


class Evidence(BaseModel):
    """Structured evidence evaluation output for answer generation."""
    query: str = Field(..., min_length=1, description="Original user query")
    doc_id: str = Field(..., description="Foreign key to Document.doc_id")
    status: EvidenceStatus = Field(..., description="4-state evidence validation status")
    results: List[RetrievalResult] = Field(default_factory=list, description="Ranked retrieval results that passed threshold")
    supporting_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks directly supporting the status")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score in the evidence sufficiency (0.0 to 1.0)")
    validator_notes: Optional[str] = Field(default=None, description="Human-readable explanation of the validation decision")
    detected_conflicts: List[str] = Field(default_factory=list, description="Descriptions of contradictory clauses if status == CONFLICTING")
    evidence_summary: Optional[str] = Field(default=None, description="Concise synthesis of evidence text for answer generation")
    total_candidates: int = Field(default=0, ge=0, description="Total candidates evaluated before thresholding")


class Citation(BaseModel):
    """Authoritative citation pointing strictly to an indexed document chunk."""
    document_id: str = Field(..., description="Foreign key to Document.doc_id")
    chunk_id: str = Field(..., description="Foreign key to Chunk.chunk_id")
    filename: str = Field(..., description="Source policy filename")
    page_number: int = Field(..., ge=1, description="Primary 1-indexed page number")
    page_numbers: List[int] = Field(..., min_length=1, description="All pages spanned by the cited chunk")
    section: Optional[str] = Field(default=None, description="Section heading")
    heading: Optional[str] = Field(default=None, description="Subsection or clause heading")
    source_snippet: str = Field(..., description="Verbatim text snippet from the cited chunk")


class GroundedAnswer(BaseModel):
    """Canonical final answer grounded strictly in verified policy evidence."""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Grounded answer text in plain English or Tamil")
    language: str = Field(default="en", description="Language code: 'en' or 'ta'")
    evidence_status: EvidenceStatus = Field(..., description="Validated evidence state")
    citations: List[Citation] = Field(default_factory=list, description="Authoritative citations derived from chunk metadata")
    supporting_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks directly supporting the answer")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score inherited from retrieval/evidence validation")
    detected_conflicts: List[str] = Field(default_factory=list, description="Descriptions of contradictory clauses if status == CONFLICTING")
    generation_model: Optional[str] = Field(default=None, description="Model used for generation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskCategory(str, Enum):
    """Categorization of restrictive or risk-associated insurance clauses."""
    WAITING_PERIOD = "WAITING_PERIOD"
    PERMANENT_EXCLUSION = "PERMANENT_EXCLUSION"
    SPECIFIC_EXCLUSION = "SPECIFIC_EXCLUSION"
    CO_PAYMENT = "CO_PAYMENT"
    DEDUCTIBLE = "DEDUCTIBLE"
    ROOM_RENT_LIMIT = "ROOM_RENT_LIMIT"
    SUB_LIMIT = "SUB_LIMIT"
    AGE_RESTRICTION = "AGE_RESTRICTION"
    DISEASE_OR_PROCEDURE_LIMIT = "DISEASE_OR_PROCEDURE_LIMIT"
    CLAIM_CONDITION = "CLAIM_CONDITION"
    RESTORATION_RESTRICTION = "RESTORATION_RESTRICTION"
    OTHER_MATERIAL_RESTRICTION = "OTHER_MATERIAL_RESTRICTION"
    # Backward compatibility aliases
    EXCLUSION = "EXCLUSION"
    RESTRICTION = "RESTRICTION"


class RiskSeverity(str, Enum):
    """Review priority level indicating importance for policyholder attention."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RedFlag(BaseModel):
    """A granular restriction or risk clause detected in the policy."""
    flag_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    category: RiskCategory = Field(..., description="Classification category")
    title: str = Field(..., description="Short descriptive title of the restriction")
    description: str = Field(..., description="Clear explanation of the restriction grounded in policy text")
    severity: RiskSeverity = Field(..., description="Priority level for policyholder review")
    document_id: str = Field(..., description="Foreign key to Document.doc_id")
    chunk_id: str = Field(..., description="Foreign key to Chunk.chunk_id")
    page_number: int = Field(..., ge=1, description="Primary 1-indexed page number")
    page_numbers: List[int] = Field(default_factory=list, description="All pages spanned by source chunk")
    section: Optional[str] = Field(default=None, description="Policy section name")
    heading: Optional[str] = Field(default=None, description="Policy heading or clause name")
    source_snippet: str = Field(..., description="Verbatim text snippet from policy chunk")
    citations: List[Citation] = Field(default_factory=list, description="Authoritative citations")


class RiskFinding(BaseModel):
    """Aggregated risk finding per category as specified in DATA_MODEL.md."""
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    doc_id: str = Field(..., description="Foreign key to Document.doc_id")
    category: RiskCategory = Field(..., description="Risk category evaluated")
    found: bool = Field(..., description="Whether clauses in this category were found")
    summary: Optional[str] = Field(default=None, description="Brief description of finding")
    source_chunks: List[Chunk] = Field(default_factory=list, description="Supporting evidence chunks")
    citations: List[Citation] = Field(default_factory=list, description="Authoritative citations")
    severity: Optional[RiskSeverity] = Field(default=None, description="Category severity assessment")


class RedFlagReport(BaseModel):
    """Comprehensive policy scan report aggregating detected red flags."""
    document_id: str = Field(..., description="Foreign key to Document.doc_id")
    filename: str = Field(..., description="Source policy filename")
    total_flags: int = Field(default=0, ge=0, description="Total detected red flags")
    findings: List[RedFlag] = Field(default_factory=list, description="List of granular detected red flags")
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Scan timestamp")
    categories_found: List[RiskCategory] = Field(default_factory=list, description="List of unique categories found")
    disclaimer: str = Field(
        default="This review highlights policy clauses requiring attention. It does not constitute legal or financial advice and does not imply the policy is deficient.",
        description="Mandatory advisory disclaimer"
    )



