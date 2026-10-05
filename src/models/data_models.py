"""Canonical Pydantic data models for Insurance Policy Explainer."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple
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

