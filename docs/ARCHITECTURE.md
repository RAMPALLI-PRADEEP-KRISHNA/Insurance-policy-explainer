# Insurance Policy Explainer — Architecture

> **Version:** 2.0 (Post-Adversarial Review)
> **Date:** 2026-10-05
> **Status:** Revised — supersedes v1.0

---

## 1. System Overview

The Insurance Policy Explainer is a Retrieval-Augmented Generation (RAG) application
that enables users to upload insurance policy PDF documents, ask questions in English
or Tamil, and receive strictly grounded answers with verifiable page-level citations.

The system is intentionally architected as a set of plain Python modules orchestrated
by a Streamlit front-end. No LangChain or LlamaIndex abstractions are introduced;
each pipeline stage is explicit, testable, and independently replaceable.

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONT-END                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
│  │  Upload UI  │  │  Q&A Panel   │  │ Risk Panel │  │ Compare  │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  └────┬─────┘  │
└─────────┼────────────────┼────────────────┼───────────────┼────────┘
          │                │                │               │
          ▼                ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER (Python)                      │
│                                                                     │
│  ┌──────────────────────┐    ┌────────────────────────────────────┐ │
│  │  INGESTION PIPELINE  │    │         QUERY PIPELINE             │ │
│  │                      │    │                                    │ │
│  │  ┌────────────────┐  │    │  ┌──────────┐  ┌──────────────┐  │ │
│  │  │  PDF Extractor │  │    │  │  Query   │  │  Retrieval   │  │ │
│  │  │  (PyMuPDF +    │  │    │  │ Processor│  │  Engine      │  │ │
│  │  │   OCR fallback)│  │    │  └─────┬────┘  └──────┬───────┘  │ │
│  │  └───────┬────────┘  │    │        │               │          │ │
│  │          │           │    │        ▼               ▼          │ │
│  │  ┌───────▼────────┐  │    │  ┌──────────────────────────┐    │ │
│  │  │    Chunker     │  │    │  │   Evidence Validator     │    │ │
│  │  └───────┬────────┘  │    │  └──────────┬───────────────┘    │ │
│  │          │           │    │             │                     │ │
│  │  ┌───────▼────────┐  │    │             ▼                     │ │
│  │  │    Embedder    │  │    │  ┌──────────────────────────┐    │ │
│  │  └───────┬────────┘  │    │  │      LLM Generator       │    │ │
│  │          │           │    │  └──────────┬───────────────┘    │ │
│  │  ┌───────▼────────┐  │    │             │                     │ │
│  │  │  FAISS Index   │  │    │             ▼                     │ │
│  │  │  (per-doc)     │  │    │  ┌──────────────────────────┐    │ │
│  │  └────────────────┘  │    │  │   Citation Assembler     │    │ │
│  └──────────────────────┘    │  └──────────────────────────┘    │ │
│                              └────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────┐   ┌────────────────────────────────────┐ │
│  │  RISK SCANNER         │   │    COMPARISON ENGINE               │ │
│  │  (clause detector)    │   │    (structured attribute diff)     │ │
│  └───────────────────────┘   └────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     EVALUATION MODULE                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────┐                ┌─────────────────────┐
│  LOCAL STORAGE   │                │   GEMINI API (LLM)  │
│  data/           │                │   (external only for│
│  ├── indexes/    │                │    generation)      │
│  ├── chunks/     │                └─────────────────────┘
│  └── documents/  │
└──────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Ingestion Pipeline

**Responsibility:** Transform a raw PDF into persisted, indexed, retrievable chunks.

| Sub-component | Responsibility |
|---|---|
| `pdf_extractor.py` | Open PDF with PyMuPDF; extract text and images per page; record page numbers in metadata; detect extraction quality. |
| `ocr_engine.py` | Triggered when a page's text character count falls below a configurable threshold; uses `pytesseract` + Pillow to render page as image and extract text; preserves page number. |
| `chunker.py` | Split extracted text into overlapping, structure-aware chunks; detect section headings; assign `chunk_id`, `chunk_type`, `section`, `page_number`. |
| `embedder.py` | Wrap `sentence-transformers` model; encode a list of chunk texts; return numpy arrays; model loaded once per process. |
| `index_store.py` | Manage per-document FAISS flat-L2 index; serialize/deserialize index + chunk metadata to `data/indexes/<doc_id>/`; never mix indexes across documents. |

**Ingestion flow (ordered):**
```
PDF bytes → PDFExtractor → [OCR if needed] → pages[]
         → Chunker → chunks[]
         → Embedder → embeddings[]
         → IndexStore.build(doc_id, chunks, embeddings) → persisted to disk
```

### 3.2 Query Pipeline

**Responsibility:** Transform a user question into a grounded, cited answer.

| Sub-component | Responsibility |
|---|---|
| `query_processor.py` | Validate document selection; sanitize query; detect query language. |
| `retrieval_engine.py` | Embed query; search FAISS (top-10) for candidate chunks *within the selected document only*; apply keyword boost for numeric signals and clause keywords; enforce evidence threshold. **Note: cross-encoder reranker is NOT in the MVP.** |
| `evidence_validator.py` | Check retrieved chunks pass minimum quality bar; classify as SUFFICIENT / MARGINAL / INSUFFICIENT / CONFLICTING; decide whether to proceed, abstain, or use conflict-aware prompt. |
| `llm_generator.py` | Construct a strict grounding prompt with evidence text; call Gemini API via `google-genai` SDK (model from `GEMINI_MODEL` env var — never hardcoded); parse structured response. |
| `citation_assembler.py` | Extract page numbers and chunk IDs from used evidence; construct `Citation` objects from real metadata — never from LLM output. |
| `answer_builder.py` | Assemble final `Answer` object from generator response + citations + language label + disclaimer. |

**Query flow (ordered):**
```
user_query + doc_id → QueryProcessor → validated_query
                    → RetrievalEngine → [top-10 FAISS candidates]
                                     → [keyword boost applied]
                                     → [threshold filter]
                    → EvidenceValidator → SUFFICIENT | MARGINAL | INSUFFICIENT | CONFLICTING
                         if INSUFFICIENT  → AbstentionAnswer (no LLM call)
                         if CONFLICTING   → LLMGenerator(evidence, conflict_prompt)
                         if SUFFICIENT/MARGINAL → LLMGenerator(evidence, query) → raw_text
                                   → CitationAssembler(evidence) → citations[]
                                   → AnswerBuilder → Answer
```

### 3.3 Risk Scanner

**Responsibility:** Identify risk-relevant clause categories within a single document.

- Operates directly on indexed chunks of the selected document.
- Applies keyword + semantic search patterns for each risk category:
  - Exclusions
  - Waiting periods
  - Co-payments
  - Sub-limits
  - Important restrictions/conditions
- Each finding is linked to the originating chunk with `page_number` and `section`.
- Does **not** call the LLM; this is a structured extraction pass using retrieval only.
- LLM summarization may be optionally applied per finding, but only with evidence text passed explicitly.

### 3.4 Comparison Engine

**Responsibility:** Compare two policies on a defined set of normalized attributes.

- Accepts exactly two `doc_id` values; never merges their indexes.
- Defines a `COMPARISON_SCHEMA` — a fixed list of attributes (e.g., `sum_insured`, `waiting_period_months`, `exclusions`, `co_payment_pct`, `room_rent_limit`, `network_hospitals`, `pre_existing_disease_coverage`).
- For each attribute, independently runs a targeted retrieval against each document.
- Assembles a `ComparisonResult` with per-document evidence per attribute.
- The LLM is invoked per-attribute with evidence from one document at a time; outputs are then assembled into a structured comparison table — not a free-form paragraph.
- If an attribute cannot be found in a document, that cell is explicitly marked as `NOT_FOUND` rather than omitted or assumed.

### 3.5 LLM Generator (`llm_generator.py`)

**Responsibility:** Generate natural-language answers grounded strictly in supplied evidence.

**Design constraints:**
- Uses the **`google-genai`** SDK (`from google import genai`). **NOT** the legacy
  `google-generativeai` package, which is EOL since November 2025.
- Model name is read from `GEMINI_MODEL` environment variable. It is **never hardcoded**
  in source code. The startup check verifies the model exists via `client.models.get()`.
- Recommended default for `.env.example`: `GEMINI_MODEL=gemini-2.5-flash`
- Evidence text is injected into the prompt verbatim; the LLM is instructed to answer only from that text.
- A special **conflict-aware prompt** is used when `EvidenceStatus.CONFLICTING`: the LLM
  is instructed to present both clauses and describe their relationship — not resolve it.
- The prompt explicitly instructs the model to say "I cannot find this information in the uploaded policy" if the evidence does not contain enough information.
- Prompt-level grounding alone is NOT treated as sufficient. The `EvidenceValidator` acts as an independent pre-generation gate.
- The generator supports two output modes: `STANDARD` and `BEGINNER_FRIENDLY`.
- Language is controlled by a `target_language` parameter (`en` or `ta`); Tamil translation is applied as a post-generation step via a dedicated Gemini call.
- The Tamil translation prompt receives the original evidence text as a reference to prevent numeric/negation errors during translation.
- Retry with exponential backoff on API failures; surfaces clear error messages on repeated failure.

### 3.6 Streamlit Front-End (`app.py`)

**Responsibility:** User-facing interface only; contains no business logic.

**Pages/Panels:**
| Panel | Features |
|---|---|
| Upload | Drag-and-drop PDF; show ingestion progress; display extracted page count and OCR usage stats |
| Q&A | Text input; language toggle; explanation-level toggle; answer display with highlighted citations; source text accordion |
| Risk Scanner | Per-category risk findings table; click to see source page |
| Compare | Two-document selector; attribute-level comparison table; per-cell evidence links |
| Evaluation (dev mode) | Load test dataset; run evaluation; display metrics |

### 3.7 Evaluation Module (`evaluation/`)

**Responsibility:** Measure system quality against a manually verified test dataset.

- Separate from the main application; invoked via CLI or Streamlit dev panel.
- See `EVALUATION_DESIGN.md` for detailed specification.

---

## 4. Module Structure (Planned)

```
insurance_explainer/
├── app.py                          # Streamlit entry point
├── config.py                       # Central config (env vars, thresholds, model names)
├── ingestion/
│   ├── __init__.py
│   ├── pdf_extractor.py
│   ├── ocr_engine.py
│   ├── chunker.py
│   ├── embedder.py
│   └── index_store.py
├── retrieval/
│   ├── __init__.py
│   ├── retrieval_engine.py         # FAISS search + keyword boost (no cross-encoder in MVP)
│   └── evidence_validator.py       # 4-state: SUFFICIENT/MARGINAL/INSUFFICIENT/CONFLICTING
├── generation/
│   ├── __init__.py
│   ├── llm_generator.py            # google-genai SDK; model from GEMINI_MODEL env var
│   ├── citation_assembler.py
│   ├── answer_builder.py
│   └── prompts.py
├── features/
│   ├── __init__.py
│   ├── risk_scanner.py
│   └── comparison_engine.py
├── models/
│   ├── __init__.py
│   └── data_models.py              # All Pydantic models (see DATA_MODEL.md)
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── test_data/
│       └── eval_dataset.json       # Built from real policy PDFs (see EVALUATION_DESIGN.md)
├── ui/
│   ├── __init__.py
│   ├── upload_panel.py
│   ├── qa_panel.py
│   ├── risk_panel.py
│   └── compare_panel.py
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   └── logging_utils.py
├── data/                           # Runtime data (gitignored)
│   ├── indexes/                    # Per-doc FAISS indexes + chunks.json
│   └── documents/                  # Per-doc metadata (meta.json)
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/                   # Test PDF fixtures
│   └── conftest.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 5. Data Flow — End-to-End

### 5.1 Ingestion

```
1. User uploads PDF via Streamlit.
2. app.py calls ingestion.pdf_extractor.extract(pdf_bytes) → List[Page]
3. For each Page where quality_score < threshold:
      ingestion.ocr_engine.extract_page(page_image) → Page (OCR)
4. ingestion.chunker.chunk(pages) → List[Chunk]
5. ingestion.embedder.encode(chunks) → np.ndarray
6. ingestion.index_store.build(doc_id, chunks, embeddings) → persisted
7. Document metadata saved to data/documents/<doc_id>/meta.json
8. UI displays: doc_id, page_count, chunk_count, ocr_page_count
```

### 5.2 Q&A

```
1. User enters query + selects doc_id + selects language + selects explanation mode.
2. query_processor.process(query, doc_id) → validates doc exists; detects lang.
3. retrieval_engine.retrieve(query, doc_id, top_k=20) → List[Chunk] (candidates)
4. reranker.rerank(query, candidates) → List[RetrievalResult] (scored, ordered)
5. evidence_validator.validate(results, threshold) → EvidenceDecision
   IF EvidenceDecision.sufficient == False:
      answer_builder.abstain(reason) → Answer (no citations, no LLM call)
   ELSE:
      evidence = top RetrievalResults within threshold
      llm_generator.generate(query, evidence, lang="en", mode) → raw_text
      citation_assembler.assemble(evidence) → List[Citation]
      IF lang == "ta":
         llm_generator.translate(raw_text, "ta") → tamil_text
      answer_builder.build(raw_text|tamil_text, citations, evidence) → Answer
6. UI renders Answer.text, Answer.citations, Answer.source_chunks, disclaimer.
```

### 5.3 Risk Scan

```
1. User selects doc_id.
2. For each risk_category in RISK_CATEGORIES:
      retrieval_engine.retrieve(category_query, doc_id, top_k=10)
      evidence_validator.validate(results, threshold)
      IF sufficient:
         risk_scanner.extract_finding(category, evidence) → RiskFinding
      ELSE:
         RiskFinding(category, found=False)
3. UI renders structured risk table with page links.
```

### 5.4 Comparison

```
1. User selects doc_id_A and doc_id_B (must be different).
2. comparison_engine.compare(doc_id_A, doc_id_B) → ComparisonResult
   For each attribute in COMPARISON_SCHEMA:
      evidence_A = retrieval_engine.retrieve(attr_query, doc_id_A)
      evidence_B = retrieval_engine.retrieve(attr_query, doc_id_B)
      value_A = llm_generator.extract_attribute(attr, evidence_A) | NOT_FOUND
      value_B = llm_generator.extract_attribute(attr, evidence_B) | NOT_FOUND
      ComparisonRow(attribute, value_A, evidence_A, value_B, evidence_B)
3. UI renders structured comparison table.
```

---

## 6. Document Isolation Guarantees

| Guarantee | Mechanism |
|---|---|
| Separate vector indexes | Each document has its own FAISS index at `data/indexes/<doc_id>/` |
| No cross-document retrieval | `retrieval_engine.retrieve()` requires `doc_id`; FAISS search is performed only on the index loaded for that `doc_id` |
| Chunk ownership | Every `Chunk` carries `doc_id` as a required field; validation at index build time |
| Comparison isolation | Comparison engine invokes retrieval separately for each doc_id; never concatenates chunk lists before retrieval |
| Session state | Streamlit session stores `active_doc_id` per user session; switching documents requires explicit user action |

---

## 7. Citation Integrity Guarantees

| Guarantee | Mechanism |
|---|---|
| Page numbers from metadata only | `Citation.page_number` is populated from `Chunk.page_number` (set during extraction from PyMuPDF/OCR) — never from LLM output |
| LLM output not parsed for citations | `citation_assembler.py` builds citations from the evidence list passed to the LLM, not from the LLM's text response |
| Source text display | The exact chunk text passed as evidence is stored in `Answer.source_chunks` and displayed in the UI |
| Chunk ID traceability | Every citation includes `chunk_id` enabling round-trip verification to the stored chunk |

---

## 8. Grounding and Abstention

- The LLM prompt contains **only** the evidence text blocks, the query, and grounding instructions.
- The `EvidenceValidator` independently evaluates retrieval quality **before** the LLM is called.
- If `EvidenceValidator` decides evidence is insufficient, an `AbstentionAnswer` is returned immediately — the LLM is never called.
- If the LLM is called but the evidence is marginal, the prompt explicitly instructs it to state the information is not clearly present.
- The system does **not** claim that RAG prevents hallucination. The validator + strict grounding prompt together reduce the risk; they are not guaranteed to eliminate it.

---

## 9. Multilingual Design

- Query language is detected automatically (English / Tamil).
- Evidence retrieval is always performed in the source language of the document.
- LLM answer generation is always performed in English first (for grounding clarity).
- Tamil output is generated by a second Gemini call: `translate(english_answer, target="ta")`.
- This separation ensures that Tamil translation does not inadvertently introduce non-grounded content.
- Tamil queries are embedded using the same multilingual embedding model (`paraphrase-multilingual-mpnet-base-v2` or similar) that handles Tamil text.

---

## 10. Security and Reliability Considerations

| Risk | Mitigation |
|---|---|
| Missing GEMINI_API_KEY | `config.py` validates env vars at startup; app shows clear error before any user interaction |
| API failure | Exponential backoff with 3 retries; `llm_generator` raises typed exceptions caught by UI |
| Malformed / empty PDF | `pdf_extractor` wraps PyMuPDF in try/except; returns error result; UI shows user-friendly message |
| Scanned PDF (image-only) | OCR fallback triggered automatically when text extraction yields < threshold characters per page |
| Very large PDF | Ingestion is chunked; configurable page limit; progress bar in UI; streaming embeddings batch |
| Unsupported file type | File type check on upload (MIME type + extension); rejected before processing |
| Prompt injection via PDF content | Evidence text is injected as a clearly demarcated block with system-level framing instructions |
| Data persistence between sessions | Per-user isolation via `doc_id`; indexes are keyed by UUID; no shared mutable state |
| Disk space | Configurable index retention; old indexes pruned after configurable TTL |

---

## 11. Consistency Audit

| Requirement | Architectural Component | Status |
|---|---|---|
| PDF upload | Streamlit upload panel + ingestion pipeline | ✅ |
| Page-number preserving extraction | `pdf_extractor.py` uses PyMuPDF page index | ✅ |
| OCR fallback | `ocr_engine.py` with quality threshold | ✅ |
| Structure-aware chunking | `chunker.py` with section heading detection | ✅ |
| Local embeddings | `embedder.py` with sentence-transformers | ✅ |
| Persistent FAISS indexes | `index_store.py` serialized per doc_id | ✅ |
| Document isolation | Per-doc FAISS + doc_id required everywhere | ✅ |
| Retrieval beyond top-k | Candidate retrieval + reranker + threshold | ✅ |
| Evidence validation | `evidence_validator.py` as pre-LLM gate | ✅ |
| Strictly grounded LLM | Strict prompt + evidence-only validator | ✅ |
| Abstention | `EvidenceDecision.sufficient==False` path | ✅ |
| Page-level citations | `citation_assembler.py` from chunk metadata | ✅ |
| Source text display | `Answer.source_chunks` → UI accordion | ✅ |
| English + Tamil | Detect → generate English → translate | ✅ |
| Beginner-friendly mode | `BEGINNER_FRIENDLY` mode in generator | ✅ |
| Risk scanner | `risk_scanner.py` per category | ✅ |
| Two-policy comparison | `comparison_engine.py` isolated per doc | ✅ |
| Evaluation | `evaluation/` module with test dataset | ✅ |
| Error handling | Typed exceptions + UI error messages | ✅ |
| Disclaimer | `answer_builder.py` always appends disclaimer | ✅ |

All 21 requirements have an identified architectural component.
