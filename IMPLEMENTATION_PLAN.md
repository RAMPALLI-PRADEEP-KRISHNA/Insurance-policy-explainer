# Insurance Policy Explainer — Implementation Plan

> **Version:** 1.0
> **Date:** 2026-10-05

---

## Overview

Development is organized into 8 phases in strict dependency order. No phase begins
until the acceptance criteria of its predecessor are met. Each phase produces working,
tested code — not stubs or placeholders.

---

## Phase 0 — Project Scaffold & Configuration
**Objective:** Establish a clean, reproducible development environment and project structure.

**Files/Modules Created:**
- `requirements.txt`
- `.env.example`
- `config.py`
- `models/__init__.py`
- `models/data_models.py` (all Pydantic models from DATA_MODEL.md)
- `utils/__init__.py`
- `utils/logging_utils.py`
- `utils/file_utils.py`
- `tests/conftest.py`
- `.gitignore`
- `README.md`
- `data/indexes/`, `data/chunks/`, `data/documents/` (empty directories with `.gitkeep`)

**Tests Required:**
- `tests/unit/test_data_models.py` — Pydantic model instantiation and invariant validation
- `tests/unit/test_config.py` — Config loads correctly; raises on missing required env vars

**Acceptance Criteria:**
- [ ] `pip install -r requirements.txt` succeeds.
- [ ] All Pydantic models instantiate without error.
- [ ] Config validation detects missing `GEMINI_API_KEY` and raises `ConfigError`.
- [ ] `pytest tests/unit/` passes with 0 failures.

**Dependencies:** None

---

## Phase 1 — PDF Ingestion Pipeline (Extraction + Chunking)
**Objective:** Reliably extract text from PDFs, apply OCR where needed, and produce
well-formed `Chunk` objects with correct page metadata.

**Files/Modules Created:**
- `ingestion/__init__.py`
- `ingestion/pdf_extractor.py`
- `ingestion/ocr_engine.py`
- `ingestion/chunker.py`

**Key Implementation Details:**
- `pdf_extractor.py`: Uses PyMuPDF (`fitz`). Extracts per-page text and page number (1-indexed). Computes `quality_score` as `(char_count / expected_chars_per_page)` clamped to [0,1]. If `quality_score < OCR_THRESHOLD` (default 0.15), marks page for OCR.
- `ocr_engine.py`: Renders page to image via PyMuPDF, runs pytesseract. Returns OCR text with `extraction_method=OCR`.
- `chunker.py`: Splits text by section headings (regex for ALL CAPS lines or numbered sections). Applies sliding window with configurable `MAX_CHUNK_CHARS` (800) and `OVERLAP_CHARS` (160). Assigns `chunk_id` (UUID4), `chunk_type`, `section`, `page_number`.

**Test Fixtures Required:**
- `tests/fixtures/normal_text.pdf` — A short, selectable-text PDF (2–5 pages)
- `tests/fixtures/scanned_page.pdf` — A PDF with at least one image-only page
- `tests/fixtures/empty.pdf` — A PDF with no text content
- `tests/fixtures/malformed.pdf` — A corrupted/malformed PDF file

**Tests Required:**
- `tests/unit/test_pdf_extractor.py`
  - Normal text extraction returns correct page count
  - Page numbers are 1-indexed
  - Scanned page triggers OCR flag
  - Empty PDF returns empty page list with graceful error
  - Malformed PDF raises `ExtractionError` (not crashes)
- `tests/unit/test_chunker.py`
  - Chunks from a normal page have correct `doc_id`, `page_number`
  - No chunk exceeds `MAX_CHUNK_CHARS`
  - Section heading detected and assigned to chunks in that section
  - Overlap is present between adjacent chunks

**Acceptance Criteria:**
- [ ] `pdf_extractor` correctly extracts text from a normal PDF with correct 1-indexed page numbers.
- [ ] `ocr_engine` successfully extracts text from an image-only page.
- [ ] Malformed PDF returns a descriptive `ExtractionError`, not an unhandled exception.
- [ ] All chunks have `doc_id`, `page_number`, `chunk_id` populated.
- [ ] `pytest tests/unit/test_pdf_extractor.py tests/unit/test_chunker.py` passes.

**Dependencies:** Phase 0

---

## Phase 2 — Embedding & FAISS Index Store
**Objective:** Encode chunks into embeddings and build persistent, per-document FAISS
indexes that can be serialized, deserialized, and queried in isolation.

**Files/Modules Created:**
- `ingestion/embedder.py`
- `ingestion/index_store.py`

**Key Implementation Details:**
- `embedder.py`: Loads `paraphrase-multilingual-mpnet-base-v2` once. Exposes `encode(texts: List[str]) -> np.ndarray`. Batches inputs with configurable `EMBED_BATCH_SIZE` (default 64).
- `index_store.py`:
  - `build(doc_id, chunks, embeddings)`: Creates `IndexFlatL2`, adds L2-normalized embeddings, writes `index.faiss` and `chunks.json` to `data/indexes/<doc_id>/`.
  - `load(doc_id) -> (faiss.Index, List[Chunk])`: Deserializes. Raises `IndexNotFoundError` if missing.
  - `search(doc_id, query_embedding, k) -> List[Tuple[Chunk, float]]`: Loads index, runs FAISS search, returns chunks with scores.
  - `delete(doc_id)`: Removes index directory.
  - LRU cache for loaded indexes (configurable capacity, default 3).

**Tests Required:**
- `tests/unit/test_embedder.py`
  - Output shape matches expected dimension (768)
  - Tamil text produces non-zero embedding
  - Batching works correctly for > EMBED_BATCH_SIZE inputs
- `tests/unit/test_index_store.py`
  - Build → serialize → deserialize → search returns same results
  - Search with doc_id_A returns only chunks with doc_id_A
  - `IndexNotFoundError` raised for unknown doc_id
  - Delete removes all files for that doc_id
- `tests/integration/test_isolation.py`
  - Build two indexes (doc_id_A, doc_id_B)
  - Search with doc_id_A: all returned chunks have doc_id_A
  - Search with doc_id_B: all returned chunks have doc_id_B
  - No cross-contamination possible via any code path

**Acceptance Criteria:**
- [ ] Embedder produces correct shape output for both English and Tamil text.
- [ ] Index build + serialize + deserialize + search round-trip works correctly.
- [ ] Cross-document isolation test passes: searching doc_id_A never returns doc_id_B chunks.
- [ ] `pytest tests/unit/test_embedder.py tests/unit/test_index_store.py tests/integration/test_isolation.py` passes.

**Dependencies:** Phase 1

---

## Phase 3 — Retrieval Pipeline (Keyword Boost + Evidence Validator)
**Objective:** Implement the multi-stage retrieval pipeline that produces validated
`Evidence` objects ready for generation. **No cross-encoder reranker in MVP.**

**Files/Modules Created:**
- `retrieval/__init__.py`
- `retrieval/retrieval_engine.py`
- `retrieval/evidence_validator.py`

**Key Implementation Details:**
- `retrieval_engine.py`:
  - `retrieve(query, doc_id, top_k_candidates, top_k_final) -> Evidence`
  - Embeds query → FAISS IndexFlatIP search (top_k_candidates=10) → keyword boost → threshold filter → EvidenceValidator → Evidence
  - Keyword boost: detects numeric patterns (%, ₹, "days", "months") and risk keywords in query; adds 0.05–0.10 to matching chunk scores
- `evidence_validator.py`:
  - `validate(results: List[RetrievalResult], query: str) -> Evidence`
  - Applies thresholds from `config.py`
  - Sets `EvidenceStatus` with **four states**: SUFFICIENT / MARGINAL / INSUFFICIENT / CONFLICTING
  - Conflict detection: checks top results for contradictory coverage/exclusion signals

**Tests Required:**
- `tests/unit/test_retrieval_engine.py`
  - Top-10 FAISS search returns only chunks from the correct doc_id
  - Keyword boost raises score for chunks with matching numerics
  - No result from doc_id_B appears when searching doc_id_A
- `tests/unit/test_evidence_validator.py`
  - High-scoring results → SUFFICIENT
  - Low-scoring results → INSUFFICIENT
  - Contradictory coverage/exclusion chunks → CONFLICTING
  - Threshold boundary conditions
- `tests/integration/test_retrieval_pipeline.py`
  - End-to-end: index built → query → Evidence returned with correct doc_id
  - All evidence results share the queried doc_id
  - INSUFFICIENT evidence does not call LLM (mock LLM to verify)

**Acceptance Criteria:**
- [ ] Retrieval pipeline returns `Evidence` with all four possible statuses testable.
- [ ] All `Evidence.results` have `chunk.doc_id == requested_doc_id`.
- [ ] A query clearly unrelated to the indexed content returns `INSUFFICIENT`.
- [ ] A query that retrieves contradictory chunks returns `CONFLICTING`.
- [ ] Integration test passes end-to-end with real (small) indexed document.

**Dependencies:** Phase 2

---

## Phase 4 — LLM Generation, Citation Assembly & Answer Building
**Objective:** Implement strictly grounded LLM generation with citation assembly from
evidence metadata, abstention handling, and the disclaimer.

**Files/Modules Created:**
- `generation/__init__.py`
- `generation/prompts.py`
- `generation/llm_generator.py`
- `generation/citation_assembler.py`
- `generation/answer_builder.py`

**Key Implementation Details:**
- `prompts.py`: Defines prompt templates for `STANDARD`, `BEGINNER_FRIENDLY`, `CONFLICT_AWARE`, and `TRANSLATE` modes. Evidence text is injected between `[EVIDENCE START]` and `[EVIDENCE END]` markers. Standard prompt instruction: "If the evidence does not contain the answer, say: 'I cannot find this information in the uploaded policy.'" Conflict-aware prompt: "The following sections may contain qualifying or conflicting clauses. Present both and describe their relationship."
- `llm_generator.py`:
  - `generate(query, evidence, mode, language) -> str`
  - Uses **`google-genai`** SDK: `from google import genai; client = genai.Client(api_key=...)`
  - Model read from `GEMINI_MODEL` environment variable — **never hardcoded**
  - Startup validation: `client.models.get(model_name)` confirms model availability
  - Exponential backoff: 3 retries with 2s, 4s, 8s delays
  - `translate(text, evidence_text, target_lang) -> str` — separate Gemini call; evidence_text provided as grounding reference for Tamil translation
- `citation_assembler.py`:
  - `assemble(evidence: Evidence, answer_id: str) -> List[Citation]`
  - Reads `chunk.page_number` and `chunk.chunk_id` directly — no LLM output parsed
- `answer_builder.py`:
  - `build(query, evidence, generated_text, citations, language, mode) -> Answer`
  - `abstain(query, doc_id, evidence, reason) -> Answer`
  - Always appends disclaimer text (configurable string, cannot be empty)

**Tests Required:**
- `tests/unit/test_prompts.py`
  - Evidence text appears verbatim in constructed prompt
  - Disclaimer instruction present in all prompts
- `tests/unit/test_citation_assembler.py`
  - Citations contain page numbers from chunk metadata, not from generated text
  - Citation count matches number of evidence results used
- `tests/unit/test_answer_builder.py`
  - ABSTENTION answer has empty citations list
  - GROUNDED answer always has non-empty disclaimer
  - ERROR answer has populated error_detail
- `tests/integration/test_generation.py` (mocked LLM)
  - Full pipeline: Evidence → Answer with citations
  - Abstention path: INSUFFICIENT Evidence → AbstentionAnswer without LLM call

**Acceptance Criteria:**
- [ ] Citation page numbers come exclusively from chunk metadata (verified by unit test).
- [ ] Abstention path is exercised and returns correct `AnswerType.ABSTENTION`.
- [ ] Disclaimer is present in all non-error answers.
- [ ] Tamil answer has `english_text` also populated.
- [ ] API retry logic is tested with a mocked failing API.

**Dependencies:** Phase 3

---

## Phase 5 — Risk Scanner & Comparison Engine
**Objective:** Implement the risk clause scanner and two-policy comparison engine,
both reusing the retrieval pipeline from Phase 3.

**Files/Modules Created:**
- `features/__init__.py`
- `features/risk_scanner.py`
- `features/comparison_engine.py`

**Key Implementation Details:**
- `risk_scanner.py`:
  - `scan(doc_id) -> List[RiskFinding]`
  - Uses template queries per `RiskCategory` (from RETRIEVAL_DESIGN.md §9)
  - Merges deduplicated results by `chunk_id`
  - Returns `RiskFinding` with `found=False` for categories with INSUFFICIENT evidence
- `comparison_engine.py`:
  - `compare(doc_id_a, doc_id_b) -> ComparisonResult`
  - Asserts `doc_id_a != doc_id_b`
  - For each attribute in `COMPARISON_SCHEMA`: retrieve from A, retrieve from B separately
  - Generate per-cell value with LLM (evidence from single doc only)
  - Constructs `ComparisonResult` with explicit `NOT_FOUND` cells

**Tests Required:**
- `tests/unit/test_risk_scanner.py`
  - Returns `RiskFinding(found=False)` for category not in document
  - All `source_chunks` in a finding belong to the correct `doc_id`
  - Each category query runs independently
- `tests/unit/test_comparison_engine.py`
  - `compare(A, A)` raises `ValueError`
  - All cells in `policy_a` have evidence from `doc_id_a` only
  - All cells in `policy_b` have evidence from `doc_id_b` only
  - `NOT_FOUND` cell is correctly set for genuinely absent attribute
- `tests/integration/test_comparison.py`
  - Two different indexed documents produce a `ComparisonResult`
  - No cross-document chunk contamination in any cell

**Acceptance Criteria:**
- [ ] Risk scan returns a finding for each category (found or not-found).
- [ ] Comparison correctly isolates retrieval per document per attribute.
- [ ] `compare(A, A)` raises an error.
- [ ] Integration test with two real (small) PDFs passes.

**Dependencies:** Phase 4

---

## Phase 6 — Streamlit Front-End
**Objective:** Build the user-facing Streamlit application that wires all backend
components into a usable interface.

**Files/Modules Created:**
- `app.py`
- `ui/__init__.py`
- `ui/upload_panel.py`
- `ui/qa_panel.py`
- `ui/risk_panel.py`
- `ui/compare_panel.py`

**Key Implementation Details:**
- `app.py`: Initializes config, validates API key, sets up session state.
- `upload_panel.py`: File uploader, triggers ingestion, shows progress spinner, displays `doc_id`, page count, OCR stats.
- `qa_panel.py`: Text input, language toggle (EN/TA), explanation mode toggle (Standard/Beginner), submit button. On answer: renders answer text, citations table (page, section, score), source text accordions, disclaimer.
- `risk_panel.py`: Document selector, scan button, renders risk table grouped by category with page links.
- `compare_panel.py`: Two document selectors (enforces A ≠ B), compare button, renders comparison table with per-cell evidence toggle.
- All UI components use `st.error()` for error states with user-friendly messages.

**Tests Required:**
- No automated UI tests in this phase (Streamlit testing is complex).
- Manual smoke test checklist in `tests/manual_test_checklist.md`.

**Manual Test Checklist:**
- [ ] Upload a normal PDF → ingestion succeeds and doc appears in selector.
- [ ] Upload the same PDF again → duplicate warning shown.
- [ ] Upload a malformed PDF → error message shown, no crash.
- [ ] Ask an in-policy question → grounded answer with citations shown.
- [ ] Ask an out-of-policy question → abstention message shown.
- [ ] Toggle Tamil → Tamil answer shown with English text available.
- [ ] Beginner mode → answer uses simpler language.
- [ ] Risk scan → table with findings per category.
- [ ] Compare two documents → structured table shown.
- [ ] Compare same document → error message shown.
- [ ] Missing API key at startup → clear error shown before any user action.

**Acceptance Criteria:**
- [ ] All items in manual test checklist pass.
- [ ] No unhandled Python exceptions visible to user (all caught and displayed via `st.error`).
- [ ] Disclaimer visible on every answer.

**Dependencies:** Phase 5

---

## Phase 7 — Evaluation Module
**Objective:** Build the evaluation framework and run the first evaluation pass against
the test dataset to establish baseline metrics.

**Files/Modules Created:**
- `evaluation/__init__.py`
- `evaluation/evaluator.py`
- `evaluation/metrics.py`
- `evaluation/test_data/eval_dataset.json` (manually authored, minimum 20 questions to start)
- `evaluation/results/` (gitignored directory for run outputs)

**Key Implementation Details:**
- `evaluator.py`:
  - CLI: `python -m evaluation.evaluator --dataset ... --output ...`
  - Loads eval dataset, runs each question through the full pipeline
  - Stores `EvaluationResult` per question
  - Calls `metrics.py` to compute aggregate scores
- `metrics.py`:
  - Implements D2 (Citation F1), D3 (Recall@5, MRR), D4 (Abstention P/R) automatically
  - D1 ROUGE-L proxy
  - Outputs summary CSV and failure analysis markdown

**Tests Required:**
- `tests/unit/test_metrics.py`
  - Citation F1 computed correctly for a known case
  - MRR computed correctly
  - Abstention precision/recall computed correctly for known TP/FP/FN counts

**Acceptance Criteria:**
- [ ] Evaluator runs end-to-end on the test dataset without crashing.
- [ ] D2, D3, D4 metrics are computed automatically and written to CSV.
- [ ] Failure analysis report lists questions below threshold.
- [ ] First evaluation results are documented in `evaluation/results/baseline_report.md` with honest metric values (no invented numbers).

**Dependencies:** Phase 6

---

## Phase 8 — Hardening, Threshold Tuning & Documentation
**Objective:** Address findings from Phase 7 evaluation, tune retrieval thresholds,
complete README, and perform final consistency check.

**Activities:**
- Analyze Phase 7 failure analysis; adjust `EVIDENCE_HIGH_THRESHOLD`, `EVIDENCE_LOW_THRESHOLD`, `RETRIEVAL_CANDIDATE_K` based on D3/D4 results.
- Add integration tests for any failure patterns discovered in Phase 7.
- Complete `README.md` with setup instructions, usage guide, and architecture summary.
- Run full `pytest` suite and confirm all tests pass.
- Verify that all 12 risks in `RISK_REGISTER.md` are addressed.
- Final audit: re-run the consistency check from ARCHITECTURE.md §11.

**Acceptance Criteria:**
- [ ] All automated tests pass (`pytest` 0 failures).
- [ ] Phase 7 evaluation shows improvement in at least D3 and D4 after threshold tuning.
- [ ] `README.md` is complete with installation, environment setup, and usage instructions.
- [ ] No `TODO` or `FIXME` comments remain in production code.
- [ ] All 21 requirements from the project spec have a verified implementation.

**Dependencies:** Phase 7

---

## Dependency Graph

```
Phase 0 (Scaffold)
    │
    ▼
Phase 1 (Ingestion: Extract + Chunk)
    │
    ▼
Phase 2 (Embedding + FAISS)
    │
    ▼
Phase 3 (Retrieval + Reranker + Validator)
    │
    ▼
Phase 4 (LLM Generation + Citations + Answers)
    │
    ▼
Phase 5 (Risk Scanner + Comparison)
    │
    ▼
Phase 6 (Streamlit UI)
    │
    ▼
Phase 7 (Evaluation)
    │
    ▼
Phase 8 (Hardening + Documentation)
```

---

## Packages to Install (requirements.txt)

```text
# PDF processing
pymupdf>=1.24.0

# OCR
pytesseract>=0.3.10
Pillow>=10.0.0  # already installed (12.3.0)

# Embeddings (multilingual — covers English and Tamil)
sentence-transformers>=3.0.0

# FAISS (CPU version)
faiss-cpu>=1.8.0

# LLM — NEW unified SDK (replaces legacy google-generativeai which is EOL)
google-genai>=1.0.0

# Data validation
pydantic>=2.5.0  # already installed (2.5.3)

# Data analysis / evaluation
pandas>=2.0.0  # already installed (3.0.6)
rouge-score>=0.1.2

# UI
streamlit>=1.40.0

# Environment
python-dotenv>=1.0.0  # already installed (1.2.4)

# Testing
pytest>=8.0.0
pytest-cov>=5.0.0

# Utilities
numpy>=1.26.0  # already installed (1.26.4)
```

> **IMPORTANT:** `google-generativeai` (legacy SDK) is NOT in this list. Use `google-genai` only.

> **Windows prerequisite (not a pip package):**
> Tesseract must be installed separately:
> ```
> winget install UB-Mannheim.TesseractOCR
> ```
> Tamil OCR requires the `tam.traineddata` language pack (downloadable from the
> Tesseract tessdata repository).

> **Already-installed packages:** `pandas`, `pydantic`, `Pillow`, `numpy`,
> `python-dotenv`, and `opencv-python` are present in the environment.
> Confirm compatibility with installed versions before Phase 0 closes.
