# Insurance Policy Explainer — Architecture Review (Adversarial)

> **Version:** 2.0 — Adversarial Review
> **Date:** 2026-10-05
> **Reviewer role:** Adversarial architect — finding weaknesses, not defending prior decisions

---

## A. Issues Discovered

### A1 — CRITICAL: Wrong SDK and Obsolete Model Names

**Finding:** The original architecture specified `google-generativeai>=0.8.0` and model
`gemini-1.5-flash`. Both are now incorrect:

- `google-generativeai` reached end-of-life for active feature development on
  **November 30, 2025**. It is now a legacy library with no new feature additions.
- The replacement SDK is **`google-genai`** (`pip install google-genai`), using the
  `google.genai.Client` pattern instead of `genai.configure()`.
- Model `gemini-1.5-flash` is a legacy model. Gemini is now on the **Gemini 3.x**
  generation. Recommended model identifiers: `gemini-2.5-flash` (fast/cost-effective)
  or `gemini-2.5-pro` (higher quality). **Because model names are moving fast, the
  architecture must not hard-code any specific model string.** Instead:
  - The model name is stored exclusively in `config.py` / `.env.example`.
  - A `GEMINI_MODEL` environment variable controls the model at runtime.
  - A startup validation step calls `client.models.get(model_name)` to verify the
    model exists before the app opens to users.

**Severity:** Critical — using the deprecated SDK will require a rewrite.
**Resolution:** Replace `google-generativeai` with `google-genai` everywhere.

---

### A2 — CRITICAL: Cross-Encoder Reranker is Over-Engineered for This Scale

**Finding:** The proposed cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
adds:
- ~90 MB model download
- A second sentence-transformers inference pass per query (~0.3–1s on CPU for 20
  candidates)
- A new hard dependency on `sentence-transformers`' cross-encoder API
- A score-combination formula with a free parameter (α = 0.7) that requires tuning
  data to justify — and we have none yet
- A fallback mode that produces different behavior from the primary mode, increasing
  test surface

At the scale of a single insurance document (typically < 500 chunks), a well-tuned
vector search plus keyword boosting will produce results that are practically
indistinguishable from cross-encoder reranking in the vast majority of queries.

**The cross-encoder provides clear benefit only when:** vector search retrieves
semantically plausible but not the most relevant chunk among many near-equal
candidates. Insurance documents have strong structural signals (section headings,
numbered clauses) that already differentiate chunks effectively.

**Decision:** **Remove the cross-encoder from the core (MVP) pipeline.**

Replace with:
1. FAISS vector search (top-10 candidates, not top-20)
2. Keyword signal boost (hard regex matching for numeric values, clause keywords)
3. Similarity threshold filter
4. Evidence validator

The cross-encoder is demoted to **OPTIONAL** in Phase 6+ and can be added later
if evaluation (D3 Recall@5) shows retrieval misses.

---

### A3 — CRITICAL: Tamil Generation — Two-Step Architecture Needs Validation

**Finding:** The current design always generates English first, then translates to
Tamil. This introduces a real risk that was not acknowledged: **Tamil insurance
terminology is not always equivalent to a back-translated English term**.

Examples of where translate-from-English fails:
- "Co-payment" → Tamil direct translation may use a phrase a Tamil speaker
  would never use in an insurance context
- Negations in Tamil have different grammar — a two-step translate risks
  changing the polarity of a conditional exclusion
- Numbers embedded in translated sentences may reorder (Tamil is SOV, English is SVO)
  — though numerals themselves are usually safe

**However**, the alternative (direct Tamil generation from grounded English evidence)
has its own risk: the LLM may produce a Tamil sentence that contains a claim not
directly present in the English evidence text, because Tamil reformulation is harder
to audit.

**Resolution:** The two-step approach (English first, then translation) is **retained**
but must be explicitly tested. An A/B evaluation (D8: Direct vs Translated Tamil)
is added to the evaluation plan. The architecture makes the translation step
replaceable without touching the core pipeline.

Additionally: the Tamil translation prompt must include the original evidence text as
a reference so the model can verify numeric accuracy during translation.

---

### A4 — HIGH: Numeric and Table Content Has No Dedicated Handling

**Finding:** The original design treats numbers identically to prose. An insurance
policy's most critical information is often numeric: co-pay 20%, room rent ₹5,000/day,
waiting period 30 days, sub-limit ₹50,000 per procedure, deductible ₹20,000.

Problems under the current design:
- A chunk containing a table row "20% co-payment for all non-network claims" may be
  embedded as prose and retrieved semantically — fine.
- But if the chunk is split mid-table, the 20% value is orphaned from its context.
- When the LLM receives two chunks — one with "20% co-payment" and one with
  "only for non-network hospitals" — as separate evidence blocks with no joining
  context, it may incorrectly combine or misstate the condition.
- The evaluation has no metric that specifically checks numeric accuracy.

**Resolutions:**
1. **Chunker**: Tables must never be split mid-row. A table is identified by PyMuPDF's
   table detection and emitted as a single `TABLE` chunk, regardless of size (up to a
   configurable max). `ChunkType.TABLE_ROW` is replaced by `ChunkType.TABLE` (whole
   table as one chunk).
2. **Prompt**: When evidence contains a chunk typed as `TABLE`, the prompt explicitly
   instructs the LLM: "This chunk is a structured table. Read all values carefully.
   Do not paraphrase numeric values or percentages."
3. **Evaluation**: A new dimension **D8: Numeric Accuracy** is added, covering
   questions with numeric expected answers. Scoring: exact match for percentages and
   rupee amounts (with ±0 tolerance for percentages, ±5% for rounded rupee values).

---

### A5 — HIGH: Chunking Strategy Lacks Explicit Table and List Protection

**Finding:** The original 800-character sliding window chunker will split tables and
bulleted lists mid-item. Insurance policies heavily use both. This is not just a UX
issue — a list item that defines an exclusion condition may be orphaned from its
category header, making it semantically meaningless when retrieved in isolation.

**Resolutions:**
1. Detect tables using `page.find_tables()` from PyMuPDF 1.24+; emit the entire table
   as a `TABLE` chunk.
2. Detect list items (regex: lines starting with `•`, `-`, `(a)`, `(i)`, numbers
   followed by `.`) and group them with their immediately preceding heading.
3. Respect section boundaries as hard split points: never let a chunk cross from one
   numbered section into another.
4. Cross-page clause continuations: if a sentence starts on page N and ends on page
   N+1, it stays in one chunk. The chunk records `page_numbers = [N, N+1]` and
   `page_number = N` (start page). **This is already in the model** — confirmed as correct.

---

### A6 — HIGH: Evidence Validator Does Not Handle Conflicting Evidence

**Finding:** The original design defines three evidence states (SUFFICIENT, MARGINAL,
INSUFFICIENT) but does not address the case where retrieved chunks **contradict each
other**. In insurance policies this is common:

- Section 4.1: "Maternity benefit is covered."
- Section 8.3 (exclusions): "Maternity complications arising from pre-existing
  conditions are excluded."

If both chunks are retrieved, the LLM receives contradictory evidence. The current
design passes both to the LLM without flagging the conflict. The LLM may silently
resolve the contradiction by omitting one clause.

**Resolution:** Add `CONFLICTING` as a fourth `EvidenceStatus` state. The evidence
validator checks for semantic contradiction signals (keywords: "excluded", "not
covered", "except", "provided that" appearing alongside positive coverage statements
for the same topic). When `CONFLICTING`, the prompt explicitly tells the LLM: "The
retrieved sections contain potentially qualifying or conflicting clauses. Present both
and note the relationship between them. Do not resolve the conflict."

---

### A7 — HIGH: "NOT_FOUND" vs "NOT_MENTIONED" Conflation in Comparison

**Finding:** The comparison engine marks a cell `NOT_FOUND` when retrieval returns
insufficient evidence. But "insufficient retrieval evidence" is not the same as
"the attribute is definitively absent from the policy." There are at least three
distinct states:

1. **Value extracted**: The attribute was found and a specific value retrieved.
2. **Topic present, value ambiguous**: Chunks about the topic were found but no
   specific value could be extracted (e.g., "co-payment may vary").
3. **Not mentioned**: No relevant chunks found — the topic may be absent or may be
   described using different terminology.
4. **Explicitly excluded**: A chunk explicitly states "this benefit is not available".

Conflating states 2, 3, 4 into `NOT_FOUND` is materially wrong for insurance advice.
The user cannot distinguish "the policy is silent about maternity" from "the policy
explicitly excludes maternity."

**Resolution:** Expand `ComparisonCellStatus`:
```python
class ComparisonCellStatus(str, Enum):
    FOUND            = "FOUND"             # Value extracted with evidence
    AMBIGUOUS        = "AMBIGUOUS"         # Topic found; value unclear
    NOT_MENTIONED    = "NOT_MENTIONED"     # No relevant chunks found
    EXPLICITLY_ABSENT = "EXPLICITLY_ABSENT" # Chunk explicitly states not covered/not applicable
```

The UI must display these states with distinct visual treatment and tooltip explanations.

---

### A8 — MEDIUM: Document Quality Check Is Underspecified

**Finding:** The extraction design mentions quality thresholds but does not define
a structured quality-gate output. Currently the system might silently ingest a
document where 80% of pages are OCR failures and then produce answers with
very-low-confidence citations.

**Resolution:** Define an explicit `ExtractionQuality` enum at the document level:

```python
class DocumentQuality(str, Enum):
    GOOD          = "GOOD"          # ≥ 90% pages with native text; quality_score ≥ 0.7
    WARNING       = "WARNING"       # 70–89% pages good; user warned before querying
    OCR_REQUIRED  = "OCR_REQUIRED"  # < 70% pages had native text; OCR was applied
    POOR          = "POOR"          # OCR applied but mean quality_score < 0.3
    FAILED        = "FAILED"        # Cannot extract usable text; ingestion blocked
```

For `POOR` quality, the system still ingests but the UI shows a persistent banner:
"⚠ This document has poor text quality. Answers may be unreliable. Check the source
PDF before relying on any response."

For `FAILED`, ingestion is blocked and a clear error is shown.

---

### A9 — MEDIUM: Score Combination Formula is Unjustified

**Finding (now moot after A2):** The formula `α * rerank_score + (1-α) * vector_score`
with α = 0.7 has no empirical basis in this project. Since the cross-encoder is being
removed from the MVP, this formula is removed entirely.

The simplified pipeline uses: `final_score = normalized_L2_similarity_score + keyword_boost`
where `keyword_boost ∈ {0.0, 0.05, 0.10}`. This is simpler, transparent, and tunable.

---

### A10 — MEDIUM: Evaluation Dataset Target of 94 Questions is Premature

**Finding:** The original design set a target of 94 questions across 11 categories
before any real policy PDFs are available. This is a number invented to look thorough.

**Problems:**
- Category counts (15 coverage, 10 exclusions, etc.) were not derived from actual
  policy document analysis.
- Tamil questions (10 listed) require a Tamil-proficient reviewer to write and verify —
  this dependency is not surfaced as a blocker.
- Some categories (claims procedure, multi-hop) were listed in the review requirements
  but not in the original EVALUATION_DESIGN.md.

**Resolution:**
- The evaluation dataset is built **from actual policy documents** after they are
  available. The structure and schema are defined now; the question count is not.
- Add missing question categories: `claims_procedure`, `multi_hop`, `numeric`,
  `table_based`, `ambiguous`.
- Set a **minimum viable evaluation set** of 20 manually verified questions to
  validate the pipeline before Phase 7 full evaluation.
- Tamil questions are gated on reviewer availability (human decision required).

---

### A11 — MEDIUM: `data/chunks/` Directory Is Unused

**Finding:** The planned directory structure includes `data/chunks/` but the design
stores chunks in `data/indexes/<doc_id>/chunks.json`. The `data/chunks/` directory
serves no purpose and will cause confusion.

**Resolution:** Remove `data/chunks/` from the planned directory structure. Chunks
are stored only alongside their index.

---

### A12 — LOW: `google-generativeai>=0.8.0` version pin is wrong

**Finding (corollary of A1):** Even if we were keeping the old SDK, `0.8.0` is an
extremely old version. The correct package is `google-genai` (new unified SDK).

---

### A13 — LOW: faiss-cpu version compatibility on Python 3.12 / Windows

**Finding:** `faiss-cpu` has historically had installation issues on Windows with
newer Python versions. The `faiss-cpu` package on PyPI generally works with Python
3.12 on Windows as of 2025, but it should be tested explicitly in Phase 0.

**Resolution:** Add a Phase 0 acceptance criterion: `import faiss; faiss.IndexFlatL2(768)` 
succeeds without error. If it fails, the fallback is `faiss-cpu` from conda or
using `numpy`-based brute-force search (acceptable given document scale).

---

### A14 — LOW: No `.gitignore` and No Git Initialization Defined

**Finding:** The workspace has no `.gitignore` and Git is not initialized. The
implementation plan mentions `.gitignore` as a Phase 0 deliverable but does not
specify its contents. The following must be explicitly gitignored:
- `data/` (indexes, documents — contains user-uploaded private policy content)
- `.env` (API keys)
- `__pycache__/`
- `.venv/`
- `evaluation/results/` (run outputs)
- `*.pyc`

---

### A15 — LOW: `ocr_page_ids` Uses 0-Indexed in Document Model but 1-Indexed Elsewhere

**Finding:** `Document.ocr_page_ids: List[int]` is documented as "0-indexed" while
`Page.page_number` is 1-indexed. This inconsistency will cause bugs when the UI
displays "OCR was used on pages: [0, 3, 7]" when it should display "[1, 4, 8]".

**Resolution:** `ocr_page_ids` is changed to store **1-indexed page numbers**, consistent
with all other user-facing page references.

---

## B. Decisions Retained

| Decision | Rationale for Retention |
|---|---|
| Plain Python modules, no LangChain/LlamaIndex | Every pipeline stage must be explicit and testable. Framework abstractions would obscure citation and isolation logic. |
| Per-document FAISS flat (exact) index | Insurance documents are small enough that exact search is appropriate. Cross-document isolation is structural. |
| `paraphrase-multilingual-mpnet-base-v2` for embeddings | Handles both English and Tamil in the same vector space. No alternative was identified that is simpler and equally capable. |
| English-first generation, then Tamil translation | Retained but with the addition of: evidence text provided to the translation prompt as a grounding reference, and a D8 A/B evaluation to validate the approach. |
| Pydantic v2 for all data models | Correct choice. Already installed, well-maintained, strict validation. |
| `IndexFlatIP` (inner product / cosine) vs `IndexFlatL2` | **Changed**: Use `IndexFlatIP` with L2-normalized embeddings (equivalent to cosine similarity). This is more standard for sentence-transformer embeddings than L2 distance. |
| Evidence validator as pre-LLM gate | Critical to grounding. Retained and enhanced with `CONFLICTING` state. |
| Citation assembly from metadata only | Non-negotiable. Retained. |
| `data/indexes/<doc_id>/` structure | Clean isolation. Retained. |
| Streamlit for UI | Appropriate for a single-user demo application. Not over-engineered. |
| pytest for testing | Correct. Already standard. |

---

## C. Decisions Changed

| Original Decision | Changed Decision | Reason |
|---|---|---|
| `google-generativeai>=0.8.0` | `google-genai` (new unified SDK) | Old SDK is EOL since Nov 2025 |
| `gemini-1.5-flash` hardcoded | `GEMINI_MODEL` env var; startup validation of model name | Model names change frequently |
| Cross-encoder reranker in MVP | **Removed from MVP**; optional in Phase 6+ | Unjustified complexity at this scale |
| 20 FAISS candidates | 10 FAISS candidates | Fewer needed without reranking |
| α=0.7 score combination | Removed | Cross-encoder removed; no combination needed |
| `ComparisonCellStatus.NOT_FOUND` only | Four-state: FOUND / AMBIGUOUS / NOT_MENTIONED / EXPLICITLY_ABSENT | Material distinction for insurance domain |
| `EvidenceStatus` 3 states | 4 states: SUFFICIENT / MARGINAL / INSUFFICIENT / **CONFLICTING** | Conflicting clauses must be surfaced |
| `ChunkType.TABLE_ROW` | `ChunkType.TABLE` (whole table) | Tables must not be split mid-row |
| `ocr_page_ids` 0-indexed | 1-indexed | Consistency with all other page references |
| `data/chunks/` directory | Removed | Unused; chunks stored with index |
| 94-question eval target | Build from real documents; 20-question minimum viable set | Premature to specify count without documents |
| No document-quality gate output | `DocumentQuality` enum added to Document model | Silent ingestion of bad PDFs is dangerous |
| BM25 mentioned as "Phase 6 planned" | **Removed from plan** | No evidence it's needed; adds complexity |

---

## D. Decisions Deferred Until Testing

| Decision | Deferred Until |
|---|---|
| Whether cross-encoder reranker improves D3 Recall@5 meaningfully | Phase 7 evaluation; add if Recall@5 < 0.75 |
| Whether 5 or 10 FAISS candidates is optimal | Threshold tuning in Phase 8 |
| Whether direct Tamil generation outperforms translate-from-English | D8 A/B evaluation in Phase 7 |
| Exact evidence thresholds (0.65 HIGH, 0.40 LOW) | Phase 8 tuning after first evaluation run |
| Whether BM25 keyword search is needed as a complement to dense retrieval | Phase 7 evaluation; only if D3 Recall@5 is unacceptably low |
| Optimal chunk size (currently 800 chars) | Phase 7/8 based on retrieval quality with real documents |

---

## E. Final MVP Architecture

**MVP = Must work reliably end-to-end with demonstrable citations and abstention.**

```
INGESTION:
  PDF → PyMuPDF extraction (text + table detection)
     → OCR (pytesseract) for low-quality pages
     → Chunker (section/table/list aware; no mid-table splits)
     → sentence-transformers (multilingual) → embeddings
     → FAISS IndexFlatIP (per-document) → persisted

QUERY:
  query + doc_id
     → Embed query (same model)
     → FAISS search top-10 candidates (doc-isolated)
     → Keyword boost (numeric values, clause keywords)
     → Evidence Validator (SUFFICIENT / MARGINAL / INSUFFICIENT / CONFLICTING)
        ↓ INSUFFICIENT → AbstentionAnswer (no LLM call)
        ↓ CONFLICTING  → LLM with conflict-aware prompt
        ↓ SUFFICIENT   → LLM with grounding prompt
     → Gemini API (google-genai SDK, model from env var)
     → Citation Assembler (from chunk metadata only)
     → Answer (with disclaimer)

TAMIL:
  English Answer → Translation prompt (evidence provided as reference)
     → Tamil Answer (displayed)
     → English Answer (stored for evaluation)

UI (Streamlit):
  Upload | Q&A | Risk Scanner | Compare
```

**MVP Phases:**
- Phase 0: Scaffold, config, models
- Phase 1: Extraction + chunking (with table protection)
- Phase 2: Embedding + FAISS
- Phase 3: Retrieval + evidence validation (no cross-encoder)
- Phase 4: LLM generation + citations + abstention
- Phase 5: UI (Streamlit)
- Phase 6: Risk scanner + comparison
- Phase 7: Evaluation

---

## F. Final Optional Architecture (Post-MVP)

| Feature | Complexity | Benefit |
|---|---|---|
| Cross-encoder reranker | Medium | Better retrieval on ambiguous queries |
| BM25 sparse retrieval | Medium | Exact keyword match complement |
| Tamil direct generation (A/B test) | Low | Potentially more natural Tamil output |
| Async ingestion with progress bar | Low | Better UX for large PDFs |
| Index TTL and cleanup | Low | Disk management |
| Multi-user session isolation | High | Not needed for single-user desktop app |
| PDF password protection handling | Low | Informational error only |

---

## G. Final Dependency List

| Package | Version | Purpose | Status |
|---|---|---|---|
| `pymupdf` | `>=1.24.0` | PDF extraction + table detection | New install required |
| `pytesseract` | `>=0.3.10` | OCR fallback | New install required |
| `Pillow` | `>=10.0.0` | Image rendering for OCR | **Already installed** (12.3.0) |
| `sentence-transformers` | `>=3.0.0` | Multilingual embeddings | New install required |
| `faiss-cpu` | `>=1.8.0` | Vector index | New install required |
| `google-genai` | `>=1.0.0` | Gemini API (new unified SDK) | New install required |
| `pydantic` | `>=2.5.0` | Data validation | **Already installed** (2.5.3) |
| `pandas` | `>=2.0.0` | Evaluation/reporting | **Already installed** (3.0.6) |
| `rouge-score` | `>=0.1.2` | D1 ROUGE proxy metric | New install required |
| `streamlit` | `>=1.40.0` | UI | New install required |
| `python-dotenv` | `>=1.0.0` | Env var loading | **Already installed** (1.2.4) |
| `numpy` | `>=1.26.0` | Array operations | **Already installed** (1.26.4) |
| `pytest` | `>=8.0.0` | Testing | New install required |
| `pytest-cov` | `>=5.0.0` | Coverage reporting | New install required |

**Removed from original list:** `google-generativeai` (EOL; replaced by `google-genai`)

**Not added:** LangChain, LlamaIndex, BM25 libraries, cloud OCR services, Redis, FastAPI
(none are needed for the MVP)

**Note on tesseract:** `pytesseract` requires the **Tesseract binary** to be installed
separately on Windows. This is a Phase 0 prerequisite that must be documented in README:
```
winget install UB-Mannheim.TesseractOCR
```
Tamil OCR specifically requires the `tam.traineddata` language pack for Tesseract.

---

## H. Exact Phase 1 Prerequisites

Before Phase 1 (PDF Extraction + Chunking) implementation begins:

1. ✅ Python 3.12.10 confirmed available.
2. ✅ Git 2.55 confirmed available.
3. ⬜ `pip install pymupdf pytesseract sentence-transformers faiss-cpu google-genai streamlit rouge-score pytest pytest-cov` succeeds.
4. ⬜ Tesseract binary installed on Windows (`winget install UB-Mannheim.TesseractOCR`).
5. ⬜ Tesseract Tamil language pack installed.
6. ⬜ `GEMINI_API_KEY` is available (human action required).
7. ⬜ `import faiss; faiss.IndexFlatIP(768)` succeeds (faiss-cpu Windows compatibility verified).
8. ⬜ `import fitz; fitz.__version__` returns ≥ 1.24.0 (table detection API available).
9. ⬜ `.gitignore` created and committed.
10. ⬜ Git repository initialized (`git init`).
11. ⬜ At least 1 real insurance policy PDF available for testing (human action required).
12. ⬜ Phase 0 acceptance criteria all pass (model instantiation, config validation).

---

## I. Human Decisions Still Required

| Decision | Options | Impact if Deferred |
|---|---|---|
| **Which Gemini model to use** | `gemini-2.5-flash` (recommended: cost-effective, fast) vs a Pro tier model | Affects API cost, response quality, and latency. The model string must go in `.env.example` before Phase 4. |
| **Gemini API key provisioning** | Personal key vs project key | Required before Phase 4 can run end-to-end. Must never be committed to Git. |
| **Real policy PDFs for testing** | Provide 2–3 real (or synthetic) insurance PDFs | Required before Phase 1 fixtures can be created. Evaluation dataset cannot be written without them. |
| **Tamil evaluation reviewer** | Must be a Tamil-proficient person | Required before Tamil evaluation (D7, D8) can run. If not available, Tamil features are tested manually without formal scoring. |
| **Tesseract installation** | Windows binary install via winget | Required before Phase 1 OCR tests can pass. |
| **Page limit for large PDFs** | 150 pages (conservative) vs 300 pages | Affects what documents the system accepts. Default: 300 pages with a warning for > 150. |
| **"Not mentioned" UI wording** | e.g., "Not mentioned in this policy" vs "Could not be confirmed" | Legal/UX choice. Must not imply exclusion when merely absent. |

---

*This review does not begin implementation. It supersedes the Phase 0 architecture
for all subsequent phases.*
