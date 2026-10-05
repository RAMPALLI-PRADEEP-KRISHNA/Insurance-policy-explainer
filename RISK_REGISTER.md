# Insurance Policy Explainer — Risk Register

> **Version:** 2.0 (Post-Adversarial Review)
> **Date:** 2026-10-05

---

## 1. Risk Classification

Each risk is rated on:
- **Likelihood:** Low / Medium / High
- **Impact:** Low / Medium / High / Critical
- **Priority = Likelihood × Impact** (from 1 to 9)

---

## 2. Technical Risk Register

### R01 — Hallucination Despite Grounding Prompt
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | Critical |
| **Priority** | 6/9 |
| **Description** | The LLM generates a claim not present in the supplied evidence, even with a strict grounding prompt. |
| **Why it matters** | Users may make insurance decisions based on incorrect information. |
| **Mitigation** | (1) Independent evidence validator gates LLM call. (2) Grounding prompt explicitly instructs abstention. (3) D6 evaluation dimension measures hallucination rate. (4) Citation assembler uses evidence metadata, not LLM output — so at minimum the citations are trustworthy even if text wanders. |
| **Residual risk** | Prompt-level grounding can be defeated by sufficiently creative user queries. Regular evaluation runs are needed to catch regressions. |

---

### R02 — OCR Quality Failure on Scanned PDFs
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | High |
| **Priority** | 6/9 |
| **Description** | Tesseract OCR fails to extract meaningful text from low-quality scanned pages. The text may be garbled, missing, or in the wrong language. |
| **Why it matters** | Many Indian insurance policy scans are low-resolution. Garbage OCR text corrupts chunks and produces poor embeddings. |
| **Mitigation** | (1) Quality score threshold for OCR acceptance. (2) Log OCR confidence per page. (3) Display OCR warning to user with affected page count. (4) Page quality score stored in Page model for post-hoc diagnosis. |
| **Residual risk** | Very low-quality scans may remain unrecoverable without better OCR tooling (e.g., Google Cloud Vision). This is noted in the UI. |

---

### R03 — Gemini API Outage or Rate Limiting
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | High |
| **Priority** | 3/9 |
| **Description** | The Gemini API is unavailable, rate-limited, or returns unexpected errors during answer generation or translation. |
| **Mitigation** | (1) Exponential backoff with 3 retries. (2) Typed exception hierarchy with UI-friendly messages. (3) Generation and translation are decoupled — a translation failure produces English-only output rather than a total failure. (4) API key validation at app startup. |
| **Residual risk** | Extended outages will prevent answer generation. Ingestion and retrieval remain functional. |

---

### R04 — Document Isolation Breach
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | Critical |
| **Priority** | 3/9 |
| **Description** | Information from one user's uploaded document contaminates retrieval for another document (or another user's session). |
| **Why it matters** | Privacy violation; incorrect citations. |
| **Mitigation** | (1) Separate FAISS index file per doc_id. (2) `retrieval_engine.retrieve()` requires explicit doc_id and loads only that index. (3) Chunk objects carry doc_id as a required field with validation at index build time. (4) Integration test specifically verifies cross-document retrieval isolation. |
| **Residual risk** | Low, provided the index loading and doc_id validation are correctly implemented. A unit test must verify this. |

---

### R05 — Invented Page Numbers in Citations
| Attribute | Value |
|---|---|
| **Likelihood** | Low (with current design) |
| **Impact** | Critical |
| **Priority** | 3/9 |
| **Description** | Page numbers in citations are incorrect, pointing users to wrong pages. |
| **Why it matters** | Destroys trust in the citation system; users cannot verify answers. |
| **Mitigation** | (1) Citation assembler reads page numbers from Chunk.page_number (set during PDF extraction), never from LLM text. (2) LLM output is not parsed for page numbers. (3) D2 citation correctness evaluation specifically measures this. |
| **Residual risk** | If PDF extractor assigns wrong page numbers (e.g., off-by-one), citations will be consistently wrong. This is a single point of failure — the extractor's page numbering logic must be unit tested with multiple PDF fixtures. |

---

### R06 — Chunking Destroys Context
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Priority** | 4/9 |
| **Description** | The chunker splits a semantically coherent clause across two chunks. Neither chunk alone answers the query; retrieval misses the full context. |
| **Mitigation** | (1) Chunk overlap (configurable; default 20% of max chunk size). (2) Section-aware chunking prefers to split at section boundaries. (3) If retrieval returns adjacent chunks, the evidence assembler may merge them before passing to LLM. |
| **Residual risk** | Some clause fragmentation is inevitable. The overlap strategy reduces but cannot eliminate this risk. |

---

### R07 — Tamil Embedding Quality
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Priority** | 4/9 |
| **Description** | The multilingual embedding model may not represent Tamil insurance vocabulary accurately. Tamil queries may not retrieve the correct English chunks. |
| **Mitigation** | (1) Multilingual model chosen specifically for Tamil support. (2) D3 evaluation measures retrieval quality for Tamil queries. (3) Tamil query expansion may be added in Phase 6 if D3 scores are poor. |
| **Residual risk** | Technical Tamil terminology may not appear in training data for the embedding model. |

---

### R08 — Very Large PDF Performance
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | Medium |
| **Priority** | 2/9 |
| **Description** | A user uploads a 500-page PDF (e.g., combined policy + brochure + terms). Ingestion takes minutes; the UI appears frozen. |
| **Mitigation** | (1) Configurable page limit with user warning. (2) Async ingestion with progress reporting. (3) Embedding batching to avoid OOM. |
| **Residual risk** | Page limit may frustrate users with legitimately long documents. |

---

### R09 — Reranker Model Download Failure
### R09 — Optional Reranker Download/Latency Failure (Post-MVP)
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | Low |
| **Priority** | 1/9 |
| **Description** | If the optional cross-encoder reranker is enabled post-MVP, model download may fail or add unwanted latency on CPU. |
| **Mitigation** | Cross-encoder was removed from the MVP pipeline in favor of exact FAISS search and keyword boosting. If enabled post-MVP, graceful fallback to vector score is automatic. |

---

### R10 — Evaluation Dataset Contamination
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | Medium |
| **Priority** | 4/9 |
| **Description** | The evaluation dataset uses the same documents that were used during development tuning. Results are optimistic and do not generalize. |
| **Mitigation** | (1) Evaluation documents are held out during threshold tuning. (2) At least one document in the eval set must be one that was never used during development. (3) Document on this constraint in EVALUATION_DESIGN.md. |

---

### R11 — Comparison Attribute Extraction & Conflation Failure
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | High |
| **Priority** | 6/9 |
| **Description** | Conflating absence of mention with explicit exclusion (e.g. interpreting unmentioned maternity benefits as "not covered"). |
| **Mitigation** | (1) Replaced single `NOT_FOUND` with 4 explicit states: `FOUND`, `AMBIGUOUS`, `NOT_MENTIONED`, and `EXPLICITLY_ABSENT`. (2) The system is architecturally forbidden from converting `NOT_MENTIONED` into `EXPLICITLY_ABSENT`. (3) Verbatim evidence text and citations are attached to every cell. |

---

### R12 — Disclaimer Bypass / Misuse
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | High |
| **Priority** | 3/9 |
| **Description** | A user treats the application's answers as authoritative financial advice and makes a poor insurance decision. |
| **Mitigation** | (1) Disclaimer is appended to every answer by `answer_builder.py` — it cannot be disabled. (2) Disclaimer is prominently displayed in the UI header. (3) Application is described as "informational only" in README and on-screen. |

---

### R13 — Numeric Value & Table Clause Context Loss
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | Critical |
| **Priority** | 6/9 |
| **Description** | An insurance table or sub-limit clause (e.g. 20% co-pay, ₹5,000 room rent) is split across chunks, severing the number from its condition. |
| **Mitigation** | (1) Chunker identifies tables via PyMuPDF table detection and preserves the whole table as a single `TABLE` chunk (no mid-row splitting). (2) Keyword boosting elevates numeric matches during retrieval. (3) D8 evaluation dimension tests numeric accuracy specifically. |

---

### R14 — Conflicting or Qualifying Clauses Concealed
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | High |
| **Priority** | 6/9 |
| **Description** | Policy states coverage in one section but specifies an exclusion or condition in another; LLM blindly picks one. |
| **Mitigation** | (1) `EvidenceValidator` includes a dedicated `CONFLICTING` status when contradictory coverage/exclusion keywords co-occur in top chunks. (2) Specialized conflict-aware prompt forces the model to explain both clauses without resolving them. |

---

### R15 — System Tesseract Binary Missing on Windows
| Attribute | Value |
|---|---|
| **Likelihood** | Medium |
| **Impact** | High |
| **Priority** | 6/9 |
| **Description** | `pytesseract` Python library is installed but the underlying Windows `tesseract.exe` executable is missing from `PATH`. |
| **Mitigation** | (1) Clear setup instructions in README (`winget install UB-Mannheim.TesseractOCR`). (2) `ocr_engine.py` validates `tesseract` binary at module initialization and falls back gracefully with a clear setup notification instead of crashing. |

---

### R16 — LLM SDK Deprecation & Model Identifier Drift
| Attribute | Value |
|---|---|
| **Likelihood** | Low |
| **Impact** | High |
| **Priority** | 3/9 |
| **Description** | Reliance on deprecated `google-generativeai` package or hardcoded deprecated model strings (e.g., `gemini-1.5-flash`). |
| **Mitigation** | (1) Adopted modern `google-genai` SDK. (2) Model identifier parameterized via `GEMINI_MODEL` with startup validation probe via `client.models.get()`. |

---

## 3. Risk Summary Matrix

| ID | Risk | Likelihood | Impact | Priority |
|---|---|---|---|---|
| R01 | Hallucination despite grounding | Medium | Critical | 6 |
| R02 | OCR quality failure | Medium | High | 6 |
| R03 | Gemini API failure | Low | High | 3 |
| R04 | Document isolation breach | Low | Critical | 3 |
| R05 | Invented page numbers | Low | Critical | 3 |
| R06 | Chunking destroys context | Medium | Medium | 4 |
| R07 | Tamil embedding quality | Medium | Medium | 4 |
| R08 | Large PDF performance | Low | Medium | 2 |
| R09 | Optional reranker download failure | Low | Low | 1 |
| R10 | Evaluation dataset contamination | Medium | Medium | 4 |
| R11 | Comparison attribute conflation | Medium | High | 6 |
| R12 | Disclaimer bypass / misuse | Low | High | 3 |
| R13 | Numeric/Table clause context loss | Medium | Critical | 6 |
| R14 | Conflicting clauses concealed | Medium | High | 6 |
| R15 | Tesseract Windows binary missing | Medium | High | 6 |
| R16 | LLM SDK/Model identifier drift | Low | High | 3 |

---

## 4. Risks Deliberately Not Mitigated (and Why)

| Risk | Reason not mitigated |
|---|---|
| Multi-user concurrency | This is a single-user desktop Streamlit app; concurrent session isolation is not required in Phase 1. |
| PDF password protection | Out of scope for initial release; blocked PDFs return a clear error message. |
| Adversarial prompt injection via PDF | Partial mitigation only (evidence demarcation in prompt). Full mitigation is disproportionate for an informational tool. |
| Accuracy on non-insurance PDFs | Out of scope; the application is domain-specific and documents this clearly. |
