# Insurance Policy Explainer

A RAG-based application that helps you understand insurance policy documents in plain language — with verifiable page-level citations, a risk clause scanner, and two-policy comparison.

> ⚠️ **Disclaimer:** This application is for informational purposes only. It is not professional insurance, legal, or financial advice. Always verify answers directly in the original policy document and consult a licensed insurance advisor for decisions.

---

## Features

- **Upload & Parse** — Upload insurance policy PDFs; scanned documents are handled via OCR fallback.
- **Ask Questions** — Ask questions in English; answers are grounded strictly in the uploaded policy.
- **Page-Level Citations** — Every answer includes verifiable page numbers sourced from document metadata.
- **Source Text Display** — See the exact policy text used to generate each answer.
- **Controlled Abstention** — When the policy doesn't contain enough information, the system explicitly abstains instead of hallucinating.
- **Red-Flag Scanner** — Automatically identifies exclusions, waiting periods, co-payments, room rent limits, and key restrictions.
- **Policy Comparison** — Compare two policies on a structured set of attributes side by side.
- **Evaluation Suite** — Built-in evaluation framework with citation correctness, retrieval quality, and abstention metrics.

---

## Architecture Overview

The application is built from explicit Python modules — no LangChain or LlamaIndex.

```
Upload → PDF Extraction (PyMuPDF + OCR fallback) → Chunking (table/section preserved)
       → Embedding (FastEmbed Arctic Embed XS) → Persistent Local Qdrant Index

Query  → Embedding → Vector Search (top-10) → Keyword Boost (numeric & clause signals)
       → Evidence Validation (4-state) → Mistral/Gemini LLM Provider Chain
       → Citation Assembly (metadata only) → Answer (grounded + disclaimer)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architectural design.

---

## Stack

| Component | Technology |
|---|---|
| Frontend UI | Streamlit |
| PDF Extraction | PyMuPDF with table detection |
| OCR Fallback | pytesseract + Pillow (requires system Tesseract) |
| Embeddings | FastEmbed (`snowflake/snowflake-arctic-embed-xs` via ONNX Runtime) |
| Vector Database | Qdrant Client (persistent local disk storage) |
| Scoring/Boost | Keyword signal boosting (numerics, risk terms, tables) |
| LLM Provider | Mistral API (primary) with Google Gemini fallback (30s bounded chain) |
| Data Models | Pydantic v2 |
| Evaluation | Deterministic evaluation suite with ground-truth citations |
| Tests | pytest + pytest-cov (193 tests) |

---

## Repository Structure

```text
Insurance-policy-explainer/
├── frontend/                         # User Interface Layer
│   ├── __init__.py
│   └── app.py                        # Streamlit web application & UI components
├── backend/                          # Core Application Services & Business Logic
│   ├── __init__.py
│   ├── config.py                     # Central configuration & environment loader
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py            # Canonical Pydantic v2 data models
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py          # PyMuPDF text & page extraction
│   │   ├── ocr_engine.py             # Tesseract OCR fallback engine
│   │   ├── chunker.py                # Structure-aware insurance chunking
│   │   ├── embedder.py               # FastEmbed Arctic-XS ONNX embedder
│   │   ├── header_footer.py          # Header and footer trimming
│   │   ├── quality_checker.py        # Page & document quality gates
│   │   └── table_extractor.py        # Table detection & markdown formatter
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py              # Dense vector retrieval & keyword boosting
│   │   ├── evidence_validator.py     # 4-state evidence validation
│   │   └── numeric_utils.py          # Numeric & limit preservation
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_provider.py           # Provider protocol interface
│   │   ├── mistral_client.py         # Mistral client (primary)
│   │   ├── gemini_client.py          # Gemini client (fallback)
│   │   ├── provider_chain.py         # Bounded-deadline provider orchestration
│   │   ├── answer_generator.py       # Grounded answer synthesis (7 safety rules)
│   │   └── citation_assembler.py     # Authoritative citation assembler
│   ├── comparison/
│   │   ├── __init__.py
│   │   ├── comparison_engine.py      # Two-policy comparison matrix engine
│   │   ├── evidence_extractor.py     # Clause extraction for comparisons
│   │   └── normalizer.py             # Terminology normalization
│   ├── risk_scanner/
│   │   ├── __init__.py
│   │   └── red_flag_scanner.py       # Evidence-grounded red-flag scanner
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py             # Typed exception hierarchy
│       ├── file_utils.py             # PDF validation & hashing
│       ├── logging_utils.py          # Structured logging
│       └── timing.py                 # Latency profiling utilities
├── database/                         # Vector Storage & Index Persistence
│   ├── __init__.py
│   ├── index_store.py                # Index persistence coordinator
│   └── qdrant_store.py               # Persistent local Qdrant vector database
├── docs/                             # Architecture & Design Specifications
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_REVIEW.md
│   ├── DATA_MODEL.md
│   ├── EVALUATION_DESIGN.md
│   ├── EVALUATION_REPORT.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── RETRIEVAL_DESIGN.md
│   └── RISK_REGISTER.md
├── data/                             # Storage Directories
│   ├── sample_policies/              # Sample policy PDF wordings
│   ├── documents/                    # Document metadata store
│   ├── indexes/                      # Vector index metadata
│   ├── qdrant_db/                    # Qdrant persistent storage
│   └── uploads/                      # Temporary user uploads
├── reports/                          # Benchmark & Quality Gate Reports
│   ├── benchmark_baseline_report.json
│   ├── benchmark_candidate_report.json
│   └── quality_gate_report.json
├── scripts/                          # Utility & Validation Scripts
│   ├── benchmark_baseline.py
│   ├── benchmark_candidate.py
│   ├── evaluate_embedding_quality_gate.py
│   └── validate_arctic_embed.py
├── tests/                            # Automated Test Suite (193 tests)
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── evaluation/                   # Deterministic evaluation runner & cases
│   └── fixtures/                     # Test fixtures
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Dependencies
└── README.md
```

---

## Environment Setup & Requirements

### System Requirements:
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** 3.12.10

### 1. Windows Tesseract OCR (Optional Prerequisite for Scanned PDFs)
Native text extraction works out of the box. For scanned PDFs requiring OCR:
```powershell
winget install UB-Mannheim.TesseractOCR
```

### 2. Virtual Environment Setup
```powershell
# Create isolated virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```powershell
copy .env.example .env
```
Configure your `MISTRAL_API_KEY` and optionally `GEMINI_API_KEY` in `.env`.

---

## Running the Application

Start the Streamlit web application:
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Running Automated Tests

Run the full test suite (193 tests):
```powershell
.\.venv\Scripts\pytest.exe
```

Run unit tests only:
```powershell
.\.venv\Scripts\pytest.exe tests/unit/
```

Run integration tests:
```powershell
.\.venv\Scripts\pytest.exe tests/integration/
```
