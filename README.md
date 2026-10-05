# Insurance Policy Explainer

A RAG-based application that helps you understand insurance policy documents in
plain language — with verifiable page-level citations, a risk clause scanner, and
two-policy comparison.

> ⚠️ **Disclaimer:** This application is for informational purposes only. It is
> not professional insurance, legal, or financial advice. Always verify answers
> directly in the original policy document and consult a licensed insurance advisor
> for decisions.

---

## Features

- **Upload & Parse** — Upload insurance policy PDFs; scanned documents are handled via OCR fallback.
- **Ask Questions** — Ask questions in English or Tamil; answers are grounded strictly in the uploaded policy.
- **Page-Level Citations** — Every answer includes verifiable page numbers sourced from document metadata.
- **Source Text Display** — See the exact policy text used to generate each answer.
- **Abstention** — When the policy doesn't contain enough information, the system says so instead of guessing.
- **Beginner Mode** — Simplified explanations for complex insurance terminology.
- **Risk Scanner** — Automatically identifies exclusions, waiting periods, co-payments, sub-limits, and key restrictions.
- **Policy Comparison** — Compare two policies on a structured set of attributes side by side.
- **Evaluation** — Built-in evaluation framework with citation correctness, retrieval quality, and abstention metrics.

---

## Architecture Overview

The application is built from explicit Python modules — no LangChain or LlamaIndex.

```
Upload → PDF Extraction (PyMuPDF + OCR fallback) → Chunking (table/section preserved)
       → Embedding (sentence-transformers) → Per-document FAISS Index (IndexFlatIP)

Query  → Embedding → FAISS Search (top-10) → Keyword Boost (numeric & clause signals)
       → Evidence Validation (4-state) → Gemini API (google-genai)
       → Citation Assembly (metadata only) → Answer (grounded + disclaimer)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

---

## Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Extraction | PyMuPDF (fitz) with table detection |
| OCR Fallback | pytesseract + Pillow (requires system Tesseract) |
| Embeddings | sentence-transformers (`paraphrase-multilingual-mpnet-base-v2`) |
| Vector Search | FAISS CPU (`IndexFlatIP`, cosine similarity via normalized vectors) |
| Scoring/Boost | Keyword signal boosting (numerics, risk terms; cross-encoder optional post-MVP) |
| LLM | Google Gemini API via official **`google-genai`** SDK |
| Data Models | Pydantic v2 |
| Evaluation | pandas + ROUGE + custom multi-dimensional metrics |
| Tests | pytest + pytest-cov |

---

---

## Project Status

> **Current Phase:** Phase 1 Completed (Project Foundation + Document Ingestion)
> **Next Phase:** Phase 2 (Embedding & FAISS Index Store)

### Implemented Capabilities (Phase 1):
- **Deterministic PDF Ingestion:** Validates PDF byte header, detects file corruption, prevents password-protected file crashes, enforces page count and file size limits.
- **Page-Aware Text Extraction:** Preserves exact 1-indexed page numbers and builds canonical `Page` records.
- **Table Detection & Preservation:** Uses PyMuPDF's table extraction engine to identify tables, detect headers/rows, and render clean markdown representations.
- **Deterministic Quality Gate:** Evaluates character counts, alphanumeric density, whitespace ratio, and garbage characters to classify pages and documents into `GOOD`, `WARNING`, `OCR_REQUIRED`, `POOR`, or `FAILED`.
- **Safe OCR Fallback:** Detects presence of Tesseract binary explicitly; attempts OCR fallback for image-only or low-yield pages; warns cleanly with installation commands if Tesseract is not installed without crashing the pipeline.
- **Document Identification & Isolation:** Generates immutable UUID4 `doc_id` and SHA-256 `file_hash` per document; saves metadata to isolated document storage.
- **Conservative Header/Footer Analysis:** Distinguishes repeated boilerplate from substantive operative policy clauses.

---

## Project Structure

```
Insurance policy explainer/
├── src/
│   ├── config.py                 # Centralized configuration & environment loader
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py        # Canonical Pydantic v2 data models
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py      # Core PDF ingestion pipeline
│   │   ├── ocr_engine.py         # Tesseract OCR fallback engine
│   │   ├── quality_checker.py    # Page and document quality gates
│   │   ├── table_extractor.py    # PyMuPDF table detection & markdown formatter
│   │   └── header_footer.py      # Non-destructive header/footer analyzer
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py         # Typed exception hierarchy
│       ├── file_utils.py         # File hashing & PDF validation
│       └── logging_utils.py      # Safe structured logger
├── tests/
│   ├── conftest.py               # Synthetic PDF fixture generators
│   ├── fixtures/
│   │   └── sample_policy_fixture.pdf
│   ├── unit/
│   │   ├── test_data_models.py
│   │   ├── test_pdf_extractor.py
│   │   ├── test_ocr_engine.py
│   │   ├── test_quality_checker.py
│   │   ├── test_table_extractor.py
│   │   ├── test_header_footer.py
│   │   └── test_file_utils.py
│   └── integration/
│       └── test_ingestion_pipeline.py
├── data/
│   ├── uploads/                  # Temporary upload area
│   ├── documents/                # Ingested document metadata
│   └── indexes/                  # Vector index storage (Phase 2)
├── logs/                         # Application logs
├── .env.example                  # Environment variable template
├── .gitignore                    # Excludes secrets, virtual environments, and data
├── requirements.txt              # Phase 1 minimal dependencies
└── README.md
```

---

## Environment Setup & Requirements

### System Requirements:
- **Operating System:** Windows (or Linux/macOS)
- **Python Version:** 3.12.10

### 1. Windows Tesseract OCR (Optional System Prerequisite)
Native text extraction works out of the box. For scanned PDFs requiring OCR, install Tesseract:
```powershell
winget install UB-Mannheim.TesseractOCR
```

### 2. Virtual Environment Setup
```powershell
# Create isolated virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Install Phase 1 dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```powershell
copy .env.example .env
```
*(No API keys are required for Phase 1. Keys will be configured in Phase 4 for LLM generation.)*

---

## Running Automated Tests

Run the full test suite with coverage reporting:
```powershell
.\.venv\Scripts\pytest.exe -v --cov=src --cov-report=term-missing
```

Run a specific test module:
```powershell
.\.venv\Scripts\pytest.exe tests/unit/test_pdf_extractor.py -v
```

Run an ingestion smoke test on a sample PDF fixture:
```powershell
.\.venv\Scripts\python.exe -c "from src.ingestion.pdf_extractor import process_pdf; doc = process_pdf('tests/fixtures/sample_policy_fixture.pdf'); print(f'Ingested: {doc.filename} | Pages: {doc.page_count} | Quality: {doc.quality.value}')"
```

