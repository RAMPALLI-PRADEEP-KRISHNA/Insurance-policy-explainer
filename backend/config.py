"""Central configuration module for Insurance Policy Explainer."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve setting from Streamlit secrets (Cloud priority) or environment variables (Local)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            s_val = st.secrets[key]
            if s_val is not None and str(s_val).strip() != "":
                return str(s_val)
    except Exception:
        pass

    val = os.getenv(key)
    if val is not None and str(val).strip() != "":
        return val

    return default


class AppConfig:
    """Application configuration settings."""

    # Project directories
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    INDEXES_DIR: Path = DATA_DIR / "indexes"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    QDRANT_STORAGE_PATH: Path = DATA_DIR / "qdrant_db"

    # Ingestion & PDF Limits
    MAX_PDF_PAGES: int = int(_get_setting("MAX_PDF_PAGES", "300"))
    MAX_FILE_SIZE_MB: int = int(_get_setting("MAX_FILE_SIZE_MB", "50"))

    # OCR Settings
    OCR_THRESHOLD: float = float(_get_setting("OCR_THRESHOLD", "0.15"))
    OCR_DPI: int = int(_get_setting("OCR_DPI", "300"))
    TESSERACT_CMD: Optional[str] = _get_setting("TESSERACT_CMD", None)

    # Chunking Configuration
    MAX_CHUNK_CHARS: int = int(_get_setting("MAX_CHUNK_CHARS", "800"))
    OVERLAP_CHARS: int = int(_get_setting("OVERLAP_CHARS", "160"))

    # Vector Store & Embedding Configuration
    VECTOR_STORE_BACKEND: str = _get_setting("VECTOR_STORE_BACKEND", "qdrant").lower()
    QDRANT_COLLECTION_NAME: str = _get_setting("QDRANT_COLLECTION_NAME", "insurance_policy_chunks")
    QDRANT_URL: Optional[str] = _get_setting("QDRANT_URL", None)
    QDRANT_API_KEY: Optional[str] = _get_setting("QDRANT_API_KEY", None)
    EMBEDDING_BACKEND: str = _get_setting("EMBEDDING_BACKEND", "fastembed").lower()
    EMBEDDING_MODEL_NAME: str = _get_setting("EMBEDDING_MODEL_NAME", "snowflake/snowflake-arctic-embed-xs")
    EMBEDDING_DIM: int = int(_get_setting("EMBEDDING_DIM", "384"))
    EMBEDDING_BATCH_SIZE: int = int(_get_setting("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_DEVICE: str = _get_setting("EMBEDDING_DEVICE", "cpu")
    INDEX_SCHEMA_VERSION: str = "2"

    # Retrieval Configuration (Phase 3)
    RETRIEVAL_CANDIDATE_K: int = int(_get_setting("RETRIEVAL_CANDIDATE_K", "10"))
    RETRIEVAL_FINAL_K: int = int(_get_setting("RETRIEVAL_FINAL_K", "5"))
    MIN_EVIDENCE_RESULTS: int = int(_get_setting("MIN_EVIDENCE_RESULTS", "2"))
    EVIDENCE_HIGH_THRESHOLD: float = float(_get_setting("EVIDENCE_HIGH_THRESHOLD", "0.65"))
    EVIDENCE_LOW_THRESHOLD: float = float(_get_setting("EVIDENCE_LOW_THRESHOLD", "0.40"))
    KEYWORD_BOOST_NUMERIC: float = float(_get_setting("KEYWORD_BOOST_NUMERIC", "0.10"))
    KEYWORD_BOOST_RISK: float = float(_get_setting("KEYWORD_BOOST_RISK", "0.10"))
    KEYWORD_BOOST_TABLE: float = float(_get_setting("KEYWORD_BOOST_TABLE", "0.05"))
    MAX_KEYWORD_BOOST: float = float(_get_setting("MAX_KEYWORD_BOOST", "0.15"))


    # LLM Provider Selection
    LLM_PROVIDER: str = _get_setting("LLM_PROVIDER", "mistral").lower()
    LLM_FALLBACK_PROVIDER: str = _get_setting("LLM_FALLBACK_PROVIDER", "gemini").lower()

    # Gemini Settings (Phase 4)
    GEMINI_API_KEY: Optional[str] = _get_setting("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = _get_setting("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_TIMEOUT_SECONDS: int = int(_get_setting("GEMINI_TIMEOUT_SECONDS", "15"))
    LLM_REQUEST_TIMEOUT_SECONDS: int = int(_get_setting("LLM_REQUEST_TIMEOUT_SECONDS", "15"))
    LLM_TOTAL_DEADLINE_SECONDS: int = int(_get_setting("LLM_TOTAL_DEADLINE_SECONDS", "30"))
    GEMINI_TEMPERATURE: float = float(_get_setting("GEMINI_TEMPERATURE", "0.0"))

    # Mistral Settings
    MISTRAL_API_KEY: Optional[str] = _get_setting("MISTRAL_API_KEY", None)
    MISTRAL_MODEL: str = _get_setting("MISTRAL_MODEL", "ministral-8b-latest")


    # Logging
    LOG_LEVEL: str = _get_setting("LOG_LEVEL", "INFO").upper()

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure runtime directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        cls.INDEXES_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.ensure_directories()

