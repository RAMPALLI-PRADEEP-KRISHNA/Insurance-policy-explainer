"""Central configuration module for Insurance Policy Explainer."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


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
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "300"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    # OCR Settings
    OCR_THRESHOLD: float = float(os.getenv("OCR_THRESHOLD", "0.15"))
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD", None)

    # Chunking Configuration
    MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "800"))
    OVERLAP_CHARS: int = int(os.getenv("OVERLAP_CHARS", "160"))

    # Vector Store & Embedding Configuration
    VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "qdrant").lower()
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "insurance_policy_chunks")
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "snowflake/snowflake-arctic-embed-xs")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    INDEX_SCHEMA_VERSION: str = "2"

    # Retrieval Configuration (Phase 3)
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "10"))
    RETRIEVAL_FINAL_K: int = int(os.getenv("RETRIEVAL_FINAL_K", "5"))
    MIN_EVIDENCE_RESULTS: int = int(os.getenv("MIN_EVIDENCE_RESULTS", "2"))
    EVIDENCE_HIGH_THRESHOLD: float = float(os.getenv("EVIDENCE_HIGH_THRESHOLD", "0.65"))
    EVIDENCE_LOW_THRESHOLD: float = float(os.getenv("EVIDENCE_LOW_THRESHOLD", "0.40"))
    KEYWORD_BOOST_NUMERIC: float = float(os.getenv("KEYWORD_BOOST_NUMERIC", "0.10"))
    KEYWORD_BOOST_RISK: float = float(os.getenv("KEYWORD_BOOST_RISK", "0.10"))
    KEYWORD_BOOST_TABLE: float = float(os.getenv("KEYWORD_BOOST_TABLE", "0.05"))
    MAX_KEYWORD_BOOST: float = float(os.getenv("MAX_KEYWORD_BOOST", "0.15"))


    # LLM Provider Selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mistral").lower()
    LLM_FALLBACK_PROVIDER: str = os.getenv("LLM_FALLBACK_PROVIDER", "gemini").lower()

    # Gemini Settings (Phase 4)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_TIMEOUT_SECONDS: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
    LLM_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "15"))
    LLM_TOTAL_DEADLINE_SECONDS: int = int(os.getenv("LLM_TOTAL_DEADLINE_SECONDS", "30"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

    # Mistral Settings
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY", None)
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")


    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

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

