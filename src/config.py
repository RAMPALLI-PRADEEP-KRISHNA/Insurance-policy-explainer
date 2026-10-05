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

    # Ingestion & PDF Limits
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "300"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    # OCR Settings
    OCR_THRESHOLD: float = float(os.getenv("OCR_THRESHOLD", "0.15"))
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD", None)

    # Chunking Configuration (Phase 2 preview)
    MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "800"))
    OVERLAP_CHARS: int = int(os.getenv("OVERLAP_CHARS", "160"))

    # Retrieval Thresholds (Phase 3 preview)
    EVIDENCE_HIGH_THRESHOLD: float = float(os.getenv("EVIDENCE_HIGH_THRESHOLD", "0.65"))
    EVIDENCE_LOW_THRESHOLD: float = float(os.getenv("EVIDENCE_LOW_THRESHOLD", "0.40"))

    # LLM Settings (Phase 4 preview - no client is initialized here)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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


config = AppConfig()
config.ensure_directories()
