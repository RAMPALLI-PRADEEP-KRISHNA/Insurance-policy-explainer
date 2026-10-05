"""OCR fallback engine detecting and utilizing Tesseract OCR when native extraction is insufficient."""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pytesseract
from PIL import Image
import pymupdf

from src.config import config
from src.utils.logging_utils import logger
from src.utils.exceptions import TesseractNotFoundError, OCRError


# Common Windows installation directories to probe if tesseract is not in system PATH
WINDOWS_DEFAULT_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


class OCREngine:
    """Manages OCR availability detection and page image text recognition."""

    def __init__(self, custom_cmd: Optional[str] = None):
        self.tesseract_cmd: Optional[str] = None
        self._initialize_tesseract_cmd(custom_cmd or config.TESSERACT_CMD)

    def _initialize_tesseract_cmd(self, configured_path: Optional[str]) -> None:
        """Resolve and configure the tesseract executable path."""
        # 1. Configured path
        if configured_path and Path(configured_path).is_file():
            self.tesseract_cmd = str(Path(configured_path).resolve())
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            logger.info("Using configured Tesseract path: %s", self.tesseract_cmd)
            return

        # 2. System PATH
        which_path = shutil.which("tesseract")
        if which_path:
            self.tesseract_cmd = which_path
            pytesseract.pytesseract.tesseract_cmd = which_path
            logger.info("Found Tesseract in system PATH: %s", which_path)
            return

        # 3. Known Windows directories
        for probe_path in WINDOWS_DEFAULT_TESSERACT_PATHS:
            if probe_path.is_file():
                self.tesseract_cmd = str(probe_path.resolve())
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                logger.info("Discovered Tesseract in default Windows location: %s", self.tesseract_cmd)
                return

        self.tesseract_cmd = None
        logger.warning(
            "Tesseract OCR is not installed or not in PATH. OCR fallback for scanned pages will be disabled."
        )

    def is_available(self) -> bool:
        """Check whether Tesseract is installed and executable."""
        if not self.tesseract_cmd:
            return False
        try:
            version = pytesseract.get_tesseract_version()
            return bool(version)
        except Exception as e:
            logger.debug("Tesseract check failed: %s", e)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status information for Tesseract OCR."""
        available = self.is_available()
        return {
            "available": available,
            "executable_path": self.tesseract_cmd,
            "install_instructions": (
                "Install Tesseract OCR on Windows via PowerShell:\n"
                "  winget install UB-Mannheim.TesseractOCR\n"
                "Or set TESSERACT_CMD in your .env file to the tesseract.exe path."
                if not available
                else "Tesseract is installed and ready."
            ),
        }

    def ocr_page(
        self,
        page: pymupdf.Page,
        dpi: int = 300,
        lang: str = "eng",
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Render a PDF page to an image and perform OCR.

        Returns:
            Tuple of (extracted_text, success_bool, optional_warning_or_error)
        """
        if not self.is_available():
            warning_msg = (
                f"Page {page.number + 1} requires OCR, but Tesseract OCR is not available on this system. "
                "Text could not be extracted via OCR fallback. "
                "To enable OCR, install Tesseract: winget install UB-Mannheim.TesseractOCR"
            )
            logger.warning(warning_msg)
            return "", False, warning_msg

        try:
            # Render page to pixmap at specified DPI
            pix = page.get_pixmap(dpi=dpi)
            # Convert pixmap bytes to PIL Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Perform OCR extraction
            text = pytesseract.image_to_string(img, lang=lang)
            cleaned_text = text.strip()

            if not cleaned_text:
                return "", False, f"OCR executed on page {page.number + 1} but yielded no text."

            logger.info("OCR successfully extracted %d characters from page %d", len(cleaned_text), page.number + 1)
            return cleaned_text, True, None

        except Exception as e:
            error_msg = f"OCR failed on page {page.number + 1}: {str(e)}"
            logger.error(error_msg)
            return "", False, error_msg


# Default singleton instance
ocr_engine = OCREngine()
