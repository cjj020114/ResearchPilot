from __future__ import annotations

from pathlib import Path

from backend.app.core.config import settings
from backend.app.ingestion.types import PdfMode


def detect_pdf_mode(path: Path) -> tuple[PdfMode, str]:
    """Decide digital vs OCR from average extractable text density."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "ocr", "pypdf unavailable; default to OCR mode"

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        return "ocr", f"pdf open failed ({exc}); default to OCR mode"

    if not reader.pages:
        return "ocr", "empty pdf; default to OCR mode"

    total_chars = 0
    for page in reader.pages:
        total_chars += len((page.extract_text() or "").strip())
    density = total_chars / max(len(reader.pages), 1)
    if density < settings.pdf_scan_text_density_threshold:
        return (
            "ocr",
            f"low text density={density:.1f} < {settings.pdf_scan_text_density_threshold}; OCR mode",
        )
    return "digital", f"extractable text density={density:.1f}; digital mode"
